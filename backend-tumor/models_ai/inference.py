"""
models_ai/inference.py

Inference engine TensorFlow/Keras untuk RSU U²-Net+ (Attention Gate).
100% konsisten dengan implementasi training di Google Colab:
- Data format: channels_first (B, 4, 160, 160, 16)
- Urutan modalitas: [img_t2f, img_t1n, img_t1c, img_t2w]
- Standarisasi per slice Z pada setiap modalitas
- Loading weights langsung tanpa remapping/scrambling
- Output: 3 channel sigmoid [0: NETC, 1: Edema, 2: ET]
"""

import os
import time
from typing import Tuple, Dict

import numpy as np
import nibabel as nib
import tensorflow as tf

from models_ai.u2net_model import build_u2net

# Cache loaded model instance
_model_instance = None
_predict_fn = None

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "checkpoints",
    "model_weights.weights.h5"
)

# Urutan modalitas PERSIS seperti di Colab (urutan file numerik):
# img_t2f = _0000, img_t1n = _0001, img_t1c = _0002, img_t2w = _0003
# mri_4ch_raw = np.stack([_0000, _0001, _0002, _0003], axis=0)
MODALITY_MAPPING = [
    "_0000",  # Index 0: t1c (di Colab dinamai img_t2f)
    "_0001",  # Index 1: t1n (di Colab dinamai img_t1n)
    "_0002",  # Index 2: t2f (di Colab dinamai img_t1c)
    "_0003",  # Index 3: t2w (di Colab dinamai img_t2w)
]


def load_u2net_model():
    global _model_instance, _predict_fn
    if _model_instance is not None:
        return _model_instance

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH}")

    tf.config.optimizer.set_jit(False)
    num_threads = max(2, os.cpu_count() or 4)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, num_threads // 2))

    print(f"[MODEL] Loading TensorFlow RSU U²-Net+ (channels_first) dari {CHECKPOINT_PATH}...")
    model = build_u2net(input_shape=(4, 160, 160, 16), num_classes=3)
    model.load_weights(CHECKPOINT_PATH)
    print(f"[MODEL] Bobot berhasil dimuat secara presisi 100% ({model.count_params():,} parameters).")

    _model_instance = model

    @tf.function(reduce_retracing=True)
    def predict_step(x):
        return model(x, training=False)

    _predict_fn = predict_step
    return model


def standarisasi(image: np.ndarray) -> np.ndarray:
    """
    Standarisasi persis seperti kode training Colab:
    Per slice Z untuk setiap channel modalitas: (image_slice - mean) / std.
    Input image shape: (4, X, Y, Z)
    """
    standarisasi_image = np.zeros(image.shape, dtype=np.float32)
    for c in range(image.shape[0]):
        for z in range(image.shape[3]):
            image_slice = image[c, :, :, z]
            std = np.std(image_slice)
            if std > 0:
                standarisasi_image[c, :, :, z] = (image_slice - np.mean(image_slice)) / std
            else:
                standarisasi_image[c, :, :, z] = 0.0
    return standarisasi_image


def load_and_stack_modalities(input_dir: str, case_id: str) -> Tuple[np.ndarray, dict]:
    """
    Memuat 4 modalitas NIfTI dan menyusunnya ke order [t2f, t1n, t1c, t2w] (4, X, Y, Z).
    """
    vols = []
    first_nii = None

    for suffix in MODALITY_MAPPING:
        file_path = os.path.join(input_dir, f"{case_id}{suffix}.nii.gz")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File modalitas tidak ditemukan: {file_path}")
        nii = nib.load(file_path)
        if first_nii is None:
            first_nii = nii
        data = nii.get_fdata().astype(np.float32)
        vols.append(data)

    stacked_volume = np.stack(vols, axis=0)  # Shape: (4, X, Y, Z)

    meta = {
        "affine": first_nii.affine,
        "header": first_nii.header,
        "shape": first_nii.shape
    }

    return stacked_volume, meta


def predict_full_volume(
    model,
    mri_4ch_raw: np.ndarray,
    patch_size=(160, 160, 16),
    num_classes=3,
    overlap=0.5,
    batch_size=4
) -> np.ndarray:
    """
    Sliding window prediction persis seperti kode Colab:
    mri_4ch_raw: array (4, X, Y, Z) mentah
    Standarisasi dilakukan PER PATCH di dalam loop.
    Returns: pred_full (3, X, Y, Z)
    """
    global _predict_fn
    _, X, Y, Z = mri_4ch_raw.shape
    px, py, pz = patch_size

    pad_x = max(0, px - X)
    pad_y = max(0, py - Y)
    pad_z = max(0, pz - Z)
    if pad_x > 0 or pad_y > 0 or pad_z > 0:
        padded_vol = np.pad(mri_4ch_raw, ((0, 0), (0, pad_x), (0, pad_y), (0, pad_z)), mode='constant')
    else:
        padded_vol = mri_4ch_raw

    _, X_p, Y_p, Z_p = padded_vol.shape

    stride_x = max(1, int(px * (1 - overlap)))
    stride_y = max(1, int(py * (1 - overlap)))
    stride_z = pz  # Stride 16 di sumbu Z untuk efisiensi CPU

    pred_full = np.zeros((num_classes, X_p, Y_p, Z_p), dtype=np.float32)
    count_full = np.zeros((1, X_p, Y_p, Z_p), dtype=np.float32)

    def get_starts(total, patch, stride):
        starts = list(range(0, max(total - patch, 0) + 1, stride))
        if not starts or starts[-1] + patch < total:
            starts.append(max(total - patch, 0))
        return sorted(set(starts))

    x_starts = get_starts(X_p, px, stride_x)
    y_starts = get_starts(Y_p, py, stride_y)
    z_starts = get_starts(Z_p, pz, stride_z)

    patch_coords = []
    for sx in x_starts:
        for sy in y_starts:
            for sz in z_starts:
                p_check = padded_vol[:, sx:sx+px, sy:sy+py, sz:sz+pz]
                if np.any(p_check > 0.05):
                    patch_coords.append((sx, sy, sz))

    total_patches = len(patch_coords)
    print(f"[INFERENCE] Total patch yang akan diproses: {total_patches}")

    t_start = time.perf_counter()

    for idx, (sx, sy, sz) in enumerate(patch_coords):
        patch_raw = padded_vol[:, sx:sx+px, sy:sy+py, sz:sz+pz]
        patch_std = standarisasi(patch_raw)
        patch_batch = np.expand_dims(patch_std, axis=0)  # (1, 4, 160, 160, 16)

        pred = model(patch_batch, training=False).numpy()[0]  # (3, 160, 160, 16)

        pred_full[:, sx:sx+px, sy:sy+py, sz:sz+pz] += pred
        count_full[:, sx:sx+px, sy:sy+py, sz:sz+pz] += 1.0

        elapsed = time.perf_counter() - t_start
        print(f"  ...patch {idx + 1}/{total_patches} selesai ({elapsed:.1f}s)")

    count_full[count_full == 0] = 1.0
    pred_full = (pred_full / count_full)[:, :X, :Y, :Z]
    return pred_full


def save_raw_binary_predictions(output_dir: str, case_id: str, prob_map: np.ndarray, threshold: float = 0.5):
    """
    Simpan 3 file NIfTI biner (netc.nii.gz, edema.nii.gz, et.nii.gz) serta raw_prob.npz.
    Sesuai ground truth training BraTS & Colab:
    - Channel 0: NETC (Class 1, Magenta)
    - Channel 1: Peritumoral Edema (Class 2, Yellow, massa pembengkakan luas)
    - Channel 2: Enhancing Tumor / ET (Class 3, Cyan, nodul tumor aktif)
    """
    os.makedirs(output_dir, exist_ok=True)
    ch_mapping = {
        "netc": 0,
        "edema": 1,
        "et": 2,
    }
    for name, idx in ch_mapping.items():
        bin_mask = (prob_map[idx] > threshold).astype(np.uint8)
        img = nib.Nifti1Image(bin_mask, np.eye(4))
        nib.save(img, os.path.join(output_dir, f"{case_id}_{name}.nii.gz"))

    # raw_bin [0: NETC, 1: Edema, 2: ET] untuk metrik evaluasi
    netc_bin = (prob_map[0] > threshold).astype(np.uint8)
    edema_bin = (prob_map[1] > threshold).astype(np.uint8)
    et_bin = (prob_map[2] > threshold).astype(np.uint8)
    raw_bin = np.stack([netc_bin, edema_bin, et_bin], axis=0)
    np.savez_compressed(os.path.join(output_dir, f"{case_id}_raw_prob.npz"), raw_bin=raw_bin)


def convert_prob_map_to_multiclass(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Mengonversi 3-channel probabilitas biner model menjadi single multiclass label map (0, 1, 2, 3)
    100% PERSIS seperti arsitektur training & tampilan Google Colab:
    - Model Channel 0 -> Class 1: NETC (Magenta, inti nekrotik)
    - Model Channel 1 -> Class 2: Peritumoral Edema (Yellow, massa pembengkakan luas luar)
    - Model Channel 2 -> Class 3: Enhancing Tumor (Cyan, nodul tumor aktif)

    Urutan assignment: Edema (2, kuning transparan) di bawah -> ET (3, cyan solid) di atas -> NETC (1, magenta) di inti
    """
    netc_bin = prob_map[0] > threshold   # Channel 0: NETC (Class 1)
    edema_bin = prob_map[1] > threshold  # Channel 1: Edema (Class 2, Kuning - area besar)
    et_bin = prob_map[2] > threshold     # Channel 2: ET (Class 3, Cyan - nodul tumor aktif)

    label_map = np.zeros(prob_map.shape[1:], dtype=np.uint8)
    label_map[edema_bin] = 2  # Edema (Kuning, area pembengkakan luas luar)
    label_map[et_bin] = 3     # ET (Cyan, nodul tumor aktif)
    label_map[netc_bin] = 1   # NETC (Magenta, inti nekrotik)

    return label_map


def crop_nonzero(volume: np.ndarray, margin: int = 8):
    """
    Mengambil bounding box dari area non-zero brain untuk memangkas background kosong.
    volume shape: (4, X, Y, Z)
    """
    C, X, Y, Z = volume.shape
    nz_mask = np.any(volume > 0.01, axis=0)  # (X, Y, Z)
    
    nz_x = np.where(np.any(nz_mask, axis=(1, 2)))[0]
    nz_y = np.where(np.any(nz_mask, axis=(0, 2)))[0]
    nz_z = np.where(np.any(nz_mask, axis=(0, 1)))[0]
    
    if len(nz_x) == 0 or len(nz_y) == 0 or len(nz_z) == 0:
        return volume, (0, X, 0, Y, 0, Z)
        
    x_min = max(0, int(nz_x[0]) - margin)
    x_max = min(X, int(nz_x[-1]) + 1 + margin)
    
    y_min = max(0, int(nz_y[0]) - margin)
    y_max = min(Y, int(nz_y[-1]) + 1 + margin)
    
    z_min = max(0, int(nz_z[0]) - margin)
    z_max = min(Z, int(nz_z[-1]) + 1 + margin)
    
    bbox = (x_min, x_max, y_min, y_max, z_min, z_max)
    cropped = volume[:, x_min:x_max, y_min:y_max, z_min:z_max]
    return cropped, bbox


def predict_segmentation(
    input_dir: str,
    output_dir: str,
    case_id: str,
    model_type: str = "u2net_attention",
    gt_label_path: str = None
) -> Tuple[str, float]:
    """
    Pipeline inferensi lengkap: muat modalitas -> crop non-zero -> sliding window -> rekonstruksi -> masking -> save NIfTI.
    """
    t0 = time.perf_counter()
    print(f"[INFERENCE] Memulai RSU U²-Net+ TensorFlow inference untuk case: {case_id}")

    stacked_volume, meta = load_and_stack_modalities(input_dir, case_id)
    print(f"[INFERENCE] Loaded stacked modalities shape: {stacked_volume.shape} (order: [t2f, t1n, t1c, t2w])")

    # Crop area non-zero brain untuk efisiensi komputasi
    cropped_vol, bbox = crop_nonzero(stacked_volume, margin=8)
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    print(f"[INFERENCE] Bounding box otak: {bbox} | Cropped shape: {cropped_vol.shape}")

    model = load_u2net_model()

    cropped_prob = predict_full_volume(
        model,
        cropped_vol,
        patch_size=(160, 160, 16),
        num_classes=3,
        overlap=0.5,
        batch_size=4
    )

    # Rekonstruksi prob_map ke koordinat volume asli
    C_out = cropped_prob.shape[0]
    X_orig, Y_orig, Z_orig = stacked_volume.shape[1:]
    prob_map = np.zeros((C_out, X_orig, Y_orig, Z_orig), dtype=np.float32)
    prob_map[:, x_min:x_max, y_min:y_max, z_min:z_max] = cropped_prob

    # Terapkan brain mask agar di luar tengkorak tidak pernah terdeteksi sebagai tumor
    brain_mask = np.any(stacked_volume > 0.01, axis=0)  # (X_orig, Y_orig, Z_orig)
    prob_map = prob_map * brain_mask[np.newaxis, ...]
    print(f"[INFERENCE] Prob map berhasil direkonstruksi dan di-mask: {prob_map.shape}")

    # Simpan probabilitas mentah untuk kalkulasi metrik
    os.makedirs(output_dir, exist_ok=True)
    save_raw_binary_predictions(output_dir, case_id, prob_map)

    # Buat single multiclass label map (ET > NETC > Edema)
    label_map = convert_prob_map_to_multiclass(prob_map, threshold=0.5)
    label_map[~brain_mask] = 0

    pred_nifti = nib.Nifti1Image(label_map, affine=meta["affine"], header=meta["header"])
    pred_nifti.set_data_dtype(np.uint8)

    pred_path = os.path.join(output_dir, f"{case_id}.nii.gz")
    nib.save(pred_nifti, pred_path)

    inference_time = time.perf_counter() - t0
    print(f"[INFERENCE] Inference selesai dalam {inference_time:.2f}s, saved to: {pred_path}")

    return pred_path, inference_time