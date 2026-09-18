# main.py
from fastapi.security import OAuth2AuthorizationCodeBearer, OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from jose import JWTError, jwt
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from urllib.parse import unquote
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session, joinedload
from database import SessionLocal, engine
from datetime import datetime
from sqlalchemy import func
import os
import re
import time
import zipfile
import auth
import models, schemas
import shutil
import uuid
import io
from PIL import Image
import pydantic
import nibabel as nib
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from skimage import measure
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_erosion
from datetime import datetime, timezone
import pytz

# pyrefly: ignore [missing-import]
from models_ai.inference import predict_segmentation


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create Database Table
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS Configuration
# Catatan: allow_origins=["*"] tidak kompatibel dengan allow_credentials=True.
# Token autentikasi dikirim via Authorization header (Bearer JWT), bukan cookie,
# sehingga allow_credentials=False aman digunakan.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.on_event("startup")
def cleanup_orphaned_scans():
    db = SessionLocal()
    try:
        orphaned = db.query(models.MRIScan).filter(
            models.MRIScan.processing_status.in_([
                "uploaded", "loading_model", "preprocessing", "running_inference",
                "computing_metrics", "rendering_3d", "saving_results"
            ])
        ).all()
        for scan in orphaned:
            scan.processing_status = "failed"
            scan.processing_progress = -1
            scan.processing_message = "Proses terhenti saat restart server. Silakan upload ulang file."
        if orphaned:
            db.commit()
            print(f"[STARTUP] Membersihkan {len(orphaned)} scan terhenti dari restart sebelumnya.")
    except Exception as e:
        print(f"[STARTUP] Cleanup error: {e}")
    finally:
        db.close()
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(status_code=401, detail="Token tidak valid", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None: raise credentials_exception
    return user

def save_log(db: Session, username: str, role: str, activity: str, details: str = ""):
    try:
        new_log = models.ActivityLog(username=username, role=role, activity=activity, details=details)
        db.add(new_log)
        db.commit()
    except Exception as e:
        print(f"Gagal menyimpan log: {e}")

# ENDPOINT AUTH & USER
@app.post("/token/")
async def login_for_access_token(from_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == from_data.username).first()
    if not user or not auth.verify_password(from_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Username atau Password Salah")
    access_token = auth.create_access_token(data={"sub": user.username, "role": user.role, "id": user.id})
    save_log(db, user.username, user.role, "Login", "Login Berhasil")
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

@app.post("/logout/")
def logout(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    save_log(db, current_user.username, current_user.role, "Logout", "Logout dari sistem")
    return {"message": "Berhasil logout"}

@app.post("/users/", response_model=schemas.UserResponse)
def create_new_user(user: schemas.UserCreate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user: raise HTTPException(status_code=400, detail="Username sudah terdaftar")
    hashed_pw = auth.get_password_hash(user.password)
    new_user = models.User(username=user.username, full_name=user.full_name, role=user.role, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    save_log(db, current_user.username, current_user.role, "Create User", f"Membuat user: {user.username} ({user.role})")
    return new_user

@app.put("/users/{user_id}", response_model=schemas.UserResponse)
def update_user(user_id: int, user_update: schemas.UserUpdate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user: raise HTTPException(status_code=404, detail="User tidak ditemukan")
    if user_update.username is not None: db_user.username = user_update.username
    if user_update.full_name is not None: db_user.full_name = user_update.full_name
    if user_update.role is not None: db_user.role = user_update.role
    if user_update.password and user_update.password.strip():
        db_user.hashed_password = auth.get_password_hash(user_update.password)
    if user_update.avatar is not None: db_user.avatar = user_update.avatar
    if user_update.is_active is not None:
        db_user.is_active = user_update.is_active
    db.commit()
    db.refresh(db_user)
    status_text = "Aktif" if db_user.is_active else "Nonaktif"
    save_log(db, current_user.username, current_user.role, "Edit User", f"Mengedit user ID: {user_id} - Status: {status_text}")
    return db_user

@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    db_user = db.query(models.User).filter(models.User.id == user_id).first()
    if not db_user: raise HTTPException(status_code=404, detail="User tidak ditemukan")
    if db_user.id == current_user.id: raise HTTPException(status_code=400, detail="Tidak dapat menghapus akun sendiri")
    target_username = db_user.username
    db_user.is_active = False
    db.commit()
    save_log(db, current_user.username, current_user.role, "Deactivate User", f"Menonaktifkan user: {target_username}")
    return {"detail": "User berhasil dinonaktifkan"}

@app.get("/users/me/", response_model=schemas.UserResponse)
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user

@app.get("/users/", response_model=List[schemas.UserResponse])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    return db.query(models.User).offset(skip).limit(limit).all()

@app.post("/users/upload-avatar/")
async def upload_avatar(file: UploadFile = File(...), current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan file: {str(e)}")
    avatar_url = f"static/{unique_filename}"
    current_user.avatar = avatar_url
    db.commit()
    db.refresh(current_user)
    save_log(db, current_user.username, current_user.role, "Update Profile", "Mengganti foto profil")
    return {"message": "Foto profil berhasil diupdate", "url": avatar_url} 

# ENDPOINT DATA PASIEN
@app.get("/patients/", response_model=List[schemas.PatientResponse])
def read_patient(db: Session = Depends(get_db)):
    return db.query(models.Patient).order_by(models.Patient.id.asc()).all()

@app.post("/patients/", response_model=schemas.PatientResponse)
def create_patient(patient: schemas.PatientCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    existing = db.query(models.Patient).filter(models.Patient.id_pasien_rs == patient.id_pasien_rs).first()
    if existing: raise HTTPException(status_code=400, detail="ID Pasien (RM) Sudah terdaftar")
    new_patient = models.Patient(id_pasien_rs=patient.id_pasien_rs, nama=patient.nama, tanggal_lahir=patient.tanggal_lahir, jenis_kelamin=patient.jenis_kelamin, status_pasien=patient.status_pasien)
    db.add(new_patient)
    db.commit()
    db.refresh(new_patient)
    save_log(db, current_user.username, current_user.role, "Create Patient", f"Menambah pasien baru: {new_patient.nama} (RM: {new_patient.id_pasien_rs})")
    return new_patient

@app.put("/patients/{patient_id}", response_model=schemas.PatientResponse)
def update_patient(patient_id: int, patient_update: schemas.PatientCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not db_patient: raise HTTPException(status_code=404, detail="Pasien tidak ditemukan")
    old_status = db_patient.status_pasien
    db_patient.id_pasien_rs = patient_update.id_pasien_rs
    db_patient.nama = patient_update.nama
    db_patient.tanggal_lahir = patient_update.tanggal_lahir
    db_patient.jenis_kelamin = patient_update.jenis_kelamin
    db_patient.status_pasien = patient_update.status_pasien
    db.commit()
    db.refresh(db_patient)
    detail_msg = f"Update data pasien: {db_patient.nama} (RM: {db_patient.id_pasien_rs})"
    if old_status != db_patient.status_pasien: detail_msg += f" | Status ubah: {old_status} -> {db_patient.status_pasien}"
    save_log(db, current_user.username, current_user.role, "Edit Patient", detail_msg)
    return db_patient

# METRIC FUNCTIONS (Dice + Sensitivity + HD95)

def _dice(p_mask, g_mask, eps=1e-5):
    p_sum, g_sum = int(p_mask.sum()), int(g_mask.sum())
    if p_sum == 0 and g_sum == 0: return 1.0
    if p_sum == 0 or g_sum == 0: return 0.0
    inter = int(np.logical_and(p_mask, g_mask).sum())
    return float((2 * inter + eps) / (p_sum + g_sum + eps))

def _sens(p_mask, g_mask, eps=1e-5):
    """Sensitivity = Recall = TP / (TP + FN)."""
    tp = int(np.logical_and(p_mask, g_mask).sum())
    fn = int(np.logical_and(~p_mask, g_mask).sum())
    if tp + fn == 0: return 1.0
    return float((tp + eps) / (tp + fn + eps))

def _hd95(p_mask, g_mask):
    """HD95 — 95th percentile bidirectional Hausdorff distance (voxel units)."""
    p_sum, g_sum = int(p_mask.sum()), int(g_mask.sum())
    if p_sum == 0 and g_sum == 0: return 0.0
    if p_sum == 0 or g_sum == 0: return float("nan")

    # Crop ke bounding box voxels aktif untuk kalkulasi EDT super cepat
    union_mask = p_mask | g_mask
    nz = np.where(union_mask)
    margin = 5
    x_min, x_max = max(0, int(nz[0].min()) - margin), min(union_mask.shape[0], int(nz[0].max()) + 1 + margin)
    y_min, y_max = max(0, int(nz[1].min()) - margin), min(union_mask.shape[1], int(nz[1].max()) + 1 + margin)
    z_min, z_max = max(0, int(nz[2].min()) - margin), min(union_mask.shape[2], int(nz[2].max()) + 1 + margin)

    p_sub = p_mask[x_min:x_max, y_min:y_max, z_min:z_max]
    g_sub = g_mask[x_min:x_max, y_min:y_max, z_min:z_max]

    ps = np.logical_xor(p_sub, binary_erosion(p_sub, border_value=1))
    ts = np.logical_xor(g_sub, binary_erosion(g_sub, border_value=1))
    if not ps.any() or not ts.any(): return float("nan")
    dt = distance_transform_edt(~ts, sampling=(1, 1, 1))
    dp = distance_transform_edt(~ps, sampling=(1, 1, 1))
    return float(np.percentile(np.concatenate([dt[ps], dp[ts]]), 95))

def _safe_round(v, ndigits):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), ndigits)

def _spec(p_mask, g_mask, eps=1e-8):
    """Specificity = TN / (TN + FP)"""
    tn = int(np.logical_and(~p_mask, ~g_mask).sum())
    fp = int(np.logical_and(p_mask, ~g_mask).sum())
    if tn + fp == 0: return 1.0
    return float((tn + eps) / (tn + fp + eps))

def _triplet(p_mask, g_mask):
    return {
        "dice": _safe_round(_dice(p_mask, g_mask), 4),
        "sens": _safe_round(_sens(p_mask, g_mask), 4),
        "spec": _safe_round(_spec(p_mask, g_mask), 4),
        "hd95": _safe_round(_hd95(p_mask, g_mask), 2),
    }

def _mean_triplet(metrics_list):
    """Compute mean dice/sens/spec/hd95 over list of metric dicts."""
    all_dice = [m["dice"] for m in metrics_list if m.get("dice") is not None]
    all_sens = [m["sens"] for m in metrics_list if m.get("sens") is not None]
    all_spec = [m["spec"] for m in metrics_list if m.get("spec") is not None]
    all_hd95 = [m["hd95"] for m in metrics_list if m.get("hd95") is not None]
    return {
        "dice": round(float(np.mean(all_dice)), 4) if all_dice else None,
        "sens": round(float(np.mean(all_sens)), 4) if all_sens else None,
        "spec": round(float(np.mean(all_spec)), 4) if all_spec else None,
        "hd95": round(float(np.mean(all_hd95)), 2) if all_hd95 else None,
    }

def load_u8(path):
    nii = nib.as_closest_canonical(nib.load(path))
    return np.rint(nii.get_fdata()).astype(np.uint8)

def calculate_metrics_if_gt_exists(pred_path, gt_file_path):
    """
    Evaluasi metrik BraTS2023 3 kelas (NETC, Edema, ET):
    - per_class: NETC (1), ED (2), ET (3) + mean
    - per_region_brats: ET, TC (NETC+ET), WT (NETC+ED+ET)
    - mean_brats_5: mean over ET, TC, WT, NETC, ED

    Metrik dihitung dari channel probabilitas biner mentah (raw_bin) jika ada,
    agar konsisten dengan evaluasi Colab (tanpa efek aturan prioritas visual).
    """
    if gt_file_path is None or not os.path.exists(gt_file_path):
        return None
    try:
        gt = load_u8(gt_file_path)

        # Cek apakah ada file raw binary prediction npz
        npz_path = pred_path.replace('.nii.gz', '_raw_prob.npz')
        if os.path.exists(npz_path):
            raw_data = np.load(npz_path)
            raw_bin = raw_data['raw_bin']  # Shape: (3, X, Y, Z)

            netc_p = (raw_bin[0] == 1)
            edema_p = (raw_bin[1] == 1)
            et_p = (raw_bin[2] == 1)

            tc_p = netc_p | et_p
            wt_p = netc_p | edema_p | et_p
        else:
            pred = load_u8(pred_path)
            if pred.shape != gt.shape:
                print(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
                return None
            netc_p = (pred == 1)
            edema_p = (pred == 2)
            et_p = (pred == 3)
            tc_p = (pred == 1) | (pred == 3)
            wt_p = (pred == 1) | (pred == 2) | (pred == 3)

        netc_g = (gt == 1)
        edema_g = (gt == 2)
        et_g = (gt == 3)
        tc_g = (gt == 1) | (gt == 3)
        wt_g = (gt == 1) | (gt == 2) | (gt == 3)

        per_region_brats = {
            "ET": _triplet(et_p, et_g),
            "TC": _triplet(tc_p, tc_g),
            "WT": _triplet(wt_p, wt_g),
        }
        per_class = {
            "NETC": _triplet(netc_p, netc_g),
            "ED":   _triplet(edema_p, edema_g),
            "ET":   _triplet(et_p, et_g),
        }
        per_class["mean"] = _mean_triplet(
            [per_class["NETC"], per_class["ED"], per_class["ET"]]
        )

        mean_brats_5 = _mean_triplet([
            per_region_brats["ET"], per_region_brats["TC"], per_region_brats["WT"],
            per_class["NETC"], per_class["ED"],
        ])

        result = {
            "default_view": "per_class",
            "per_class": per_class,
            "per_region_brats": per_region_brats,
            "mean_brats_6": mean_brats_5,  # Key tetap mean_brats_6 untuk kompatibilitas UI frontend
        }
        return json.dumps(result)
    except Exception as e:
        print(f"Error saat menghitung metric: {e}")
        return None

def norm01(x):
    v = x[np.isfinite(x)]
    lo, hi = np.percentile(v, [2, 98]) if v.size > 0 else (0.0, 1.0)
    if hi <= lo: hi = lo + 1e-8
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)

def make_mesh(mask, color, opacity, name, smooth_sigma=None):
    mask = mask.astype(np.uint8)

    if smooth_sigma:
        mask = gaussian_filter(mask.astype(float), sigma=smooth_sigma)
        level = 0.3
    else:
        level = 0.5

    if mask.max() < level:
        return None

    try:
        verts, faces, normals, _ = measure.marching_cubes(mask, level=level)
        x, y, z = verts.T
        i, j, k = faces.T

        return go.Mesh3d(
            x=x, y=y, z=z, i=i, j=j, k=k,
            color=color,
            opacity=opacity,
            name=name,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
            lightposition=dict(x=100, y=200, z=150),
            flatshading=False
        )
    except Exception as e:
        print(f"[WARN] Error marching cubes for {name}: {e}")
        return None

def generate_all_3d_views(volume_ds, pred_ds, upload_dir, scan_id, ds=2):
    # ===== Permukaan otak (persis seperti Colab) =====
    brain_mesh = None
    brain_thresh = np.percentile(volume_ds[volume_ds > 0], 40) if (volume_ds > 0).any() else 0
    brain_mask = volume_ds > brain_thresh

    brain_mesh = make_mesh(
        brain_mask, color='rgb(230,230,230)', opacity=0.12, name='Jaringan Otak', smooth_sigma=1.2
    )

    # ===== Mesh tumor per kelas (persis seperti Colab) =====
    # Opacity berbeda per kelas: Edema dibuat transparan (0.35) supaya kelas di dalamnya terlihat
    color_map = {
        1: ('magenta', 'Necrotic / Non-Enhancing Tumor Core', 0.95),
        2: ('yellow', 'Peritumoral Edema', 0.35),
        3: ('cyan', 'Enhancing Tumor', 0.95),
    }

    # Render order: Edema dulu (2), baru Core (1), baru Enhancing (3)
    render_order = [2, 1, 3]

    tumor_meshes = {}
    if pred_ds is not None:
        for class_val in render_order:
            color, name, opacity = color_map[class_val]
            mask = (pred_ds == class_val)
            if mask.sum() < 5:
                continue
            t_mesh = make_mesh(mask, color=color, opacity=opacity, name=name.split(' (')[0], smooth_sigma=0.3)
            if t_mesh is not None:
                tumor_meshes[class_val] = t_mesh

    legend_text_parts = ["<b>Keterangan Segmentasi:</b>"]
    legend_text_parts.append("⬜ Jaringan Otak — struktur otak secara keseluruhan (transparan)")
    for class_val in [2, 1, 3]:
        if class_val in tumor_meshes:
            color, name, _ = color_map[class_val]
            swatch = {'magenta': '🟣', 'yellow': '🟡', 'cyan': '🔵'}.get(color, '⬛')
            legend_text_parts.append(f"{swatch} {name}")
    legend_text = "<br>".join(legend_text_parts)

    camera = dict(eye=dict(x=1.5, y=1.5, z=1.0), up=dict(x=0, y=0, z=1))
    scene_config = dict(
        aspectmode="data",
        xaxis=dict(visible=False, showgrid=False, zeroline=False, showbackground=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False, showbackground=False),
        zaxis=dict(visible=False, showgrid=False, zeroline=False, showbackground=False),
        bgcolor="white",
        camera=camera,
    )

    path_3d_db = {}
    base_labels = {"all": 0, "netc": 1, "edema": 2, "et": 3}

    for key, target_label in base_labels.items():
        for show_brain in [True, False]:
            fig_3d = go.Figure()

            # Gunakan mesh brain
            if show_brain and brain_mesh is not None:
                fig_3d.add_trace(brain_mesh)

            labels_to_draw = [2, 1, 3] if target_label == 0 else [target_label]
            for lbl in labels_to_draw:
                if lbl in tumor_meshes:
                    fig_3d.add_trace(tumor_meshes[lbl])

            fig_3d.update_layout(
                title="Visualisasi 3D",
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(color='black'),
                scene=scene_config,
                margin=dict(l=0, r=0, b=0, t=30),
                showlegend=True,
                legend=dict(
                    title=dict(text="<b>Kelas Segmentasi</b>", font=dict(size=14)),
                    font=dict(size=12),
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='lightgray',
                    borderwidth=1,
                    x=0.75, y=0.9
                ),
                annotations=[
                    dict(
                        text=legend_text,
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.02, y=0.02,
                        align="left",
                        font=dict(size=11, color="black"),
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="lightgray",
                        borderwidth=1,
                        borderpad=8
                    )
                ]
            )

            fname_suffix = "" if show_brain else "_nobrain"
            fname_3d = f"result_{scan_id}_3d_{key}{fname_suffix}.html"
            out_path = os.path.join(upload_dir, fname_3d)

            fig_3d.write_html(out_path, full_html=True, include_plotlyjs='cdn')
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                clean_html = re.sub(r'\s+integrity="[^"]*"', '', html_content)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(clean_html)
            except Exception as e:
                print(f"[WARN] Gagal membersihkan integrity HTML 3D: {e}")

            db_key = f"{key}{fname_suffix}"
            path_3d_db[db_key] = f"static/{fname_3d}"

    path_3d_db["snfh"] = path_3d_db["edema"]
    path_3d_db["snfh_nobrain"] = path_3d_db["edema_nobrain"]

    return path_3d_db


# ── Progress helper ──
def _update_scan_progress(db, scan, status: str, progress: int, message: str = None):
    """Update progress field tanpa block proses utama. Best-effort."""
    try:
        scan.processing_status = status
        scan.processing_progress = progress
        if message:
            scan.processing_message = message
        db.commit()
    except Exception as e:
        print(f"[WARN] Gagal update progress: {e}")
        db.rollback()

# AI PROCESSOR (BACKGROUND TASK)
def process_mri_ai(scan_id: int, input_dir: str, output_dir: str, case_id: str, gt_file_path: str, model_type: str = "u2net_attention"):
    db = SessionLocal()
    scan = db.query(models.MRIScan).filter(models.MRIScan.id == scan_id).first()
    if not scan:
        db.close()
        return

    print(f"[PROSES MULAI] AI ({model_type}) sedang membedah pasien: {case_id}...")

    try:
        # Stage 1: Loading model
        _update_scan_progress(db, scan, "loading_model", 15,
                             "Memuat model RSU U²-Net+ (Attention Gate)...")

        # Stage 2: Preprocessing
        _update_scan_progress(db, scan, "preprocessing", 25,
                             "Memproses data MRI (z-score standarisasi per-patch)...")

        # Stage 3: Inference
        _update_scan_progress(db, scan, "running_inference", 60,
                             "Menjalankan AI segmentasi (sliding window inference)...")

        pred_path, inference_time = predict_segmentation(
            input_dir=input_dir,
            output_dir=output_dir,
            case_id=case_id,
            model_type=model_type,
            gt_label_path=gt_file_path,
        )

        print(f"[INFERENCE TIME] {inference_time:.2f} detik (model: {model_type})")

        mri_path_3d = os.path.join(input_dir, f"{case_id}_0000.nii.gz")

        pred_data = nib.load(pred_path).get_fdata()
        unique_labels = np.unique(pred_data).astype(int).tolist()
        if 0 in unique_labels: unique_labels.remove(0)
        scan.detected_regions = json.dumps(unique_labels)

        # Stage 4: Metrics
        _update_scan_progress(db, scan, "computing_metrics", 75,
                             "Menghitung metrik evaluasi (Dice, Sensitivity, HD95)...")

        metrics_json = calculate_metrics_if_gt_exists(pred_path, gt_file_path)

        original_notes = scan.catatan_teknis or "-"

        scan.catatan_teknis = json.dumps({
            "catatan": original_notes,
            "metrics": json.loads(metrics_json) if metrics_json else None,
            "shape": list(pred_data.shape),
            "case_id": case_id,
            "model_type": model_type,
            "inference_time_seconds": round(inference_time, 2),
        })

        # Stage 5: 3D Rendering (Pre-computed mesh optimization)
        _update_scan_progress(db, scan, "rendering_3d", 85,
                             "Membuat visualisasi 3D interaktif...")

        mri_3d = nib.as_closest_canonical(nib.load(mri_path_3d)).get_fdata().astype(np.float32)
        pred_3d = nib.as_closest_canonical(nib.load(pred_path)).get_fdata().astype(np.uint8)

        ds = 2
        volume_ds = mri_3d[::ds, ::ds, ::ds]
        pred_ds = pred_3d[::ds, ::ds, ::ds]

        path_3d_db = generate_all_3d_views(volume_ds, pred_ds, UPLOAD_DIR, scan_id, ds=ds)

        scan.filepath_3d = json.dumps(path_3d_db)
        scan.filepath_2d = "dynamic" 
        scan.hasil_prediksi = "Tumor Terdeteksi" if unique_labels else "Normal"

        # Stage 6: Saving
        _update_scan_progress(db, scan, "saving_results", 95,
                             "Menyimpan hasil...")

        target_roles = ["Dokter", "Radiolog"]
        for target in target_roles:
            db.add(models.Notification(
                target_role=target,
                title="Analisis AI Selesai!",
                message=f"Model: {model_type.upper()} | Waktu: {inference_time:.1f}s | Hasil pemindaian 3D dan Slice 2D dapat dilihat.",
                analysis_id=scan.id
            ))

        # Stage 7: Complete
        _update_scan_progress(db, scan, "completed", 100, "Analisis selesai!")
        db.commit()
    
    except Exception as e:
        import traceback
        print(f"[ERROR AI] Gagal memproses: {e}")
        traceback.print_exc()
        scan.hasil_prediksi = "Gagal Diproses AI"
        try:
            scan.processing_status = "failed"
            scan.processing_progress = -1
            scan.processing_message = f"Error: {str(e)[:200]}"
            db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()

@app.get("/scan/{scan_id}/status")
def get_scan_status(scan_id: int, db: Session = Depends(get_db)):
    """
    Return current processing status. Frontend poll endpoint ini setiap 1-2s
    selama scan masih processing.
    """
    scan = db.query(models.MRIScan).filter(models.MRIScan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan tidak ditemukan")
    
    return {
        "scan_id": scan.id,
        "status": scan.processing_status or "uploaded",
        "progress": scan.processing_progress if scan.processing_progress is not None else 0,
        "message": scan.processing_message or "",
        "is_complete": scan.processing_status == "completed",
        "is_failed": scan.processing_status == "failed",
        "hasil_prediksi": scan.hasil_prediksi,
    }

# ENDPOINT ANALISIS & FILE
@app.post("/upload-mri/")
async def upload_mri_smart(
    background_tasks: BackgroundTasks, nama: str = Form(...), id_pasien: str = Form(...), tgl_lahir: str = Form(...),
    status: str = Form(...), jenis_mri: str = Form(...), catatan: str = Form(default="-"),
    model_type: str = Form(default="u2net_attention"),
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)
):
    if not file.filename.endswith('.zip'): raise HTTPException(status_code=400, detail="Harus file .zip yang berisi 4 modalitas MRI!")

    pasien_db = db.query(models.Patient).filter(models.Patient.id_pasien_rs == id_pasien).first()
    if not pasien_db:
        pasien_db = models.Patient(nama=nama, id_pasien_rs=id_pasien, tanggal_lahir=tgl_lahir, status_pasien=status)
        db.add(pasien_db)
        db.commit()
        db.refresh(pasien_db)
    else:
        pasien_db.nama = nama
        pasien_db.status_pasien = status
        db.commit()
    
    new_scan = models.MRIScan(patient_id=pasien_db.id, jenis_mri="MRI Otak", catatan_teknis=catatan, filepath_raw="", hasil_prediksi="Sedang Dianalisis...", processing_status="uploaded", processing_progress=5, processing_message="Berkas diterima, antri untuk diproses...")
    db.add(new_scan)
    db.commit()
    db.refresh(new_scan)

    # Folder Scan
    scan_folder = os.path.join(UPLOAD_DIR, f"scan_{new_scan.id}")
    input_dir = os.path.join(scan_folder, "input")
    output_dir = os.path.join(scan_folder, "output")

    # Retry logic untuk mengatasi OSError: [Errno 5] pada Docker/WSL2 volume mount
    for attempt in range(3):
        try:
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            break
        except OSError as e:
            if attempt < 2:
                print(f"[WARN] os.makedirs gagal (attempt {attempt+1}/3): {e}. Retry dalam 1 detik...")
                time.sleep(1)
            else:
                raise HTTPException(status_code=500, detail=f"Gagal membuat folder scan: {e}. Coba restart Docker Desktop.")

    # Ekstrak ZIP dengan proteksi Zip-Slip (Path Traversal)
    zip_path = os.path.join(scan_folder, "temp.zip")
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            # Validasi absolut path agar tidak melompat keluar dari input_dir
            target_path = os.path.abspath(os.path.join(input_dir, member.filename))
            if not target_path.startswith(os.path.abspath(input_dir)):
                raise HTTPException(status_code=400, detail="Zip file contains illegal path traversal attempt.")
            zip_ref.extract(member, input_dir)
    os.remove(zip_path)

    # Flatten nested folder: kalau semua file ada di subfolder, pindahkan ke input_dir
    for root, dirs, files in os.walk(input_dir):
        if root == input_dir:
            continue
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(input_dir, f)
            if not os.path.exists(dst):
                shutil.move(src, dst)
    # Hapus subfolder kosong
    for root, dirs, files in os.walk(input_dir, topdown=False):
        if root == input_dir:
            continue
        if not os.listdir(root):
            os.rmdir(root)

    extracted_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    
    t1n_file = None
    t1c_file = None
    t2w_file = None
    t2f_file = None
    seg_file = None

    # Pencocokan nama file secara fleksibel (BraTS & nnUNet formats)
    # Urutan if-elif sangat penting untuk mencegah greediness (misal '-t2' mencocokkan '-t2f')
    for f in extracted_files:
        f_lower = f.lower()
        if any(x in f_lower for x in ["-seg", "_seg", "-gt", "_gt"]):
            seg_file = f
        elif any(x in f_lower for x in ["_0000", "-t1c", "_t1c", "-t1ce", "_t1ce"]):
            t1c_file = f
        elif any(x in f_lower for x in ["_0001", "-t1n", "_t1n"]) or (any(x in f_lower for x in ["-t1", "_t1"]) and not any(x in f_lower for x in ["t1c", "t1ce"])):
            t1n_file = f
        elif any(x in f_lower for x in ["_0002", "-t2f", "_t2f", "-flair", "_flair"]):
            t2f_file = f
        elif any(x in f_lower for x in ["_0003", "-t2w", "_t2w", "-t2", "_t2"]):
            t2w_file = f

    if not (t1n_file and t1c_file and t2w_file and t2f_file):
        raise HTTPException(
            status_code=400,
            detail="Format ZIP salah! Harus berisi 4 modalitas MRI (T1, T1c, T2, FLAIR). "
                   f"Ditemukan: T1c={t1c_file}, T1={t1n_file}, FLAIR={t2f_file}, T2={t2w_file}"
        )


    # Tentukan case_id dari prefix file T1c
    case_id = t1c_file
    for suff in [".nii.gz", "-t1c", "_t1c", "-t1ce", "_t1ce", "_0000", "_0001"]:
        if case_id.lower().endswith(suff):
            case_id = case_id[:-len(suff)]
    
    # Standarisasi nama file ke format internal dataset.json (_0000=t1c, _0001=t1n, _0002=t2f, _0003=t2w)
    shutil.move(os.path.join(input_dir, t1c_file), os.path.join(input_dir, f"{case_id}_0000.nii.gz"))
    shutil.move(os.path.join(input_dir, t1n_file), os.path.join(input_dir, f"{case_id}_0001.nii.gz"))
    shutil.move(os.path.join(input_dir, t2f_file), os.path.join(input_dir, f"{case_id}_0002.nii.gz"))
    shutil.move(os.path.join(input_dir, t2w_file), os.path.join(input_dir, f"{case_id}_0003.nii.gz"))

    gt_file_path = None
    if seg_file:
        gt_file_path = os.path.join(scan_folder, f"{case_id}_GT.nii.gz")
        shutil.move(os.path.join(input_dir, seg_file), gt_file_path)
        print(f"File GT ditemukan dan diamankan ke: {gt_file_path}")


    new_scan.filepath_raw = scan_folder
    db.commit()

    new_notif = models.Notification(target_role="Dokter", title="Data MRI Otak Diterima", message=f"File ZIP Pasien {nama} berhasil diekstrak dan masuk antrean AI (Model: {model_type.upper()}).", analysis_id=new_scan.id)
    db.add(new_notif)
    db.commit()

    save_log(db, current_user.username, current_user.role, "Upload MRI", f"Upload scan untuk pasien: {nama} (Model: {model_type})")

    background_tasks.add_task(process_mri_ai, new_scan.id, input_dir, output_dir, case_id, gt_file_path, model_type)

    return {
        "status": "sukses",
        "pesan": f"ZIP terekstrak dan masuk antrean AI (model: {model_type})",
        "scan_id": new_scan.id,
        "model_type": model_type,
    }

@app.get("/analisis/{analysis_id}/slice")
def get_mri_slice(analysis_id: int, axis: int = 2, idx: int = None, label: str = "all", db: Session = Depends(get_db)):
    scan = db.query(models.MRIScan).filter(models.MRIScan.id == analysis_id).first()
    if not scan: raise HTTPException(status_code=404, detail="Scan tidak ditemukan")
    if scan.processing_status != "completed":
        raise HTTPException(status_code=400, detail="Proses AI segmentasi belum selesai")
    try:
        meta = json.loads(scan.catatan_teknis) if scan.catatan_teknis else {}
        case_id = meta.get("case_id")
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        meta = {}
        case_id = None

    scan_folder = scan.filepath_raw
    input_dir = os.path.join(scan_folder, "input")
    output_dir = os.path.join(scan_folder, "output")

    # 1. Cari pred_path secara fleksibel
    pred_path = None
    if case_id:
        p = os.path.join(output_dir, f"{case_id}.nii.gz")
        if os.path.exists(p): pred_path = p
    if not pred_path and os.path.exists(output_dir):
        files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".nii.gz") and not any(x in f for x in ["_edema", "_et", "_netc", "_nobrain", "_GT"])]
        if files: pred_path = files[0]

    # 2. Cari mri_path_2d secara fleksibel (prioritaskan T1c _0000 atau _0001, kemudian modalitas lain)
    mri_path_2d = None
    if case_id:
        for suffix in ["_0000.nii.gz", "_0001.nii.gz", "_0002.nii.gz", "_0003.nii.gz"]:
            p = os.path.join(input_dir, f"{case_id}{suffix}")
            if os.path.exists(p):
                mri_path_2d = p
                break
    if not mri_path_2d and os.path.exists(input_dir):
        files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".nii.gz") and not any(x in f for x in ["-seg", "_seg", "-gt", "_gt"])]
        if files: mri_path_2d = files[0]

    if not pred_path or not mri_path_2d or not os.path.exists(pred_path) or not os.path.exists(mri_path_2d):
        raise HTTPException(status_code=400, detail="File segmentasi atau MRI belum tersedia")

    fig = None
    try:
        mri_vol = nib.load(mri_path_2d).get_fdata().astype(np.float32)
        pred_vol = nib.load(pred_path).get_fdata().astype(np.uint8)

        axis_size = mri_vol.shape[axis]

        # Dynamic slice index selection jika idx tidak diberikan atau bernilai negatif:
        # Cari slice dengan area tumor terbesar pada axis tersebut
        if idx is None or idx < 0:
            tumor_mask = pred_vol > 0
            if tumor_mask.any():
                sum_axes = tuple(i for i in range(3) if i != axis)
                tumor_area_per_slice = tumor_mask.sum(axis=sum_axes)
                idx = int(np.argmax(tumor_area_per_slice))
            else:
                idx = axis_size // 2

        # Clamp idx to safe boundary [0, axis_size - 1]
        idx = max(0, min(int(idx), axis_size - 1))

        def take_slice(vol, axis, idx):
            if axis == 0: return vol[idx, :, :]
            if axis == 1: return vol[:, idx, :] 
            if axis == 2: return vol[:, :, idx]
            raise HTTPException(status_code=400, detail=f"Axis tidak valid: {axis}. Harus 0, 1, atau 2.")

        raw_mri_slice = take_slice(mri_vol, axis, idx)
        mri_s = norm01(raw_mri_slice)
        pred_s = take_slice(pred_vol, axis, idx)

        # Filter label sesuai pilihan user
        if label == "netc": pred_s = np.where(pred_s == 1, 1, 0)
        elif label in ("edema", "snfh"): pred_s = np.where(pred_s == 2, 2, 0)
        elif label == "et": pred_s = np.where(pred_s == 3, 3, 0)

        # Colormap 100% Identik Colab:
        # 1: NETC (Magenta #FF00FF), 2: Edema (Yellow #FFD700), 3: ET (Cyan #00FFFF)
        colors_hex = {
            1: "#FF00FF",  # NETC (Magenta)
            2: "#FFD700",  # Edema (Yellow)
            3: "#00FFFF",  # ET (Cyan)
        }
        mask_cmap = ListedColormap(["none", colors_hex[1], colors_hex[2], colors_hex[3]])

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(np.rot90(mri_s), cmap="gray")
        ax.imshow(np.rot90(pred_s), cmap=mask_cmap, alpha=0.6, vmin=0, vmax=3)
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-cache"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if fig is not None:
            plt.close(fig)

@app.get("/analisis/{analysis_id}/info")
def get_analysis_info(analysis_id: int, db: Session = Depends(get_db)):
    scan = db.query(models.MRIScan).filter(models.MRIScan.id == analysis_id).first()
    if not scan: 
        raise HTTPException(status_code=404, detail="Scan tidak ditemukan")
    try:
        meta = json.loads(scan.catatan_teknis)
        return {
            "case_id": meta.get("case_id"),
            "model_type": meta.get("model_type", "unknown"),
            "inference_time_seconds": meta.get("inference_time_seconds"),
            "metrics": meta.get("metrics"),
            "shape": meta.get("shape"),
        }
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        raise HTTPException(status_code=400, detail="Data belum siap")

@app.get("/riwayat-semua/")
def get_all_history(db: Session = Depends(get_db)):
    scans = db.query(models.MRIScan).options(joinedload(models.MRIScan.patient)).order_by(models.MRIScan.upload_date.desc()).all()
    
    tz_jkt = pytz.timezone('Asia/Jakarta')
    results = []
    
    for scan in scans:
        if not scan.upload_date:
            tgl_cantik = "-"
        else:
            if scan.upload_date.tzinfo is None:
                waktu_utc = scan.upload_date.replace(tzinfo=pytz.utc)
                waktu_lokal = waktu_utc.astimezone(tz_jkt)
            else:
                waktu_lokal = scan.upload_date.astimezone(tz_jkt)
            
            tgl_cantik = waktu_lokal.strftime("%d/%m/%Y")

        # Parse catatan teknis
        catatan_val = "-"
        if scan.catatan_teknis:
            if scan.catatan_teknis.strip().startswith("{"):
                try:
                    import json
                    meta = json.loads(scan.catatan_teknis)
                    catatan_val = meta.get("catatan", "-")
                except Exception:
                    catatan_val = scan.catatan_teknis
            else:
                catatan_val = scan.catatan_teknis

        results.append({
            "id": scan.id, "jenis_mri": scan.jenis_mri, "tanggal_periksa": tgl_cantik,
            "hasil_prediksi": scan.hasil_prediksi, "nama_pasien": scan.patient.nama if scan.patient else "Tanpa Nama",
            "id_rm": scan.patient.id_pasien_rs if scan.patient else "-",
            "catatan_teknis": catatan_val
        })
    return results

@app.get("/analisis/{analysis_id}")
def get_analysis_detail(analysis_id: int, db: Session = Depends(get_db)):
    scan = db.query(models.MRIScan).options(joinedload(models.MRIScan.patient)).filter(models.MRIScan.id == analysis_id).first()
    if not scan: raise HTTPException(status_code=404, detail="Data MRI tidak ditemukan")

    detected_list = []
    if scan.detected_regions:
        try:
            detected_list = json.loads(scan.detected_regions)
        except (json.JSONDecodeError, TypeError):
            pass
    
    meta = {}
    if scan.catatan_teknis and "{" in scan.catatan_teknis:
        try:
            meta = json.loads(scan.catatan_teknis)
        except (json.JSONDecodeError, TypeError):
            pass

    paths_3d = {}
    if scan.filepath_3d and "{" in scan.filepath_3d:
        try:
            paths_3d = json.loads(scan.filepath_3d)
        except (json.JSONDecodeError, TypeError):
            paths_3d = {"all": scan.filepath_3d}
    else:
        paths_3d = {"all": scan.filepath_3d}

    # Konversi Zona Waktu
    tz_jkt = pytz.timezone('Asia/Jakarta')
    if scan.upload_date:
        if scan.upload_date.tzinfo is None:
            waktu_utc = scan.upload_date.replace(tzinfo=pytz.utc)
            waktu_lokal = waktu_utc.astimezone(tz_jkt)
        else:
            waktu_lokal = scan.upload_date.astimezone(tz_jkt)
        waktu_scan_cantik = waktu_lokal.strftime("%d/%m/%Y • %H:%M WIB")
    else:
        waktu_scan_cantik = "-"

    # Hitung atau sediakan peak tumor slice per axis [Sagittal, Coronal, Axial]
    peak_slices = meta.get("peak_slices")
    if not peak_slices:
        try:
            pred_path = os.path.join(scan.filepath_raw, "output", f"{meta.get('case_id')}.nii.gz")
            if not os.path.exists(pred_path) and os.path.exists(os.path.join(scan.filepath_raw, "output")):
                cands = [os.path.join(scan.filepath_raw, "output", f) for f in os.listdir(os.path.join(scan.filepath_raw, "output")) if f.endswith(".nii.gz") and not any(x in f for x in ["_edema", "_et", "_netc", "_nobrain", "_GT"])]
                if cands: pred_path = cands[0]
            if os.path.exists(pred_path):
                pred_vol = nib.load(pred_path).get_fdata()
                tumor_mask = pred_vol > 0
                peak_slices = []
                for ax in range(3):
                    sum_axes = tuple(i for i in range(3) if i != ax)
                    tumor_area = tumor_mask.sum(axis=sum_axes)
                    if tumor_area.any():
                        peak_slices.append(int(np.argmax(tumor_area)))
                    else:
                        peak_slices.append(int(pred_vol.shape[ax] // 2))
                meta["peak_slices"] = peak_slices
                scan.catatan_teknis = json.dumps(meta)
                db.commit()
        except Exception as e:
            print(f"[WARN] Gagal menghitung peak_slices: {e}")
            peak_slices = [155, 119, 42]

    return {
        "id": scan.id,
        "image_url": "dynamic", 
        "paths_3d": paths_3d,
        "result": scan.hasil_prediksi,

        "waktu_scan": waktu_scan_cantik,
        "nama_pasien": scan.patient.nama if scan.patient else "-",
        "id_rm": scan.patient.id_pasien_rs if scan.patient else "-",
        "tgl_lahir": scan.patient.tanggal_lahir if scan.patient else "-",
        "jenis_kelamin": scan.patient.jenis_kelamin if scan.patient else "-",
        "notes_radiolog": meta.get("catatan", "-"), 
        "notes_dokter": getattr(scan, "catatan_dokter", "Belum ada catatan dokter"),
        "detected_regions": detected_list,
        "metrics": meta.get("metrics"),
        "shape": meta.get("shape", [240, 240, 155]),
        "peak_slices": peak_slices or [155, 119, 42],
        "model_type": meta.get("model_type", "unknown"),
        "inference_time_seconds": meta.get("inference_time_seconds"),
    }

@app.get("/get-image/{filename}")
async def get_image_manual(filename: str, db: Session = Depends(get_db)):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-cache"})
    return {"error": "File tidak ditemukan"}

@app.put("/analisis/{analysis_id}/update-notes/")
async def update_doctor_notes(analysis_id: int, data: dict, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    scan = db.query(models.MRIScan).filter(models.MRIScan.id == analysis_id).first()
    if not scan: raise HTTPException(status_code=404, detail="Data MRI tidak ditemukan")
    
    new_notes = data.get("notes_dokter", "")
    scan.catatan_dokter = new_notes
    db.commit()
    db.refresh(scan)

    nama_pasien = scan.patient.nama if scan.patient else "Tanpa Nama"
    db.add(models.Notification(target_role="Radiolog", title="Catatan Dokter", message=f"Dokter telah menambahkan catatan untuk pasien {nama_pasien}.", analysis_id=scan.id))
    db.commit()

    save_log(db, current_user.username, current_user.role, "Update Notes", f"Update catatan dokter untuk Scan ID: {analysis_id}")
    return {"status": "sukses", "message": "Catatan berhasil diperbarui", "data": new_notes}

# ENDPOINT SUMMARY
@app.get("/dashboard-summary/", response_model=schemas.DashboardSummary)
def get_summary(db: Session = Depends(get_db)):
    total_p = db.query(models.Patient).count()
    menunggu = db.query(models.MRIScan).filter(models.MRIScan.processing_status.in_(["uploaded", "processing"])).count()
    selesai = db.query(models.MRIScan).filter(models.MRIScan.processing_status == "completed").count()
    return {"total_pasien": total_p, "total_menunggu": menunggu, "total_selesai": selesai}

@app.get("/logs/", response_model=List[schemas.LogResponse])
def get_logs(role: str = None, start_date: str = None, end_date: str = None, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    query = db.query(models.ActivityLog)
    if role and role != "Semua": query = query.filter(models.ActivityLog.role == role)
    if start_date and end_date:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            query = query.filter(models.ActivityLog.timestamp >= start)
            query = query.filter(models.ActivityLog.timestamp <= end)
        except ValueError: pass
        
    logs = query.order_by(models.ActivityLog.timestamp.desc()).limit(200).all()
    tz_jkt = pytz.timezone('Asia/Jakarta')
    results = []
    
    for log in logs:
        if log.timestamp:
            if log.timestamp.tzinfo is None:
                waktu_utc = log.timestamp.replace(tzinfo=pytz.utc)
                waktu_lokal = waktu_utc.astimezone(tz_jkt)
            else:
                waktu_lokal = log.timestamp.astimezone(tz_jkt)
        else:
            waktu_lokal = None

        results.append({
            "id": log.id,
            "username": log.username,
            "role": log.role,
            "activity": log.activity,
            "details": log.details,
            "timestamp": waktu_lokal
        })
        
    return results

@app.get("/notifications/")
def get_notifications(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    notifs = db.query(models.Notification).filter(func.lower(models.Notification.target_role) == func.lower(current_user.role)).order_by(models.Notification.created_at.desc()).limit(100).all()
    
    tz_jkt = pytz.timezone('Asia/Jakarta')
    results = []
    
    for n in notifs:
        if not n.created_at:
            tgl_cantik = "-"
        else:
            if n.created_at.tzinfo is None:
                waktu_utc = n.created_at.replace(tzinfo=pytz.utc)
                waktu_lokal = waktu_utc.astimezone(tz_jkt)
            else:
                waktu_lokal = n.created_at.astimezone(tz_jkt)
            
            tgl_cantik = waktu_lokal.strftime("%d/%m/%Y • %H:%M WIB")

        results.append({
            "id": n.id, 
            "title": n.title, 
            "message": n.message, 
            "analysis_id": n.analysis_id, 
            "is_read": n.is_read, 
            "created_at": tgl_cantik
        })
        
    return results

@app.put("/notifications/{notif_id}/read")
def mark_notification_read(notif_id: int, db: Session = Depends(get_db)):
    notif = db.query(models.Notification).filter(models.Notification.id == notif_id).first()
    if notif:
        notif.is_read = True
        db.commit()
    return {"status": "sukses"}