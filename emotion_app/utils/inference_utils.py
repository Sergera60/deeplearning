# utils/inference_utils.py
# ----------------------------------------------------------
# Inference helpers aligned to notebook training:
#  - T=16 frames (evenly sampled)
#  - Grayscale 112x112
#  - Normalization MEAN/STD from training set
#  - CNN (ResNet18, frozen) -> FeatureAE(128) -> LSTM(256x2) -> Classifier
#  - Weights saved as state_dicts: models/feature_ae.pt, models/best_model.pt
# ----------------------------------------------------------

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from torchvision import models

# ========= Constants (from your notebook logs) =========
T = 16
H, W = 112, 112

# Class order as trained
EMOTIONS = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

# Use your dataset normalization
MEAN, STD = 0.4982, 0.2325

AE_PATH = os.path.join("models", "feature_ae.pt")
MODEL_PATH = os.path.join("models", "best_model.pt")

# ========= Face detection (largest face, padded) =========
# utils/inference_utils.py

_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _crop_face_bgr(img_bgr, pad_ratio=0.12, max_side=1600):
    """
    Crop the largest detected face with padding.
    For very large images, first downscale to speed up detection,
    then map the face box back to original resolution.
    """
    h, w = img_bgr.shape[:2]

    # Optional downscale for speed on huge images
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img_small = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        img_small = img_bgr

    gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray_small, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

    if len(faces) == 0:
        # Fallback: centered square crop
        side = min(h, w)
        y0 = (h - side)//2
        x0 = (w - side)//2
        return img_bgr[y0:y0+side, x0:x0+side]

    # Largest face on the *downscaled* image
    x, y, wf, hf = max(faces, key=lambda b: b[2]*b[3])

    # Map back to original coords
    if scale != 1.0:
        x  = int(x  / scale); y  = int(y  / scale)
        wf = int(wf / scale);  hf = int(hf / scale)

    pad = int(pad_ratio * max(wf, hf))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w, x + wf + pad); y1 = min(h, y + hf + pad)
    return img_bgr[y0:y1, x0:x1]

def _to_gray_112(img_bgr, use_face_crop=True, pad_ratio=0.12, max_side=1600):
    if use_face_crop:
        img_bgr = _crop_face_bgr(img_bgr, pad_ratio=pad_ratio, max_side=max_side)
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    g = (g - MEAN) / (STD if STD > 1e-6 else 1.0)
    return g

# ========= Model definitions (match notebook) =========
class ResNetFrameEncoder(nn.Module):
    """
    Input:  [B,1,112,112]
    Output: [B,512] (global pooled)
    """
    def __init__(self, train_backbone: bool = False):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # [B,512,1,1]
        self.out_dim = 512
        if not train_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x):  # x: [B,1,112,112]
        x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
        x = x.repeat(1,3,1,1)        # gray -> 3ch
        x = self.features(x).flatten(1)
        return x                     # [B,512]

class FeatureAE(nn.Module):
    """512 -> 128 -> 512 MLP Autoencoder (encode-only used at inference)."""
    def __init__(self, d_in=512, d_z=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, d_z)
        )
        self.dec = nn.Sequential(
            nn.Linear(d_z, 256), nn.ReLU(),
            nn.Linear(256, d_in)
        )
    def encode(self, f): return self.enc(f)
    def decode(self, z): return self.dec(z)
    def forward(self, f): return self.decode(self.encode(f))

class CNN_AE_LSTM_Base(nn.Module):
    """
    Frozen ResNet -> (frozen) FeatureAE.encode -> LSTM over time.
    Returns last hidden by default.
    """
    def __init__(self, cnn_extractor, feature_ae, freeze_ae=True, hidden=256, num_layers=2, dropout=0.3, d_z=128):
        super().__init__()
        self.cnn = cnn_extractor
        self.ae = feature_ae
        self.freeze_ae = freeze_ae
        self.lstm = nn.LSTM(
            input_size=d_z, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):  # x: [B,T,1,H,W]
        B, Tt, C, Hh, Ww = x.shape
        x = x.view(B*Tt, C, Hh, Ww)
        with torch.no_grad():             # CNN frozen
            f = self.cnn(x)               # [B*T, 512]
        if self.freeze_ae:
            with torch.no_grad():
                z = self.ae.encode(f)     # [B*T, 128]
        else:
            z = self.ae.encode(f)
        z = z.view(B, Tt, -1)             # [B,T,128]
        out, (hn, _) = self.lstm(z)       # hn: [layers,B,hidden]
        h = hn[-1]                        # [B,hidden]
        return h

class ClassifierHead(nn.Module):
    """Use attribute name 'layers' to match checkpoint keys: head.layers.*"""
    def __init__(self, input_dim=256, num_classes=6, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.layers(x)

class CNN_LSTM_Model(nn.Module):
    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self.head = head
    def forward(self, x):  # x: [B,T,1,H,W]
        return self.head(self.base(x))

# ========= Loader (handles possible key-name drift) =========
@st.cache_resource
def load_models():
    device = torch.device("cpu")  # Streamlit usually runs on CPU; works for all

    cnn  = ResNetFrameEncoder(train_backbone=False)
    ae   = FeatureAE(d_in=cnn.out_dim, d_z=128)
    base = CNN_AE_LSTM_Base(cnn, ae, freeze_ae=True, hidden=256, num_layers=2, dropout=0.3, d_z=128)
    head = ClassifierHead(input_dim=256, num_classes=len(EMOTIONS), dropout=0.3)
    model = CNN_LSTM_Model(base, head)

    # Load state_dicts (strict to catch mismatches)
    ae_sd    = torch.load(AE_PATH, map_location=device)
    model_sd = torch.load(MODEL_PATH, map_location=device)

    # If someone saved with head.net.*, remap to head.layers.*
    if any(k.startswith("head.net.") for k in model_sd.keys()):
        model_sd = {k.replace("head.net.", "head.layers."): v for k, v in model_sd.items()}

    ae.load_state_dict(ae_sd, strict=True)
    model.load_state_dict(model_sd, strict=True)

    ae.eval(); model.eval()
    return ae.to(device), model.to(device)

# ========= Preprocessing =========
def preprocess_frames(video_path: str, frame_count: int = T, use_face_crop: bool = True, pad_ratio: float = 0.12, max_side: int = 1600) -> torch.Tensor:

    """Read video, sample T evenly spaced frames, convert to model-ready tensor [1,T,1,112,112]."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total <= 0:
        cap.release()
    idx = np.linspace(0, max(total - 1, 0), frame_count).astype(int)
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(_to_gray_112(frame, use_face_crop=use_face_crop, pad_ratio=pad_ratio))
    cap.release()

    if len(frames) == 0:
        frames.append(_to_gray_112(frame, use_face_crop=use_face_crop,pad_ratio=pad_ratio, max_side=max_side))
    while len(frames) < frame_count:
        frames.append(frames[-1])

    arr = np.stack(frames[:frame_count], axis=0)   # [T,H,W]
    arr = np.ascontiguousarray(arr)                # avoid non-writable warnings
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(2)  # [1,T,1,H,W]
    return x

def preprocess_image(img_bgr: np.ndarray, use_face_crop: bool = True, pad_ratio: float = 0.12, max_side: int = 1600) -> torch.Tensor:
    """Single image → repeat to T frames to simulate a clip. Returns [1,T,1,112,112]."""
    g = _to_gray_112(img_bgr, use_face_crop=use_face_crop, pad_ratio=pad_ratio, max_side=max_side)
    arr = np.stack([g] * T, axis=0)                # [T,H,W]
    arr = np.ascontiguousarray(arr)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(2)  # [1,T,1,H,W]
    return x

# ========= Inference (with optional TTA) =========
@torch.inference_mode()
def predict_emotion(model: nn.Module, ae: nn.Module, x: torch.Tensor, tta: bool = False):
    """
    x: [1,T,1,112,112]
    If tta=True, average logits over a few simple test-time transforms.
    """
    if not tta:
        logits = model(x)
    else:
        # simple, safe TTAs that preserve training distribution
        xs = [x]
        # Horizontal flip
        xs.append(torch.flip(x, dims=[4]))             # flip width
        # Slightly different padding ratios baked into preprocessing isn't available here,
        # so we just use flip TTA. (More TTAs would require re-preprocessing.)
        logits = torch.stack([model(xi) for xi in xs], dim=0).mean(0)

    probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    label = EMOTIONS[pred_idx]
    prob_dict = {EMOTIONS[i]: float(p) for i, p in enumerate(probs)}
    return label, prob_dict
