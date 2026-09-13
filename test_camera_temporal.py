#!/usr/bin/env python3
"""
Teste em tempo real com camera - XGBoost + confianca.
Usa predict_proba() e exibe "DESCONHECIDO" quando confianca < 80%.
"""

import sys
from pathlib import Path
from typing import Optional
import mediapipe as mp
import cv2
import pickle
import numpy as np
from collections import deque

PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "modelo_rotulado_xgb.pkl"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"
N_FRAMES: int = 30
N_LANDMARKS: int = 21
N_COORDS: int = 3
FEAT_DIM: int = N_LANDMARKS * N_COORDS
CONFIDENCE_THRESHOLD: float = 0.8

print("=" * 50)
print("PREDICAO EM TEMPO REAL - MODELO OTIMIZADO")
print("=" * 50)

print("Carregando modelo...")
with open(MODEL_PATH, 'rb') as f:
    data = pickle.load(f)
model = data['model']
scaler = data['scaler']
le = data['label_encoder']

print(f"Classes: {list(le.classes_)}")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(hand_options)


def normalize_landmarks(seq: np.ndarray) -> np.ndarray:
    """Converte coordenadas absolutas para posicoes relativas ao punho e a
    escala da propria mao (deve espelhar normalize_landmarks do treino)."""
    r = seq.reshape(-1, N_LANDMARKS, N_COORDS)
    r = r - r[:, 0:1]

    scale = np.linalg.norm(r[:, 9], axis=1, keepdims=True)
    scale = np.clip(scale, 1e-6, None)
    r = r / scale

    return r.reshape(-1, FEAT_DIM)


def compute_finger_angles(seq: np.ndarray) -> np.ndarray:
    """Calcula angulos das articulacoes dos dedos. Retorna (T, 10)."""
    r = seq.reshape(-1, N_LANDMARKS, N_COORDS)

    triples = [
        (1, 2, 3), (2, 3, 4),
        (5, 6, 7), (6, 7, 8),
        (9, 10, 11), (10, 11, 12),
        (13, 14, 15), (14, 15, 16),
        (17, 18, 19), (18, 19, 20),
    ]

    all_angles = []
    for a, b, c in triples:
        v1 = r[:, a] - r[:, b]
        v2 = r[:, c] - r[:, b]
        dot = np.sum(v1 * v2, axis=1)
        norm = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        norm = np.clip(norm, 1e-8, None)
        cos_ang = np.clip(dot / norm, -1.0, 1.0)
        all_angles.append(np.arccos(cos_ang))

    return np.column_stack(all_angles)


def add_relative_features(seq: np.ndarray) -> np.ndarray:
    """Distancias relativas entre landmarks."""
    r = seq.reshape(-1, N_LANDMARKS, N_COORDS)
    extra = []

    wrist = r[:, 0]
    for tip in [4, 8, 12, 16, 20]:
        dist = np.linalg.norm(r[:, tip] - wrist, axis=1)
        extra.append(dist)

    for a, b in [(4, 8), (8, 12), (12, 16), (16, 20),
                 (4, 3), (8, 7), (12, 11), (16, 15), (20, 19)]:
        dist = np.linalg.norm(r[:, a] - r[:, b], axis=1)
        extra.append(dist)

    return np.column_stack(extra) if extra else np.zeros((seq.shape[0], 1))


def compute_enhanced_features(seq: np.ndarray) -> np.ndarray:
    """Vetor de features: estatisticas temporais + relativas + angulares."""
    T, D = seq.shape
    features = []

    for d in range(D):
        col = seq[:, d]
        features.extend([
            np.mean(col), np.std(col), np.min(col), np.max(col),
            np.max(col) - np.min(col), col[-1] - col[0],
            np.mean(np.abs(np.diff(col))), np.median(col),
        ])

    centroids = seq.reshape(T, N_LANDMARKS, N_COORDS).mean(axis=1)
    for c in range(N_COORDS):
        traj = centroids[:, c]
        features.extend([
            np.mean(traj), np.std(traj), np.max(traj) - np.min(traj),
            traj[-1] - traj[0], np.mean(np.abs(np.diff(traj))),
        ])

    rel = add_relative_features(seq)
    for d in range(rel.shape[1]):
        col = rel[:, d]
        features.extend([
            np.mean(col), np.std(col), np.max(col) - np.min(col),
            col[-1] - col[0], np.mean(np.abs(np.diff(col))),
        ])

    angles = compute_finger_angles(seq)
    for d in range(angles.shape[1]):
        col = angles[:, d]
        features.extend([
            np.mean(col), np.std(col), np.max(col) - np.min(col),
            col[-1] - col[0], np.mean(np.abs(np.diff(col))),
        ])

    return np.array(features)


print("Inicializando camera...")
# API padrao do OpenCV: funciona em Windows, Linux e macOS
# (CAP_V4L2 e exclusivo do Linux e falhava em outros sistemas)
cap = None
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera encontrada no indice {i}")
        break
    cap.release()
else:
    print("Nenhuma camera!")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

window: deque = deque(maxlen=N_FRAMES)
frame_count: int = 0
last_pred: str = ""
pred_count: int = 0

print("\n>>> FACA UM SINAL EM LIBRAS EM FRENTE A CAMERA <<<")
print("Pressione 'q' para sair")
print("-" * 40)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    features_list: list[float] = []
    hand_detected = False
    if result and result.hand_landmarks:
        hand = result.hand_landmarks[0]
        for lm in hand:
            features_list.extend([lm.x, lm.y, lm.z])
        hand_detected = True

        h, w = frame.shape[:2]
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    window.append(features_list if len(features_list) == FEAT_DIM else None)

    prediction: str = ""
    confidence: float = 0.0

    if hand_detected and len(window) == N_FRAMES and all(f is not None for f in window):
        seq = np.array(window)
        seq = normalize_landmarks(seq)
        feat = compute_enhanced_features(seq).reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        try:
            probas = model.predict_proba(feat_scaled)[0]
            pred_idx = int(np.argmax(probas))
            conf = float(probas[pred_idx])

            if conf >= CONFIDENCE_THRESHOLD:
                palavra = le.inverse_transform([pred_idx])[0]
                if palavra == last_pred:
                    pred_count += 1
                else:
                    last_pred = palavra
                    pred_count = 1
                if pred_count >= 3:
                    prediction = palavra.upper()
                    confidence = conf
            else:
                last_pred = ""
                pred_count = 0
                prediction = "DESCONHECIDO"
                confidence = conf
        except Exception as e:
            print(f"Erro na predicao: {e}")

    if prediction:
        if prediction == "DESCONHECIDO":
            cv2.putText(frame, ">>> DESCONHECIDO", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            cv2.putText(frame, f">>> {prediction} ({confidence*100:.0f}%)",
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    cv2.putText(frame, "Q = Sair", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Libras - Reconhecimento Otimizado', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Encerrado!")
