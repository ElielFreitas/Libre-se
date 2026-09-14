#!/usr/bin/env python3
"""
Treino de ALFABETO (sinais estaticos) com Random Forest.

Diferente do train_rotulado.py (palavras dinamicas, sequencias de 30
frames), aqui cada letra e uma POSE: um unico frame de landmarks
normalizados. Extrai todos os frames de mao parada de cada video
palavras2/<LETRA>.mp4, filtra os frames de transicao pela velocidade
da mao e treina um classificador simples.

Uso: python3 train_alfabeto.py
"""

import pickle
import numpy as np
from pathlib import Path

import cv2
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report

from libres_features import N_LANDMARKS, N_COORDS, FEAT_DIM, normalize_landmarks

PROJECT_DIR = Path(__file__).parent
PALAVRAS_DIR = PROJECT_DIR / "palavras2"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"
OUTPUT_MODEL = PROJECT_DIR / "modelo_alfabeto.pkl"

# Letras de 1 caractere seguidas de .mp4 (A.mp4, B.mp4 ...). Z nao
# aparece: em Libras, J e Z envolvem movimento, nao pose estatica.
LETTER_RE = None  # filtro feito via nome do arquivo abaixo

# frames de transicao (mao se mexendo) viram lixo: mantemos so os 60%
# mais parados de cada video
VELOCITY_KEEP_PCT = 0.60

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

print("=" * 60)
print("TREINO ALFABETO - Random Forest (poses estaticas)")
print("=" * 60)

landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
))


def extract_static_poses(video: Path) -> np.ndarray:
    """Extrai landmarks normalizados dos frames de mao parada do video.

    Retorna (N, 63): uma linha por frame estatico. Usa a velocidade da mao
    para descartar a transicao de entrada/saida do sinal.
    """
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(mp.ImageFormat.SRGB, data=rgb))
        if res and res.hand_landmarks:
            frames.append([c for lm in res.hand_landmarks[0]
                           for c in (lm.x, lm.y, lm.z)])
    cap.release()

    if len(frames) < 3:
        return np.empty((0, FEAT_DIM))

    seq = normalize_landmarks(np.array(frames)).reshape(-1, N_LANDMARKS, N_COORDS)

    # velocidade media da maquinaria de pontos entre frames consecutivos
    vel = np.linalg.norm(np.diff(seq, axis=0), axis=2).mean(axis=1)
    vel = np.concatenate([[vel[0]], vel])
    keep = vel <= np.quantile(vel, VELOCITY_KEEP_PCT)
    return seq[keep].reshape(-1, FEAT_DIM)


def main() -> None:
    videos = sorted(v for v in PALAVRAS_DIR.glob("*.mp4")
                    if len(v.stem) == 1 and v.stem.isalpha())
    print(f"\n[1/3] Extraindo poses estaticas de {len(videos)} videos de letra...")

    X, y, groups = [], [], []
    for v in videos:
        letter = v.stem.upper()
        poses = extract_static_poses(v)
        print(f"    {letter}: {len(poses):3d} poses de {v.name}")
        for p in poses:
            X.append(p)
            y.append(letter)
            groups.append(str(v))  # 1 grupo por video: split honesto N/A

    X = np.array(X)
    y = np.array(y)
    print(f"\n  Total: {len(X)} poses, {len(np.unique(y))} letras")

    # Split aleatorio simples: os frames de um mesmo video sao da mesma
    # pessoa, mas com 1 video por letra nao ha outro jeito. O teste abaixo
    # serve de SANITY CHECK apenas; a metrica real do alfabeto e na camera.
    print("\n[2/3] Treino/teste 80/20 (sanity check)...")
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, te = idx[:cut], idx[cut:]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])

    print("\n[3/3] Treinando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42,
    )
    model.fit(Xtr, y[tr])

    acc_tr = accuracy_score(y[tr], model.predict(Xtr))
    acc_te = accuracy_score(y[te], model.predict(Xte))
    print(f"\n  >>> ACURACIA TREINO: {acc_tr*100:.1f}% <<<")
    print(f"  >>> ACURACIA TESTE: {acc_te*100:.1f}% <<<")
    print("\n  Relatorio por letra (teste):")
    print(classification_report(y[te], model.predict(Xte), zero_division=0))

    # retreina com TUDO pro modelo final (padrao quando o teste e so sanity)
    model.fit(scaler.fit_transform(X), y)

    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "letters": sorted(np.unique(y).tolist()),
        }, f)
    print(f"Modelo salvo em: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
