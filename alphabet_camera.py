#!/usr/bin/env python3
"""
Reconhecimento de ALFABETO em tempo real (modelo estatico por frame,
modelo_alfabeto.pkl). Soletra letra a letra e monta a palavra na tela.

Uso: python3 alphabet_camera.py

Teclas:
  Q           = sair
  ESPACO      = limpa a palavra montada
  BACKSPACE   = apaga ultima letra
  C           = entra/sai do MODO COLETA (grave amostras suas)
                no modo coleta, digite a letra (A-Y) para escolher o alvo
                e segure a pose: grava N frames bons e salva em
                coleta_alfabeto/<LETRA>/ como .npy
"""

import pickle, sys, time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from libres_features import FEAT_DIM, N_LANDMARKS, N_COORDS, normalize_landmarks

PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "modelo_alfabeto.pkl"
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"
COLETA_DIR = PROJECT_DIR / "coleta_alfabeto"

CONFIDENCE_THRESHOLD = 0.55
STABLE_FRAMES = 8          # predicao so vira "letra" depois de 8 frames iguais
COLETA_FRAMES = 40         # frames gravados por rodada de coleta

print("=" * 50)
print("ALFABETO EM TEMPO REAL + COLETA")
print("=" * 50)

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
model, scaler = data["model"], data["scaler"]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
))

cap = None
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera no indice {i}")
        break
    cap.release()
else:
    sys.exit("Nenhuma camera!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

word: list[str] = []
stable = ""
stable_count = 0
modo_coleta = False
coleta_target: str | None = None
coleta_buffer: list[np.ndarray] = []
coleta_n = 0

print("\n>>> FACA A LETRA EM FRENTE A CAMERA <<<")
print("C = coleta | ESPACO = limpar palavra | Q = sair\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(mp.ImageFormat.SRGB, data=rgb))

    pred_letra, pred_conf = "", 0.0

    if res and res.hand_landmarks:
        hand = res.hand_landmarks[0]
        coords = np.array([c for lm in hand for c in (lm.x, lm.y, lm.z)],
                          dtype=np.float32).reshape(1, N_LANDMARKS, N_COORDS)
        feat = normalize_landmarks(coords.reshape(1, -1))
        proba = model.predict_proba(scaler.transform(feat))[0]
        top = int(proba.argmax())
        pred_letra = model.classes_[top]
        pred_conf = float(proba[top])

        # desenha esqueleto
        h, w = frame.shape[:2]
        for lm in hand:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

        # ---- suavizacao temporal ----
        if pred_conf >= CONFIDENCE_THRESHOLD and pred_letra == stable:
            stable_count += 1
        else:
            stable, stable_count = pred_letra, 1

        # ---- modo coleta ----
        if modo_coleta and coleta_target:
            coleta_buffer.append(coords)
            if len(coleta_buffer) >= COLETA_FRAMES:
                pasta = COLETA_DIR / coleta_target
                pasta.mkdir(parents=True, exist_ok=True)
                arq = pasta / f"{coleta_target}_{int(time.time())}_{coleta_n}.npy"
                # guarda a pose MEDIDA (mais estavel do que 1 frame so)
                np.save(arq, np.mean(np.concatenate(coleta_buffer), axis=0))
                coleta_n += 1
                coleta_buffer.clear()
                print(f"  salvo: {arq.name}")

        # ---- confirmacao de letra na palavra ----
        if stable_count == STABLE_FRAMES and (not word or word[-1] != stable):
            word.append(stable)
    else:
        stable, stable_count = "", 0
        coleta_buffer.clear()

    # ---- HUD ----
    txt_palavra = "".join(word)
    cv2.putText(frame, f"Palavra: {txt_palavra}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    if stable_count >= 2 and pred_conf >= CONFIDENCE_THRESHOLD:
        cv2.putText(frame, f"{stable} ({pred_conf*100:.0f}%)", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    if modo_coleta:
        msg = f"[COLETA] alvo: {coleta_target or '?'} | gravados: {coleta_n}"
        cv2.putText(frame, msg, (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 255), 2)

    cv2.imshow("Libras - Alfabeto", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        word.clear()
    elif key == 8:  # backspace
        word = word[:-1]
    elif key == ord("c"):
        modo_coleta = not modo_coleta
        coleta_target = None
        coleta_buffer.clear()
        coleta_n = 0
    elif modo_coleta and chr(key).upper() in "ABCDEFGHIKLMNOPQRSTUVWXY":
        coleta_target = chr(key).upper()
        coleta_n = 0
        print(f"  coletando letra: {coleta_target}")

cap.release()
cv2.destroyAllWindows()
print(f"Palavra final: {''.join(word) or '(vazia)'}")
print("Encerrado!")
