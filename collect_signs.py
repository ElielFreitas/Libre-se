#!/usr/bin/env python3
"""
Coletor de sinais individuais via webcam.
Grava os landmarks de cada sinal para treinar o modelo.
"""
import sys, os, json, time
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp

PROJECT_DIR = Path(__file__).parent
COLETADOS_DIR = PROJECT_DIR / "coletados"
HAND_MODEL = Path.home() / "hand_landmarker.task"

if not HAND_MODEL.exists():
    HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"

N_FRAMES = 30
FRAMES_POR_TAKE = N_FRAMES

TODOS_SINAIS = []
alfabeto = [chr(l) for l in range(ord('A'), ord('Z') + 1)]
numeros = [str(n) for n in range(1, 11)]
TODOS_SINAIS.extend(alfabeto)
TODOS_SINAIS.extend(numeros)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def init_landmarker():
    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        num_hands=2,
        running_mode=VisionRunningMode.IMAGE,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
    )
    return HandLandmarker.create_from_options(opts)

def contar_takes(sinal):
    pasta = COLETADOS_DIR / sinal
    if not pasta.exists():
        return 0
    return len(list(pasta.glob("*.npy")))

def coletar():
    landmarker = init_landmarker()
    print("Procurando camera...")
    cap = None
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} encontrada")
            break
        cap.release()
    else:
        print("NENHUMA CAMERA ENCONTRADA")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    COLETADOS_DIR.mkdir(exist_ok=True)

    idx = 0
    coletando = False
    frames_buffer = []
    gravando = False
    take_atual = 0

    print("\n" + "="*50)
    print("COLETOR DE SINAIS")
    print("="*50)
    print("  ← →  : navegar entre sinais")
    print("  ESPAÇO: gravar um take")
    print("  R     : resetar takes desse sinal")
    print("  Q     : sair")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        mao_detectada = False
        if result and result.hand_landmarks:
            mao_detectada = True
            h, w = frame.shape[:2]
            hand = result.hand_landmarks[0]
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            if gravando:
                feat = []
                for lm in hand:
                    feat.extend([lm.x, lm.y, lm.z])
                frames_buffer.append(feat)

                if len(frames_buffer) >= FRAMES_POR_TAKE:
                    pasta = COLETADOS_DIR / TODOS_SINAIS[idx]
                    pasta.mkdir(exist_ok=True)
                    fname = pasta / f"take_{take_atual:03d}.npy"
                    np.save(fname, np.array(frames_buffer[:FRAMES_POR_TAKE]))
                    print(f"  ✓ {TODOS_SINAIS[idx]} take {take_atual} salvo")
                    gravando = False
                    frames_buffer = []
                    take_atual += 1

        sinal = TODOS_SINAIS[idx]
        total_takes = contar_takes(sinal)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10),  (400, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        cv2.putText(frame, f"SINAL: {sinal}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Takes: {total_takes}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if gravando:
            prog = len(frames_buffer) / FRAMES_POR_TAKE
            bar_w = int(prog * 300)
            cv2.rectangle(frame, (20, 100), (20 + bar_w, 115), (0, 0, 255), -1)
            cv2.putText(frame, f"GRAVANDO... {len(frames_buffer)}/{FRAMES_POR_TAKE}",
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Coletor de Sinais Libras", frame)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if not gravando and mao_detectada:
                gravando = True
                frames_buffer = []
                take_atual = total_takes + 1
        elif key == ord('r'):
            pasta = COLETADOS_DIR / sinal
            if pasta.exists():
                import shutil
                shutil.rmtree(pasta)
                print(f"  ✗ Takes de {sinal} removidos")
        elif key == 81 or key == ord(','):  # seta esquerda
            idx = (idx - 1) % len(TODOS_SINAIS)
            gravando = False
            frames_buffer = []
        elif key == 83 or key == ord('.'):  # seta direita
            idx = (idx + 1) % len(TODOS_SINAIS)
            gravando = False
            frames_buffer = []

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    print("\nResumo final:")
    total = 0
    for s in TODOS_SINAIS:
        n = contar_takes(s)
        if n > 0:
            print(f"  {s:4s}: {n} takes")
            total += n
    print(f"  Total: {total} takes de {sum(1 for s in TODOS_SINAIS if contar_takes(s) > 0)} sinais")

if __name__ == "__main__":
    coletar()
