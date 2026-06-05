#!/usr/bin/env python3
"""
Detecta e corta sinais individuais de videos compilados automaticamente.
Usa MediaPipe para detectar movimento da mao e identifica pausas entre sinais.
Processa no maximo 1 frame a cada FRAME_SKIP para ser viavel.
"""

import subprocess, sys, json, re
from pathlib import Path
import numpy as np
import mediapipe as mp
import cv2

PROJECT_DIR = Path(__file__).parent
YOUTUBE_DIR = PROJECT_DIR / "youtube_videos"
PALAVRAS_DIR = PROJECT_DIR / "palavras2"
HAND_MODEL = Path.home() / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"

PALAVRAS_DIR.mkdir(exist_ok=True)

# Parametros de deteccao
FRAME_SKIP = 5
MIN_SEGMENT_FRAMES = 3
PAUSE_SKIP_COUNT = 3
MOVEMENT_THRESHOLD = 0.012
MAX_DURATION_SECONDS = 300

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


def get_video_info(video_path: Path) -> tuple[float, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error",
         "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate,duration",
         "-of", "default=noprint_wrappers=1:nokey=1",
         str(video_path)],
        capture_output=True, text=True, timeout=30
    )
    lines = result.stdout.strip().split('\n')
    fps = 30.0
    if len(lines) > 0 and '/' in lines[0]:
        num, den = lines[0].split('/')
        fps = float(num) / float(den)
    duration = 0
    if len(lines) > 1 and lines[1]:
        duration = float(lines[1])
    return fps, int(duration)


def extract_motion_segments(video_path: Path, fps: float) -> list[tuple[float, float]]:
    """Detecta segmentos de movimento usando diferenca entre frames (mais rapido que MediaPipe)."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    prev_gray = None
    motion_frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0 and frame is not None and frame.size > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_pct = np.count_nonzero(thresh) / thresh.size
                motion_frames.append((frame_idx / fps, motion_pct))

            prev_gray = gray

        frame_idx += 1
        if frame_idx % 1000 == 0:
            pct = frame_idx / total_frames * 100
            print(f"\r  Detectando movimento: {frame_idx}/{total_frames} ({pct:.0f}%)",
                  end="", flush=True)

    cap.release()
    print()

    # Converte pct de movimento em segmentos
    segments = []
    in_motion = False
    start_time = 0
    pause_count = 0

    for time_s, motion in motion_frames:
        is_moving = motion > MOVEMENT_THRESHOLD

        if is_moving and not in_motion:
            in_motion = True
            start_time = time_s
            pause_count = 0
        elif not is_moving and in_motion:
            pause_count += 1
            if pause_count >= PAUSE_SKIP_COUNT:
                seg_duration = time_s - start_time
                if seg_duration >= 0.5 and seg_duration <= 10.0:
                    segments.append((start_time, time_s))
                in_motion = False
                pause_count = 0

    if in_motion:
        end_time = motion_frames[-1][0]
        seg_duration = end_time - start_time
        if seg_duration >= 0.5 and seg_duration <= 10.0:
            segments.append((start_time, end_time))

    return segments


def verify_segment_with_mediapipe(video_path: Path, start: float, end: float,
                                   label: str) -> bool:
    """Verifica se o segmento tem mao via MediaPipe (rapido, poucos frames)."""
    landmarker = HandLandmarker.create_from_options(hand_options)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    max_frames = int((end - start) * fps)
    hand_count = 0
    checked = 0

    for i in range(0, max_frames, FRAME_SKIP * 2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps) + i)
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue
        checked += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if result and result.hand_landmarks:
            hand_count += 1

    cap.release()
    landmarker.close()

    if checked == 0:
        return False
    return (hand_count / checked) >= 0.3


def cut_segment_ffmpeg(video_path: Path, output_path: Path,
                       start_time: float, end_time: float) -> bool:
    duration = end_time - start_time
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", str(start_time),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        str(output_path)
    ], capture_output=True, text=True, timeout=300)
    return result.returncode == 0


def main() -> None:
    videos = sorted(YOUTUBE_DIR.glob("*.mp4"))
    if not videos:
        print("Nenhum video encontrado em youtube_videos/")
        return

    print(f"Encontrados {len(videos)} videos.")

    for v in videos:
        print(f"\n{'='*60}")
        print(f"Processando: {v.name}")
        print(f"{'='*60}")

        # Pula videos muito longos
        try:
            fps, duration = get_video_info(v)
        except Exception as e:
            print(f"  Erro lendo info: {e}")
            continue

        if duration > MAX_DURATION_SECONDS:
            print(f"  Pulando ({duration}s > {MAX_DURATION_SECONDS}s limite)")
            continue

        print(f"  Duracao: {duration}s, FPS: {fps:.1f}")

        # Extrai label base do nome
        m = re.match(r'^\d+_(\w+)_', v.stem)
        label_base = m.group(1) if m else v.stem

        # Detecta segmentos por movimento
        segments = extract_motion_segments(v, fps)
        print(f"  Encontrados {len(segments)} segmentos candidatos")

        if not segments:
            continue

        # Inicializa landmarker uma vez
        landmarker = HandLandmarker.create_from_options(hand_options)

        saved = 0
        for i, (start_time, end_time) in enumerate(segments):
            # Verifica com MediaPipe
            cap = cv2.VideoCapture(str(v))
            vfps = cap.get(cv2.CAP_PROP_FPS)
            if vfps <= 0:
                vfps = 30

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * vfps))
            max_frames = int((end_time - start_time) * vfps)
            hand_count = 0
            checked = 0

            for fi in range(0, max_frames, FRAME_SKIP):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * vfps) + fi)
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    continue
                checked += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)
                if result and result.hand_landmarks:
                    hand_count += 1

            cap.release()

            if checked == 0 or (hand_count / checked) < 0.3:
                continue

            dur = end_time - start_time
            output_name = f"{label_base}_auto_{i+1:03d}.mp4"
            output_path = PALAVRAS_DIR / output_name

            hand_pct = hand_count / checked * 100
            print(f"  Seg {i+1}: {start_time:.1f}s-{end_time:.1f}s "
                  f"({dur:.1f}s, {hand_pct:.0f}% mao) salvando...", end=" ", flush=True)

            if cut_segment_ffmpeg(v, output_path, start_time, end_time):
                print(f"OK -> {output_name}")
                saved += 1
            else:
                print("FALHOU")

        landmarker.close()

        if saved == 0:
            print("  Nenhum segmento valido encontrado.")

    print(f"\nConcluido! Segmentos salvos em {PALAVRAS_DIR}")
    total = len(list(PALAVRAS_DIR.glob("*_auto_*.mp4")))
    print(f"Total: {total} clips extraidos")
    print("\nProximo passo: renomeie os arquivos em palavras2/")
    print("Ex: saudacao_auto_001.mp4 -> oi.mp4")
    print("    saudacao_auto_002.mp4 -> tchau.mp4")
    print("    comida_auto_001.mp4 -> comer.mp4")
    print("Depois rode: python3 train_rotulado.py")


if __name__ == '__main__':
    main()
