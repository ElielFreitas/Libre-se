#!/usr/bin/env python3
"""
Corta trechos de videos do YouTube compilados em clips individuais.
Uso: python3 slice_signs.py
     Navega pelos videos em youtube_videos/, permite marcar inicio/fim
     e salva o trecho em palavras2/ com o nome da classe.
"""

import subprocess, sys, time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
YOUTUBE_DIR = PROJECT_DIR / "youtube_videos"
PALAVRAS_DIR = PROJECT_DIR / "palavras2"

PALAVRAS_DIR.mkdir(exist_ok=True)

def get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, timeout=30
    )
    return float(result.stdout.strip())

def cut_segment(video_path: Path, output_path: Path, start: float, end: float) -> bool:
    duration = end - start
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", str(start),
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
    print()

    for v in videos:
        print(f"  [{v.name}]")
        try:
            dur = get_video_duration(v)
            print(f"  Duracao: {dur:.1f}s")
        except Exception as e:
            print(f"  Erro ao ler duracao: {e}")
            continue

        print(f"  Informe os cortes no formato: inicio fim label")
        print(f"  Ex: 0.5 3.5 azul   (corta de 0.5s a 3.5s, salva como azul_yt.mp4)")
        print(f"  Ex: 5.2 8.1 vermelho")
        print(f"  Linha vazia = pular video")
        print(f"  'q' = sair")
        print()

        while True:
            linha = input(f"  > ").strip()
            if not linha:
                break
            if linha.lower() == 'q':
                print("Encerrando.")
                return

            partes = linha.split()
            if len(partes) < 3:
                print("    Formato: inicio_seg fim_seg label")
                continue

            try:
                start = float(partes[0])
                end = float(partes[1])
                label = partes[2].lower().replace(" ", "_")
            except ValueError:
                print("    inicio e fim devem ser numeros")
                continue

            if start < 0 or end > dur or start >= end:
                print(f"    Intervalo invalido (video tem {dur:.1f}s)")
                continue

            output = PALAVRAS_DIR / f"{label}_yt.mp4"
            if output.exists():
                resp = input(f"    {output.name} ja existe. Sobrescrever? (s/N): ")
                if resp.lower() != 's':
                    print("    Pulando.")
                    continue

            print(f"    Cortando {start:.1f}s -> {end:.1f}s como '{label}'...", end=" ", flush=True)
            if cut_segment(v, output, start, end):
                print("OK")
            else:
                print("FALHOU")

        print()

if __name__ == "__main__":
    main()
