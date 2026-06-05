#!/usr/bin/env python3
"""
Extrai sequencias de landmarks do dataset MINDS e salva como .npy
pra usar no treino junto com os dados coletados.
"""
import sys, os, csv, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_DIR = Path(__file__).parent
MINDS_CSV = PROJECT_DIR / "minds" / "libras_minds.csv"
OUTPUT_DIR = PROJECT_DIR / "coletados"

N_FRAMES = 30
N_LANDMARKS = 21
N_COORDS = 3
FEATS_PER_FRAME = N_LANDMARKS * N_COORDS

print("="*50)
print("INTEGRACAO DATASET MINDS")
print("="*50)

if not MINDS_CSV.exists():
    print(f"ERRO: {MINDS_CSV} nao encontrado")
    exit(1)

print("Lendo CSV...")
videos = defaultdict(lambda: {'frames': [], 'category': ''})

with open(MINDS_CSV) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        vname = row['video_name']
        videos[vname]['category'] = row['category']
        hand = []
        missing_hand = row.get('missing_hand', 'False') == 'True'
        if not missing_hand:
            for j in range(N_LANDMARKS):
                x = float(row.get(f'hand_0_{j}_x', 0))
                y = float(row.get(f'hand_0_{j}_y', 0))
                z = float(row.get(f'hand_0_{j}_z', 0))
                hand.extend([x, y, z])
        else:
            hand = [0.0] * FEATS_PER_FRAME
        videos[vname]['frames'].append(hand)

print(f"Total videos: {len(videos)}")

print("\nProcessando videos...")
total_salvos = 0
classes_salvas = set()

for vname, vdata in videos.items():
    label = vdata['category'].lower().replace(' ', '_')
    frames = np.array(vdata['frames'])

    valid_frames = frames[~np.all(frames == 0, axis=1)]
    if len(valid_frames) < 10:
        continue

    n_segments = max(1, len(valid_frames) // N_FRAMES)
    for seg in range(n_segments):
        start = seg * N_FRAMES
        end = start + N_FRAMES
        segment = valid_frames[start:end]

        if len(segment) < N_FRAMES:
            pad = np.tile(segment[-1:], (N_FRAMES - len(segment), 1))
            segment = np.vstack([segment, pad])

        pasta = OUTPUT_DIR / label
        pasta.mkdir(parents=True, exist_ok=True)

        fname = pasta / f"minds_{Path(vname).stem}_seg{seg:03d}.npy"
        np.save(fname, np.array(segment[:N_FRAMES]))
        total_salvos += 1
        classes_salvas.add(label)

print(f"\nResumo:")
print(f"  Total segmentos salvos: {total_salvos}")
print(f"  Classes: {len(classes_salvas)}")
for c in sorted(classes_salvas):
    pasta = OUTPUT_DIR / c
    n = len(list(pasta.glob("minds_*.npy")))
    print(f"    {c}: {n} segmentos")
