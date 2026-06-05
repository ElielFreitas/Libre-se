#!/usr/bin/env python3
"""
Treino XGBoost com landmarks normalizados, features angulares
e divisao treino/teste antes da augmentacao (sem vazamento de dados).
"""

import sys, os, re, pickle, random
from typing import Optional
import numpy as np
from pathlib import Path
from collections import Counter

import mediapipe as mp
import cv2

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

PROJECT_DIR = Path(__file__).parent
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"

PALAVRAS_DIR = PROJECT_DIR / "palavras2"
YOUTUBE_DIR = PROJECT_DIR / "youtube_videos"
COLETADOS_DIR = PROJECT_DIR / "coletados"
OUTPUT_MODEL = PROJECT_DIR / "modelo_rotulado_xgb.pkl"

N_FRAMES: int = 30
N_LANDMARKS: int = 21
N_COORDS: int = 3
FEAT_DIM: int = N_LANDMARKS * N_COORDS
AUG_MULTIPLIER: int = 3
CONFIDENCE_THRESHOLD: float = 0.8

random.seed(42)
np.random.seed(42)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

print("=" * 60)
print("TREINO OTIMIZADO - AUGMENTACAO + XGBoost")
print("=" * 60)

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(hand_options)

WORD_GROUPS: dict[str, list[str]] = {
    'amigo': ['amigo', 'amiga'],
    'avo': ['avó', 'avô'],
    'porfavor': ['porfavor', 'por_favor'],
}


def normalize_label(label: str) -> str:
    label = label.lower().strip().replace(' ', '_')
    for canonical, variants in WORD_GROUPS.items():
        if label in variants:
            return canonical
    return label


def extract_label_palavras2(filename: Path) -> Optional[str]:
    name = filename.stem
    parts = name.split('_', 1)
    return normalize_label(parts[0])


def extract_label_youtube(filename: Path) -> Optional[str]:
    name = filename.stem
    m = re.match(r'^\d+_(\w+)_\w{11}$', name)
    if m:
        return m.group(1)
    name_lower = name.lower().replace('_', ' ').replace('-', ' ')
    multi_sign = ['sinais de', '30 sinais', 'verbos em libras', 'cores em libras',
                  'dias da semana', 'meses do ano', 'numeros cardinais',
                  'aprenda todas as cores', 'minha familia', 'eu nao entendo',
                  'hello goodbye', '8 hours']
    for pat in multi_sign:
        if pat in name_lower:
            return None
    keywords = {
        'obrigado': 'obrigado', 'tchau': 'tchau',
        'oi': 'oi', 'ola': 'oi', 'olá': 'oi',
        'bom dia': 'bomdia', 'boa tarde': 'boatarde', 'boa noite': 'boanoite',
        'familia': 'familia', 'mae': 'mae', 'mãe': 'mae', 'pai': 'pai',
        'amigo': 'amigo', 'amiga': 'amigo',
        'irmao': 'irmao', 'irmão': 'irmao',
        'filho': 'filho', 'filha': 'filha',
        'comprar': 'comprar', 'preco': 'preco',
        'por favor': 'porfavor', 'prazer': 'prazer',
        'comer': 'comer', 'comida': 'comida',
        'beber': 'beber', 'casa': 'casa',
        'numero': 'numero', 'alfabeto': 'alfabeto', 'nome': 'nome',
        'saudacao': 'saudacao', 'saudacoes': 'saudacao',
        'cor': 'cor', 'verbo': 'verbo', 'animal': 'animal',
        'roupa': 'roupa', 'semana': 'semana',
        'mes': 'mes', 'mês': 'mes',
    }
    for key, label in keywords.items():
        if key in name_lower:
            return normalize_label(label)
    return None


def normalize_landmarks(seq: np.ndarray) -> np.ndarray:
    """Converte coordenadas absolutas para posicoes relativas ao punho."""
    r = seq.reshape(-1, N_LANDMARKS, N_COORDS)
    r = r - r[:, 0:1]
    return r.reshape(-1, FEAT_DIM)


def compute_finger_angles(seq: np.ndarray) -> np.ndarray:
    """Calcula angulos das articulacoes dos dedos.
    Retorna (T, 10): 2 angulos por dedo.
    """
    r = seq.reshape(-1, N_LANDMARKS, N_COORDS)
    T = r.shape[0]

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


def extract_landmarks_from_video(video_path: Path) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_read = 0
    max_total_read = N_FRAMES * 2

    while len(frames) < N_FRAMES and total_read < max_total_read:
        ret, frame = cap.read()
        if not ret:
            break
        total_read += 1
        if frame is not None and frame.size > 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            if result and result.hand_landmarks:
                hand = result.hand_landmarks[0]
                frames.append([coord for lm in hand for coord in (lm.x, lm.y, lm.z)])

    cap.release()
    if not frames:
        return None
    while len(frames) < N_FRAMES:
        frames.append(frames[-1])
    seq = np.array(frames[:N_FRAMES])
    return normalize_landmarks(seq)


def augment_sequence(seq: np.ndarray) -> list[np.ndarray]:
    T, D = seq.shape
    augs = []

    for _ in range(AUG_MULTIPLIER):
        aug = seq.copy()
        noise_scale = np.random.uniform(0.002, 0.008)
        aug += np.random.normal(0, noise_scale, aug.shape)

        scale = np.random.uniform(0.92, 1.08)
        aug *= scale

        shift = np.random.uniform(-0.02, 0.02, D)
        aug += shift

        if np.random.random() < 0.4:
            t_warp = np.random.uniform(0.85, 1.15)
            new_len = int(T * t_warp)
            indices = np.linspace(0, T - 1, new_len)
            warped = np.zeros((new_len, D))
            for d in range(D):
                warped[:, d] = np.interp(indices, np.arange(T), seq[:, d])
            if new_len >= T:
                aug = warped[:T]
            else:
                aug = np.zeros((T, D))
                aug[:new_len] = warped
                aug[new_len:] = warped[-1]

        if np.random.random() < 0.3:
            drop_pct = np.random.uniform(0.05, 0.15)
            n_drop = int(T * drop_pct)
            drop_idx = np.random.choice(T, n_drop, replace=False)
            for idx in drop_idx:
                if idx > 0:
                    aug[idx] = aug[idx - 1]

        augs.append(aug)

    return augs


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


def collect_data() -> tuple[np.ndarray, np.ndarray]:
    """Carrega dados brutos (landmarks) sem augmentacao."""
    X_raw: list[np.ndarray] = []
    y_raw_labels: list[str] = []

    print("\n[1/4] Extraindo landmarks dos videos...")

    if COLETADOS_DIR.exists():
        print(f"\n  coletados/:")
        pastas = sorted([p for p in COLETADOS_DIR.iterdir() if p.is_dir()])
        for pasta in pastas:
            label = pasta.name
            npy_files = sorted(pasta.glob("*.npy"))
            for npy_f in npy_files:
                seq = np.load(npy_f)
                if seq.shape == (N_FRAMES, FEAT_DIM):
                    seq = normalize_landmarks(seq)
                    X_raw.append(seq)
                    y_raw_labels.append(label)
            print(f"    {label:15s}: {len(npy_files)} takes")
        total_coletados = sum(len(list(p.glob('*.npy'))) for p in pastas)
        print(f"  >> Total coletados: {total_coletados}")

    # Sempre processa palavras2/ (ignora arquivos _auto_ nao renomeados)
    if PALAVRAS_DIR.exists():
        print(f"\n  palavras2/:")
        videos = sorted([v for v in PALAVRAS_DIR.glob("*.mp4") if "_auto_" not in v.stem])
        auto_count = len(list(PALAVRAS_DIR.glob("*_auto_*.mp4")))
        if auto_count > 0:
            print(f"    ({auto_count} arquivos _auto_ ignorados - renomeie-os primeiro)")
        for i, v in enumerate(videos):
            label = extract_label_palavras2(v)
            if not label:
                continue
            label = label.replace(' ', '_')
            print(f"    [{i+1}/{len(videos)}] {label:15s} <- {v.name[:30]}...",
                  end=" ", flush=True)
            seq = extract_landmarks_from_video(v)
            if seq is not None:
                X_raw.append(seq)
                y_raw_labels.append(label)
                print("OK")
            else:
                print("SEM MAO")

    # youtube_videos/ processado apenas se MINDS nao estiver carregado
    processar_youtube = not (COLETADOS_DIR.exists() and (COLETADOS_DIR / "barulho").exists())
    if processar_youtube and YOUTUBE_DIR.exists():
        print(f"\n  youtube_videos/:")
        videos = [v for v in YOUTUBE_DIR.glob("*.mp4") if not v.name.endswith('.webm')]
        for i, v in enumerate(videos):
            label = extract_label_youtube(v)
            if not label:
                continue
            label = label.replace(' ', '_')
            print(f"    [{i+1}/{len(videos)}] {label:15s} <- {v.name[:30]}...",
                  end=" ", flush=True)
            seq = extract_landmarks_from_video(v)
            if seq is not None:
                X_raw.append(seq)
                y_raw_labels.append(label)
                print("OK")
            else:
                print("SEM MAO")

    X_raw_arr = np.array(X_raw)
    y_raw_arr = np.array(y_raw_labels)
    n_real = len(X_raw_arr)

    print(f"\n  Total amostras reais: {n_real}")
    print(f"  Shape: {X_raw_arr.shape}")
    print(f"  Classes: {len(np.unique(y_raw_arr))}")

    return X_raw_arr, y_raw_arr


def build_feature_names() -> list[str]:
    """Gera nomes alinhados com compute_enhanced_features."""
    names = []
    for d in range(FEAT_DIM):
        for stat in ['media', 'std', 'min', 'max', 'range', 'diff', 'veloc', 'mediana']:
            names.append(f"lm{d}_{stat}")
    for c in range(N_COORDS):
        for stat in ['ctr_media', 'ctr_std', 'ctr_range', 'ctr_diff', 'ctr_veloc']:
            names.append(f"centro{c}_{stat}")
    for d in range(14):
        for stat in ['rel_media', 'rel_std', 'rel_range', 'rel_diff', 'rel_veloc']:
            names.append(f"rel{d}_{stat}")
    for d in range(10):
        for stat in ['ang_media', 'ang_std', 'ang_range', 'ang_diff', 'ang_veloc']:
            names.append(f"ang{d}_{stat}")
    return names


def main() -> None:
    X_raw, y_raw_labels = collect_data()

    if len(X_raw) == 0:
        print("ERRO: Nenhum dado!")
        sys.exit(1)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw_labels)
    indices = np.arange(len(X_raw))

    # Classes com 1 amostra vao inteiras pro treino
    class_counts = Counter(y_enc)
    single_sample = {c for c, n in class_counts.items() if n < 2}
    multi_sample = [i for i in indices if y_enc[i] not in single_sample]
    single_idx = [i for i in indices if y_enc[i] in single_sample]

    if multi_sample and len(set(y_enc[multi_sample])) >= 2:
        train_multi, test_multi = train_test_split(
            multi_sample, test_size=0.2, random_state=42,
            stratify=y_enc[multi_sample]
        )
        train_idx = np.concatenate([train_multi, single_idx])
        test_idx = test_multi
    else:
        # Tudo vai pro treino, teste usa 20% do total aleatorio
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42
        )

    if single_sample:
        print(f"\n  {len(single_sample)} classes com 1 amostra - todas no treino")

    print(f"\n[2/4] Aumentando dados (x{AUG_MULTIPLIER} no treino)...")
    X_train_list: list[np.ndarray] = []
    y_train_list: list[int] = []

    for i, idx in enumerate(train_idx):
        seq = X_raw[idx]
        label = y_enc[idx]
        X_train_list.append(compute_enhanced_features(seq))
        y_train_list.append(label)

        for aug_seq in augment_sequence(seq):
            X_train_list.append(compute_enhanced_features(aug_seq))
            y_train_list.append(label)

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(train_idx)} processadas...", flush=True)

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    print(f"\n  Treino apos augmentacao: {len(X_train)} amostras")

    print(f"\n[3/4] Processando teste (sem augmentacao)...")
    X_test_list = [compute_enhanced_features(X_raw[idx]) for idx in test_idx]
    X_test = np.array(X_test_list)
    y_test = y_enc[test_idx]

    print(f"  Teste: {len(X_test)} amostras")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")
    print(f"  Distribuicao (treino+teste):")
    for cls_name in sorted(le.classes_):
        cls_enc = le.transform([cls_name])[0]
        n_train = (y_train == cls_enc).sum()
        n_test = (y_test == cls_enc).sum()
        print(f"    {cls_name:15s}: {n_train:3d} treino, {n_test:3d} teste")

    print(f"\n[4/4] Treinando XGBoost...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    min_w = max(1, len(y_train) // (len(np.unique(y_train)) * 2))
    sample_weights: np.ndarray = np.ones(len(y_train))
    for cls in np.unique(y_train):
        mask = y_train == cls
        n_cls = mask.sum()
        if n_cls < min_w:
            sample_weights[mask] = min_w / n_cls

    model = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
    )
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights, verbose=True)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  >>> ACURACIA NO TESTE: {acc*100:.1f}% <<<")

    y_pred_train = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_pred_train, y_train)
    print(f"  >>> ACURACIA NO TREINO: {train_acc*100:.1f}% <<<")

    print(f"\n  Relatorio por classe (teste):")
    real_labels = le.inverse_transform(np.unique(y_test))
    print(classification_report(y_test, y_pred, labels=np.unique(y_test),
                                target_names=real_labels, zero_division=0))

    feature_names = build_feature_names()
    importances = model.feature_importances_
    top_n = 30
    top_idx = np.argsort(importances)[-top_n:]
    print(f"\n  Top {top_n} features:")
    for idx in reversed(top_idx):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"    {name}: {importances[idx]:.4f}")

    data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'normalize_fn': normalize_landmarks,
        'feature_fn': compute_enhanced_features,
    }
    with open(OUTPUT_MODEL, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nModelo salvo em: {OUTPUT_MODEL}")
    print(f"Classes: {list(le.classes_)}")
    print(f"Features totais: {X_train.shape[1]}")


if __name__ == '__main__':
    main()
