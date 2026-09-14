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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

from libres_features import (N_FRAMES, N_LANDMARKS, N_COORDS, FEAT_DIM,
                              normalize_landmarks, compute_finger_angles,
                              add_relative_features, compute_enhanced_features)

PROJECT_DIR = Path(__file__).parent
HAND_MODEL = PROJECT_DIR / "hand_landmarker.task"
if not HAND_MODEL.exists():
    HAND_MODEL = Path.home() / "hand_landmarker.task"

PALAVRAS_DIR = PROJECT_DIR / "palavras2"
YOUTUBE_DIR = PROJECT_DIR / "youtube_videos"
COLETADOS_DIR = PROJECT_DIR / "coletados"
OUTPUT_MODEL = PROJECT_DIR / "modelo_rotulado_xgb.pkl"

AUG_MULTIPLIER: int = 8
CONFIDENCE_THRESHOLD: float = 0.8
MIN_SAMPLES_PER_CLASS: int = 5

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

# Mapeamento de labels com encoding/index incorreto
LABEL_FIXES: dict[str, str] = {
    'com_license': None,      # descartado, classe estranha
    'nãosei': 'naos_i',
    'n�osei': 'nao_sei',
}


# Classes consideradas ruido/lixo: nao sao sinais reais de Libras e so
# adicionam confusao. Sao descartadas durante a normalizacao dos labels.
DROP_LABELS: set[str] = {
    'barulho', 'com_license', 'ruim',
}


def normalize_label(label: str) -> str:
    label = label.lower().strip().replace(' ', '_')
    label = label.replace('\ufffd', '')

    # Corrige encoding quebrado de "nao sei" (acentos perdidos virando \ufffd)
    if label in ('naosei', 'nosei', 'nao_sei', 'não_sei', 'não_sei', 'nÃ£o_sei'):
        return 'nao_sei'

    for canonical, variants in WORD_GROUPS.items():
        if label in variants:
            return canonical

    if label in DROP_LABELS:
        return ''

    return label


def extract_label_palavras2(filename: Path) -> Optional[str]:
    name = filename.stem
    parts = name.split('_', 1)
    label = normalize_label(parts[0])
    return label or None


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
            return label or None
    return None


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

        # Escala simula variacao de tamanho da mao; shift removido:
        # apos normalizar pelo punho, o punho e sempre 0 na inferencia,
        # entao deslocar tudo criaria uma distribuicao que nunca ocorre.
        scale = np.random.uniform(0.92, 1.08)
        aug *= scale

        if np.random.random() < 0.4:
            t_warp = np.random.uniform(0.85, 1.15)
            new_len = int(T * t_warp)
            indices = np.linspace(0, T - 1, new_len)
            warped = np.zeros((new_len, D))
            for d in range(D):
                warped[:, d] = np.interp(indices, np.arange(T), aug[:, d])
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


def collect_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carrega dados brutos (landmarks) sem augmentacao.

    Retorna tambem o grupo de origem de cada amostra (video ou take),
    para que segmentos do mesmo video nunca se dividam entre treino e teste.
    """
    X_raw: list[np.ndarray] = []
    y_raw_labels: list[str] = []
    groups: list[str] = []

    print("\n[1/4] Extraindo landmarks dos videos...")

    if COLETADOS_DIR.exists():
        print(f"\n  coletados/:")
        pastas = sorted([p for p in COLETADOS_DIR.iterdir() if p.is_dir()])
        for pasta in pastas:
            # normaliza para minusculas: pastas 'A' (webcam) e labels 'a'
            # (videos) devem ser a mesma classe
            label = normalize_label(pasta.name)
            if not label:
                continue
            npy_files = sorted(pasta.glob("*.npy"))
            for npy_f in npy_files:
                seq = np.load(npy_f)
                if seq.shape == (N_FRAMES, FEAT_DIM):
                    seq = normalize_landmarks(seq)
                    X_raw.append(seq)
                    y_raw_labels.append(label)
                    m = re.match(r'^minds_(.+)_seg\d+$', npy_f.stem)
                    groups.append(f"minds_{m.group(1)}" if m else str(npy_f))
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
                groups.append(str(v))
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
                groups.append(str(v))
                print("OK")
            else:
                print("SEM MAO")

    X_raw_arr = np.array(X_raw)
    y_raw_arr = np.array(y_raw_labels)
    groups_arr = np.array(groups)
    n_real = len(X_raw_arr)

    print(f"\n  Total amostras reais: {n_real}")
    print(f"  Shape: {X_raw_arr.shape}")
    print(f"  Classes: {len(np.unique(y_raw_arr))}")

    return X_raw_arr, y_raw_arr, groups_arr


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


def grouped_split(indices: np.ndarray, y_enc: np.ndarray, groups: np.ndarray,
                  test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split por grupo: amostras do mesmo video/take nunca se separam.

    Classes que cairiam so no lado menor sao movidas de volta ao treino,
    garantindo que o modelo conheca todas as classes.
    """
    indices = np.asarray(indices)
    if len(indices) < 4 or len(np.unique(groups[indices])) < 2:
        return indices, np.array([], dtype=int)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(indices, y_enc[indices], groups[indices]))
    train_idx, test_idx = indices[tr], indices[te]

    train_classes = set(y_enc[train_idx].tolist())
    only_test = np.array([y_enc[i] not in train_classes for i in test_idx])
    if only_test.any():
        train_idx = np.concatenate([train_idx, test_idx[only_test]])
        test_idx = test_idx[~only_test]

    return train_idx, test_idx


def main() -> None:
    X_raw, y_raw_labels, groups = collect_data()

    if len(X_raw) == 0:
        print("ERRO: Nenhum dado!")
        sys.exit(1)

    # Classes com poucas amostras so adicionam ruido e nunca sao
    # avaliadas de forma confiavel: melhor descartar e coletar mais dados
    counts = Counter(y_raw_labels.tolist())
    dropped = sorted(lbl for lbl, n in counts.items() if n < MIN_SAMPLES_PER_CLASS)
    if dropped:
        print(f"\n  {len(dropped)} classes com < {MIN_SAMPLES_PER_CLASS} "
              f"amostras descartadas:")
        print(f"    {', '.join(dropped)}")
        keep = np.array([counts[lbl] >= MIN_SAMPLES_PER_CLASS for lbl in y_raw_labels])
        X_raw = X_raw[keep]
        y_raw_labels = y_raw_labels[keep]
        groups = groups[keep]

    if len(X_raw) == 0:
        print("ERRO: nenhuma classe com amostras suficientes!")
        sys.exit(1)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw_labels)
    indices = np.arange(len(X_raw))

    # Teste separado por grupo (sem segmentos do mesmo video nos dois lados)
    train_idx, test_idx = grouped_split(indices, y_enc, groups,
                                        test_size=0.2, seed=42)
    # Validacao para early stopping, tambem por grupo, tirada do treino
    train_idx, val_idx = grouped_split(train_idx, y_enc, groups,
                                       test_size=0.15, seed=43)

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

    print(f"\n[3/4] Processando validacao e teste (sem augmentacao)...")
    X_val = np.array([compute_enhanced_features(X_raw[idx]) for idx in val_idx]) \
        if len(val_idx) else np.empty((0, X_train.shape[1]))
    y_val = y_enc[val_idx]

    X_test_list = [compute_enhanced_features(X_raw[idx]) for idx in test_idx]
    X_test = np.array(X_test_list) if X_test_list else np.empty((0, X_train.shape[1]))
    y_test = y_enc[test_idx]

    print(f"  Validacao: {len(X_val)} amostras")
    print(f"  Teste: {len(X_test)} amostras")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y_train))}")
    print(f"  Distribuicao (treino/val/teste):")
    for cls_name in sorted(le.classes_):
        cls_enc = le.transform([cls_name])[0]
        n_train = (y_train == cls_enc).sum()
        n_val = (y_val == cls_enc).sum()
        n_test = (y_test == cls_enc).sum()
        print(f"    {cls_name:15s}: {n_train:3d} treino, "
              f"{n_val:3d} val, {n_test:3d} teste")

    print(f"\n[4/4] Treinando XGBoost...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if len(X_test) else X_test

    min_w = max(1, len(y_train) // (len(np.unique(y_train)) * 2))
    sample_weights: np.ndarray = np.ones(len(y_train))
    for cls in np.unique(y_train):
        mask = y_train == cls
        n_cls = mask.sum()
        # Reduz peso de classes com muitos dados (MINDS) e aumenta o peso
        # de classes com poucos (webcam), equilibrando o gradiente do modelo
        if n_cls < min_w:
            sample_weights[mask] = min_w / n_cls
        elif n_cls > len(X_train) * 0.5:
            sample_weights[mask] = 0.5

    use_early_stopping = len(X_val) > 0
    model = xgb.XGBClassifier(
        n_estimators=200,           # reduzido: menos arvores = menos overfit
        max_depth=3,                # reduzido de 4: modelos mais rasos generalizam melhor
        learning_rate=0.08,          # reduzido de 0.1: aprendizado mais gradual
        subsample=0.75,              # mais aleatoriedade na escolha de amostras
        colsample_bytree=0.7,       # menos features por arvore
        reg_lambda=5.0,             # regularizacao L2 forte (default 1.0)
        reg_alpha=0.5,              # regularizacao L1 (esparsidade)
        min_child_weight=3,         # folhas precisam de mais amostras
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        early_stopping_rounds=30 if use_early_stopping else None,
    )
    if use_early_stopping:
        X_val_scaled = scaler.transform(X_val)
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights,
                  eval_set=[(X_val_scaled, y_val)], verbose=False)
        print(f"  Early stopping: melhor iteracao = {model.best_iteration}")
    else:
        print("  (sem validacao: poucos grupos, treinando sem early stopping)")
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights,
                  verbose=False)

    y_pred_train = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"\n  >>> ACURACIA NO TREINO: {train_acc*100:.1f}% <<<")

    if len(X_test):
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"  >>> ACURACIA NO TESTE: {acc*100:.1f}% <<<")

        print(f"\n  Relatorio por classe (teste):")
        real_labels = le.inverse_transform(np.unique(y_test))
        print(classification_report(y_test, y_pred, labels=np.unique(y_test),
                                    target_names=real_labels, zero_division=0))
    else:
        print("  (sem conjunto de teste: poucos grupos para separar)")

    feature_names = build_feature_names()
    importances = model.feature_importances_
    top_n = 30
    top_idx = np.argsort(importances)[-top_n:]
    print(f"\n  Top {top_n} features:")
    for idx in reversed(top_idx):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"    {name}: {importances[idx]:.4f}")

    # so objetos picklable seguros: nunca funcoes locais (quebram o
    # unpickle dependendo de onde o modelo e carregado)
    data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_names': build_feature_names(),
        'n_frames': N_FRAMES,
        'feat_dim': FEAT_DIM,
    }
    with open(OUTPUT_MODEL, 'wb') as f:
        pickle.dump(data, f)

    print(f"\nModelo salvo em: {OUTPUT_MODEL}")
    print(f"Classes: {list(le.classes_)}")
    print(f"Features totais: {X_train.shape[1]}")


if __name__ == '__main__':
    main()
