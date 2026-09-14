"""
libres_features.py
------------------
Modulo unico de extracao de features a partir das sequencias de landmarks
MediaPipe Hands usadas no Libre-se.

FONTE UNICA DA VERDADE: serve tanto o treino (train_rotulado.py) quanto a
inferencia em tempo real (test_camera_temporal.py). Altere aqui uma vez e
os dois lados ficam em sync -- evita o bug classico: a normalizacao muda
no treino mas a inferencia fica presa numa versao diferente, gerando
predicoes fora da distribuicao que o modelo conhece.
"""

import numpy as np

N_FRAMES: int = 30
N_LANDMARKS: int = 21
N_COORDS: int = 3
FEAT_DIM: int = N_LANDMARKS * N_COORDS


def normalize_landmarks(seq: np.ndarray) -> np.ndarray:
    """Converte coordenadas absolutas para posicoes relativas ao punho e a
    escala da propria mao (independe de posicao, distancia da camera e
    resolucao). Sem isso, a mesma forma feita perto/longe gera features
    diferentes e causa falsos positivos (ex: 'a' -> 'familia').
    """
    r = seq.reshape(-1, N_LANDMARKS, N_COORDS)
    r = r - r[:, 0:1]

    # Escala: distancia punho(0) -> MCP do dedo medio(9). Estabiliza o
    # tamanho da mao dentro do frame, igualando sinais em qualquer distancia.
    scale = np.linalg.norm(r[:, 9], axis=1)[:, None, None]
    scale = np.clip(scale, 1e-6, None)
    r = r / scale

    return r.reshape(-1, FEAT_DIM)


def compute_finger_angles(seq: np.ndarray) -> np.ndarray:
    """Calcula angulos das articulacoes dos dedos.
    Retorna (T, 10): 2 angulos por dedo.
    """
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
