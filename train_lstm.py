#!/usr/bin/env python3
"""
Treino LSTM para classificacao de sinais de Libras sobre sequencias
de landmarks MediaPipe.

Foco em generalizar entre sinalizadores com pouco dado:
 - rede pequena (64 unidades, 1 camada) + dropout alto
 - augmentacao moderada (mesma familia do pipeline XGBoost)
 - LOSO: deixa 1 sinalizador inteiro fora do treino

Uso:
  python3 train_lstm.py            # LOSO completo (12 folds)
  python3 train_lstm.py --only s01 # 1 fold so (sanity check)
"""

import re, random, sys, time
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import train_rotulado as tr

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = "cpu"
HIDDEN = 64
DROPOUT = 0.4
EPOCHS = 60
PATIENCE = 10
BATCH = 256
LR = 1e-3
AUG = 4                # augmentacao moderada pra CPU
FRAMES = 15            # downsample temporal: 30 -> 15 frames

SIGNER_RE = re.compile(r'Sinalizador(\d+)')


def extract_signer(group: str) -> str:
    m = SIGNER_RE.search(group)
    return f"s{m.group(1)}" if m else group


def downsample(seq: np.ndarray) -> np.ndarray:
    """30 frames -> 15 frames (pega 1 a cada 2). Acelera a CPU e
    mantem a coreografia do sinal."""
    return seq[::2]


class SignLSTM(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN, num_layers=1,
                            batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, n_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])


def augment_one(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augmentacao leve: ruido + escala (as mais seguras pos-normalizacao)."""
    aug = seq.copy()
    aug += rng.normal(0, rng.uniform(0.002, 0.008), aug.shape)
    aug *= rng.uniform(0.92, 1.08)
    return aug


def run_fold(X, y_enc, signers, tr_idx, te_idx, n_classes: int) -> tuple[float, float]:
    rng = np.random.default_rng(42)

    # validacao = 1 sinalizador do TREINO, pro early stopping nunca
    # tocar no sinalizador de teste (evita vazamento)
    train_signers = np.unique(signers[tr_idx])
    val_signer = rng.choice(train_signers)
    va_mask = signers[tr_idx] == val_signer
    va_idx_local = np.where(va_mask)[0]
    tr_idx_local = np.where(~va_mask)[0]

    Xtr_full = [downsample(s) for s in X[tr_idx]]
    ytr_full = list(y_enc[tr_idx])

    Xva = np.stack([Xtr_full[i] for i in va_idx_local]).astype(np.float32)
    yva = np.array([ytr_full[i] for i in va_idx_local])

    Xtr = [Xtr_full[i] for i in tr_idx_local]
    ytr = [ytr_full[i] for i in tr_idx_local]
    for _ in range(AUG):
        Xtr += [downsample(augment_one(s, rng)) for s in X[tr_idx][tr_idx_local]]
        ytr += list(ytr_full[i] for i in tr_idx_local)
    Xtr = np.stack(Xtr).astype(np.float32)
    ytr = np.array(ytr)

    Xte = np.stack([downsample(s) for s in X[te_idx]]).astype(np.float32)
    yte = y_enc[te_idx]

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    model = SignLSTM(Xtr.shape[2], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_state, best_epoch, best_va = None, -1, 0.0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(Xva)).argmax(1).numpy()
        acc = accuracy_score(yva, pred)
        if acc > best_va:
            best_va, best_state, best_epoch = acc, model.state_dict(), epoch
        if epoch - best_epoch >= PATIENCE:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).argmax(1).numpy()
    return accuracy_score(yte, pred), balanced_accuracy_score(yte, pred)


def main():
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    print("=" * 60)
    print("LSTM - " + (f"SANITY CHECK ({only})" if only else "LOSO COMPLETO"))
    print("=" * 60)

    X_raw, y_labels, groups = tr.collect_data()
    counts = Counter(y_labels.tolist())
    keep = np.array([counts[l] >= tr.MIN_SAMPLES_PER_CLASS for l in y_labels])
    X_raw = X_raw[keep]
    y_arr = np.array(y_labels)[keep]
    signers = np.array([extract_signer(g) for g in groups[keep]])

    le = LabelEncoder()
    y_enc = le.fit_transform(y_arr)
    signer_ids = [only] if only else sorted(np.unique(signers))

    print(f"\n  Amostras: {len(X_raw)} | Classes: {len(le.classes_)} "
          f"| Folds: {len(signer_ids)}")

    accs, baccs = [], []
    for fold, sid in enumerate(signer_ids, 1):
        te_mask = signers == sid
        tr_classes = set(y_enc[~te_mask])
        te_idx = np.where(te_mask)[0][
            [c in tr_classes for c in y_enc[te_mask]]]
        tr_idx = np.where(~te_mask)[0]
        if len(te_idx) == 0:
            continue

        tf = time.time()
        acc, bacc = run_fold(X_raw, y_enc, signers, tr_idx, te_idx, len(le.classes_))
        accs.append(acc)
        baccs.append(bacc)
        print(f"  [{fold:2d}/{len(signer_ids)}] {sid}: acc={acc*100:5.1f}%  "
              f"bal_acc={bacc*100:5.1f}%  ({time.time()-tf:.0f}s)", flush=True)

    print("\n" + "=" * 60)
    print(f">>> LSTM LOSO MEDIA  acc={np.mean(accs)*100:.1f}%  "
          f"bal_acc={np.mean(baccs)*100:.1f}% <<<")
    print(f"    (min={min(accs)*100:.1f}%  max={max(accs)*100:.1f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()
