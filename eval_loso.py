#!/usr/bin/env python3
"""
Avaliacao Leave-One-Signer-Out (LOSO).

Cada rodada deixa 1 sinalizador inteiro fora do treino e testa nele.
E a metrica honesta de generalizacao: mede se o modelo reconhece o sinal
quando feito por uma pessoa que ele nunca viu.

Otimizacao: as features aumentadas nao dependem do fold, entao sao
pre-computadas uma unica vez e reutilizadas nas 12 rodadas.

Uso: python3 eval_loso.py
"""

import re, random, time
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import xgboost as xgb

import train_rotulado as tr

random.seed(42)
np.random.seed(42)

SIGNER_RE = re.compile(r'Sinalizador(\d+)')


def extract_signer(group: str) -> str:
    """Extrai o id do sinalizador do grupo (video/take)."""
    m = SIGNER_RE.search(group)
    if m:
        return f"s{m.group(1)}"
    return group  # fontes sem sinalizador (palavras2): vira "signer" proprio


def collect_with_signers():
    """Igual collect_data do treino, mas retorna signer_id em vez de grupo."""
    X_raw, y_labels, groups = tr.collect_data()

    # Mesmo filtro do treino: descarta classes com poucas amostras
    counts = Counter(y_labels.tolist())
    keep = np.array([counts[l] >= tr.MIN_SAMPLES_PER_CLASS for l in y_labels])
    X_raw, y_arr, groups_arr = X_raw[keep], np.array(y_labels)[keep], groups[keep]

    signers = np.array([extract_signer(g) for g in groups_arr])
    return X_raw, y_arr, signers


def precompute_features(X_raw):
    """Features sem e com augmentacao; cada amostra gera 1 + AUG variantes.

    Retorna:
      X_feat: (N, F) features das amostras reais
      X_aug:  (N, AUG, F) features das variantes augmentadas
    """
    n = len(X_raw)
    X_feat = np.array([tr.compute_enhanced_features(s) for s in X_raw])
    X_aug = np.empty((n, tr.AUG_MULTIPLIER, X_feat.shape[1]))
    for i, seq in enumerate(X_raw):
        for j, aug_seq in enumerate(tr.augment_sequence(seq)):
            X_aug[i, j] = tr.compute_enhanced_features(aug_seq)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{n} features...", flush=True)
    return X_feat, X_aug


def train_one_fold(X_feat, X_aug, y_enc, tr_idx, te_idx) -> tuple[float, float]:
    X_train = np.concatenate([X_feat[tr_idx], X_aug[tr_idx].reshape(-1, X_feat.shape[1])])
    y_train = np.concatenate([y_enc[tr_idx], np.repeat(y_enc[tr_idx], tr.AUG_MULTIPLIER)])
    X_test = X_feat[te_idx]
    y_test = y_enc[te_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.08,
        subsample=0.75, colsample_bytree=0.7,
        reg_lambda=5.0, reg_alpha=0.5, min_child_weight=3,
        random_state=42, n_jobs=-1, eval_metric='mlogloss',
    )
    model.fit(X_train_s, y_train, verbose=False)

    y_pred = model.predict(X_test_s)
    return accuracy_score(y_test, y_pred), balanced_accuracy_score(y_test, y_pred)


def main():
    print("=" * 60)
    print("AVALIACAO LEAVE-ONE-SIGNER-OUT (LOSO)")
    print("=" * 60)

    X_raw, y_labels, signers = collect_with_signers()

    le = LabelEncoder()
    y_enc = le.fit_transform(y_labels)

    signer_ids = sorted(np.unique(signers))
    print(f"\n  Amostras: {len(X_raw)} | Classes: {len(le.classes_)} "
          f"| Sinalizadores: {len(signer_ids)}")

    print("\n  Pre-computando features (uma vez so)...", flush=True)
    t0 = time.time()
    X_feat, X_aug = precompute_features(X_raw)
    print(f"  Features prontas em {(time.time()-t0)/60:.1f} min:"
          f" real={X_feat.shape}, aug={X_aug.shape}")

    accs, baccs = [], []

    for fold, sid in enumerate(signer_ids, 1):
        te_mask = signers == sid
        tr_mask = ~te_mask

        # classes que so existem no teste nao tem como acertar: remove
        tr_classes = set(y_enc[tr_mask])
        valid_te = np.array([c in tr_classes for c in y_enc[te_mask]])
        te_idx = np.where(te_mask)[0][valid_te]
        tr_idx = np.where(tr_mask)[0]

        if len(te_idx) == 0:
            print(f"  [{fold:2d}/{len(signer_ids)}] {sid}: sem amostras validas")
            continue

        tf = time.time()
        acc, bacc = train_one_fold(X_feat, X_aug, y_enc, tr_idx, te_idx)
        accs.append(acc)
        baccs.append(bacc)
        print(f"  [{fold:2d}/{len(signer_ids)}] {sid}: "
              f"acc={acc*100:5.1f}%  bal_acc={bacc*100:5.1f}%  "
              f"({len(te_idx)} amostras, {time.time()-tf:.0f}s)", flush=True)

    print("\n" + "=" * 60)
    print(f">>> LOSO MEDIA  acc={np.mean(accs)*100:.1f}%  "
          f"bal_acc={np.mean(baccs)*100:.1f}% <<<")
    print(f"    (min={min(accs)*100:.1f}%  max={max(accs)*100:.1f}%)")
    print("=" * 60)


if __name__ == '__main__':
    main()
