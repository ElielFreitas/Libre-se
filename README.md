# 🤟 Libre-se

Reconhecimento de sinais em **Libras** (Língua Brasileira de Sinais) usando visão computacional e machine learning.

Extrai landmarks das mãos com **MediaPipe Hands**, calcula features temporais (estatísticas, ângulos dos dedos, distâncias relativas) e classifica com **XGBoost**.

## Funcionalidades

- Reconhecimento de letras do alfabeto (A-Z), números (1-10) e palavras específicas em Libras
- Pipeline completo: coleta → extração → treino → inferência em tempo real
- Aumentação de dados (ruído, escala, deslocamento, warp, dropout) para classes com poucas amostras
- Classificação com threshold de confiança (80%) para evitar falsos positivos
- Normalização dos landmarks centralizada no punho (torna independente da posição da mão)
- 639 features por signo: estatísticas dos landmarks, centróide, distâncias relativas e ângulos dos dedos

## Estrutura

```
Libre-se/
├── train_rotulado.py          # Pipeline de treino (extração → aumento → XGBoost)
├── test_camera_temporal.py    # Inferência em tempo real com webcam
├── collect_signs.py           # Coleta de sinais via webcam (A-Z, 1-10)
├── auto_split_signs.py        # Divisão automática de vídeos YouTube por movimento
├── slice_signs.py             # Divisão manual de vídeos com timestamps
├── integrate_minds.py         # Extração do dataset MINDS para formato .npy
├── modelo_rotulado_xgb.pkl    # Modelo treinado (61 classes, 639 features)
├── hand_landmarker.task       # Modelo MediaPipe de detecção de mãos
├── requirements.txt           # Dependências
└── README.md
```

## Instalação

```bash
git clone https://github.com/ElielFreitas/Libre-se.git
cd Libre-se
pip install -r requirements.txt
```

Baixe o modelo `hand_landmarker.task` do [MediaPipe Models](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) e coloque na raiz do projeto (já incluso).

## Como usar

### Coletar sinais (A-Z e 1-10)

```bash
python collect_signs.py
```

Grava 5 takes por letra/número via webcam. Os arquivos `.npy` são salvos em `coletados/{classe}/`.

### Extrair dataset MINDS

```bash
python integrate_minds.py
```

Extrai os segmentos do dataset MINDS (CSV) e salva em `coletados/`.

### Treinar modelo

```bash
python train_rotulado.py
```

Processa vídeos em `palavras2/` e dados em `coletados/`, aplica aumentaçãono treino (3x), treina XGBoost com 639 features e salva `modelo_rotulado_xgb.pkl`.

### Testar em tempo real

```bash
python test_camera_temporal.py
```

Abre a webcam e exibe o sinal reconhecido com a confiança em tempo real.

### Processar vídeos YouTube

```bash
python auto_split_signs.py
```

Detecta segmentos de movimento em vídeos de compilações, verifica presença de mão com MediaPipe e corta com FFmpeg. Os clips salvos em `palavras2/` precisam ser renomeados antes do treino.

```bash
python slice_signs.py
```

Versão manual: digite `inicio fim label` por vídeo.

## Dataset MINDS

O modelo usa o dataset [MINDS](https://universe.robots.cloud.lncc.br) (Libras), que contém 4.518 amostras de 20 classes (~226 por classe), com landmarks extraídos por MediaPipe. Execute `integrate_minds.py` para extrair os segmentos do CSV.

## Features

Para cada frame de 21 landmarks (x, y, z), o pipeline calcula:

| Grupo | Features | Descrição |
|-------|----------|-----------|
| Landmarks normalizados | 504 | Média, mediana, std, min, max, range, diff para cada landmark (x,y,z) |
| Centróide | 15 | Média, mediana, std, min, max da posição do centróide |
| Distâncias relativas | 70 | Média, mediana, std, min, max, range, diff entre cada landmark e os demais |
| Ângulos dos dedos | 50 | 10 ângulos (MCP/PIP) × 5 estatísticas |

**Total: 639 features** por signo.

## Modelo atual

- **61 classes**: A-Y, 20 classes MINDS, 16 palavras avulsas
- **Algoritmo**: XGBoost (80 estimators, max_depth=4, learning_rate=0.15)
- **Acurácia no teste**: 45% (20 classes MINDS)
- **Acurácia no treino**: 96%

## Licença

MIT
