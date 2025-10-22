# Transfer Learning na Prática - Cats vs Dogs (DIO Challenge)

Desafio DIO - Transfer Learning em Deep Learning (Python / Google Colab)

Entendendo o Desafio

A missão é aplicar Transfer Learning em um problema real de classificação de imagens, documentar o processo de ponta a ponta e publicar no GitHub.

# O que foi feito neste projeto

Tarefa: Classificação binária (cats vs dogs).

Abordagem: Transfer Learning com MobileNetV2 pré-treinada no ImageNet:

Fase 1: base congelada como feature extractor + topo leve (GAP - Dense(256, ReLU) - Dropout(0.3) - Dense(2, Softmax)).

Fase 2: Fine-Tuning parcial das últimas ~20 camadas da MobileNetV2 com learning rate baixo (1e-5).

Métricas & Visualizações: accuracy, AUC, loss, curvas de treino, matriz de confusão, classification report.

Reprodutibilidade: seeds fixadas, logs limpos, checkpoints salvos.

Objetivos de Aprendizagem (DIO)

Ao concluir, ser capaz de:

Aplicar Transfer Learning em um problema real de visão computacional.

Documentar processos técnicos de forma clara (README + notebook).

Usar o GitHub para compartilhar documentação técnica e resultados.

Datasets

Cats vs Dogs (filtered) — 2 classes (gatos e cachorros).

O notebook baixa automaticamente via tf.keras.utils.get_file.
Você pode substituir por qualquer dataset com 2 classes do seu interesse.

# Resultados

Validação (pré - pós Fine-Tuning)

Accuracy: 0.9833 - 0.9933 (+1.00 pp)

AUC: 0.9987 - 0.9989

Val loss: ~0.046–0.047 - ~0.040–0.043

Teste (300 imagens)

Accuracy: 0.9800 - 0.9833

AUC: 0.9954 - 0.9987

Loss: 0.0557 - 0.0511

Erros totais: 6 - 5

Matrizes de confusão (resumo)

Fase 1: 5 cats-dogs, 1 dog-cats

Fase 2: 2 cats-dogs, 3 dogs-cats

O Fine-Tuning melhorou recall de cats (menos gatos classificados como dogs) e deixou os erros mais equilibrados.

Como rodar (Google Colab ou local)
Pré-requisitos

Python 3.10+

TensorFlow 2.12+

numpy, matplotlib, scikit-learn, tqdm

# Passo a passo

Abra o notebook no Google Colab ou localmente (Jupyter).

Execute as células na ordem:

* Setup e seeds

* Download e extração do dataset

* Carregamento e pré-processamento (MobileNetV2)

* Treino Fase 1 (base congelada)

* Treino Fase 2 (fine-tuning parcial)

* Avaliação no teste

* Predição unitária (demo)

# Checkpoints:

* models/mobilenetv2_feature_extractor.keras

* models/mobilenetv2_finetuned.keras


# Estrutura do Repositório
.
├── notebooks/

│   └── transfer-learning_cats-vs-dogs.ipynb

├── models/

│   ├── mobilenetv2_feature_extractor.keras

│   └── mobilenetv2_finetuned.keras

├── images/ 

│   ├── curves_freeze.png

│   ├── curves_ft.png

│   ├── confmat_phase1.png

│   └── confmat_phase2.png

├── requirements.txt

└── README.md


# Metodologia

Transfer Learning reaproveita conhecimento do ImageNet (bordas, texturas, formas).

Congelar - depois FT parcial dá treino rápido e evita overfitting com dataset moderado.

GAP (GlobalAveragePooling) reduz parâmetros e melhora generalização (vs. Flatten).

Dropout(0.3) no topo segura overfitting da cabeça do modelo.

LR baixo no FT (1e-5) previne catastrophic forgetting.
