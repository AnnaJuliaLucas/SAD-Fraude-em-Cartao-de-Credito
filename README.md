# Detecção de Fraude em Cartão de Crédito como Apoio à Priorização de Auditoria

**TVC3 — Sistemas de Apoio à Decisão (DCC166-2026.1)**  
Universidade Federal de Juiz de Fora — Departamento de Ciência da Computação

---

## Pergunta-Problema

> Como um modelo de classificação pode apoiar a priorização de transações a serem investigadas por uma equipe de auditoria com capacidade limitada?

---

## Sobre o Projeto

Sistema de apoio à decisão baseado em classificação supervisionada para priorização de transações suspeitas de fraude em cartão de crédito. Combina dois algoritmos (Regressão Logística e Random Forest) com uma análise de capacidade *what-if* que permite simular diferentes cenários operacionais de auditoria.

**Dataset real:** [Credit Card Fraud Detection — Worldline / MLG-ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
284.807 transações reais · 492 fraudes confirmadas (0,17%) · setembro/2013

---

## Dashboard Interativo (Streamlit)

O projeto inclui um dashboard interativo com três abas:

- **📋 Visão Geral** — Definição de fraude, métricas dos modelos, curva PR
- **🔍 Fila de Auditoria** — Transações priorizadas por score de suspeita
- **📊 Análise de Capacidade** — Simulação *what-if* de cenários operacionais

### Executar localmente

```bash
streamlit run app.py
```

---

## Como Executar

**1. Clone o repositório e instale as dependências**
```bash
git clone https://github.com/AnnaJuliaLucas/SAD-Fraude-em-Cartao-de-Credito.git
cd SAD-Fraude-em-Cartao-de-Credito
pip install -r requirements.txt
```

**2. Baixe o dataset** e coloque o arquivo `creditcard.csv` na raiz do projeto
```bash
# Kaggle CLI (requer conta):
kaggle datasets download mlg-ulb/creditcardfraud
```

**3. Execute o pipeline de análise e modelagem**
```bash
python pipeline.py
```
Isso gera o modelo treinado, as figuras e as métricas em `artefatos/` e `figures/`.

**4. Rode o dashboard interativo**
```bash
streamlit run app.py
```

---

## Estrutura do Projeto

```
├── app.py               # Dashboard Streamlit (análise de capacidade interativa)
├── pipeline.py          # EDA, pré-processamento, modelagem e avaliação
├── requirements.txt     # Dependências Python
├── .streamlit/
│   └── config.toml      # Tema visual do dashboard
├── artefatos/
│   ├── fraud_model.joblib     # Modelo treinado (gerado pelo pipeline)
│   ├── metrics.json           # Métricas dos modelos
│   ├── scenario_capacity.csv  # Tabela de cenários de capacidade
│   └── test_scored.csv        # Dados de teste pré-pontuados (gerado pelo pipeline)
└── figures/
    ├── eda_distribuicao_classes.png
    ├── eda_padrao_temporal.png
    ├── eda_distribuicao_amount.png
    ├── modelo_curva_pr.png
    ├── modelo_matriz_confusao.png
    ├── modelo_importancia_features.png
    └── cenario_capacidade.png
```

---

## Resultados Principais

| Modelo | PR-AUC | Precisão | Recall | F1 |
|---|---|---|---|---|
| Regressão Logística (baseline) | 0,776 | 0,027 | 0,907 | 0,052 |
| **Random Forest** | **0,815** | **0,988** | 0,732 | **0,840** |

Com **85 revisões/dia**, a equipe captura **75,9 % das fraudes** com **96,5 % de precisão**.

---

## Autores

- **Anna Julia Lucas**  
- **João Victor Senra** 
