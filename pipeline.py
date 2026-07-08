"""
============================================================
TVC3 — Sistemas de Apoio à Decisão (DCC166-2026.1) — UFJF
============================================================
Tema   : Detecção de Fraude em Cartão de Crédito como
         Apoio à Priorização de Auditoria

PERGUNTA-PROBLEMA:
    Como um modelo de classificação pode apoiar a priorização
    de transações a serem investigadas por uma equipe de
    auditoria com capacidade limitada?

DEFINIÇÃO OPERACIONAL DE FRAUDE:
    Neste trabalho, considera-se fraude (Class = 1) toda
    transação de cartão de crédito realizada por um terceiro
    NÃO AUTORIZADO — ou seja, uma compra ou operação
    financeira executada com os dados ou o cartão físico de
    um cliente legítimo SEM o seu conhecimento ou consentimento.
    Em termos simples: outra pessoa realizou uma compra no
    cartão do cliente.

    Essa modalidade é conhecida como "uso não autorizado de
    cartão de crédito" e se distingue de outras formas de
    fraude financeira (ex.: fraude de identidade para obtenção
    de crédito, lavagem de dinheiro ou fraude do próprio
    titular). Aqui, o portador legítimo do cartão não reconhece
    a transação como sua e a contesta junto à instituição.

    Os rótulos (Class = 1) foram atribuídos de forma
    retrospectiva pela equipe de investigação da Worldline:
    após a ocorrência das transações (setembro/2013), analistas
    cruzaram contestações dos clientes com os registros das
    operações e confirmaram caso a caso se a transação havia
    sido realizada pelo titular ou por um fraudador.

    Implicações para a modelagem:
    (a) O modelo aprende padrões comportamentais de transações
        JÁ CONFIRMADAS como fraudulentas — não estima suspeita.
    (b) As features V1-V28 (PCA) provavelmente codificam
        atributos como perfil de gasto, tipo de comércio e
        contexto temporal, anonimizados por confidencialidade.
    (c) O ground-truth é binário: 0 = legítima, 1 = fraude
        confirmada. Não há graus intermediários.
    (d) Toda fraude não detectada representa uma transação
        real em que outra pessoa usou indevidamente o cartão
        de um cliente — perda financeira confirmada.

Dataset : Credit Card Fraud Detection
          Worldline × MLG-ULB (setembro/2013)
          kaggle.com/datasets/mlg-ulb/creditcardfraud
          Dal Pozzolo et al. (2015), CIDM/IEEE
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score,
    precision_score, recall_score
)

warnings.filterwarnings("ignore")

# ── Caminhos ────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "creditcard.csv")
FIGS   = os.path.join(BASE, "figures")
ARTS   = os.path.join(BASE, "artefatos")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(ARTS, exist_ok=True)

# ── Estilo visual (paleta da apresentação) ───────────────────────────
NAVY, TEAL, MUTED = "#1C2B4B", "#00A896", "#6B7A8D"
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F4F8FB",
    "axes.grid":        True,
    "grid.color":       "#D0D8E4",
    "grid.alpha":       0.6,
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.labelcolor":  NAVY,
    "xtick.color":      MUTED,
    "ytick.color":      MUTED,
})

RANDOM_STATE = 42

# ════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — Carregamento e validação dos dados
# ════════════════════════════════════════════════════════════════════
print("=" * 60)
print("SEÇÃO 1 — Carregamento e validação dos dados")
print("=" * 60)

df = pd.read_csv(DATA)
df["Hour"] = (df["Time"] // 3600) % 24   # hora do dia (feature derivada)

N_TOTAL  = len(df)
N_FRAUD  = df["Class"].sum()
N_LEGIT  = N_TOTAL - N_FRAUD
PCT_FRAUD = N_FRAUD / N_TOTAL * 100

print(f"Transações totais : {N_TOTAL:,}")
print(f"  → Legítimas     : {N_LEGIT:,}  ({100-PCT_FRAUD:.4f} %)")
print(f"  → Fraudes       : {N_FRAUD:,}  ({PCT_FRAUD:.4f} %)")
print(f"Valores nulos     : {df.isnull().sum().sum()}")
print(f"Período (dias)    : {df['Time'].max() / 86400:.2f}")

fraud   = df[df["Class"] == 1]
legit   = df[df["Class"] == 0]

# ════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — Análise Exploratória de Dados (EDA)
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SEÇÃO 2 — Análise Exploratória de Dados")
print("=" * 60)

# ── 2.1  Distribuição das classes (desbalanceamento) ─────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(["Legítima\n(Class = 0)", "Fraude\n(Class = 1)"],
              [N_LEGIT, N_FRAUD],
              color=[NAVY, TEAL], width=0.45, zorder=3)
for bar, val, pct in zip(bars, [N_LEGIT, N_FRAUD],
                         [100 - PCT_FRAUD, PCT_FRAUD]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2500,
            f"{val:,}\n({pct:.2f} %)",
            ha="center", va="bottom", fontsize=11,
            color=bar.get_facecolor(), fontweight="bold")

ax.set_title("Distribuição das Classes — Desbalanceamento Extremo",
             fontsize=13, fontweight="bold", color=NAVY, pad=15)
ax.set_ylabel("Número de transações")
ax.set_ylim(0, N_LEGIT * 1.18)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "eda_distribuicao_classes.png"), dpi=150)
plt.close(fig)
print("  [OK] eda_distribuicao_classes.png")

# ── 2.2  Padrão temporal: volume por hora do dia ─────────────────
fraud_hour  = fraud.groupby("Hour").size() / N_FRAUD * 100
legit_hour  = legit.groupby("Hour").size() / N_LEGIT * 100
all_hours   = range(24)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(all_hours, [legit_hour.get(h, 0) for h in all_hours],
        color=NAVY, linewidth=2.0, label="Legítimas", zorder=3)
ax.plot(all_hours, [fraud_hour.get(h, 0) for h in all_hours],
        color=TEAL, linewidth=2.0, linestyle="--",
        marker="o", markersize=4, label="Fraudes", zorder=4)
ax.set_title("Padrão Temporal: Distribuição por Hora do Dia\n"
             "(% do total de cada classe)",
             fontsize=13, fontweight="bold", color=NAVY, pad=12)
ax.set_xlabel("Hora do dia")
ax.set_ylabel("% de transações da classe")
ax.set_xticks(all_hours)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "eda_padrao_temporal.png"), dpi=150)
plt.close(fig)
print("  [OK] eda_padrao_temporal.png")
# Insight textual
peak_fraud = int(fraud_hour.idxmax())
peak_legit = int(legit_hour.idxmax())
print(f"  Pico de fraudes : {peak_fraud}h  |  Pico de legítimas: {peak_legit}h")

# ── 2.3  Distribuição de valores (Amount) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Boxplot em escala log
for i, (cls, clabel, color) in enumerate([
    (legit["Amount"], "Legítimas", NAVY),
    (fraud["Amount"], "Fraudes",  TEAL)
]):
    axes[0].boxplot(np.log1p(cls), positions=[i],
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.7),
                    medianprops=dict(color="white", linewidth=2),
                    flierprops=dict(marker=".", markersize=2,
                                   markerfacecolor=color, alpha=0.3))
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(["Legítimas", "Fraudes"])
axes[0].set_ylabel("log(Amount + 1)")
axes[0].set_title("Distribuição do Valor\n(escala log)", fontsize=12,
                  fontweight="bold", color=NAVY)

# Histograma do Amount (fraudes, sem log)
axes[1].hist(fraud["Amount"], bins=40, color=TEAL, alpha=0.85, edgecolor="white")
axes[1].set_xlabel("Valor da transação (€)")
axes[1].set_ylabel("Frequência")
axes[1].set_title("Histograma dos Valores\n(somente fraudes confirmadas)",
                  fontsize=12, fontweight="bold", color=NAVY)
axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

fig.tight_layout()
fig.savefig(os.path.join(FIGS, "eda_distribuicao_amount.png"), dpi=150)
plt.close(fig)
print("  [OK] eda_distribuicao_amount.png")
print(f"  Amount (fraude) — mediana: €{fraud['Amount'].median():.2f}  "
      f"máximo: €{fraud['Amount'].max():.2f}")
print(f"  Amount (legít.) — mediana: €{legit['Amount'].median():.2f}  "
      f"máximo: €{legit['Amount'].max():.2f}")

# ════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — Pré-processamento
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SEÇÃO 3 — Pré-processamento")
print("=" * 60)

# Split temporal: ordena por Time, treina nos 70 % mais antigos.
# Justificativa: simula o cenário real de auditoria — o modelo é
# treinado com dados históricos e aplicado a transações futuras.
# Um split aleatório causaria vazamento temporal (data leakage).
df_sorted = df.sort_values("Time").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.70)
train_df  = df_sorted.iloc[:split_idx]
test_df   = df_sorted.iloc[split_idx:].copy()

FEATURES = [c for c in df.columns if c not in ("Class", "Time")]

X_train = train_df[FEATURES].copy()
y_train = train_df["Class"]
X_test  = test_df[FEATURES].copy()
y_test  = test_df["Class"]

# Padronização apenas das variáveis contínuas de escala livre.
# V1-V28 já estão na mesma escala (resultado de PCA).
scaler = StandardScaler()
X_train[["Amount", "Hour"]] = scaler.fit_transform(X_train[["Amount", "Hour"]])
X_test[["Amount", "Hour"]]  = scaler.transform(X_test[["Amount", "Hour"]])

print(f"  Treino : {len(train_df):,} transações "
      f"({int(y_train.sum())} fraudes, {y_train.mean()*100:.3f} %)")
print(f"  Teste  : {len(test_df):,} transações  "
      f"({int(y_test.sum())} fraudes, {y_test.mean()*100:.3f} %)")
print(f"  Features: {len(FEATURES)}")
print(f"  Estratégia de split: TEMPORAL (70 % passado / 30 % futuro)")
print(f"  Desbalanceamento: class_weight='balanced' (sem oversampling artificial)")

# ════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — Modelagem
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SEÇÃO 4 — Modelagem")
print("=" * 60)

models = {
    "Regressão Logística (baseline)": LogisticRegression(
        max_iter=2000, class_weight="balanced",
        solver="lbfgs", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    ),
}

results = {}
probas  = {}

for name, model in models.items():
    print(f"\n  Treinando: {name} ...", end=" ", flush=True)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc  = average_precision_score(y_test, proba)
    prec    = precision_score(y_test, preds, zero_division=0)
    rec     = recall_score(y_test, preds, zero_division=0)
    f1      = f1_score(y_test, preds, zero_division=0)
    cm      = confusion_matrix(y_test, preds).tolist()

    results[name] = {
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC":  round(pr_auc, 4),
        "Precisao": round(prec, 4),
        "Recall":   round(rec, 4),
        "F1":       round(f1, 4),
        "confusion_matrix": cm,
    }
    probas[name] = proba
    print(f"OK — PR-AUC: {pr_auc:.4f}")

# ── Modelo principal: melhor PR-AUC ──────────────────────────────
best_name  = max(results, key=lambda k: results[k]["PR_AUC"])
best_proba = probas[best_name]
best_model = models[best_name]
test_df["score"] = best_proba
print(f"\n  ★ Modelo principal: {best_name}")

# ════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — Avaliação e figuras do relatório
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SEÇÃO 5 — Avaliação")
print("=" * 60)

print("\n  Métricas no conjunto de teste:")
header = f"  {'Modelo':<35} {'ROC-AUC':>8} {'PR-AUC':>8} {'Prec.':>8} {'Recall':>8} {'F1':>8}"
print(header)
print("  " + "-" * (len(header) - 2))
for name, r in results.items():
    star = " ★" if name == best_name else ""
    print(f"  {name+star:<35} {r['ROC_AUC']:>8.4f} {r['PR_AUC']:>8.4f} "
          f"{r['Precisao']:>8.4f} {r['Recall']:>8.4f} {r['F1']:>8.4f}")

# ── 5.1  Curva Precisão-Recall ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, proba in probas.items():
    prec_c, rec_c, _ = precision_recall_curve(y_test, proba)
    pr_auc = results[name]["PR_AUC"]
    lw = 2.5 if name == best_name else 1.5
    ls = "-" if name == best_name else "--"
    ax.plot(rec_c, prec_c, linewidth=lw, linestyle=ls,
            label=f"{name}  (PR-AUC = {pr_auc:.3f})")

ax.set_xlabel("Recall  (fraudes capturadas / total de fraudes)")
ax.set_ylabel("Precisão  (fraudes confirmadas / alertas emitidos)")
ax.set_title("Curva Precisão-Recall — Comparação dos Modelos",
             fontsize=13, fontweight="bold", color=NAVY, pad=12)
ax.legend(frameon=False, fontsize=10)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "modelo_curva_pr.png"), dpi=150)
plt.close(fig)
print("\n  [OK] modelo_curva_pr.png")

# ── 5.2  Matriz de confusão (modelo principal) ────────────────────
preds_best = (best_proba >= 0.5).astype(int)
cm_best    = confusion_matrix(y_test, preds_best)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legítima\n(prevista)", "Fraude\n(prevista)"],
            yticklabels=["Legítima\n(real)", "Fraude\n(real)"],
            ax=ax, linewidths=0.5, annot_kws={"size": 13})
ax.set_title(f"Matriz de Confusão — {best_name}",
             fontsize=12, fontweight="bold", color=NAVY, pad=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "modelo_matriz_confusao.png"), dpi=150)
plt.close(fig)
print("  [OK] modelo_matriz_confusao.png")

tn, fp, fn, tp = cm_best.ravel()
print(f"  TN={tn:,}  FP={fp:,}  FN={fn}  TP={tp}")
print(f"  Fraudes capturadas: {tp}/{int(y_test.sum())} = {tp/y_test.sum()*100:.1f} %")

# ── 5.3  Importância das features (Random Forest) ────────────────
if hasattr(best_model, "feature_importances_"):
    fi = (pd.Series(best_model.feature_importances_, index=FEATURES)
            .sort_values(ascending=False).head(12))
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(fi.index[::-1], fi.values[::-1],
                   color=[TEAL if i < 3 else NAVY for i in range(len(fi)-1, -1, -1)],
                   zorder=3)
    ax.set_xlabel("Importância relativa (Gini)")
    ax.set_title("Top 12 Variáveis mais Importantes\n(Random Forest)",
                 fontsize=13, fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "modelo_importancia_features.png"), dpi=150)
    plt.close(fig)
    print("  [OK] modelo_importancia_features.png")

# ════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — Análise de Capacidade (Cenários)
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SEÇÃO 6 — Análise de Capacidade (Cenários)")
print("=" * 60)

# Pergunta-problema: priorizar transações dado capacidade limitada.
# Simulação: para cada capacidade diária K, ordenamos as transações
# por score descendente (maior probabilidade de fraude primeiro),
# pegamos as K×dias mais suspeitas e calculamos quantas fraudes
# a equipe capturaria.

n_days = max(1, round(
    (test_df["Time"].max() - test_df["Time"].min()) / 86400
))
print(f"  Período de teste: ~{n_days} dia(s)")
print(f"  Fraudes totais no teste: {int(y_test.sum())}")

ranked = test_df.sort_values("score", ascending=False).reset_index(drop=True)
total_frauds = int(y_test.sum())
capacities   = list(range(5, 205, 5))
rows = []
for cap in capacities:
    K           = cap * n_days
    top_k       = ranked.iloc[:K]
    caught      = int(top_k["Class"].sum())
    recall_k    = caught / total_frauds if total_frauds else 0
    precision_k = caught / K if K else 0
    rows.append({
        "capacidade_por_dia": cap,
        "total_revisados": K,
        "fraudes_capturadas": caught,
        "recall_%": round(recall_k * 100, 1),
        "precisao_%": round(precision_k * 100, 1),
    })

scenario_df = pd.DataFrame(rows)

# Ponto de "cobertura suficiente" (primeiro recall ≥ 75%)
cov75 = scenario_df[scenario_df["recall_%"] >= 75.0].iloc[0] if any(scenario_df["recall_%"] >= 75) else None
cov90 = scenario_df[scenario_df["recall_%"] >= 90.0].iloc[0] if any(scenario_df["recall_%"] >= 90) else None

print(f"\n  Para cobrir ≥ 75 % das fraudes: "
      f"{int(cov75['capacidade_por_dia']) if cov75 is not None else 'N/A'} casos/dia")
print(f"  Para cobrir ≥ 90 % das fraudes: "
      f"{int(cov90['capacidade_por_dia']) if cov90 is not None else 'N/A'} casos/dia")

# Figura: recall e precisão por capacidade
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

ax1.plot(scenario_df["capacidade_por_dia"], scenario_df["recall_%"],
         color=TEAL, linewidth=2.5, label="Recall (fraudes capturadas)")
ax2.plot(scenario_df["capacidade_por_dia"], scenario_df["precisao_%"],
         color=NAVY, linewidth=2.0, linestyle="--",
         label="Precisão (% de alertas corretos)")

# marcos de recall
for meta, cap_row, color in [
    ("75 %", cov75, TEAL),
    ("90 %", cov90, NAVY),
]:
    if cap_row is not None:
        ax1.axvline(cap_row["capacidade_por_dia"], color=color,
                    linestyle=":", alpha=0.7)
        ax1.text(cap_row["capacidade_por_dia"] + 2,
                 float(meta.replace(" %", "")) - 4,
                 f"{int(cap_row['capacidade_por_dia'])} casos/dia\n(recall {meta})",
                 fontsize=9, color=color)

ax1.set_xlabel("Capacidade da equipe (casos revisados por dia)")
ax1.set_ylabel("Recall — % de fraudes capturadas", color=TEAL)
ax2.set_ylabel("Precisão dos alertas (%)", color=NAVY)
ax1.set_ylim(0, 105)
ax2.set_ylim(0, 105)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="center right")

ax1.set_title("Análise de Capacidade: Cobertura de Fraudes vs Capacidade Diária",
              fontsize=13, fontweight="bold", color=NAVY, pad=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "cenario_capacidade.png"), dpi=150)
plt.close(fig)
print("  [OK] cenario_capacidade.png")

# ════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — Exportação de artefatos
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SEÇÃO 7 — Exportação de artefatos")
print("=" * 60)

# Modelo, scaler e metadados
joblib.dump(
    {"model": best_model, "scaler": scaler, "features": FEATURES,
     "model_name": best_name, "n_days_test": n_days},
    os.path.join(ARTS, "fraud_model.joblib")
)

# Métricas
metrics_out = {
    "pergunta_problema": (
        "Como um modelo de classificação pode apoiar a priorização de "
        "transações a serem investigadas por uma equipe de auditoria "
        "com capacidade limitada?"
    ),
    "definicao_fraude": (
        "Transação confirmada como fraudulenta pela equipe de "
        "investigação da Worldline (Class=1), com base em análise "
        "individual retrospectiva de casos reais (set/2013)."
    ),
    "resultados": results,
    "modelo_principal": best_name,
    "dataset": {
        "total": N_TOTAL,
        "fraudes": int(N_FRAUD),
        "legitimas": int(N_LEGIT),
        "taxa_fraude_%": round(PCT_FRAUD, 4),
        "periodo": "setembro/2013 (~2 dias)",
        "fonte": "Worldline x MLG-ULB — kaggle.com/datasets/mlg-ulb/creditcardfraud",
    }
}
with open(os.path.join(ARTS, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, indent=2, ensure_ascii=False)

# Tabela de cenários
scenario_df.to_csv(os.path.join(ARTS, "scenario_capacity.csv"), index=False)

print(f"  Artefatos salvos em: {ARTS}")
print(f"  Figuras salvas em:   {FIGS}")
print("\n  Figuras geradas:")
for fname in sorted(os.listdir(FIGS)):
    print(f"    {fname}")

print("\n" + "=" * 60)
print("Pipeline concluído com sucesso.")
print("=" * 60)
