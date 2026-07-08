"""
TVC3 — SAD (DCC166-2026.1) — UFJF
Dashboard: Priorização de Auditoria de Fraude em Cartão de Crédito

Execute: streamlit run app.py
(certifique-se de rodar pipeline.py antes para gerar os artefatos)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# ── Configuração da página ────────────────────────────────────────
st.set_page_config(
    page_title="Auditoria de Fraude — SAD TVC3",
    page_icon="🔍",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NAVY, TEAL, MUTED = "#1C2B4B", "#00A896", "#6B7A8D"

# ── Carregamento ──────────────────────────────────────────────────
@st.cache_resource
def carregar():
    bundle = joblib.load(os.path.join(BASE_DIR, "artefatos", "fraud_model.joblib"))
    df = pd.read_csv(os.path.join(BASE_DIR, "creditcard.csv"))
    df["Hour"] = (df["Time"] // 3600) % 24
    df = df.sort_values("Time").reset_index(drop=True)
    split_idx = int(len(df) * 0.70)
    test_df = df.iloc[split_idx:].copy()
    features = bundle["features"]
    scaler = bundle["scaler"]
    X_test = test_df[features].copy()
    X_test[["Amount", "Hour"]] = scaler.transform(X_test[["Amount", "Hour"]])
    test_df["score"] = bundle["model"].predict_proba(X_test)[:, 1]
    with open(os.path.join(BASE_DIR, "artefatos", "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    scenario_df = pd.read_csv(os.path.join(BASE_DIR, "artefatos", "scenario_capacity.csv"))
    n_days = bundle["n_days_test"]
    return test_df, metrics, scenario_df, n_days, bundle["model_name"]

test_df, metrics, scenario_df, n_days, model_name = carregar()
total_frauds = int(test_df["Class"].sum())
ranked = test_df.sort_values("score", ascending=False).reset_index(drop=True)

# ── Cabeçalho ─────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style='background:{NAVY};padding:20px 28px;border-radius:10px;margin-bottom:6px'>
      <p style='color:#A8C1E0;font-size:13px;margin:0'>
        UFJF · DCC · Sistemas de Apoio à Decisão / DCC166-2026.1 · TVC3
      </p>
      <h2 style='color:white;margin:6px 0 2px 0'>
        Priorização de Auditoria — Fraude em Cartão de Crédito
      </h2>
      <p style='color:{TEAL};font-size:14px;margin:0;font-style:italic'>
        "{metrics['pergunta_problema']}"
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Abas ──────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📋  Visão Geral",
    "🔍  Fila de Auditoria",
    "📊  Análise de Capacidade",
])


# ══════════════════════════════════════════════════════════════════
# ABA 1 — Visão Geral
# ══════════════════════════════════════════════════════════════════
with tab1:
    col_def, col_met = st.columns([1, 1], gap="large")

    with col_def:
        st.subheader("📌 Definição Operacional de Fraude")
        st.info(
            "**Fraude por uso não autorizado de cartão de crédito**\n\n"
            "Neste trabalho, considera-se fraude (Class = 1) toda transação "
            "realizada por um **terceiro não autorizado** — ou seja, uma compra "
            "ou operação financeira executada com os dados ou o cartão físico "
            "de um **cliente legítimo sem o seu conhecimento ou consentimento**. "
            "Em termos diretos: outra pessoa realizou uma compra no cartão do "
            "cliente.\n\n"
            "Essa modalidade distingue-se de outras formas de fraude financeira "
            "(fraude de identidade, lavagem de dinheiro, fraude pelo próprio "
            "titular). Aqui, o portador legítimo **não reconhece a transação "
            "como sua** e a contesta junto à instituição.\n\n"
            "Os rótulos foram atribuídos retrospectivamente pela equipe de "
            "investigação da **Worldline**: analistas cruzaram as contestações "
            "dos clientes com os registros das operações e confirmaram "
            "caso a caso se a transação havia sido realizada pelo titular "
            "ou por um fraudador (setembro/2013, portadores europeus).\n\n"
            "O ground-truth é **binário**: `0` = legítima · `1` = fraude confirmada. "
            "Não há graus intermediários de suspeita."
        )

        st.subheader("🗄️ Dataset")
        ds = metrics["dataset"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Transações", f"{ds['total']:,}")
        c2.metric("Fraudes confirmadas", f"{ds['fraudes']:,}")
        c3.metric("Taxa de fraude", f"{ds['taxa_fraude_%']:.4f} %")
        st.caption(f"Fonte: {ds['fonte']}")

    with col_met:
        st.subheader("🤖 Desempenho dos Modelos (conjunto de teste)")
        st.caption("Split temporal 70/30 — dados ordenados cronologicamente")
        res = metrics["resultados"]
        rows_tbl = []
        for name, r in res.items():
            marker = " ★" if name == metrics["modelo_principal"] else ""
            rows_tbl.append({
                "Modelo": name + marker,
                "ROC-AUC": r["ROC_AUC"],
                "PR-AUC": r["PR_AUC"],
                "Precisão": r["Precisao"],
                "Recall": r["Recall"],
                "F1": r["F1"],
            })
        tbl = pd.DataFrame(rows_tbl).set_index("Modelo")
        st.dataframe(
            tbl.style.highlight_max(subset=["PR-AUC", "F1"],
                                    color="#D0F3EE", axis=0)
                     .format("{:.4f}"),
            use_container_width=True,
        )
        st.markdown(
            f"> **Por que PR-AUC?** Com apenas {ds['taxa_fraude_%']:.2f} % de fraudes, "
            "um modelo que classifica *tudo* como legítimo teria ~99,83 % de "
            "acurácia — mas capturaria **0 fraudes**. A PR-AUC mede a capacidade "
            "real de priorização da classe de interesse."
        )

        fig_pr = os.path.join(BASE_DIR, "figures", "modelo_curva_pr.png")
        if os.path.exists(fig_pr):
            st.image(fig_pr, caption="Curva Precisão-Recall — comparação dos modelos",
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# ABA 2 — Fila de Auditoria
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🔍 Fila de Transações Priorizadas")
    st.markdown(
        "O modelo classifica cada transação pela probabilidade de ser fraude. "
        "A **fila de auditoria** exibe as mais suspeitas no topo — "
        "a equipe trabalha de cima para baixo até esgotar sua capacidade diária."
    )

    col_ctrl, col_info = st.columns([1, 2], gap="large")

    with col_ctrl:
        cap = st.slider("Capacidade da equipe (casos/dia)", 5, 300, 80, step=5)
        threshold = st.slider("Limiar de alerta (probabilidade mínima)", 0.0, 1.0, 0.5, step=0.05)

    K = cap * n_days
    top_k = ranked.iloc[:K]
    caught = int(top_k["Class"].sum())
    recall_k = caught / total_frauds if total_frauds else 0
    prec_k = caught / K if K else 0

    with col_info:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Casos revisados", f"{K:,}")
        m2.metric("Fraudes capturadas", f"{caught} / {total_frauds}")
        m3.metric("Cobertura (Recall)", f"{recall_k*100:.1f} %")
        m4.metric("Precisão dos alertas", f"{prec_k*100:.1f} %")

    st.divider()
    st.markdown(f"**Top {min(50, K)} casos da fila de auditoria** "
                f"(score ≥ {threshold:.2f} exibidos)")
    exibir = top_k[top_k["score"] >= threshold].head(50).copy()
    exibir = exibir[["Time", "Amount", "Hour", "score", "Class"]].rename(columns={
        "score": "prob_fraude",
        "Class": "fraude_real (ground-truth)",
        "Amount": "valor_transacao_EUR",
    })
    exibir["prob_fraude"] = exibir["prob_fraude"].round(4)
    exibir["valor_transacao_EUR"] = exibir["valor_transacao_EUR"].round(2)

    st.dataframe(
        exibir.style.background_gradient(
            subset=["prob_fraude"], cmap="Greens"
        ).format({"prob_fraude": "{:.4f}", "valor_transacao_EUR": "€ {:.2f}"}),
        use_container_width=True,
        height=420,
    )
    st.caption(
        "fraude_real = 1 → fraude confirmada · 0 → legítima. "
        "A equipe de auditoria *não* enxerga essa coluna em produção — "
        "é usada aqui apenas para avaliar o modelo."
    )


# ══════════════════════════════════════════════════════════════════
# ABA 3 — Análise de Capacidade
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Análise de Capacidade: Cobertura de Fraudes")
    st.markdown(
        "Dado que a equipe tem capacidade limitada, o gráfico mostra "
        "**quantas fraudes são capturadas** e **qual a precisão dos alertas** "
        "para cada nível de capacidade diária. "
        "A linha vertical indica a capacidade selecionada na Aba 2."
    )

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.set_facecolor("#F4F8FB")
    fig.patch.set_facecolor("white")

    ax1.plot(scenario_df["capacidade_por_dia"], scenario_df["recall_%"],
             color=TEAL, linewidth=2.5, label="Recall — fraudes capturadas (%)")
    ax2.plot(scenario_df["capacidade_por_dia"], scenario_df["precisao_%"],
             color=NAVY, linewidth=2.0, linestyle="--",
             label="Precisão — alertas corretos (%)")

    ax1.axvline(cap, color="orange", linestyle=":", linewidth=2, alpha=0.8,
                label=f"Capacidade atual ({cap}/dia)")

    row_cur = scenario_df[scenario_df["capacidade_por_dia"] == cap]
    if not row_cur.empty:
        ax1.scatter([cap], [row_cur.iloc[0]["recall_%"]],
                    color="orange", zorder=5, s=80)

    ax1.set_xlabel("Casos revisados por dia")
    ax1.set_ylabel("Recall — % de fraudes capturadas", color=TEAL)
    ax2.set_ylabel("Precisão dos alertas (%)", color=NAVY)
    ax1.set_ylim(0, 108)
    ax2.set_ylim(0, 108)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False,
               loc="lower right", fontsize=10)

    ax1.set_title("Capacidade da Equipe × Cobertura de Fraudes", fontsize=13,
                  fontweight="bold", color=NAVY, pad=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.subheader("Tabela de Cenários")
    st.dataframe(
        scenario_df.rename(columns={
            "capacidade_por_dia": "Casos/dia",
            "total_revisados": "Total revisados",
            "fraudes_capturadas": "Fraudes capturadas",
            "recall_%": "Recall (%)",
            "precisao_%": "Precisão (%)",
        }).style.background_gradient(subset=["Recall (%)"], cmap="Greens")
                .format({"Recall (%)": "{:.1f}", "Precisão (%)": "{:.1f}"}),
        use_container_width=True,
        height=380,
    )
