from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Panorama das Licitações 2015–2025",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_ROOT / "outputs"

TEAM_MEMBERS = [
    "Vinicius Lummertz",
    "Jenifer da silva",
    "Paola da silva",
    "hummm",
    "Tem 5?",
]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(193, 145, 78, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(53, 89, 70, 0.10), transparent 24%),
                linear-gradient(180deg, #f7f1e7 0%, #f3ece2 100%);
        }
        .hero-card {
            padding: 1.8rem 1.8rem 1.3rem 1.8rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #17352e 0%, #274a40 60%, #486459 100%);
            color: #f7f2ea;
            box-shadow: 0 18px 40px rgba(19, 38, 33, 0.18);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            line-height: 1.05;
            margin-bottom: 0.5rem;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            line-height: 1.6;
            opacity: 0.95;
            max-width: 900px;
        }
        .tag-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }
        .tag {
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.15);
            font-size: 0.88rem;
        }
        .section-card {
            background: rgba(255,255,255,0.84);
            border: 1px solid rgba(52, 74, 63, 0.08);
            box-shadow: 0 14px 34px rgba(63, 53, 34, 0.08);
            border-radius: 22px;
            padding: 1.2rem 1.2rem 0.9rem 1.2rem;
            margin-bottom: 1rem;
        }
        .section-kicker {
            color: #91632d;
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .section-title {
            color: #18342d;
            font-size: 1.45rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .section-copy {
            color: #43534e;
            line-height: 1.55;
            font-size: 0.98rem;
        }
        .mini-note {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: #efe4d4;
            border-left: 5px solid #b17d38;
            color: #43362a;
            margin: 0.6rem 0 1rem 0;
        }
        .takeaway {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: #edf4f0;
            border-left: 5px solid #2f6c56;
            color: #224236;
            margin: 0.5rem 0 1rem 0;
        }
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.76);
            border: 1px solid rgba(42, 70, 58, 0.08);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 10px 24px rgba(63, 53, 34, 0.06);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #efe4d5 0%, #f5eee6 100%);
        }
        .footer-box {
            padding: 1rem 1.2rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.72);
            border: 1px dashed rgba(42, 70, 58, 0.2);
            color: #33423d;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def br_currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    formatted = f"{float(value):,.2f}"
    formatted = formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {formatted}"


def br_number(value: float | int | None, decimals: int = 0) -> str:
    if value is None or pd.isna(value):
        return "-"
    formatted = f"{float(value):,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def br_percent(value: float | int | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{br_number(value, decimals)}%"


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, object]:
    annual_compare = pd.read_csv(OUTPUTS_DIR / "inflation_forecast" / "comparativo_inflacao_custos.csv")
    annual_costs = pd.read_csv(OUTPUTS_DIR / "inflation_forecast" / "custos_anuais_licitacao.csv")
    forecast = pd.read_csv(OUTPUTS_DIR / "inflation_forecast" / "previsao_2026.csv")
    model_scores = pd.read_csv(OUTPUTS_DIR / "inflation_forecast" / "avaliacao_modelos_previsao_2026.csv")
    correlations = pd.read_csv(OUTPUTS_DIR / "correlations" / "correlacoes_filtradas.csv")
    all_correlations = pd.read_csv(OUTPUTS_DIR / "correlations" / "correlacoes_completas.csv")
    normality = pd.read_csv(OUTPUTS_DIR / "target_normality" / "teste_normalidade_transformacoes.csv")
    with open(OUTPUTS_DIR / "unified_base" / "relatorio_qualidade.json", "r", encoding="utf-8") as handle:
        quality = json.load(handle)
    return {
        "annual_compare": annual_compare,
        "annual_costs": annual_costs,
        "forecast": forecast,
        "model_scores": model_scores,
        "correlations": correlations,
        "all_correlations": all_correlations,
        "normality": normality,
        "quality": quality,
    }


def section_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_theme(chart: alt.Chart) -> alt.Chart:
    return chart.configure_view(stroke=None).configure_axis(
        labelColor="#33423d",
        titleColor="#18342d",
        gridColor="#d8d2c7",
    ).configure_title(
        color="#18342d",
        fontSize=18,
        anchor="start",
    )


def build_hero(data: dict[str, object]) -> None:
    quality = data["quality"]
    forecast = data["forecast"].iloc[0]
    correlations = data["correlations"]
    annual_compare = data["annual_compare"]
    years = annual_compare["ano_dados"].astype(int)

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">Panorama das Licitações de Criciúma: 2015–2025</div>
            <div class="hero-subtitle">
                Este painel resume a construção da base, a análise estatística e a comparação entre o custo das licitações e a inflação.
                O foco foi entender a variância do custo homologado ao longo de 10 anos e produzir uma estimativa exploratória para 2026.
            </div>
            <div class="tag-row">
                <span class="tag">Base unificada em Python</span>
                <span class="tag">{int(years.min())}–{int(years.max())}</span>
                <span class="tag">{br_number(quality['registros_finais'])} registros analíticos</span>
                <span class="tag">{len(correlations)} variáveis bem correlacionadas</span>
                <span class="tag">Previsão exploratória para 2026</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registros Analíticos", br_number(quality["registros_finais"]))
    col2.metric("Licitações com Alvo Válido", br_number(sum(quality["estatisticas_por_ano"][year]["processos_com_participantes"] for year in quality["estatisticas_por_ano"])))
    col3.metric("Mediana Nominal 2025", br_currency(annual_compare.loc[annual_compare["ano_dados"] == 2025, "mediana_nominal"].iloc[0]))
    col4.metric("Previsão Mediana 2026", br_currency(forecast["mediana_nominal_prevista"]), br_percent(forecast["variacao_nominal_mediana_vs_2025_pct"]))


def overview_section(data: dict[str, object]) -> None:
    annual_compare = data["annual_compare"].copy()
    quality = data["quality"]

    section_header(
        "Visão Geral",
        "O que foi analisado",
        "A equipe construiu uma base única de licitações públicas, unificou mais de 10 anos de arquivos heterogêneos e transformou esse material em uma narrativa que combina contexto, estatística e apoio à decisão orçamentária.",
    )

    st.markdown(
        """
        <div class="takeaway">
            <strong>Leitura rápida:</strong> o custo homologado cresceu no período, e mesmo após retirar o efeito do IPCA o crescimento real da mediana continuou positivo.
            Isso indica que a pressão sobre o custo das licitações não foi explicada apenas pela inflação.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        trend_df = annual_compare.melt(
            id_vars="ano_dados",
            value_vars=["mediana_nominal", "mediana_real_2025"],
            var_name="serie",
            value_name="valor",
        )
        trend_df["serie"] = trend_df["serie"].map(
            {
                "mediana_nominal": "Mediana nominal",
                "mediana_real_2025": "Mediana real (base 2025)",
            }
        )
        chart = (
            alt.Chart(trend_df)
            .mark_line(point=True, strokeWidth=4)
            .encode(
                x=alt.X("ano_dados:O", title="Ano"),
                y=alt.Y("valor:Q", title="Valor"),
                color=alt.Color(
                    "serie:N",
                    title="Série",
                    scale=alt.Scale(range=["#b17831", "#234d3f"]),
                ),
                tooltip=[
                    alt.Tooltip("ano_dados:O", title="Ano"),
                    alt.Tooltip("serie:N", title="Série"),
                    alt.Tooltip("valor:Q", title="Valor", format=",.2f"),
                ],
            )
            .properties(height=350, title="Evolução da mediana anual do custo")
        )
        st.altair_chart(chart_theme(chart), use_container_width=True)

    with col2:
        process_rows = pd.DataFrame(
            {
                "Ano": list(quality["registros_por_ano"].keys()),
                "Itens de proposta": list(quality["registros_por_ano"].values()),
            }
        )
        chart = (
            alt.Chart(process_rows)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#91632d")
            .encode(
                x=alt.X("Ano:O"),
                y=alt.Y("Itens de proposta:Q", title="Registros"),
                tooltip=["Ano:O", alt.Tooltip("Itens de proposta:Q", format=",.0f")],
            )
            .properties(height=350, title="Volume de registros por ano")
        )
        st.altair_chart(chart_theme(chart), use_container_width=True)

    st.markdown(
        """
        <div class="mini-note">
            <strong>Como ler o painel:</strong> quando mostramos <em>mediana</em>, estamos usando a medida mais robusta para esta base,
            porque os custos são muito assimétricos e têm outliers relevantes. A média continua disponível, mas não é a protagonista.
        </div>
        """,
        unsafe_allow_html=True,
    )


def data_method_section(data: dict[str, object], show_technical: bool) -> None:
    quality = data["quality"]
    annual_compare = data["annual_compare"]

    section_header(
        "Base & Método",
        "Como a análise foi construída",
        "A apresentação foi desenhada para ser compreensível para quem nunca viu o projeto, mas mantendo o nível técnico necessário para justificar as escolhas estatísticas.",
    )

    steps = [
        ("1", "Auditoria dos CSVs", "Mapeamento das colunas por ano para descobrir mudanças de esquema e diferenças de encoding."),
        ("2", "Base unificada", "Expansão da licitação em participante e item de proposta, chegando à unidade analítica necessária."),
        ("3", "Estatística", "Correlação com a variável-alvo, teste de normalidade e comparação com inflação oficial."),
        ("4", "Projeção", "Deflação para base 2025 e previsão exploratória da mediana nominal para 2026."),
    ]
    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        col.markdown(
            f"""
            <div class="section-card" style="height: 220px;">
                <div class="section-kicker">Etapa {num}</div>
                <div class="section-title" style="font-size: 1.1rem;">{title}</div>
                <div class="section-copy">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    quality_cols = st.columns(4)
    quality_cols[0].metric("Registros finais", br_number(quality["registros_finais"]))
    quality_cols[1].metric("Duplicidades", br_number(quality["duplicidades_chave_primaria"]))
    quality_cols[2].metric("Anos cobertos", f"{annual_compare['ano_dados'].min()}–{annual_compare['ano_dados'].max()}")
    quality_cols[3].metric("Variáveis correlacionadas", br_number(len(data["correlations"])))

    year_stats = (
        pd.DataFrame(quality["estatisticas_por_ano"]).T.reset_index().rename(columns={"index": "ano"})
    )
    year_stats["ano"] = year_stats["ano"].astype(int)
    year_stats = year_stats.sort_values("ano")

    chart = (
        alt.Chart(year_stats)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#234d3f")
        .encode(
            x=alt.X("ano:O", title="Ano"),
            y=alt.Y("processos_com_participantes:Q", title="Licitações aproveitadas"),
            tooltip=[
                alt.Tooltip("ano:O", title="Ano"),
                alt.Tooltip("processos_lidos:Q", title="Processos lidos"),
                alt.Tooltip("processos_com_participantes:Q", title="Com participantes"),
                alt.Tooltip("itens_lidos:Q", title="Itens de proposta", format=",.0f"),
            ],
        )
        .properties(height=320, title="Aproveitamento da base ao longo dos anos")
    )
    st.altair_chart(chart_theme(chart), use_container_width=True)

    if show_technical:
        with st.expander("Notas técnicas da modelagem de dados", expanded=False):
            st.markdown(
                """
                - O arquivo principal de cada ano foi o `27_processoslicitatorios-61218.csv`.
                - A unidade de registro escolhida foi `1 item de proposta`, porque o trabalho exigia pelo menos 20 mil registros.
                - A variável-alvo é `valor_homologado`, replicada nas linhas derivadas da mesma licitação.
                - A série de normalidade foi testada por `licitação única`, para não superpesar licitações com mais itens.
                - Para a comparação com inflação, os valores foram deflacionados para `base 2025`.
                """
            )


def cost_evolution_section(data: dict[str, object]) -> None:
    annual_compare = data["annual_compare"].copy()

    section_header(
        "Custos",
        "Evolução do custo ao longo do tempo",
        "Aqui o foco sai do pipeline e vai para a história dos dados: como o custo se moveu ao longo do tempo, em termos nominais e reais.",
    )

    measure = st.radio(
        "Medida principal",
        options=["Mediana", "Média"],
        horizontal=True,
    )

    if measure == "Mediana":
        nominal_col = "mediana_nominal"
        real_col = "mediana_real_2025"
        std_col = "desvio_padrao_nominal"
    else:
        nominal_col = "media_nominal"
        real_col = "media_real_2025"
        std_col = "desvio_padrao_nominal"

    long_df = annual_compare[["ano_dados", nominal_col, real_col]].melt(
        id_vars="ano_dados",
        var_name="tipo",
        value_name="valor",
    )
    long_df["tipo"] = long_df["tipo"].map(
        {
            nominal_col: f"{measure} nominal",
            real_col: f"{measure} real (base 2025)",
        }
    )

    col1, col2 = st.columns([1.45, 1])
    with col1:
        chart = (
            alt.Chart(long_df)
            .mark_line(point=True, strokeWidth=4)
            .encode(
                x=alt.X("ano_dados:O", title="Ano"),
                y=alt.Y("valor:Q", title="Valor"),
                color=alt.Color("tipo:N", scale=alt.Scale(range=["#ad7330", "#22493d"]), title="Série"),
                tooltip=["ano_dados:O", "tipo:N", alt.Tooltip("valor:Q", format=",.2f")],
            )
            .properties(height=360, title=f"{measure} anual do custo homologado")
        )
        st.altair_chart(chart_theme(chart), use_container_width=True)

    with col2:
        chart = (
            alt.Chart(annual_compare)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#6c8a7b")
            .encode(
                x=alt.X("ano_dados:O", title="Ano"),
                y=alt.Y(f"{std_col}:Q", title="Desvio padrão nominal"),
                tooltip=["ano_dados:O", alt.Tooltip(f"{std_col}:Q", format=",.2f")],
            )
            .properties(height=360, title="Dispersão anual do custo")
        )
        st.altair_chart(chart_theme(chart), use_container_width=True)

    show_cols = [
        "ano_dados",
        "quantidade_licitacoes",
        nominal_col,
        real_col,
        "ipca_percentual",
        f"{nominal_col}_crescimento_pct",
        f"{real_col}_crescimento_pct",
    ]
    table = annual_compare[show_cols].copy()
    table.columns = [
        "Ano",
        "Qtd. licitações",
        f"{measure} nominal",
        f"{measure} real (base 2025)",
        "IPCA (%)",
        f"Crescimento {measure.lower()} nominal (%)",
        f"Crescimento {measure.lower()} real (%)",
    ]
    st.dataframe(table, use_container_width=True, hide_index=True)


def stats_section(data: dict[str, object], show_technical: bool) -> None:
    correlations = data["correlations"].copy()
    all_correlations = data["all_correlations"].copy()
    normality = data["normality"].copy()

    section_header(
        "Estatística",
        "Normalidade e correlação",
        "Esta seção reúne as duas peças centrais da análise estatística: o comportamento da variável-alvo e as variáveis que mais se relacionam com ela.",
    )

    col1, col2 = st.columns([1, 1.2])
    with col1:
        best_transformation = normality.sort_values("shapiro_w", ascending=False).iloc[0]
        original = normality.loc[normality["transformacao"] == "original"].iloc[0]
        st.metric("Shapiro-Wilk da série original", f"W = {original['shapiro_w']:.3f}", f"p = {original['shapiro_pvalue']:.2e}")
        st.metric("Melhor transformação", best_transformation["transformacao"], f"W = {best_transformation['shapiro_w']:.3f}")

        norm_chart = (
            alt.Chart(normality.sort_values("shapiro_w", ascending=False))
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("transformacao:N", sort="-y", title="Transformação"),
                y=alt.Y("shapiro_w:Q", title="Estatística W"),
                color=alt.condition(
                    alt.datum.transformacao == best_transformation["transformacao"],
                    alt.value("#234d3f"),
                    alt.value("#b17831"),
                ),
                tooltip=[
                    "transformacao:N",
                    alt.Tooltip("shapiro_w:Q", format=".4f"),
                    alt.Tooltip("shapiro_pvalue:Q", format=".2e"),
                    alt.Tooltip("skewness:Q", format=".4f"),
                ],
            )
            .properties(height=320, title="Comparação das transformações testadas")
        )
        st.altair_chart(chart_theme(norm_chart), use_container_width=True)

    with col2:
        corr_chart = (
            alt.Chart(correlations.head(15))
            .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
            .encode(
                y=alt.Y("variavel:N", sort="-x", title="Variável"),
                x=alt.X("correlacao_pearson:Q", title="Correlação de Pearson"),
                color=alt.condition(
                    alt.datum.correlacao_pearson > 0,
                    alt.value("#234d3f"),
                    alt.value("#b17831"),
                ),
                tooltip=[
                    "variavel:N",
                    alt.Tooltip("correlacao_pearson:Q", format=".4f"),
                    alt.Tooltip("amostra_utilizada:Q", format=",.0f"),
                ],
            )
            .properties(height=420, title="15 variáveis mais correlacionadas com o custo homologado")
        )
        st.altair_chart(chart_theme(corr_chart), use_container_width=True)

    st.markdown(
        """
        <div class="mini-note">
            <strong>Resumo estatístico:</strong> a série original do custo homologado não é normal.
            Mesmo após as transformações, a distribuição não passou no teste a 5%, mas o Box-Cox reduziu muito a assimetria.
            Isso não invalida a análise; apenas pede cuidado na interpretação e preferência por medidas robustas como a mediana.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_technical:
        with st.expander("Tabela técnica de normalidade e correlação", expanded=False):
            st.dataframe(
                normality[["transformacao", "shapiro_w", "shapiro_pvalue", "skewness", "kurtosis"]]
                .sort_values("shapiro_w", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
            st.dataframe(
                all_correlations[["variavel", "correlacao_pearson", "correlacao_absoluta", "p_valor"]]
                .sort_values("correlacao_absoluta", ascending=False)
                .head(25),
                use_container_width=True,
                hide_index=True,
            )


def inflation_forecast_section(data: dict[str, object]) -> None:
    annual_compare = data["annual_compare"].copy()
    forecast = data["forecast"].iloc[0]
    model_scores = data["model_scores"].copy()
    first_year = annual_compare.iloc[0]
    last_year = annual_compare.iloc[-1]

    base_nominal = float(first_year["mediana_nominal"])
    base_index = float(first_year["indice_precos"])
    annual_compare["mediana_seguindo_ipca"] = base_nominal * (annual_compare["indice_precos"] / base_index)
    annual_compare["indice_custo_observado"] = (annual_compare["mediana_nominal"] / base_nominal) * 100.0
    annual_compare["indice_somente_ipca"] = (annual_compare["indice_precos"] / base_index) * 100.0
    annual_compare["gap_reais_vs_ipca"] = annual_compare["mediana_nominal"] - annual_compare["mediana_seguindo_ipca"]

    years_span = int(last_year["ano_dados"] - first_year["ano_dados"])
    real_cagr_pct = ((float(last_year["mediana_real_2025"]) / float(first_year["mediana_real_2025"])) ** (1 / years_span) - 1) * 100.0
    ipca_2026_pct = float(forecast["ipca_cenario_2026_percentual"])
    direct_real_2026 = float(last_year["mediana_real_2025"]) * (1 + real_cagr_pct / 100.0)
    direct_nominal_2026 = direct_real_2026 * (1 + ipca_2026_pct / 100.0)
    inflation_only_2026 = float(last_year["mediana_nominal"]) * (1 + ipca_2026_pct / 100.0)

    section_header(
        "Inflação & Previsão",
        "O custo cresceu só por causa da inflação?",
        "A comparação abaixo foi construída com o IPCA anual informado para o período. O objetivo é separar crescimento nominal de crescimento real e produzir uma estimativa exploratória para 2026.",
    )

    st.markdown(
        """
        <div class="takeaway">
            <strong>Leitura mais direta:</strong> a linha de referência abaixo mostra onde o custo estaria se tivesse crescido apenas no ritmo do IPCA.
            Quando a curva observada sobe mais do que essa referência, temos evidência visual de aumento acima da inflação.
        </div>
        """,
        unsafe_allow_html=True,
    )

    compare_df = annual_compare[["ano_dados", "ipca_percentual", "mediana_nominal_crescimento_pct", "gap_mediana_nominal_menos_ipca"]].copy().dropna()

    col1, col2 = st.columns([1.35, 1])
    with col1:
        direct_compare = annual_compare[["ano_dados", "mediana_nominal", "mediana_seguindo_ipca"]].melt(
            id_vars="ano_dados",
            var_name="serie",
            value_name="valor",
        )
        direct_compare["serie"] = direct_compare["serie"].map(
            {
                "mediana_nominal": "Custo observado",
                "mediana_seguindo_ipca": "Custo se seguisse só o IPCA",
            }
        )
        chart = (
            alt.Chart(direct_compare)
            .mark_line(point=True, strokeWidth=4)
            .encode(
                x=alt.X("ano_dados:O", title="Ano"),
                y=alt.Y("valor:Q", title="Mediana nominal (R$)"),
                color=alt.Color(
                    "serie:N",
                    title="Série",
                    scale=alt.Scale(range=["#234d3f", "#c48c48"]),
                ),
                tooltip=[
                    alt.Tooltip("ano_dados:O", title="Ano"),
                    alt.Tooltip("serie:N", title="Série"),
                    alt.Tooltip("valor:Q", title="Valor", format=",.2f"),
                ],
            )
            .properties(height=360, title="Custo observado vs custo explicado apenas pela inflação")
        )
        st.altair_chart(chart_theme(chart), use_container_width=True)

    with col2:
        gap_chart = (
            alt.Chart(annual_compare)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("ano_dados:O", title="Ano"),
                y=alt.Y("gap_reais_vs_ipca:Q", title="Diferença em R$"),
                color=alt.condition(
                    alt.datum.gap_reais_vs_ipca > 0,
                    alt.value("#234d3f"),
                    alt.value("#b17831"),
                ),
                tooltip=[
                    alt.Tooltip("ano_dados:O", title="Ano"),
                    alt.Tooltip("mediana_nominal:Q", title="Observado", format=",.2f"),
                    alt.Tooltip("mediana_seguindo_ipca:Q", title="Somente IPCA", format=",.2f"),
                    alt.Tooltip("gap_reais_vs_ipca:Q", title="Diferença", format=",.2f"),
                ],
            )
            .properties(height=360, title="Quanto o custo ficou acima do cenário só-inflacionário")
        )
        st.altair_chart(chart_theme(gap_chart), use_container_width=True)

    index_compare = annual_compare[["ano_dados", "indice_custo_observado", "indice_somente_ipca"]].melt(
        id_vars="ano_dados",
        var_name="serie",
        value_name="indice",
    )
    index_compare["serie"] = index_compare["serie"].map(
        {
            "indice_custo_observado": "Custo observado (base 100 em 2015)",
            "indice_somente_ipca": "Inflação acumulada (base 100 em 2015)",
        }
    )
    index_chart = (
        alt.Chart(index_compare)
        .mark_line(point=True, strokeWidth=4)
        .encode(
            x=alt.X("ano_dados:O", title="Ano"),
            y=alt.Y("indice:Q", title="Índice base 100"),
            color=alt.Color("serie:N", scale=alt.Scale(range=["#234d3f", "#b17831"]), title="Comparação"),
            tooltip=["ano_dados:O", "serie:N", alt.Tooltip("indice:Q", format=".2f")],
        )
        .properties(height=330, title="Aumento acumulado: custo real do projeto vs inflação")
    )
    st.altair_chart(chart_theme(index_chart), use_container_width=True)

    section_header(
        "Previsão 2026",
        "Cenários para o próximo ano",
        "A projeção foi reorganizada para ficar mais direta. Em vez de mostrar só o modelo estatístico, o painel agora separa um cenário simples e explicável de um cenário modelado.",
    )

    projection_points = pd.DataFrame(
        [
            {"ano": int(last_year["ano_dados"]), "cenario": "Histórico observado", "valor": float(last_year["mediana_nominal"])},
            {"ano": int(forecast["ano_previsao"]), "cenario": "Somente inflação", "valor": inflation_only_2026},
            {"ano": int(forecast["ano_previsao"]), "cenario": "Cenário direto", "valor": direct_nominal_2026},
            {"ano": int(forecast["ano_previsao"]), "cenario": "Modelo estatístico", "valor": float(forecast["mediana_nominal_prevista"])},
        ]
    )

    projected_history = annual_compare[["ano_dados", "mediana_nominal"]].copy()
    projected_history["cenario"] = "Histórico observado"
    projected_history = projected_history.rename(columns={"ano_dados": "ano", "mediana_nominal": "valor"})
    forecast_chart_df = pd.concat([projected_history, projection_points[projection_points["ano"] == int(forecast["ano_previsao"])]], ignore_index=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        lines = (
            alt.Chart(projected_history)
            .mark_line(point=True, strokeWidth=4, color="#22493d")
            .encode(
                x=alt.X("ano:O", title="Ano"),
                y=alt.Y("valor:Q", title="Mediana nominal (R$)"),
                tooltip=[alt.Tooltip("ano:O", title="Ano"), alt.Tooltip("valor:Q", title="Valor", format=",.2f")],
            )
        )
        points = (
            alt.Chart(projection_points)
            .mark_point(size=180, filled=True)
            .encode(
                x=alt.X("ano:O"),
                y=alt.Y("valor:Q"),
                color=alt.Color(
                    "cenario:N",
                    title="Cenário 2026",
                    scale=alt.Scale(
                        range=["#c48c48", "#5f8a76", "#8a3d2c", "#22493d"],
                    ),
                ),
                tooltip=["cenario:N", alt.Tooltip("valor:Q", title="Valor projetado", format=",.2f")],
            )
        )
        projection_chart = (lines + points).properties(height=360, title="2025 observado e cenários para 2026")
        st.altair_chart(chart_theme(projection_chart), use_container_width=True)

    with col2:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-kicker">Base da projeção</div>
                <div class="section-title" style="font-size:1.15rem;">Cenário direto</div>
                <div class="section-copy">
                    A mediana real de 2026 foi estimada usando o <strong>crescimento real médio anual composto</strong> entre 2015 e 2025.
                    Depois, aplicamos um cenário de inflação para 2026 igual à <strong>média do IPCA de 2023–2025</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        summary_df = pd.DataFrame(
            [
                {"Indicador": "Crescimento real médio anual (2015–2025)", "Valor": br_percent(real_cagr_pct)},
                {"Indicador": "IPCA usado para 2026", "Valor": br_percent(ipca_2026_pct)},
                {"Indicador": "2026 se seguisse só inflação", "Valor": br_currency(inflation_only_2026)},
                {"Indicador": "2026 - cenário direto", "Valor": br_currency(direct_nominal_2026)},
                {"Indicador": "2026 - modelo estatístico", "Valor": br_currency(forecast["mediana_nominal_prevista"])},
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns([1, 1.15])
    with col1:
        chart = (
            alt.Chart(model_scores)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#234d3f")
            .encode(
                x=alt.X("modelo:N", sort="-y", title="Modelo"),
                y=alt.Y("mape:Q", title="MAPE (%)"),
                tooltip=["modelo:N", alt.Tooltip("mape:Q", format=".2f"), alt.Tooltip("mae:Q", format=",.2f")],
            )
            .properties(height=320, title="Comparação dos modelos de previsão")
        )
        st.altair_chart(chart_theme(chart), use_container_width=True)

    with col2:
        model_df = pd.DataFrame(
            [
                {
                    "Indicador": "Mediana nominal 2025",
                    "Valor": br_currency(annual_compare.loc[annual_compare["ano_dados"] == 2025, "mediana_nominal"].iloc[0]),
                },
                {
                    "Indicador": "Mediana nominal prevista 2026",
                    "Valor": br_currency(forecast["mediana_nominal_prevista"]),
                },
                {
                    "Indicador": "Variação nominal prevista",
                    "Valor": br_percent(forecast["variacao_nominal_mediana_vs_2025_pct"]),
                },
                {
                    "Indicador": "Variação real prevista",
                    "Valor": br_percent(forecast["variacao_real_mediana_vs_2025_pct"]),
                },
                {
                    "Indicador": "Erro médio do modelo (MAPE)",
                    "Valor": br_percent(model_scores.iloc[0]["mape"]),
                },
            ]
        )
        st.dataframe(model_df, use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="mini-note">
                <strong>Como apresentar isso:</strong> o cenário direto é o mais fácil de explicar para a banca,
                porque combina crescimento médio real do período com inflação esperada.
                O modelo estatístico entra como uma segunda referência, mais técnica, mas com erro histórico ainda relevante.
            </div>
            """,
            unsafe_allow_html=True,
        )


def conclusion_section(data: dict[str, object]) -> None:
    annual_compare = data["annual_compare"]
    forecast = data["forecast"].iloc[0]
    correlations = data["correlations"]

    section_header(
        "Fechamento",
        "O que o professor deve levar desta apresentação",
        "A narrativa final do trabalho precisa ser simples de seguir: construímos a base, testamos hipóteses, comparamos os custos à inflação e produzimos uma previsão coerente com os limites do método.",
    )

    first = annual_compare.iloc[0]
    last = annual_compare.iloc[-1]
    cumulative_nominal = ((last["mediana_nominal"] / first["mediana_nominal"]) - 1) * 100
    cumulative_real = ((last["mediana_real_2025"] / first["mediana_real_2025"]) - 1) * 100

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-copy">
                <strong>Conclusões principais:</strong><br><br>
                1. A base consolidada permitiu observar o comportamento do custo homologado entre 2015 e 2025 com granularidade suficiente para análise estatística.<br>
                2. O custo nominal mediano cresceu <strong>{br_percent(cumulative_nominal)}</strong> no período.<br>
                3. Mesmo descontando o IPCA, o crescimento real da mediana foi de <strong>{br_percent(cumulative_real)}</strong>, sugerindo aumento acima da inflação.<br>
                4. A variável-alvo não apresentou normalidade no SW Teste; por isso a mediana foi adotada como medida principal e a leitura da previsão foi feita com cautela.<br>
                5. A projeção exploratória indica uma mediana nominal de aproximadamente <strong>{br_currency(forecast["mediana_nominal_prevista"])}</strong> em 2026.<br>
                6. A análise encontrou <strong>{len(correlations)}</strong> variáveis com correlação forte o suficiente para sustentar a etapa estatística exigida no trabalho.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="footer-box">
            <strong>Equipe:</strong> {" • ".join(TEAM_MEMBERS)}<br>
            <strong>Observação:</strong> edite a lista <code>TEAM_MEMBERS</code> no arquivo <code>streamlit_app.py</code> antes da apresentação para colocar os nomes reais do grupo.
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls(data: dict[str, object]) -> tuple[str, bool]:
    st.sidebar.title("Navegação")
    section = st.sidebar.radio(
        "Ir para",
        [
            "Visão Geral",
            "Base & Método",
            "Custos",
            "Estatística",
            "Inflação & Previsão",
            "Fechamento",
        ],
    )
    st.sidebar.divider()
    st.sidebar.markdown("### Leitura")
    show_technical = st.sidebar.toggle("Mostrar notas técnicas", value=True)
    st.sidebar.caption(
        "Use essa chave para alternar entre uma apresentação mais amigável e uma visão mais técnica para o professor."
    )
    st.sidebar.divider()
    st.sidebar.markdown("### Resumo Rápido")
    st.sidebar.metric("Período", "2015–2025")
    st.sidebar.metric("Registros analíticos", br_number(data["quality"]["registros_finais"]))
    st.sidebar.metric("Correlação forte", br_number(len(data["correlations"])))
    st.sidebar.metric("Previsão mediana 2026", br_currency(data["forecast"].iloc[0]["mediana_nominal_prevista"]))
    return section, show_technical


def main() -> None:
    inject_styles()
    data = load_data()
    section, show_technical = sidebar_controls(data)

    build_hero(data)

    if section == "Visão Geral":
        overview_section(data)
    elif section == "Base & Método":
        data_method_section(data, show_technical)
    elif section == "Custos":
        cost_evolution_section(data)
    elif section == "Estatística":
        stats_section(data, show_technical)
    elif section == "Inflação & Previsão":
        inflation_forecast_section(data)
    elif section == "Fechamento":
        conclusion_section(data)


if __name__ == "__main__":
    main()
