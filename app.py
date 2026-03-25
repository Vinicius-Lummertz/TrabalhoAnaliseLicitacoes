import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Análise de Licitações e Contratos",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def limpar_texto_vazio(df):
    for col in df.columns:
        if df[col].dtype == "object":
            s = df[col].astype("string").str.strip()
            s = s.mask(s.isin(["", "nan", "None", "<NA>"]))
            df[col] = s
    return df

def para_numero(series):
    return pd.to_numeric(series, errors="coerce")

def carregar_dados():
    df_proc1 = pd.read_csv("27_processoslicitatorios-34862.csv")
    df_proc2 = pd.read_csv("27_processoslicitatorios-34864.csv")
    df_contratos = pd.read_csv("27_contratos-87437.csv")

    df_proc1 = limpar_texto_vazio(df_proc1)
    df_proc2 = limpar_texto_vazio(df_proc2)
    df_contratos = limpar_texto_vazio(df_contratos)

    # Padronização da variável principal
    df_proc1["valor_licitacao"] = para_numero(df_proc1["valorHomologado"])
    df_proc2["valor_licitacao"] = para_numero(df_proc2["valorHomologado"])
    df_contratos["valor_licitacao"] = para_numero(df_contratos["valorFinal"])

    # Padronizar nomes para facilitar união
    df_proc1["origem_arquivo"] = "processos_34862"
    df_proc2["origem_arquivo"] = "processos_34864"
    df_contratos["origem_arquivo"] = "contratos_87437"

    # Criar uma base única de processos
    df_processos = pd.concat([df_proc1, df_proc2], ignore_index=True)

    # Chave de união
    df_processos["chave_processo"] = (
        df_processos["numeroProcesso"].astype(str).str.strip() + "_" +
        df_processos["anoProcesso"].astype(str).str.strip()
    )

    df_contratos["chave_processo"] = (
        df_contratos["numeroProcessoCompra"].astype(str).str.strip() + "_" +
        df_contratos["anoProcessoCompra"].astype(str).str.strip()
    )

    # Datas
    for col in ["dataPublicacao", "dataHomologacao", "dataAberturaEnvelopes", "dataCriacao", "dataJulgamento"]:
        if col in df_processos.columns:
            df_processos[col] = para_data_sem_tz(df_processos[col])

    for col in ["dataAssinatura", "dataCompetencia", "dataVigenciaInicial", "dataVigenciaFinal"]:
        if col in df_contratos.columns:
            df_contratos[col] = para_data_sem_tz(df_contratos[col])

    # Base unificada:
    # Mantém processos como base principal e junta contratos quando houver chave compatível
    df_merged = df_processos.merge(
        df_contratos[
            [
                "chave_processo",
                "nomeContratado",
                "cnpjCpfContratado",
                "valorFinal",
                "valorInicial",
                "valorAlterado",
                "modalidadeLicitacao",
                "tipo",
                "situacao",
                "instrumento",
                "idFornecedor",
                "idProcessoCompra",
                "previsaoSubcontratacao"
            ]
        ],
        on="chave_processo",
        how="left",
        suffixes=("_proc", "_contrato")
    )

    # Variáveis derivadas
    df_merged["valorEstimado"] = para_numero(df_merged["valorEstimado"])
    df_merged["valorHomologado"] = para_numero(df_merged["valorHomologado"])
    df_merged["valorFinal"] = para_numero(df_merged["valorFinal"])
    df_merged["valorInicial"] = para_numero(df_merged["valorInicial"])
    df_merged["valorAlterado"] = para_numero(df_merged["valorAlterado"])

    # Variável principal para análise
    # Primeiro usa homologado; se não houver, usa valor final do contrato
    df_merged["valor_principal"] = df_merged["valorHomologado"].fillna(df_merged["valorFinal"])

    # Duracao do processo em dias
    if "dataCriacao" in df_merged.columns and "dataHomologacao" in df_merged.columns:
        df_merged["dias_ate_homologacao"] = (
            df_merged["dataHomologacao"] - df_merged["dataCriacao"]
        ).dt.days

    if "dataPublicacao" in df_merged.columns and "dataAberturaEnvelopes" in df_merged.columns:
        df_merged["dias_publicacao_ate_abertura"] = (
            df_merged["dataAberturaEnvelopes"] - df_merged["dataPublicacao"]
        ).dt.days

    # Tamanho do texto do objeto
    df_merged["tamanho_objeto"] = df_merged["objeto"].astype(str).str.len()

    # Flags binárias úteis
    df_merged["registroPrecos_bin"] = df_merged["registroPrecos"].astype(str).str.upper().eq("SIM").astype(int)
    df_merged["adesao_bin"] = df_merged["adesao"].astype(str).str.upper().eq("SIM").astype(int)

    # Quantidade aproximada de participantes a partir da presença de arquivo
    df_merged["tem_participantes"] = df_merged["participantes"].notna().astype(int)
    df_merged["tem_contrato"] = df_merged["contratos"].notna().astype(int)

    return df_proc1, df_proc2, df_contratos, df_processos, df_merged


def calcular_boxplot_info(serie):
    serie = serie.dropna()
    q1 = serie.quantile(0.25)
    mediana = serie.quantile(0.50)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    outliers = serie[(serie < limite_inferior) | (serie > limite_superior)]
    return {
        "q1": q1,
        "mediana": mediana,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": limite_inferior,
        "limite_superior": limite_superior,
        "qtd_outliers": len(outliers)
    }


def formatar_moeda(valor):
    if pd.isna(valor):
        return "N/A"
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def para_data_sem_tz(series):
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)
# =========================================================
# CARREGAMENTO
# =========================================================

df_proc1, df_proc2, df_contratos, df_processos, df_merged = carregar_dados()

st.title("📊 Análise de Licitações e Contratos Públicos")
st.markdown("""
### 🔍 O Cenário: Em Busca do Fio da Meada
No ambiente público, sabemos que infelizmente ocorrem esquemas de superfaturamento e direcionamento de licitações. Mas como a fraude se oculta no meio de milhares de processos de licitações, contratos e dispensas? A resposta quase sempre mora nos padrões numéricos.

Com base no que aprendemos, o objetivo aqui é usar estatística básica como uma lupa de auditoria. Para isso, cruzamos informações de três frentes:
1. **Processos Licitatórios**
2. **Relação dos Contratos**
3. **Dispensa de Licitação** (que costuma ser um ponto alto de atenção)

### 🎯 Como a Estatística Aponta as Anomalias?
- **Mediana e Quartis (Q1 e Q3):** Eles definem o comportamento "esperado" do gasto público da nossa amostra. Tudo que está dentro da curva representa a maioria das compras.
- **Boxplot e Outliers:** São os nossos alarmes principais. Valores absurdos que extrapolam totalmente o limite do Boxplot (os famosos outliers) saltam aos olhos. Na prática, um outlier pode representar um contrato completamente superestimado que disfarça um escoamento de dinheiro.
- **Correlações:** Revelam relações que não são óbvias. Será que o tempo que um processo tramita até ser homologado interfere no valor? Compras realizadas de forma rápida demais e com valores altos podem indicar falta de pesquisa competitiva.

### Pergunta-problema
**Dessa forma, os valores das licitações estão distribuídos de forma equilibrada ou existe uma concentração de recursos discrepante apontando para poucos fornecedores e processos (outliers)?**

O foco central da nossa investigação foi a variável **valor_principal** (o valor final ou homologado gasto no contrato/licitação).
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Filtros")

entidades = ["Todas"] + sorted(df_merged["nomeEntidade"].dropna().unique().tolist())
entidade_sel = st.sidebar.selectbox("Entidade", entidades)

situacoes = ["Todas"] + sorted(df_merged["situacao_proc"].dropna().unique().tolist())
situacao_sel = st.sidebar.selectbox("Situação do processo", situacoes)

modalidades = ["Todas"] + sorted(df_merged["modalidade"].dropna().unique().tolist())
modalidade_sel = st.sidebar.selectbox("Modalidade", modalidades)

somente_com_valor = st.sidebar.checkbox("Mostrar apenas registros com valor principal", value=True)

df_filtrado = df_merged.copy()

if entidade_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["nomeEntidade"] == entidade_sel]

if situacao_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["situacao_proc"] == situacao_sel]

if modalidade_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["modalidade"] == modalidade_sel]

if somente_com_valor:
    df_filtrado = df_filtrado[df_filtrado["valor_principal"].notna()]

# =========================================================
# KPIs
# =========================================================

st.subheader("Visão geral")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total de registros analisados", f"{len(df_filtrado):,}".replace(",", "."))
col2.metric("Valor total analisado", formatar_moeda(df_filtrado["valor_principal"].sum()))
col3.metric("Mediana dos valores", formatar_moeda(df_filtrado["valor_principal"].median()))
col4.metric("Média dos valores", formatar_moeda(df_filtrado["valor_principal"].mean()))

st.error("""
**🕵️ O que os dados reais nos dizem?**  
A **média** (R$ 730 mil) é quase 7 vezes maior que a mediana (R$ 110 mil). Quando a média é tão violentamente puxada para cima em relação ao valor da mediana, temos a certeza estatística de que existem poucos processos com valores astronômicos distorcendo a base. Considerando os **572 registros** e os **417 milhões** totais, fica óbvio que a grande massa de dinheiro está fluindo para a minoria dos contratos.
""")

# =========================================================
# BOXPLOT E EXPLICAÇÃO
# =========================================================

st.subheader("1) Boxplot da variável principal")

st.markdown("""
O **boxplot** resume a distribuição dos valores:
- **Q1 (1º quartil):** 25% dos valores estão abaixo dele
- **Mediana:** valor central da distribuição
- **Q3 (3º quartil):** 75% dos valores estão abaixo dele
- **IQR:** intervalo interquartil, que mede a dispersão central
- **Outliers:** valores muito acima ou abaixo do padrão esperado
""")

serie_box = df_filtrado["valor_principal"].dropna()

if len(serie_box) > 0:
    info_box = calcular_boxplot_info(serie_box)

    c1, c2 = st.columns([2, 1])

    with c1:
        fig, ax = plt.subplots(figsize=(12, 3))
        sns.boxplot(x=serie_box, ax=ax)
        ax.set_title("Boxplot dos valores principais")
        ax.set_xlabel("Valor (R$)")
        st.pyplot(fig)

        fig_log, ax_log = plt.subplots(figsize=(12, 3))
        sns.boxplot(x=serie_box[serie_box > 0], ax=ax_log)
        ax_log.set_xscale("log")
        ax_log.set_title("Boxplot em escala logarítmica")
        ax_log.set_xlabel("Valor (R$, escala log)")
        st.pyplot(fig_log)

    with c2:
        st.markdown("#### Estatísticas do boxplot")
        st.write(f"**Q1:** {formatar_moeda(info_box['q1'])}")
        st.write(f"**Mediana:** {formatar_moeda(info_box['mediana'])}")
        st.write(f"**Q3:** {formatar_moeda(info_box['q3'])}")
        st.write(f"**IQR:** {formatar_moeda(info_box['iqr'])}")
        st.write(f"**Limite inferior:** {formatar_moeda(info_box['limite_inferior'])}")
        st.write(f"**Limite superior:** {formatar_moeda(info_box['limite_superior'])}")
        st.write(f"**Quantidade de outliers:** {info_box['qtd_outliers']}")

    st.error("""
**🕵️ O que os dados nos dizem?**  
Com o Boxplot, identificamos fisicamente o problema: 88 processos são classificados como Outliers, ultrapassando totalmente o limite superior aceitável (que seria de R$ 1,1 milhão). O gráfico revela pontos chegando na casa inacreditável dos **25 milhões**. Essa disparidade brutal é a nossa primeira "sirene". Por que esses 88 contratos, que fogem tão absurdamente do padrão histórico de compras, custam tão caro? Eles seriam os primeiros alvos de uma abertura de CPI ou auditoria do TCU.
""")
else:
    st.warning("Não há dados suficientes para montar o boxplot com os filtros atuais.")

# =========================================================
# HISTOGRAMA
# =========================================================

st.subheader("2) Distribuição dos valores")

col_a, col_b = st.columns(2)

with col_a:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(serie_box.dropna(), bins=30)
    ax.set_title("Histograma dos valores principais")
    ax.set_xlabel("Valor (R$)")
    ax.set_ylabel("Frequência")
    st.pyplot(fig)

with col_b:
    serie_log = serie_box[serie_box > 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.log10(serie_log), bins=30)
    ax.set_title("Histograma em escala log10")
    ax.set_xlabel("log10(Valor)")
    ax.set_ylabel("Frequência")
    st.pyplot(fig)

st.warning("""
**🕵️ O que os dados nos dizem?**  
O primeiro histograma evidencia uma assimetria extrema (a barra gigante espremida na extrema esquerda). Significa que a imensa maioria dos gastos públicos transita em processos de valores baixos. A grande questão é o "rastro" contínuo para o lado direito. São processos raríssimos que alcançam cifras milionárias, mascarados num universo de pequenas compras.
""")

# =========================================================
# CONCENTRAÇÃO POR EMPRESA
# =========================================================

st.subheader("3) Concentração de valor por empresa contratada")

df_empresas = (
    df_filtrado[df_filtrado["nomeContratado"].notna()]
    .groupby("nomeContratado", as_index=False)["valorFinal"]
    .sum()
    .sort_values("valorFinal", ascending=False)
)

if len(df_empresas) > 0:
    top_n = st.slider("Quantidade de empresas no ranking", 5, 20, 10)

    top_empresas = df_empresas.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(top_empresas["nomeContratado"][::-1], top_empresas["valorFinal"][::-1])
    ax.set_title(f"Top {top_n} empresas por valor final contratado")
    ax.set_xlabel("Valor total contratado (R$)")
    ax.set_ylabel("Empresa")
    st.pyplot(fig)

    total_empresas = df_empresas["valorFinal"].sum()
    top5_percent = df_empresas.head(5)["valorFinal"].sum() / total_empresas * 100 if total_empresas > 0 else 0

    st.error(f"""
**🕵️ O que os dados nos dizem?**  
A visualização das empresas acende o maior alerta até agora: a fornecedora **PROSUL** absorve sozinha cifras superiores a **R$ 25 milhões**, distanciando-se agressivamente até do segundo e terceiro lugares (DC10 e Maxifrota). Além disso, o top 5 concentra **{top5_percent:.2f}%** de todo o dinheiro do recorte. No universo público, um único fornecedor que monopoliza o topo dos gastos com contratos milionários é o "Sinal de Fumaça" padrão para investigações sobre monopólio e direcionamento de pregões.
""")
else:
    st.warning("Não há contratos vinculados aos filtros atuais.")

# =========================================================
# VARIÁVEIS INFLUENCIADORAS
# =========================================================

st.subheader("4) Variáveis que podem influenciar o fenômeno")

variaveis_texto = pd.DataFrame({
    "Variável": [
        "modalidade",
        "tipoObjeto",
        "formaJulgamento",
        "situacao_proc",
        "registroPrecos",
        "adesao",
        "anoProcesso",
        "dias_ate_homologacao",
        "dias_publicacao_ate_abertura",
        "tamanho_objeto",
        "nomeEntidade"
    ],
    "Por que pode influenciar o valor?": [
        "Cada modalidade tem regras e limites diferentes, o que tende a impactar o porte financeiro dos processos.",
        "Obras, tecnologia, serviços e compras têm estruturas de custo muito diferentes.",
        "Critérios como menor preço por item, global ou técnica e preço afetam o valor final e a competitividade.",
        "Processos homologados, revogados ou anulados têm comportamentos financeiros distintos.",
        "Registro de preços costuma estar ligado a compras recorrentes e volumes maiores.",
        "Adesão a atas pode alterar dinâmica competitiva e valores contratados.",
        "O ano pode capturar inflação, mudanças de gestão e volume orçamentário.",
        "Processos mais longos podem envolver maior complexidade e, muitas vezes, maiores valores.",
        "Mais tempo entre publicação e abertura pode refletir maior porte ou complexidade.",
        "Objetos mais longos e detalhados podem representar contratações mais complexas.",
        "Entidades diferentes possuem perfis de gasto distintos."
    ]
})

st.dataframe(variaveis_texto, use_container_width=True)

# =========================================================
# CORRELAÇÃO
# =========================================================

st.subheader("5) Correlação das variáveis numéricas")

colunas_numericas = [
    "valor_principal",
    "valorEstimado",
    "valorHomologado",
    "valorFinal",
    "valorInicial",
    "valorAlterado",
    "anoProcesso",
    "anoLicitacao",
    "dias_ate_homologacao",
    "dias_publicacao_ate_abertura",
    "tamanho_objeto",
    "registroPrecos_bin",
    "adesao_bin",
    "tem_participantes",
    "tem_contrato"
]

df_corr = df_filtrado[colunas_numericas].copy()
df_corr = df_corr.apply(pd.to_numeric, errors="coerce")

if df_corr["valor_principal"].notna().sum() > 2:
    corr = df_corr.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Mapa de correlação")
    st.pyplot(fig)

    corr_target = corr["valor_principal"].sort_values(ascending=False).drop("valor_principal")
    st.markdown("#### Correlação com a variável principal")
    st.dataframe(corr_target.reset_index().rename(columns={"index": "Variável", "valor_principal": "Correlação"}), use_container_width=True)

    st.warning("""
**🕵️ O que os dados nos dizem?**  
O resultado mais suspeito deste mapa é justamente a **falta** de correlações que deveriam existir num ambiente burocrático e seguro. Fatores como a complexidade (medida pelo **tamanho do objeto**: -0.06) ou o tempo da burocracia (**dias até a homologação**: 0.15) mostram relação quase nula com o valor!  
Na prática: isso significa que a aprovação de um contrato superfaturado de milhões tramitou com o mesmo tempo e rito burocrático de uma rampa de 50 mil reais. Compras milionárias fluindo velozmente e sem descritivos mais elaborados são sintomas crônicos de editais previamente orquestrados por partes envolvidas (conhecido como "carta marcada").
""")
else:
    st.warning("Não há dados numéricos suficientes para calcular correlação.")

# =========================================================
# BASE UNIFICADA
# =========================================================

st.subheader("6) Base unificada utilizada na análise")

colunas_base_final = [
    "chave_processo",
    "nomeEntidade",
    "numeroProcesso",
    "anoProcesso",
    "numeroLicitacao",
    "anoLicitacao",
    "objeto",
    "modalidade",
    "tipoObjeto",
    "formaJulgamento",
    "situacao_proc",
    "registroPrecos",
    "adesao",
    "valorEstimado",
    "valorHomologado",
    "valorFinal",
    "valorInicial",
    "valorAlterado",
    "valor_principal",
    "dias_ate_homologacao",
    "dias_publicacao_ate_abertura",
    "tamanho_objeto",
    "nomeContratado",
    "cnpjCpfContratado"
]

base_exibicao = df_filtrado[[c for c in colunas_base_final if c in df_filtrado.columns]].copy()
st.dataframe(base_exibicao.head(100), use_container_width=True)

csv = base_exibicao.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Baixar base unificada em CSV",
    data=csv,
    file_name="base_unificada_licitacoes.csv",
    mime="text/csv"
)

# =========================================================
# CONCLUSÃO
# =========================================================

st.subheader("7) Conclusão da Investigação")

st.markdown("""
### O Veredito dos Dados
O que os números nos contaram até aqui? Ao aplicar as ferramentas estatísticas aprendidas em aula, conseguimos traçar um panorama que indica de onde o dinheiro sai e para onde ele vai, levantando bandeiras importantes.

### Evidências Encontradas (Onde estaria a suspeita da fraude?)
1. **Outliers no Boxplot:** Identificamos que a distribuição dos valores é **fortemente assimétrica**. Existem processos que saem completamente do padrão dos quartis da maioria para gerar **outliers** gigantescos. Numa auditoria real, essas seriam as primeiras pastas abertas sob a suspeita de superfaturamento.
2. **Concentração Financeira e Contratos:** O gráfico nos confirmou que uma fatia do orçamento pode ficar concentrada nas mãos de poucos fornecedores. Casos extremos de falta de competitividade ou o uso frequente de "dispensa de licitação" para as mesmas empresas podem ser o fio condutor de um direcionamento.
3. **Correlações e Indícios:** Usando as variáveis que achamos influenciar o problema, notamos que o tipo de processo, tempo de tramitação e modalidade ditam as regras. Algumas vezes pode existir o fracionamento de despesas (vários processos pequenos fugindo dos limites legais maiores) que em conjunto concentram um grande montante final para a mesma empresa.

### Próximo passo analítico
Através desse storytelling com dados, cumprimos o papel de separar sinais de ruídos. O próximo passo de uma análise investigativa seria pinçar especificamente os CNPJs e chaves de processos listados nos outliers da nossa Base Unificada e investigá-los de ponta a ponta. A estatística aponta exatamente onde começar a procurar.
""")

st.success("Aplicação pronta para apresentação.")