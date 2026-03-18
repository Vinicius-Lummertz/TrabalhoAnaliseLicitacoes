import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar arquivos
df1 = pd.read_csv('27_processoslicitatorios-34862.csv')
df2 = pd.read_csv('27_processoslicitatorios-34864.csv')
df3 = pd.read_csv('27_contratos-87437.csv')

# =========================
# PADRONIZAR VALORES
# =========================

# Processos (34862)
df1['valor_licitacao'] = df1['valorHomologado']

# Processos (34864)
df2['valor_licitacao'] = df2['valorHomologado']

# Contratos (87437)
df3['valor_licitacao'] = df3['valorFinal']

# =========================
# UNIR BASES DE PROCESSOS
# =========================

df_processos = pd.concat([df1, df2], ignore_index=True)

# =========================
# LIMPEZA
# =========================

# Remover nulos
df_processos = df_processos[df_processos['valor_licitacao'].notnull()]
df3 = df3[df3['valor_licitacao'].notnull()]

# Converter para número (caso venha como string)
df_processos['valor_licitacao'] = pd.to_numeric(df_processos['valor_licitacao'], errors='coerce')
df3['valor_licitacao'] = pd.to_numeric(df3['valor_licitacao'], errors='coerce')

# =========================
# BOXPLOT
# =========================

plt.figure(figsize=(10,5))
sns.boxplot(x=df_processos['valor_licitacao'])
plt.title('Distribuição dos Valores das Licitações')
plt.xlabel('Valor')
plt.show()

top_empresas = df3.groupby('nomeContratado')['valor_licitacao'].sum().sort_values(ascending=False)
print(top_empresas.head(10))