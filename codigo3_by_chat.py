import pandas as pd
import numpy as np

# lê a base
df = pd.read_csv("base_final_analise.csv")

TARGET = "valorUnitarioVencedor"

# remove linhas sem target
df = df.dropna(subset=[TARGET])

# separa categóricas
categoricas = df.select_dtypes(include=["object"]).columns.tolist()

# remove target se estiver ali
if TARGET in categoricas:
    categoricas.remove(TARGET)

# transforma texto em números (One Hot Encoding)
df = pd.get_dummies(
    df,
    columns=categoricas,
    drop_first=True,
    dtype=int
)

# pega só numéricas
df_num = df.select_dtypes(include=[np.number])

# calcula correlação de Pearson
corr = (
    df_num
    .corr(numeric_only=True)[TARGET]
    .drop(TARGET)
    .dropna()
)

# filtra correlações fortes
fortes = corr[
    abs(corr) >= 0.3
].sort_values(key=abs, ascending=False)

# output
print("\nVARIÁVEIS COM |CORR| >= 0.3:\n")
print(fortes)

print(f"\nTotal: {len(fortes)}")