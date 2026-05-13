# OLS (análise de amostra + regressão)

Script para:
1. Calcular tamanho de amostra recomendado para a série (Cochran com correção finita).
2. Selecionar variável alvo e variáveis numéricas correlacionadas.
3. Ajustar regressão linear por OLS e salvar resultados em tabelas/texto.

## Como rodar

```bash
python OLS/run_ols_analysis.py
```

## Saídas

Em `OLS/outputs/`:
- `resumo_ols.txt`
- `features_correlacionadas.csv`
- `coeficientes_ols.csv`
- `amostra_real_vs_previsto.csv`
