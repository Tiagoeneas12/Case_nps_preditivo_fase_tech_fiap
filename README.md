## Tratamento de dados

O tratamento está no notebook `notebooks/01-cleansing.ipynb`. Ele lê a base bruta (`data/raw/desafio_nps_fase_1.csv`) e salva a base tratada em `data/processed/desafio_nps_fase_1_clean.csv`, que é a usada na análise exploratória, no modelo e na API.

### O que foi verificado

| Verificação | Resultado |
|---|---|
| Base original | 2.500 linhas e 19 colunas |
| Valores nulos | Nenhum |
| Linhas duplicadas (e `customer_id` / `order_id` repetidos) | Nenhuma |
| Estatísticas descritivas | Todas as colunas em faixas plausíveis (idades de 18 a 69 anos, 1 a 6 itens por pedido, sem atrasos negativos) |
| Padronização de `customer_region` | Sem variações de escrita |
| Regras de negócio (nota de 0 a 10, valores positivos, `repeat_purchase_30d` só 0 ou 1 etc.) | Apenas uma inconsistência encontrada (abaixo) |

### Inconsistência encontrada e decisão tomada

- **Problema:** 35 pedidos (1,4% da base) tinham `discount_value` maior que `order_value`. A diferença ia de cerca de 1,1 a 3,9 vezes o valor do pedido, o que não é arredondamento.
- **Decisão:** as 35 linhas foram removidas. Elas são uma fração pequena da base e não há como inferir com segurança o desconto correto.
- **Resultado:** a base final tem **2.465 linhas**. Depois da remoção, a checagem foi refeita e não restou nenhuma inconsistência.

### Pontos de atenção

- **Valor do pedido:** a documentação não informa se `order_value` é bruto (antes do desconto) ou líquido. Tratamos o desconto maior que o pedido como erro de dado. Se o valor for líquido, o desconto implícito dessas linhas seria de 51,6% a 79%.
- **Arredondamento da nota (`nps_score`):** a nota não foi arredondada no tratamento. Isso é tratado na análise exploratória.

## API

A API (`api.py`) foi feita com FastAPI. Ela mostra os dados tratados, traz os principais resultados da análise exploratória e entrega a probabilidade de um cliente ser **detrator** usando o modelo treinado.

> **Observação:** o desafio não pede uma API. Ela é um complemento para mostrar como o modelo poderia ser usado na prática.

### Como rodar

1. Instale as bibliotecas (se ainda não tiver):
   ```
   pip install fastapi uvicorn pandas scikit-learn joblib
   ```
2. Na pasta raiz do projeto, ligue a API:
   ```
   uvicorn api:app --reload
   ```
3. Abra a documentação interativa em http://127.0.0.1:8000/docs. Nela dá para testar todos os endpoints clicando em "Try it out" e depois em "Execute".

A API lê dois arquivos do repositório:
- `data/processed/desafio_nps_fase_1_clean.csv`: base tratada.
- Em `models/`, o arquivo `.joblib` mais recente (hoje `nps_detrator_logreg.joblib`). Sem ele, só os endpoints de previsão ficam indisponíveis.

### Endpoints

| Método e rota | Para que serve |
|---|---|
| `GET /health` | Mostra se a API está no ar e se a base e o modelo foram carregados |
| `GET /dados` | Lista os pedidos, com paginação (`limite`, `deslocamento`) e filtros por `regiao` e `categoria` (detrator, neutro ou promotor) |
| `GET /dados/{order_id}` | Mostra um pedido específico |
| `GET /resumo` | Total de clientes, NPS geral, divisão por categoria e recompra em 30 dias por categoria |
| `GET /insights/nps-por/{variavel}` | NPS médio por `delivery_delay_days`, `complaints_count`, `customer_service_contacts`, `resolution_time_days` ou `customer_region` |
| `GET /insights/correlacoes` | Correlação de cada variável numérica com o `nps_score` |
| `GET /modelo` | Descreve o modelo e mostra o peso de cada variável |
| `POST /predict` | Probabilidade de um cliente ser detrator, com a classe e o nível de risco |
| `POST /predict/lote` | Mesma previsão para vários pedidos de uma vez (até 1.000) |

### Exemplo de previsão

Enviar para `POST /predict`:

```json
{
  "delivery_delay_days": 4,
  "complaints_count": 5,
  "customer_service_contacts": 3,
  "resolution_time_days": 8
}
```

Resposta:

```json
{
  "probabilidade_detrator": 0.9839,
  "classe": "detrator",
  "corte_usado": 0.5,
  "nivel_risco": "alto"
}
```

### Como ler o resultado

- **Classe:** "detrator" quando a probabilidade chega ao corte. O corte padrão é 0,5, o mesmo que foi salvo junto do modelo. Para mudar em uma chamada, use `?corte=0.8`.
- **Nível de risco:** baixo (abaixo de 40%), médio (de 40% a 70%) e alto (acima de 70%).
- **Faixas do NPS:** detrator de 0 a 6, neutro 7 e 8, promotor 9 e 10.
- **Modelo:** regressão logística com `class_weight="balanced"`. As variáveis de entrada são `delivery_delay_days`, `complaints_count`, `customer_service_contacts` e `resolution_time_days`. Ficaram de fora `repeat_purchase_30d` e `csat_internal_score`, para evitar vazamento de informação.

### Limitações

- O modelo mostra associação, não causa.
- A previsão só vale para clientes que respondem à pesquisa de NPS, então pode haver viés de seleção.
- Os números da API usam a base tratada (2.465 linhas) e as faixas do NPS acima. Por isso diferem dos de materiais que usaram a base bruta com a regra abaixo de 7.