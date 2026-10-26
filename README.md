# Case NPS Preditivo — Tech Challenge FIAP (Fase 1)

## 1. Apresentação do projeto

Este projeto acadêmico de Ciência de Dados foi desenvolvido no contexto do **Tech Challenge da FIAP — Fase 1**. O objetivo é investigar fatores associados à experiência e à satisfação dos clientes de um e-commerce e avaliar como dados operacionais podem ajudar a identificar clientes com maior probabilidade de se tornarem detratores de NPS.

A análise considera informações relacionadas a pedidos, logística, atendimento e comportamento do cliente. O trabalho reúne preparação e validação dos dados, análise exploratória (EDA), experimentos de modelagem e uma API REST demonstrativa para consultar dados e disponibilizar previsões.

> **Importante:** os resultados mostram padrões e associações presentes na base analisada; não comprovam relações de causa e efeito. A API é um complemento técnico de demonstração, não o objetivo central do desafio.

## 2. Problema de negócio

O NPS (*Net Promoter Score*) é um indicador utilizado para compreender a percepção dos clientes sobre uma empresa, produto ou serviço. No e-commerce, a experiência pode ser influenciada por fatores como atraso na entrega, problemas no pedido, quantidade de reclamações, contatos com o atendimento e tempo de resolução.

O projeto busca responder a perguntas como:

- Quais características operacionais aparecem associadas a notas de NPS mais baixas?
- Como o NPS se distribui entre diferentes perfis e experiências de compra?
- Quais padrões merecem investigação pelas áreas de logística e atendimento?
- É possível construir um modelo inicial para estimar a probabilidade de um registro pertencer à classe de detratores?

As respostas devem apoiar a investigação e a priorização de oportunidades de melhoria. As conclusões precisam ser interpretadas à luz do tamanho e das limitações da base disponível.

## 3. Objetivos

- Compreender a estrutura, a qualidade e as limitações dos dados fornecidos.
- Tratar inconsistências identificadas e gerar uma base preparada para análise.
- Realizar análise exploratória sem duplicar análises entre os notebooks individuais e a EDA consolidada.
- Investigar relações entre indicadores operacionais e a nota de NPS.
- Comparar abordagens de classificação e avaliar seu desempenho com métricas apropriadas.
- Documentar o processo e disponibilizar uma API demonstrativa para consulta de dados e previsões.

## 4. Etapas do trabalho

1. **Tratamento de dados:** inspeção da base original, avaliação de valores ausentes, duplicidades, faixas e regras de negócio; registro das decisões de limpeza.
2. **Análise exploratória (EDA):** análise da distribuição das variáveis, perfil dos clientes e pedidos, logística, atendimento e relações com o NPS. A versão consolidada está em `notebooks/02-eda-.ipynb`; os notebooks individuais permanecem em `notebooks/edas_auxiliares/` como material de referência.
3. **Modelagem:** experimentos de classificação nos notebooks `03-model-01.ipynb` e `03-model-02.ipynb`, com abordagens distintas para comparar resultados.
4. **Treinamento reproduzível:** o script `src/train_model.py` treina e salva o modelo utilizado pela API.
5. **API demonstrativa:** a aplicação em `api.py`, construída com FastAPI, expõe rotas para consultar os dados, obter resumos e insights e solicitar previsões.

## 5. Estrutura do repositório

```text
Case_nps_preditivo_fase_tech_fiap/
├── api.py
├── enviroment.yml
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── desafio_nps_fase_1.csv
│   └── processed/
│       └── desafio_nps_fase_1_clean.csv
├── models/
│   ├── baseline_regressao_logistica.joblib
│   ├── nps_detrator_logreg.joblib
│   └── nps_detrator_logreg.json
├── notebooks/
│   ├── 01-cleansing.ipynb
│   ├── 02-eda-.ipynb
│   ├── 03-model-01.ipynb
│   ├── 03-model-02.ipynb
│   ├── edas_auxiliares/
│   │   ├── 02-felipe-eda.ipynb
│   │   ├── 02-marcela-eda.ipynb
│   │   ├── 02-thelson-eda.ipynb
│   │   └── 02-tiago-eda.ipynb
│   └── src/
│       ├── __init__.py
│       └── config.py
├── references/
│   └── 01_dicionario_de_dados.md
├── reports/
│   └── images/
└── src/
    └── train_model.py
```

## 6. Dados e preparação

O repositório contém a base original em `data/raw/desafio_nps_fase_1.csv` e a base tratada em `data/processed/desafio_nps_fase_1_clean.csv`. A base bruta contém **2.500 registros e 19 colunas**. O notebook de tratamento documenta a remoção de 35 registros em que `discount_value` era maior que `order_value`, resultando em **2.465 registros** na base tratada incluída.

A decisão de remover esses registros foi tomada porque a inconsistência não permitia inferir com segurança o valor correto. Essa decisão e suas limitações devem ser consideradas ao interpretar as análises.

O arquivo `references/01_dicionario_de_dados.md` documenta as variáveis e oferece contexto para a interpretação dos campos.

### Variáveis usadas pelo classificador da API

O modelo demonstrativo utiliza as seguintes variáveis de entrada:

- `delivery_delay_days`: dias de atraso na entrega;
- `complaints_count`: quantidade de reclamações;
- `customer_service_contacts`: contatos com o atendimento;
- `resolution_time_days`: tempo de resolução em dias.

A classe de detrator é derivada de `nps_score`. A implementação do projeto usa os limites `nps_score <= 6.5` para detrator, `6.5 < nps_score <= 8.5` para neutro e `nps_score > 8.5` para promotor. Preserve essa convenção ao comparar resultados entre componentes.

## 7. Ambiente e instalação

O projeto inclui o arquivo Conda `enviroment.yml` (essa é a grafia atual do nome do arquivo).

### Usando Conda

Na raiz do repositório, execute:

```bash
conda env create -f enviroment.yml
conda activate machine_learning
python -m pip install -r requirements.txt
```

Se o ambiente já existir e precisar ser atualizado:

```bash
conda env update -f enviroment.yml --prune
```

Para disponibilizar o ambiente no Jupyter:

```bash
python -m ipykernel install --user --name machine_learning --display-name "Python (machine_learning)"
```

Se alguma dependência da API não estiver incluída no ambiente, instale-a no ambiente ativado:

```bash
python -m pip install fastapi uvicorn
```

## 8. Como executar os notebooks

### 8.1 Tratamento de dados

1. Ative o ambiente `machine_learning`.
2. Inicie o Jupyter na raiz do projeto ou em uma subpasta dele.
3. Abra `notebooks/01-cleansing.ipynb`.
4. Execute as células em ordem e confirme que o arquivo `data/processed/desafio_nps_fase_1_clean.csv` foi gerado.

O notebook localiza a raiz do projeto a partir do diretório atual, desde que o Jupyter seja iniciado na raiz do repositório ou em uma pasta dentro dele.

### 8.2 Análise exploratória

Abra `notebooks/02-eda-.ipynb` e execute as células em ordem. Esse é o notebook de EDA consolidada. Os arquivos de `notebooks/edas_auxiliares/` preservam análises individuais e podem ser consultados como referência; não é necessário executar todos para acompanhar a EDA consolidada.

### 8.3 Modelagem exploratória

Os notebooks `notebooks/03-model-01.ipynb` e `notebooks/03-model-02.ipynb` apresentam experimentos de modelagem distintos. Execute primeiro o tratamento de dados para garantir que a base processada exista.

## 9. Treinar o modelo da API

A partir da raiz do projeto, execute:

```bash
python src/train_model.py
```

O script utiliza a base tratada e grava o artefato do modelo em `models/`, junto aos metadados de treino. Se a base tratada não existir, execute primeiro o notebook de limpeza.

## 10. Executar a API

A API é implementada com FastAPI. Na raiz do projeto, execute:

```bash
uvicorn api:app --reload
```

Depois de iniciar o servidor, acesse:

- **Documentação interativa (Swagger):** http://127.0.0.1:8000/docs
- **Documentação alternativa (ReDoc):** http://127.0.0.1:8000/redoc
- **Verificação de disponibilidade:** http://127.0.0.1:8000/health

A API é um recurso complementar para demonstrar uma forma de servir os dados e o modelo. Para previsões, é necessário que exista um artefato de modelo compatível em `models/` e que a base tratada esteja disponível para as rotas que dependem dela.

### Principais recursos da API

| Método e rota | Finalidade |
|---|---|
| `GET /` | Apresenta as rotas disponíveis. |
| `GET /health` | Informa o estado de carregamento da base e do modelo. |
| `GET /dados` | Consulta registros com paginação e filtros. |
| `GET /dados/{order_id}` | Consulta um pedido específico. |
| `GET /resumo` | Retorna indicadores resumidos relacionados ao NPS. |
| `GET /insights/nps-por/{variavel}` | Resume o NPS por variável suportada. |
| `GET /insights/correlacoes` | Consulta correlações entre variáveis numéricas e `nps_score`. |
| `GET /modelo` | Apresenta informações do modelo carregado. |
| `POST /predict` | Estima a probabilidade de um registro pertencer à classe detrator. |
| `POST /predict/lote` | Realiza previsões para uma lista de registros. |

Consulte `/docs` para visualizar os parâmetros, os formatos de entrada e saída e testar os endpoints disponíveis na versão atual do código.

## 11. Avaliação e interpretação

As métricas de classificação devem ser analisadas em conjunto, considerando o possível desequilíbrio entre classes e o custo de falsos positivos e falsos negativos. Para um problema de identificação de detratores, *recall* ajuda a avaliar quantos detratores foram encontrados, enquanto precisão indica quantas das previsões de detrator estavam corretas no conjunto avaliado. ROC-AUC pode complementar a avaliação da capacidade de ordenação do classificador.

Os valores registrados em `models/nps_detrator_logreg.json` descrevem uma avaliação realizada durante o treinamento incluído no projeto; não garantem o mesmo desempenho em novos dados. Antes de qualquer uso operacional, seria necessário validar o modelo em dados representativos, revisar o ponto de corte e monitorar mudanças no comportamento dos clientes e nos processos.

## 12. Limitações e cuidados

- **Associação não significa causalidade:** correlações e padrões exploratórios não provam que uma variável causou a nota de NPS.
- **Representatividade:** os resultados dependem da base disponível e podem não se generalizar para outros períodos, regiões ou públicos.
- **Qualidade dos dados:** a remoção de registros inconsistentes altera a população analisada; a decisão deve ser considerada na interpretação dos resultados.
- **Disponibilidade temporal das variáveis:** antes de usar um modelo preventivamente, é necessário confirmar que todos os campos de entrada estão disponíveis no momento em que a previsão seria feita.
- **Compatibilidade do modelo:** arquivos `.joblib` dependem das versões das bibliotecas usadas no treinamento. Treinamento e inferência devem ocorrer em ambiente compatível.
- **Ponto de corte:** o limiar de classificação padrão é uma configuração inicial e pode não ser o mais adequado para uma operação real.

## 13. Tecnologias

- **Linguagem:** Python
- **Análise e preparação:** Pandas, NumPy e bibliotecas de visualização usadas nos notebooks
- **Machine Learning:** scikit-learn e ferramentas presentes no ambiente do projeto
- **Notebooks:** Jupyter
- **API:** FastAPI e Uvicorn
- **Versionamento:** Git

Consulte `enviroment.yml` e `requirements.txt` para as dependências declaradas no repositório.

---

**Projeto acadêmico — Tech Challenge FIAP, Fase 1.**
