"""
API do Case NPS Preditivo (Tech Challenge - Fase 1).

O que ela faz
-------------
1. Exibe os dados: base tratada, resumo do NPS e os insights da EDA.
2. Serve o modelo: recebe os indicadores operacionais de um pedido e devolve a
   probabilidade de o cliente ser DETRATOR (nota de NPS de 0 a 6) antes de ele
   responder a pesquisa.

Como rodar (na raiz do projeto)
-------------------------------
    pip install fastapi uvicorn
    uvicorn api:app --reload

Documentação interativa (Swagger): http://127.0.0.1:8000/docs

Configuração opcional por variável de ambiente
----------------------------------------------
    NPS_MODEL_PATH  caminho de um .joblib específico (padrão: o .joblib mais
                    recente dentro de models/)
    NPS_CORTE       ponto de corte padrão da probabilidade (padrão: 0.5)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# Caminhos e constantes
# --------------------------------------------------------------------------
PROJECT_DIR = Path(os.environ.get("NPS_PROJECT_DIR", Path(__file__).resolve().parent))
DATA_PATH = PROJECT_DIR / "data" / "processed" / "desafio_nps_fase_1_clean.csv"
MODELS_DIR = PROJECT_DIR / "models"

CORTE_PADRAO = float(os.environ.get("NPS_CORTE", 0.5))

# Variáveis de entrada do modelo (as mesmas dos notebooks 03-model).
# Só dependem de informações disponíveis ANTES da pesquisa de NPS.
FEATURES = [
    "delivery_delay_days",
    "complaints_count",
    "customer_service_contacts",
    "resolution_time_days",
]

# Variáveis que podem ser usadas em /insights/nps-por/{variavel}
VARIAVEIS_INSIGHT = FEATURES + ["customer_region"]

# Categorias de NPS do projeto. A nota da base é contínua (ex.: 6,9), então as
# faixas oficiais (0-6, 7-8, 9-10) viram intervalos: <= 6,5 | <= 8,5 | acima.
CORTE_DETRATOR = 6.5
CORTE_NEUTRO = 8.5


def categoriza_nps(nota: pd.Series) -> pd.Series:
    """Converte a nota contínua de NPS em detrator / neutro / promotor."""
    return pd.cut(
        nota,
        bins=[-np.inf, CORTE_DETRATOR, CORTE_NEUTRO, np.inf],
        labels=["detrator", "neutro", "promotor"],
    ).astype(str)


# --------------------------------------------------------------------------
# Carga da base e do modelo (uma vez, na subida do servidor)
# --------------------------------------------------------------------------
def encontra_modelo() -> Path | None:
    """Usa NPS_MODEL_PATH, se existir; senão pega o .joblib mais recente de models/."""
    caminho_env = os.environ.get("NPS_MODEL_PATH")
    if caminho_env:
        return Path(caminho_env)
    candidatos = sorted(MODELS_DIR.glob("*.joblib"), key=lambda p: p.stat().st_mtime)
    return candidatos[-1] if candidatos else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Base tratada (saída do notebook 01-cleansing)
    app.state.df = None
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        df["nps_categoria"] = categoriza_nps(df["nps_score"])
        app.state.df = df

    # Modelo treinado (.joblib). Se ainda não existir, só a predição fica indisponível.
    app.state.modelo = None
    app.state.modelo_path = None
    app.state.metadados = {}
    app.state.erro_modelo = None
    caminho = encontra_modelo()
    if caminho is None:
        app.state.erro_modelo = f"Nenhum arquivo .joblib encontrado em {MODELS_DIR}."
    else:
        try:
            objeto = joblib.load(caminho)
            # O arquivo do Theo guarda um dicionário: {"modelo": Pipeline, "metadados": {...}}.
            # Também aceita um modelo "puro" (Pipeline/estimador salvo direto).
            if isinstance(objeto, dict) and "modelo" in objeto:
                app.state.modelo = objeto["modelo"]
                app.state.metadados = objeto.get("metadados") or {}
            else:
                app.state.modelo = objeto
                app.state.metadados = {}
            app.state.modelo_path = caminho
        except Exception as erro:  # arquivo corrompido, versão do sklearn diferente etc.
            app.state.erro_modelo = f"Falha ao carregar {caminho.name}: {erro}"
    yield


app = FastAPI(
    title="API - Case NPS Preditivo",
    description=(
        "Exibe os dados tratados e os insights da análise exploratória e serve o "
        "modelo que estima a probabilidade de um cliente ser **detrator** "
        "(nota de NPS de 0 a 6) antes da pesquisa de satisfação."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def pega_base(request: Request) -> pd.DataFrame:
    df = request.app.state.df
    if df is None:
        raise HTTPException(
            status_code=503,
            detail=f"Base tratada não encontrada em {DATA_PATH}. Rode o notebook 01-cleansing.",
        )
    return df


def pega_modelo(request: Request):
    modelo = request.app.state.modelo
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail=request.app.state.erro_modelo or "Modelo não carregado.",
        )
    return modelo


def colunas_do_modelo(modelo) -> list[str]:
    """Ordem das colunas que o modelo espera (usa feature_names_in_ quando existir)."""
    nomes = getattr(modelo, "feature_names_in_", None)
    return list(nomes) if nomes is not None else FEATURES


def limpa_json(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> lista de dicts compatível com JSON (NaN vira null)."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


# --------------------------------------------------------------------------
# Esquemas de entrada e saída da predição
# --------------------------------------------------------------------------
class PedidoEntrada(BaseModel):
    """Indicadores operacionais de um pedido, disponíveis antes da pesquisa de NPS."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "delivery_delay_days": 3,
                "complaints_count": 2,
                "customer_service_contacts": 1,
                "resolution_time_days": 4,
            }
        }
    )

    delivery_delay_days: float = Field(ge=0, le=60, description="Dias de atraso na entrega.")
    complaints_count: int = Field(ge=0, le=100, description="Reclamações registradas.")
    customer_service_contacts: int = Field(ge=0, le=100, description="Contatos com o atendimento.")
    resolution_time_days: float = Field(ge=0, le=365, description="Dias para resolver o problema.")


class PredicaoSaida(BaseModel):
    probabilidade_detrator: float = Field(description="Probabilidade (0 a 1) de o cliente ser detrator.")
    classe: Literal["detrator", "nao_detrator"] = Field(description="Classe pelo ponto de corte.")
    corte_usado: float
    nivel_risco: Literal["baixo", "medio", "alto"]


def nivel_de_risco(probabilidade: float) -> str:
    if probabilidade >= 0.7:
        return "alto"
    if probabilidade >= 0.4:
        return "medio"
    return "baixo"


def corte_efetivo(request: Request, corte: float | None) -> float:
    """Ordem: valor passado na chamada (?corte=) > variável NPS_CORTE > limiar salvo no modelo > 0,5."""
    if corte is not None:
        return corte
    if "NPS_CORTE" in os.environ:
        return CORTE_PADRAO
    return float(request.app.state.metadados.get("limiar_decisao", CORTE_PADRAO))


def calcula_predicoes(request: Request, pedidos: list[PedidoEntrada], corte: float | None) -> list[PredicaoSaida]:
    modelo = pega_modelo(request)
    colunas = colunas_do_modelo(modelo)

    X = pd.DataFrame([p.model_dump() for p in pedidos])
    faltando = [c for c in colunas if c not in X.columns]
    if faltando:
        raise HTTPException(
            status_code=500,
            detail=(
                f"O modelo carregado espera as colunas {colunas}, mas a API só recebe {FEATURES}. "
                f"Faltam: {faltando}. Ajuste FEATURES e PedidoEntrada em api.py."
            ),
        )
    X = X[colunas]
    corte = corte_efetivo(request, corte)

    # Coluna da classe positiva (1 = detrator)
    classes = list(getattr(modelo, "classes_", [0, 1]))
    if 1 not in classes:
        raise HTTPException(status_code=500, detail=f"O modelo não tem a classe 1 (detrator): classes={classes}.")
    probas = modelo.predict_proba(X)[:, classes.index(1)]

    return [
        PredicaoSaida(
            probabilidade_detrator=round(float(p), 4),
            classe="detrator" if p >= corte else "nao_detrator",
            corte_usado=corte,
            nivel_risco=nivel_de_risco(float(p)),
        )
        for p in probas
    ]


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/", tags=["Geral"])
def raiz():
    """Mapa da API."""
    return {
        "projeto": "Case NPS Preditivo - Tech Challenge Fase 1",
        "documentacao": "/docs",
        "endpoints": {
            "saude": "GET /health",
            "dados": "GET /dados",
            "pedido": "GET /dados/{order_id}",
            "resumo": "GET /resumo",
            "insight_por_variavel": "GET /insights/nps-por/{variavel}",
            "correlacoes": "GET /insights/correlacoes",
            "modelo": "GET /modelo",
            "predicao": "POST /predict",
            "predicao_em_lote": "POST /predict/lote",
        },
    }


@app.get("/health", tags=["Geral"])
def saude(request: Request):
    """Mostra se a base e o modelo foram carregados."""
    df = request.app.state.df
    return {
        "status": "ok",
        "base_carregada": df is not None,
        "linhas_na_base": None if df is None else len(df),
        "modelo_carregado": request.app.state.modelo is not None,
        "arquivo_do_modelo": None if request.app.state.modelo_path is None else request.app.state.modelo_path.name,
        "aviso_modelo": request.app.state.erro_modelo,
    }


# ---- Dados ----------------------------------------------------------------
@app.get("/dados", tags=["Dados"])
def lista_dados(
    request: Request,
    limite: int = Query(20, ge=1, le=500, description="Registros por página."),
    deslocamento: int = Query(0, ge=0, description="Quantos registros pular (paginação)."),
    regiao: str | None = Query(None, description="Filtra por customer_region (ex.: Sul)."),
    categoria: Literal["detrator", "neutro", "promotor"] | None = Query(None, description="Filtra pela categoria de NPS."),
):
    """Lista os pedidos da base tratada, com paginação e filtros simples."""
    df = pega_base(request)
    if regiao:
        df = df[df["customer_region"].str.lower() == regiao.lower()]
    if categoria:
        df = df[df["nps_categoria"] == categoria]
    pagina = df.iloc[deslocamento : deslocamento + limite]
    return {
        "total_filtrado": len(df),
        "limite": limite,
        "deslocamento": deslocamento,
        "itens": limpa_json(pagina),
    }


@app.get("/dados/{order_id}", tags=["Dados"])
def busca_pedido(order_id: int, request: Request):
    """Devolve um pedido pelo order_id."""
    df = pega_base(request)
    linha = df[df["order_id"] == order_id]
    if linha.empty:
        raise HTTPException(status_code=404, detail=f"Pedido {order_id} não encontrado.")
    return limpa_json(linha)[0]


@app.get("/resumo", tags=["Insights"])
def resumo(request: Request):
    """Visão geral: distribuição das categorias de NPS, NPS geral e recompra por categoria."""
    df = pega_base(request)
    contagem = df["nps_categoria"].value_counts()
    prop = df["nps_categoria"].value_counts(normalize=True)
    categorias = {
        c: {"clientes": int(contagem.get(c, 0)), "percentual": round(float(prop.get(c, 0)) * 100, 1)}
        for c in ["detrator", "neutro", "promotor"]
    }
    return {
        "clientes": len(df),
        "nps_medio": round(float(df["nps_score"].mean()), 2),
        "nps_geral": round((categorias["promotor"]["percentual"] - categorias["detrator"]["percentual"]), 1),
        "categorias": categorias,
        "recompra_30d_por_categoria_percentual": {
            c: round(float(v) * 100, 1)
            for c, v in df.groupby("nps_categoria")["repeat_purchase_30d"].mean().items()
        },
        "observacao": "Faixas do NPS: detrator 0 a 6 | neutro 7 e 8 | promotor 9 e 10.",
    }


@app.get("/insights/nps-por/{variavel}", tags=["Insights"])
def nps_por_variavel(
    variavel: Literal[
        "delivery_delay_days", "complaints_count", "customer_service_contacts",
        "resolution_time_days", "customer_region",
    ],
    request: Request,
    agrupar_a_partir_de: int = Query(5, ge=1, le=20, description="Valores >= a este número viram a faixa 'N+' (variáveis numéricas)."),
):
    """NPS médio e % de detratores por valor da variável (base dos gráficos da EDA)."""
    df = pega_base(request)
    if variavel == "customer_region":
        grupo = df["customer_region"]
    else:
        limitado = df[variavel].round().astype(int).clip(upper=agrupar_a_partir_de)
        grupo = limitado.map(lambda v: f"{v}+" if v >= agrupar_a_partir_de else str(v))
    tabela = (
        df.assign(faixa=grupo, detrator=(df["nps_categoria"] == "detrator"))
        .groupby("faixa")
        .agg(clientes=("nps_score", "size"), nps_medio=("nps_score", "mean"), pct_detratores=("detrator", "mean"))
        .reset_index()
    )
    if variavel != "customer_region":
        tabela["ordem"] = tabela["faixa"].str.replace("+", "", regex=False).astype(int)
        tabela = tabela.sort_values("ordem").drop(columns="ordem")
    else:
        tabela = tabela.sort_values("nps_medio")
    tabela["nps_medio"] = tabela["nps_medio"].round(2)
    tabela["pct_detratores"] = (tabela["pct_detratores"] * 100).round(1)
    return {"variavel": variavel, "faixas": limpa_json(tabela)}


@app.get("/insights/correlacoes", tags=["Insights"])
def correlacoes(request: Request):
    """Correlação de cada variável numérica com a nota de NPS, da mais negativa para a mais positiva."""
    df = pega_base(request)
    numericas = df.select_dtypes("number").drop(columns=["customer_id", "order_id"], errors="ignore")
    corr = numericas.corr()["nps_score"].drop("nps_score").sort_values()
    return {
        "correlacao_com_nps_score": {k: round(float(v), 3) for k, v in corr.items()},
        "observacao": "Correlação mede relação linear e não prova causa.",
    }


# ---- Modelo ---------------------------------------------------------------
@app.get("/modelo", tags=["Modelo"])
def info_modelo(request: Request):
    """Descreve o modelo carregado. Se for regressão logística, mostra o peso de cada fator."""
    modelo = pega_modelo(request)
    final = modelo.steps[-1][1] if hasattr(modelo, "steps") else modelo
    colunas = colunas_do_modelo(modelo)
    info = {
        "arquivo": request.app.state.modelo_path.name,
        "tipo": type(final).__name__,
        "pipeline": hasattr(modelo, "steps"),
        "variaveis_de_entrada": colunas,
        "corte_padrao": corte_efetivo(request, None),
        "metadados": request.app.state.metadados or None,
    }
    if hasattr(final, "coef_") and len(colunas) == final.coef_.shape[-1]:
        coef = final.coef_[0]
        info["fatores"] = sorted(
            [
                {"variavel": c, "coeficiente": round(float(b), 3), "razao_de_chances": round(float(np.exp(b)), 2)}
                for c, b in zip(colunas, coef)
            ],
            key=lambda f: f["coeficiente"],
            reverse=True,
        )
        info["leitura"] = (
            "Variáveis padronizadas: a razão de chances mostra quantas vezes a chance de o cliente "
            "ser detrator se multiplica a cada desvio padrão a mais na variável."
        )
    elif hasattr(final, "feature_importances_") and len(colunas) == len(final.feature_importances_):
        info["importancias"] = {c: round(float(i), 3) for c, i in zip(colunas, final.feature_importances_)}
    return info


@app.post("/predict", response_model=PredicaoSaida, tags=["Modelo"])
def prediz(
    pedido: PedidoEntrada,
    request: Request,
    corte: float | None = Query(None, ge=0, le=1, description="Probabilidade a partir da qual o cliente é classificado como detrator."),
):
    """Probabilidade de um pedido gerar um cliente detrator."""
    return calcula_predicoes(request, [pedido], corte)[0]


@app.post("/predict/lote", response_model=list[PredicaoSaida], tags=["Modelo"])
def prediz_em_lote(
    pedidos: list[PedidoEntrada],
    request: Request,
    corte: float | None = Query(None, ge=0, le=1),
):
    """Mesma predição para vários pedidos de uma vez (até 1.000)."""
    if not pedidos:
        raise HTTPException(status_code=422, detail="Envie ao menos um pedido.")
    if len(pedidos) > 1000:
        raise HTTPException(status_code=422, detail="Máximo de 1.000 pedidos por chamada.")
    return calcula_predicoes(request, pedidos, corte)