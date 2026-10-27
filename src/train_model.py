"""Treina o modelo de risco de detrator e salva o artefato em models/ para uso na API.

Reproduz o modelo final do notebook 03-model- (regressão logística sem pesos
por classe, com o corte escolhido na validação cruzada). Uso, a partir da raiz do projeto:

    python src/train_model.py
"""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_DADOS = RAIZ / "data" / "processed" / "desafio_nps_fase_1_clean.csv"
PASTA_MODELOS = RAIZ / "models"
NOME_MODELO = "baseline_regressao_logistica"
VERSAO_MODELO = "2.0.0"

RANDOM_STATE = 42
LIMIAR_NOTA_DETRATOR = 6.5
# Corte que equilibra o recall das duas classes na validação cruzada do treino (notebook 03-model-).
LIMIAR_DECISAO = 0.79
FEATURES = [
    "delivery_delay_days",
    "complaints_count",
    "customer_service_contacts",
    "resolution_time_days",
]


def carregar_dados() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(CAMINHO_DADOS)
    categoria = pd.cut(
        df["nps_score"],
        bins=[-np.inf, LIMIAR_NOTA_DETRATOR, 8.5, np.inf],
        labels=["detrator", "neutro", "promotor"],
    )
    y = (categoria == "detrator").astype(int)
    return df[FEATURES], y, categoria


def criar_modelo():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    )


def avaliar(modelo, X, y) -> dict:
    proba = modelo.predict_proba(X)[:, 1]
    pred = (proba >= LIMIAR_DECISAO).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "recall_detrator": round(recall_score(y, pred, pos_label=1), 4),
        "precisao_detrator": round(precision_score(y, pred, pos_label=1), 4),
        "recall_nao_detrator": round(recall_score(y, pred, pos_label=0), 4),
    }


def main() -> None:
    X, y, categoria = carregar_dados()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=categoria, random_state=RANDOM_STATE,
    )

    # Métricas no teste guardado; o modelo final é então treinado com a base inteira.
    metricas_teste = avaliar(criar_modelo().fit(X_train, y_train), X_test, y_test)
    modelo_final = criar_modelo().fit(X, y)

    metadados = {
        "nome": NOME_MODELO,
        "versao": VERSAO_MODELO,
        "descricao": "Regressão logística (sem pesos por classe) que estima a "
                     "probabilidade de o cliente ser detrator (nota_nps <= 6,5).",
        "features": FEATURES,
        "alvo": f"detrator (nps_score <= {LIMIAR_NOTA_DETRATOR})",
        "limiar_decisao": LIMIAR_DECISAO,
        "metricas_teste": metricas_teste,
        "linhas_treino": len(X),
        "treinado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }

    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    caminho_modelo = PASTA_MODELOS / f"{NOME_MODELO}.joblib"
    joblib.dump({"modelo": modelo_final, "metadados": metadados}, caminho_modelo)
    (PASTA_MODELOS / f"{NOME_MODELO}.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"Modelo salvo em: {caminho_modelo}")
    print(json.dumps(metricas_teste, indent=2))


if __name__ == "__main__":
    main()
