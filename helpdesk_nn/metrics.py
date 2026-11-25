"""Métricas de classificação sem dependências externas."""

import numpy as np


def matriz_confusao(y_true, y_pred, n_classes: int) -> np.ndarray:
    m = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def relatorio(y_true, y_pred, nomes: list[str]) -> dict:
    """Acurácia, precisão/recall/F1 por classe, F1 macro e matriz de confusão."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = matriz_confusao(y_true, y_pred, len(nomes))
    por_classe = {}
    f1s = []
    for i, nome in enumerate(nomes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        por_classe[nome] = {
            "precisao": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "suporte": int(cm[i, :].sum()),
        }
        f1s.append(f1)
    return {
        "acuracia": round(float(np.mean(y_true == y_pred)), 4),
        "f1_macro": round(float(np.mean(f1s)), 4),
        "por_classe": por_classe,
        "matriz_confusao": cm.tolist(),
        "classes": list(nomes),
    }


def tabela_confusao_md(rel: dict) -> str:
    nomes = rel["classes"]
    linhas = ["| real \\ previsto | " + " | ".join(nomes) + " |", "|---|" + "---|" * len(nomes)]
    for nome, linha in zip(nomes, rel["matriz_confusao"]):
        linhas.append(f"| **{nome}** | " + " | ".join(str(v) for v in linha) + " |")
    return "\n".join(linhas)
