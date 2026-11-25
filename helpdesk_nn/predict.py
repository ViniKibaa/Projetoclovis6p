"""Inferência: carrega o modelo treinado e classifica textos de chamados."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import MultiTaskMLP, carregar_modelo
from .vectorizer import TfidfVectorizer

LIMIAR_CONFIANCA = 0.50  # abaixo disso, recomenda triagem humana


@dataclass
class Predicao:
    texto: str
    categoria: str
    urgencia: str
    confianca_categoria: float
    confianca_urgencia: float
    prob_categoria: dict[str, float] = field(default_factory=dict)
    prob_urgencia: dict[str, float] = field(default_factory=dict)

    @property
    def revisao_humana(self) -> bool:
        return min(self.confianca_categoria, self.confianca_urgencia) < LIMIAR_CONFIANCA

    def to_dict(self) -> dict:
        return {
            "texto": self.texto,
            "categoria": self.categoria,
            "urgencia": self.urgencia,
            "confianca_categoria": round(self.confianca_categoria, 4),
            "confianca_urgencia": round(self.confianca_urgencia, 4),
            "prob_categoria": {k: round(v, 4) for k, v in self.prob_categoria.items()},
            "prob_urgencia": {k: round(v, 4) for k, v in self.prob_urgencia.items()},
            "revisao_humana": self.revisao_humana,
        }


class Classificador:
    def __init__(self, modelo: MultiTaskMLP, vetorizador: TfidfVectorizer, meta: dict):
        self.modelo = modelo
        self.vetorizador = vetorizador
        self.meta = meta
        self.categorias: list[str] = meta["categorias"]
        self.urgencias: list[str] = meta["urgencias"]

    @classmethod
    def carregar(cls, caminho) -> "Classificador":
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em '{caminho}'. Rode primeiro: python -m helpdesk_nn treinar"
            )
        modelo, meta = carregar_modelo(caminho)
        return cls(modelo, TfidfVectorizer.from_dict(meta["vetorizador"]), meta)

    def prever(self, textos: list[str]) -> list[Predicao]:
        if not textos:
            return []
        X = self.vetorizador.transform(textos)
        pc, pu = self.modelo.predict_proba(X)
        saida = []
        for t, c, u in zip(textos, pc, pu):
            ic, iu = int(np.argmax(c)), int(np.argmax(u))
            saida.append(
                Predicao(
                    texto=t,
                    categoria=self.categorias[ic],
                    urgencia=self.urgencias[iu],
                    confianca_categoria=float(c[ic]),
                    confianca_urgencia=float(u[iu]),
                    prob_categoria=dict(zip(self.categorias, map(float, c))),
                    prob_urgencia=dict(zip(self.urgencias, map(float, u))),
                )
            )
        return saida

    def prever_um(self, texto: str) -> Predicao:
        return self.prever([texto])[0]
