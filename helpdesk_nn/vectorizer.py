"""Vetorização TF-IDF implementada em NumPy puro.

Reproduz o comportamento do ``TfidfVectorizer`` do scikit-learn com as opções
usadas no projeto (n-gramas, ``min_df``, ``sublinear_tf``, ``smooth_idf`` e
normalização L2), mas sem depender de bibliotecas que podem ainda não ter
pacotes prontos para versões novíssimas do Python.
"""

import math
import re
import unicodedata
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Lista curta de propósito: palavras como "nao" e "sem" carregam sentido.
STOPWORDS = frozenset(
    "a o as os de do da dos das em no na nos nas um uma uns umas e ou que "
    "para por com se meu minha meus minhas ao aos".split()
)


def normalizar(texto: str) -> str:
    """Minúsculas e remoção de acentos (robusto a digitação sem acento)."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def tokenizar(texto: str) -> list[str]:
    return [
        t
        for t in _TOKEN_RE.findall(normalizar(texto))
        if len(t) > 1 and t not in STOPWORDS
    ]


class TfidfVectorizer:
    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_features: int = 3000,
        sublinear_tf: bool = True,
    ):
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.min_df = int(min_df)
        self.max_features = int(max_features)
        self.sublinear_tf = bool(sublinear_tf)
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray = np.zeros(0)

    # ------------------------------------------------------------------ análise
    def _analisar(self, texto: str) -> list[str]:
        tokens = tokenizar(texto)
        lo, hi = self.ngram_range
        saida: list[str] = []
        for n in range(lo, hi + 1):
            saida.extend(
                " ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
            )
        return saida

    # ---------------------------------------------------------------------- fit
    def fit(self, textos: list[str]) -> "TfidfVectorizer":
        if not textos:
            raise ValueError("Não é possível ajustar o vetorizador sem textos.")
        df: Counter[str] = Counter()
        for texto in textos:
            df.update(set(self._analisar(texto)))
        termos = [t for t, c in df.items() if c >= self.min_df]
        if not termos:
            raise ValueError("Nenhum termo atingiu min_df; reduza min_df.")
        termos.sort(key=lambda t: (-df[t], t))  # determinístico
        termos = sorted(termos[: self.max_features])
        self.vocabulary_ = {t: i for i, t in enumerate(termos)}
        n = len(textos)
        self.idf_ = np.array(
            [math.log((1 + n) / (1 + df[t])) + 1.0 for t in termos], dtype=np.float64
        )
        return self

    # ---------------------------------------------------------------- transform
    def transform(self, textos: list[str]) -> np.ndarray:
        if not self.vocabulary_:
            raise RuntimeError("Vetorizador não ajustado.")
        X = np.zeros((len(textos), len(self.vocabulary_)), dtype=np.float64)
        for i, texto in enumerate(textos):
            for termo, c in Counter(self._analisar(texto)).items():
                j = self.vocabulary_.get(termo)
                if j is None:
                    continue
                tf = 1.0 + math.log(c) if self.sublinear_tf else float(c)
                X[i, j] = tf * self.idf_[j]
        normas = np.linalg.norm(X, axis=1, keepdims=True)
        normas[normas == 0.0] = 1.0
        return X / normas

    def fit_transform(self, textos: list[str]) -> np.ndarray:
        return self.fit(textos).transform(textos)

    # -------------------------------------------------------------- serialização
    def to_dict(self) -> dict:
        termos = sorted(self.vocabulary_, key=self.vocabulary_.get)
        return {
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "termos": termos,
            "idf": self.idf_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TfidfVectorizer":
        v = cls(
            ngram_range=tuple(d["ngram_range"]),
            min_df=d["min_df"],
            max_features=d["max_features"],
            sublinear_tf=d["sublinear_tf"],
        )
        v.vocabulary_ = {t: i for i, t in enumerate(d["termos"])}
        v.idf_ = np.array(d["idf"], dtype=np.float64)
        return v
