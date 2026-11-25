"""Rede neural multitarefa (MLP) em NumPy puro.

Arquitetura::

    TF-IDF -> [Dense+ReLU+Dropout] x N -> cabeça "categoria"  (softmax)
                                       -> cabeça "urgência"   (softmax)

As duas cabeças compartilham as camadas ocultas (aprendizado multitarefa) e a
perda total é ``CE_categoria + w * CE_urgencia + (l2/2) * ||W||^2``.
O treino usa Adam, mini-batches, dropout invertido e *early stopping*.
"""

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

_EPS = 1e-12


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class TrainConfig:
    hidden: tuple[int, ...] = (64,)
    lr: float = 2e-3
    epochs: int = 300
    batch_size: int = 32
    l2: float = 1e-3
    dropout: float = 0.5
    patience: int = 25
    urgency_weight: float = 1.0
    seed: int = 42


class _Adam:
    def __init__(self, params: dict[str, np.ndarray], b1=0.9, b2=0.999, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
        self.b1, self.b2, self.eps = b1, b2, eps

    def step(self, params: dict, grads: dict, lr: float) -> None:
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1**self.t)
            vh = self.v[k] / (1 - self.b2**self.t)
            params[k] -= lr * mh / (np.sqrt(vh) + self.eps)


class MultiTaskMLP:
    def __init__(
        self,
        n_features: int,
        hidden: tuple[int, ...],
        n_categorias: int,
        n_urgencias: int,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self.hidden = tuple(int(h) for h in hidden)
        self.n_features = int(n_features)
        self.n_categorias = int(n_categorias)
        self.n_urgencias = int(n_urgencias)
        self.params: dict[str, np.ndarray] = {}
        tamanhos = [self.n_features, *self.hidden]
        for i, (a, b) in enumerate(zip(tamanhos[:-1], tamanhos[1:])):
            self.params[f"W{i}"] = rng.standard_normal((a, b)) * np.sqrt(2.0 / a)  # He
            self.params[f"b{i}"] = np.zeros(b)
        ultimo = tamanhos[-1]
        for nome, n in (("c", self.n_categorias), ("u", self.n_urgencias)):
            self.params[f"W{nome}"] = rng.standard_normal((ultimo, n)) * np.sqrt(1.0 / ultimo)
            self.params[f"b{nome}"] = np.zeros(n)

    # ------------------------------------------------------------------ forward
    def _forward(self, X, dropout=0.0, rng=None):
        P = self.params
        acts, zs, masks = [X], [], []
        h = X
        for i in range(len(self.hidden)):
            z = h @ P[f"W{i}"] + P[f"b{i}"]
            h = np.maximum(z, 0.0)
            if dropout > 0.0:
                m = (rng.random(h.shape) >= dropout) / (1.0 - dropout)
                h = h * m
            else:
                m = None
            zs.append(z)
            masks.append(m)
            acts.append(h)
        lc = h @ P["Wc"] + P["bc"]
        lu = h @ P["Wu"] + P["bu"]
        return acts, zs, masks, lc, lu

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        *_, lc, lu = self._forward(np.asarray(X, dtype=np.float64))
        return softmax(lc), softmax(lu)

    # --------------------------------------------------------------- perda/grad
    def loss_and_grads(self, X, yc, yu, l2=0.0, uw=1.0, dropout=0.0, rng=None):
        P = self.params
        n = len(X)
        acts, zs, masks, lc, lu = self._forward(X, dropout, rng)
        pc, pu = softmax(lc), softmax(lu)
        idx = np.arange(n)
        loss = -np.mean(np.log(pc[idx, yc] + _EPS)) - uw * np.mean(np.log(pu[idx, yu] + _EPS))
        loss += 0.5 * l2 * sum(float(np.sum(v * v)) for k, v in P.items() if k.startswith("W"))

        dlc = pc.copy()
        dlc[idx, yc] -= 1.0
        dlc /= n
        dlu = pu.copy()
        dlu[idx, yu] -= 1.0
        dlu *= uw / n

        h = acts[-1]
        grads = {
            "Wc": h.T @ dlc + l2 * P["Wc"],
            "bc": dlc.sum(axis=0),
            "Wu": h.T @ dlu + l2 * P["Wu"],
            "bu": dlu.sum(axis=0),
        }
        dh = dlc @ P["Wc"].T + dlu @ P["Wu"].T
        for i in reversed(range(len(self.hidden))):
            if masks[i] is not None:
                dh = dh * masks[i]
            dz = dh * (zs[i] > 0.0)
            grads[f"W{i}"] = acts[i].T @ dz + l2 * P[f"W{i}"]
            grads[f"b{i}"] = dz.sum(axis=0)
            dh = dz @ P[f"W{i}"].T
        return float(loss), grads

    def _avaliar(self, X, yc, yu, uw):
        pc, pu = self.predict_proba(X)
        idx = np.arange(len(X))
        loss = -np.mean(np.log(pc[idx, yc] + _EPS)) - uw * np.mean(np.log(pu[idx, yu] + _EPS))
        return float(loss), float(np.mean(pc.argmax(1) == yc)), float(np.mean(pu.argmax(1) == yu))

    # ---------------------------------------------------------------------- fit
    def fit(self, Xtr, yc_tr, yu_tr, Xva, yc_va, yu_va, cfg: TrainConfig, verbose=False):
        rng = np.random.default_rng(cfg.seed)
        opt = _Adam(self.params)
        n = len(Xtr)
        melhor, melhor_params, espera = np.inf, None, 0
        historico = []
        for epoca in range(1, cfg.epochs + 1):
            ordem = rng.permutation(n)
            for ini in range(0, n, cfg.batch_size):
                b = ordem[ini : ini + cfg.batch_size]
                _, g = self.loss_and_grads(
                    Xtr[b], yc_tr[b], yu_tr[b], cfg.l2, cfg.urgency_weight, cfg.dropout, rng
                )
                opt.step(self.params, g, cfg.lr)
            tr_loss, tr_ac, tr_au = self._avaliar(Xtr, yc_tr, yu_tr, cfg.urgency_weight)
            va_loss, va_ac, va_au = self._avaliar(Xva, yc_va, yu_va, cfg.urgency_weight)
            historico.append(
                {
                    "epoca": epoca,
                    "loss_treino": tr_loss,
                    "loss_val": va_loss,
                    "acc_cat_val": va_ac,
                    "acc_urg_val": va_au,
                }
            )
            if verbose and (epoca == 1 or epoca % 10 == 0):
                print(
                    f"época {epoca:3d} | loss treino {tr_loss:.4f} | loss val {va_loss:.4f} "
                    f"| acc val cat {va_ac:.3f} urg {va_au:.3f}"
                )
            if va_loss < melhor - 1e-4:
                melhor = va_loss
                melhor_params = {k: v.copy() for k, v in self.params.items()}
                espera = 0
            else:
                espera += 1
                if espera >= cfg.patience:
                    break
        if melhor_params is not None:
            self.params = melhor_params
        return historico


# ------------------------------------------------------------------ persistência
def salvar_modelo(caminho, modelo: MultiTaskMLP, meta: dict) -> None:
    """Salva em JSON puro (sem zip/pickle), direto no arquivo final.

    Sem arquivo temporário + rename: pastas sincronizadas (OneDrive/Dropbox)
    podem remover o temporário antes do rename.
    """
    meta = dict(meta)
    meta["arquitetura"] = {
        "n_features": modelo.n_features,
        "hidden": list(modelo.hidden),
        "n_categorias": modelo.n_categorias,
        "n_urgencias": modelo.n_urgencias,
    }
    conteudo = {"meta": meta, "pesos": {k: v.tolist() for k, v in modelo.params.items()}}
    with open(os.fspath(caminho), "w", encoding="utf-8") as f:
        json.dump(conteudo, f, ensure_ascii=False)


def carregar_modelo(caminho) -> tuple[MultiTaskMLP, dict]:
    with open(caminho, encoding="utf-8") as f:
        conteudo = json.load(f)
    meta = conteudo["meta"]
    arq = meta["arquitetura"]
    modelo = MultiTaskMLP(
        arq["n_features"], tuple(arq["hidden"]), arq["n_categorias"], arq["n_urgencias"]
    )
    for k in modelo.params:
        modelo.params[k] = np.array(conteudo["pesos"][k], dtype=np.float64)
    return modelo, meta


def config_para_dict(cfg: TrainConfig) -> dict:
    d = asdict(cfg)
    d["hidden"] = list(cfg.hidden)
    return d
