"""Pipeline de treino e avaliação (dados -> TF-IDF -> rede neural -> relatório)."""

import json
import time
from pathlib import Path

import numpy as np

from . import CATEGORIAS_PADRAO, URGENCIAS_PADRAO, __version__
from .data import carregar_csv, dividir, gerar_dataset, ordenar_rotulos, salvar_csv
from .metrics import relatorio, tabela_confusao_md
from .model import MultiTaskMLP, TrainConfig, config_para_dict, salvar_modelo
from .vectorizer import TfidfVectorizer


def treinar(
    caminho_csv,
    saida,
    cfg: TrainConfig | None = None,
    max_features: int = 3000,
    verbose: bool = True,
) -> dict:
    cfg = cfg or TrainConfig()
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    dados = carregar_csv(caminho_csv)
    categorias = ordenar_rotulos(dados["categorias"], CATEGORIAS_PADRAO)
    urgencias = ordenar_rotulos(dados["urgencias"], URGENCIAS_PADRAO)
    ic = {c: i for i, c in enumerate(categorias)}
    iu = {u: i for i, u in enumerate(urgencias)}
    y_c = np.array([ic[c] for c in dados["categorias"]])
    y_u = np.array([iu[u] for u in dados["urgencias"]])
    textos = dados["textos"]

    idx = dividir(dados["categorias"], dados["urgencias"], dados["grupos"], seed=cfg.seed)
    tr, va, te = idx["treino"], idx["val"], idx["teste"]
    if verbose:
        modo = "por grupo de frase" if dados["grupos"] else "aleatória estratificada"
        print(f"Dados: {len(textos)} exemplos | treino {len(tr)} | val {len(va)} | teste {len(te)} (divisão {modo})")

    # O vetorizador é ajustado SOMENTE no treino (evita vazamento de dados).
    vet = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features)
    Xtr = vet.fit_transform([textos[i] for i in tr])
    Xva = vet.transform([textos[i] for i in va])
    Xte = vet.transform([textos[i] for i in te])
    if verbose:
        print(f"Vocabulário TF-IDF: {len(vet.vocabulary_)} termos")

    modelo = MultiTaskMLP(Xtr.shape[1], cfg.hidden, len(categorias), len(urgencias), cfg.seed)
    hist = modelo.fit(Xtr, y_c[tr], y_u[tr], Xva, y_c[va], y_u[va], cfg, verbose=verbose)

    pc, pu = modelo.predict_proba(Xte)
    pred_c, pred_u = pc.argmax(1), pu.argmax(1)
    rel_c = relatorio(y_c[te], pred_c, categorias)
    rel_u = relatorio(y_u[te], pred_u, urgencias)
    ambos = float(np.mean((pred_c == y_c[te]) & (pred_u == y_u[te])))
    base_c = float(np.mean(y_c[te] == np.bincount(y_c[tr]).argmax()))
    base_u = float(np.mean(y_u[te] == np.bincount(y_u[tr]).argmax()))

    metricas = {
        "versao": __version__,
        "n_exemplos": len(textos),
        "n_treino": int(len(tr)),
        "n_val": int(len(va)),
        "n_teste": int(len(te)),
        "epocas_executadas": len(hist),
        "melhor_epoca": int(min(hist, key=lambda h: h["loss_val"])["epoca"]),
        "acuracia_conjunta_teste": round(ambos, 4),
        "baseline_classe_majoritaria": {"categoria": round(base_c, 4), "urgencia": round(base_u, 4)},
        "categoria": rel_c,
        "urgencia": rel_u,
        "tempo_treino_s": round(time.time() - t0, 2),
    }

    meta = {
        "categorias": categorias,
        "urgencias": urgencias,
        "vetorizador": vet.to_dict(),
        "config": config_para_dict(cfg),
        "metricas_teste": {
            "acc_categoria": rel_c["acuracia"],
            "acc_urgencia": rel_u["acuracia"],
            "f1_macro_categoria": rel_c["f1_macro"],
            "f1_macro_urgencia": rel_u["f1_macro"],
        },
    }
    salvar_modelo(saida / "modelo.json", modelo, meta)
    (saida / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    (saida / "historico.json").write_text(json.dumps(hist, ensure_ascii=False), encoding="utf-8")
    (saida / "relatorio.md").write_text(_relatorio_md(metricas, cfg), encoding="utf-8")

    if verbose:
        print(f"\nTeste — categoria: acc {rel_c['acuracia']:.3f}, F1 macro {rel_c['f1_macro']:.3f} "
              f"(baseline {base_c:.3f})")
        print(f"Teste — urgência : acc {rel_u['acuracia']:.3f}, F1 macro {rel_u['f1_macro']:.3f} "
              f"(baseline {base_u:.3f})")
        print(f"Acerto conjunto (categoria e urgência): {ambos:.3f}")
        print(f"Artefatos salvos em: {saida.resolve()}")
    return metricas


def _linhas_classe(rel: dict) -> str:
    linhas = ["| Classe | Precisão | Recall | F1 | Suporte |", "|---|---|---|---|---|"]
    for nome, m in rel["por_classe"].items():
        linhas.append(f"| {nome} | {m['precisao']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['suporte']} |")
    return "\n".join(linhas)


def _relatorio_md(m: dict, cfg: TrainConfig) -> str:
    b = m["baseline_classe_majoritaria"]
    return f"""# Relatório de treino — Triagem de chamados

## Dados
- Exemplos: **{m['n_exemplos']}** (treino {m['n_treino']} · validação {m['n_val']} · teste {m['n_teste']})
- Treino em **{m['epocas_executadas']}** épocas (melhor época: {m['melhor_epoca']}), {m['tempo_treino_s']} s

## Configuração
`hidden={list(cfg.hidden)}`, `lr={cfg.lr}`, `batch={cfg.batch_size}`, `l2={cfg.l2}`, `dropout={cfg.dropout}`, `seed={cfg.seed}`

## Resultados no conjunto de teste
| Tarefa | Acurácia | F1 macro | Baseline (classe majoritária) |
|---|---|---|---|
| Categoria | {m['categoria']['acuracia']:.3f} | {m['categoria']['f1_macro']:.3f} | {b['categoria']:.3f} |
| Urgência | {m['urgencia']['acuracia']:.3f} | {m['urgencia']['f1_macro']:.3f} | {b['urgencia']:.3f} |

Acerto conjunto (categoria **e** urgência): **{m['acuracia_conjunta_teste']:.3f}**

### Categoria
{_linhas_classe(m['categoria'])}

{tabela_confusao_md(m['categoria'])}

### Urgência
{_linhas_classe(m['urgencia'])}

{tabela_confusao_md(m['urgencia'])}
"""


def gerar_e_salvar(caminho, n: int = 480, seed: int = 7) -> int:
    linhas = gerar_dataset(n, seed)
    salvar_csv(linhas, caminho)
    return len(linhas)
