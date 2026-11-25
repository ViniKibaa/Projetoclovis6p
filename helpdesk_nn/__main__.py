"""Interface de linha de comando: ``python -m helpdesk_nn <comando>``."""

import argparse
import json
import sys
from pathlib import Path

from .model import TrainConfig

RAIZ = Path.cwd()
CSV_PADRAO = RAIZ / "data" / "chamados.csv"
SAIDA_PADRAO = RAIZ / "artifacts"


def _cmd_gerar(a):
    from .train import gerar_e_salvar

    n = gerar_e_salvar(a.saida, a.n, a.seed)
    print(f"{n} chamados sintéticos salvos em {a.saida}")


def _cmd_treinar(a):
    from .train import gerar_e_salvar, treinar

    if not Path(a.dados).exists():
        print(f"'{a.dados}' não existe; gerando dataset sintético ({a.n} exemplos)...")
        gerar_e_salvar(a.dados, a.n, a.seed)
    cfg = TrainConfig(
        hidden=tuple(a.hidden), lr=a.lr, epochs=a.epocas, batch_size=a.batch,
        l2=a.l2, dropout=a.dropout, patience=a.paciencia, seed=a.seed,
    )
    treinar(a.dados, a.saida, cfg, max_features=a.max_features)


def _cmd_prever(a):
    from .predict import Classificador

    clf = Classificador.carregar(Path(a.saida) / "modelo.json")
    textos = a.texto or [l.strip() for l in sys.stdin if l.strip()]
    if not textos:
        print("Informe o texto do chamado como argumento ou via stdin.", file=sys.stderr)
        return 2
    for p in clf.prever(textos):
        if a.json:
            print(json.dumps(p.to_dict(), ensure_ascii=False))
        else:
            aviso = "  [revisão humana recomendada]" if p.revisao_humana else ""
            print(f"- {p.texto}\n    categoria: {p.categoria} ({p.confianca_categoria:.0%}) | "
                  f"urgência: {p.urgencia} ({p.confianca_urgencia:.0%}){aviso}")


def _cmd_servir(a):
    from .predict import Classificador
    from .server import servir

    clf = Classificador.carregar(Path(a.saida) / "modelo.json")
    servir(clf, a.host, a.porta)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="helpdesk_nn", description="Triagem de chamados com rede neural.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gerar-dados", help="gera dataset sintético em CSV")
    g.add_argument("--saida", default=str(CSV_PADRAO))
    g.add_argument("--n", type=int, default=480, help="número de exemplos (200–500 recomendado)")
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(func=_cmd_gerar)

    t = sub.add_parser("treinar", help="treina e avalia o modelo")
    t.add_argument("--dados", default=str(CSV_PADRAO))
    t.add_argument("--saida", default=str(SAIDA_PADRAO))
    t.add_argument("--n", type=int, default=480, help="tamanho do dataset se for preciso gerá-lo")
    t.add_argument("--hidden", type=int, nargs="+", default=[64])
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--epocas", type=int, default=300)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--l2", type=float, default=1e-3)
    t.add_argument("--dropout", type=float, default=0.5)
    t.add_argument("--paciencia", type=int, default=25)
    t.add_argument("--max-features", type=int, default=3000)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=_cmd_treinar)

    r = sub.add_parser("prever", help="classifica um ou mais textos")
    r.add_argument("texto", nargs="*")
    r.add_argument("--saida", default=str(SAIDA_PADRAO))
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=_cmd_prever)

    s = sub.add_parser("servir", help="sobe a interface web + API local")
    s.add_argument("--saida", default=str(SAIDA_PADRAO))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--porta", type=int, default=8000)
    s.set_defaults(func=_cmd_servir)

    args = p.parse_args(argv)
    try:
        return args.func(args) or 0
    except (FileNotFoundError, ValueError) as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
