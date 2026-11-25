"""Servidor HTTP local (somente biblioteca padrão): interface web + API JSON.

Rotas:
    GET  /                  interface web
    GET  /api/saude         verificação de saúde
    GET  /api/info          classes, métricas de teste e versão
    POST /api/classificar   {"texto": "..."} -> categoria e urgência
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from . import __version__
from .predict import Classificador

TAMANHO_MAX_CORPO = 16 * 1024
TAMANHO_MAX_TEXTO = 2000
CABECALHOS_SEGURANCA = {
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'",
}


def _html() -> bytes:
    return resources.files("helpdesk_nn").joinpath("web/index.html").read_bytes()


def criar_servidor(clf: Classificador, host: str = "127.0.0.1", porta: int = 8000) -> ThreadingHTTPServer:
    pagina = _html()

    class Handler(BaseHTTPRequestHandler):
        server_version = "HelpdeskNN/" + __version__

        def _responder(self, status: int, corpo: bytes, tipo: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(corpo)))
            for k, v in CABECALHOS_SEGURANCA.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(corpo)

        def _json(self, status: int, payload: dict) -> None:
            self._responder(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                            "application/json; charset=utf-8")

        def do_GET(self):
            rota = self.path.split("?", 1)[0]
            if rota in ("/", "/index.html"):
                self._responder(200, pagina, "text/html; charset=utf-8")
            elif rota == "/api/saude":
                self._json(200, {"status": "ok"})
            elif rota == "/api/info":
                self._json(200, {
                    "versao": __version__,
                    "categorias": clf.categorias,
                    "urgencias": clf.urgencias,
                    "metricas_teste": clf.meta.get("metricas_teste", {}),
                })
            else:
                self._json(404, {"erro": "rota não encontrada"})

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/api/classificar":
                return self._json(404, {"erro": "rota não encontrada"})
            try:
                tamanho = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._json(400, {"erro": "Content-Length inválido"})
            if tamanho <= 0 or tamanho > TAMANHO_MAX_CORPO:
                return self._json(413 if tamanho > TAMANHO_MAX_CORPO else 400,
                                  {"erro": "corpo ausente ou grande demais"})
            try:
                dados = json.loads(self.rfile.read(tamanho).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return self._json(400, {"erro": "JSON inválido"})
            texto = dados.get("texto") if isinstance(dados, dict) else None
            if not isinstance(texto, str) or not texto.strip():
                return self._json(422, {"erro": "campo 'texto' é obrigatório"})
            if len(texto) > TAMANHO_MAX_TEXTO:
                return self._json(422, {"erro": f"texto acima de {TAMANHO_MAX_TEXTO} caracteres"})
            self._json(200, clf.prever_um(texto.strip()).to_dict())

        def log_message(self, fmt, *args):  # log enxuto no stderr
            super().log_message(fmt, *args)

    return ThreadingHTTPServer((host, porta), Handler)


def servir(clf: Classificador, host: str = "127.0.0.1", porta: int = 8000) -> None:
    srv = criar_servidor(clf, host, porta)
    print(f"Servidor em http://{host}:{srv.server_address[1]}  (Ctrl+C para encerrar)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando...")
    finally:
        srv.server_close()
