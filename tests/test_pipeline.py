import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from helpdesk_nn.data import carregar_csv, dividir, gerar_dataset, salvar_csv
from helpdesk_nn.model import TrainConfig
from helpdesk_nn.predict import Classificador
from helpdesk_nn.server import criar_servidor
from helpdesk_nn.train import treinar


class TestDados(unittest.TestCase):
    def test_gerador_e_deterministico_e_balanceado(self):
        a, b = gerar_dataset(240, seed=1), gerar_dataset(240, seed=1)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 240)
        contagem = {}
        for l in a:
            contagem[(l["categoria"], l["urgencia"])] = contagem.get((l["categoria"], l["urgencia"]), 0) + 1
        self.assertEqual(len(contagem), 12)
        self.assertEqual(set(contagem.values()), {20})

    def test_divisao_por_grupo_nao_vaza(self):
        linhas = gerar_dataset(480, seed=3)
        cats = [l["categoria"] for l in linhas]
        urgs = [l["urgencia"] for l in linhas]
        grupos = [l["grupo"] for l in linhas]
        idx = dividir(cats, urgs, grupos, seed=1)
        g = {k: {grupos[i] for i in v} for k, v in idx.items()}
        self.assertFalse(g["treino"] & g["teste"])
        self.assertFalse(g["treino"] & g["val"])
        self.assertFalse(g["val"] & g["teste"])
        total = sum(len(v) for v in idx.values())
        self.assertEqual(total, len(linhas))

    def test_csv_invalido(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            p.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                carregar_csv(p)


class TestFimAFim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        salvar_csv(gerar_dataset(480, seed=7), base / "d.csv")
        cls.metricas = treinar(base / "d.csv", base / "out", TrainConfig(seed=42), verbose=False)
        cls.clf = Classificador.carregar(base / "out" / "modelo.json")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_modelo_supera_baseline_com_folga(self):
        m = self.metricas
        self.assertGreater(m["categoria"]["acuracia"], m["baseline_classe_majoritaria"]["categoria"] + 0.35)
        self.assertGreater(m["urgencia"]["acuracia"], m["baseline_classe_majoritaria"]["urgencia"] + 0.30)

    def test_exemplos_obvios(self):
        casos = {
            "A internet caiu em todo o prédio e ninguém consegue trabalhar": ("Redes", "Alto"),
            "Esqueci minha senha, preciso entrar no sistema hoje": ("Acessos/Senhas", None),
            "o mouse não funciona, mas não tem pressa": ("Hardware", "Baixo"),
        }
        for texto, (cat, urg) in casos.items():
            p = self.clf.prever_um(texto)
            self.assertEqual(p.categoria, cat, texto)
            if urg:
                self.assertEqual(p.urgencia, urg, texto)

    def test_probabilidades_validas(self):
        p = self.clf.prever_um("texto totalmente fora do vocabulário xyzxyz")
        self.assertAlmostEqual(sum(p.prob_categoria.values()), 1.0, places=6)
        self.assertAlmostEqual(sum(p.prob_urgencia.values()), 1.0, places=6)

    def test_api_http(self):
        srv = criar_servidor(self.clf, "127.0.0.1", 0)
        porta = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{porta}"
        try:
            self.assertIn(b"Triagem de chamados", urllib.request.urlopen(base + "/").read())
            self.assertEqual(json.load(urllib.request.urlopen(base + "/api/saude"))["status"], "ok")

            def post(corpo: bytes):
                req = urllib.request.Request(base + "/api/classificar", data=corpo,
                                             headers={"Content-Type": "application/json"})
                return urllib.request.urlopen(req)

            r = json.load(post(json.dumps({"texto": "a VPN caiu e a equipe está parada"}).encode()))
            self.assertEqual(r["categoria"], "Redes")
            self.assertIn("prob_urgencia", r)
            for ruim, codigo in ((b"{", 400), (b'{"texto": ""}', 422), (b'{"texto": 5}', 422)):
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    post(ruim)
                self.assertEqual(cm.exception.code, codigo)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
