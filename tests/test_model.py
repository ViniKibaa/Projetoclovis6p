import unittest

import numpy as np

from helpdesk_nn.model import MultiTaskMLP, TrainConfig, carregar_modelo, salvar_modelo


class TestModel(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.random((12, 7))
        self.yc = rng.integers(0, 3, 12)
        self.yu = rng.integers(0, 3, 12)

    def test_gradientes_batem_com_diferencas_finitas(self):
        m = MultiTaskMLP(7, (5, 4), 3, 3, seed=1)
        _, grads = m.loss_and_grads(self.X, self.yc, self.yu, l2=1e-2, uw=0.7)
        eps = 1e-6
        rng = np.random.default_rng(3)
        for nome, g in grads.items():
            P = m.params[nome]
            for _ in range(4):
                idx = tuple(rng.integers(0, s) for s in P.shape)
                orig = P[idx]
                P[idx] = orig + eps
                lp, _ = m.loss_and_grads(self.X, self.yc, self.yu, 1e-2, 0.7)
                P[idx] = orig - eps
                lm, _ = m.loss_and_grads(self.X, self.yc, self.yu, 1e-2, 0.7)
                P[idx] = orig
                num = (lp - lm) / (2 * eps)
                self.assertAlmostEqual(num, g[idx], places=6, msg=f"{nome}{idx}")

    def test_probabilidades_somam_um(self):
        m = MultiTaskMLP(7, (6,), 3, 3)
        pc, pu = m.predict_proba(self.X)
        np.testing.assert_allclose(pc.sum(1), 1.0)
        np.testing.assert_allclose(pu.sum(1), 1.0)

    def test_overfit_em_lote_pequeno(self):
        m = MultiTaskMLP(7, (32,), 3, 3, seed=2)
        cfg = TrainConfig(hidden=(32,), epochs=300, batch_size=12, dropout=0.0, l2=0.0, lr=1e-2, patience=300)
        m.fit(self.X, self.yc, self.yu, self.X, self.yc, self.yu, cfg)
        pc, pu = m.predict_proba(self.X)
        self.assertGreater(np.mean(pc.argmax(1) == self.yc), 0.95)
        self.assertGreater(np.mean(pu.argmax(1) == self.yu), 0.95)

    def test_salvar_e_carregar(self):
        import tempfile, os

        m = MultiTaskMLP(7, (6,), 3, 3, seed=5)
        with tempfile.TemporaryDirectory() as d:
            caminho = os.path.join(d, "m.json")
            salvar_modelo(caminho, m, {"x": 1})
            m2, meta = carregar_modelo(caminho)
        self.assertEqual(meta["x"], 1)
        np.testing.assert_allclose(m.predict_proba(self.X)[0], m2.predict_proba(self.X)[0])


if __name__ == "__main__":
    unittest.main()
