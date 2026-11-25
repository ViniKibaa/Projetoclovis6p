import unittest

import numpy as np

from helpdesk_nn.vectorizer import TfidfVectorizer, normalizar, tokenizar


class TestVectorizer(unittest.TestCase):
    TEXTOS = [
        "A impressora não liga",
        "Impressora com defeito e não liga",
        "Minha senha expirou",
        "Senha expirou ontem",
    ]

    def test_normalizacao_remove_acentos(self):
        self.assertEqual(normalizar("Conexão Médio"), "conexao medio")
        self.assertNotIn("a", tokenizar("a internet caiu"))

    def test_linhas_normalizadas_e_forma(self):
        v = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
        X = v.fit_transform(self.TEXTOS)
        self.assertEqual(X.shape, (4, len(v.vocabulary_)))
        np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0)

    def test_termo_desconhecido_gera_vetor_zero(self):
        v = TfidfVectorizer(min_df=2).fit(self.TEXTOS)
        self.assertEqual(float(np.abs(v.transform(["zzzz qqqq"])).sum()), 0.0)

    def test_serializacao_ida_e_volta(self):
        v = TfidfVectorizer(min_df=2).fit(self.TEXTOS)
        v2 = TfidfVectorizer.from_dict(v.to_dict())
        np.testing.assert_allclose(v.transform(self.TEXTOS), v2.transform(self.TEXTOS))

    def test_erros_claros(self):
        with self.assertRaises(ValueError):
            TfidfVectorizer().fit([])
        with self.assertRaises(RuntimeError):
            TfidfVectorizer().transform(["x"])


if __name__ == "__main__":
    unittest.main()
