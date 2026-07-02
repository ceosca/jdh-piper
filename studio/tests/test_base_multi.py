# studio/tests/test_base_multi.py
import unittest
import torch
from studio.base_multi import hparams_multi, fusionar_pesos


class TestBaseMulti(unittest.TestCase):
    def test_hparams_multi_fija_speakers_y_gin(self):
        hp = {"num_symbols": 256, "num_speakers": 1, "gin_channels": 0,
              "inter_channels": 192, "no_valido_xyz": 1}
        out = hparams_multi(hp, num_speakers=8, gin_channels=256)
        self.assertEqual(out["num_speakers"], 8)
        self.assertEqual(out["gin_channels"], 256)
        self.assertEqual(out["inter_channels"], 192)
        self.assertNotIn("no_valido_xyz", out)  # se filtra a args de VitsModel

    def test_fusionar_copia_coincidentes_y_conserva_nuevas(self):
        mono = {"a": torch.ones(3), "b": torch.ones(2, 2)}
        multi = {"a": torch.zeros(3), "b": torch.zeros(2, 2),
                 "emb_g": torch.zeros(5), "cond": torch.zeros(4)}
        merged, n_cop, n_new = fusionar_pesos(mono, multi)
        self.assertTrue(torch.equal(merged["a"], torch.ones(3)))      # copiada
        self.assertTrue(torch.equal(merged["b"], torch.ones(2, 2)))   # copiada
        self.assertTrue(torch.equal(merged["emb_g"], torch.zeros(5)))  # nueva (init)
        self.assertTrue(torch.equal(merged["cond"], torch.zeros(4)))   # nueva (init)
        self.assertEqual(n_cop, 2)
        self.assertEqual(n_new, 2)

    def test_fusionar_ignora_shape_distinta(self):
        mono = {"a": torch.ones(3)}
        multi = {"a": torch.zeros(9)}          # misma clave, otra forma
        merged, n_cop, n_new = fusionar_pesos(mono, multi)
        self.assertTrue(torch.equal(merged["a"], torch.zeros(9)))  # NO copia
        self.assertEqual(n_cop, 0)
        self.assertEqual(n_new, 1)


if __name__ == "__main__":
    unittest.main()
