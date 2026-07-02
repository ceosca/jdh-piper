import unittest
from dataset_builder import fila_multi


class TestDatasetMulti(unittest.TestCase):
    def test_fila_multi_formato(self):
        self.assertEqual(fila_multi("clip_0", "silvio", "hola mundo"),
                         "clip_0|silvio|hola mundo")

    def test_fila_multi_limpia_pipes_del_texto(self):
        # el texto no debe romper el CSV con '|'
        self.assertEqual(fila_multi("c1", "ana", "uno | dos"), "c1|ana|uno  dos")


if __name__ == "__main__":
    unittest.main()
