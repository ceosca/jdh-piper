import unittest
from dataset_builder import contar_duplicados, fila_multi


class TestDatasetMulti(unittest.TestCase):
    def test_fila_multi_formato(self):
        self.assertEqual(fila_multi("clip_0", "silvio", "hola mundo"),
                         "clip_0|silvio|hola mundo")

    def test_fila_multi_limpia_pipes_del_texto(self):
        # el texto no debe romper el CSV con '|'
        self.assertEqual(fila_multi("c1", "ana", "uno | dos"), "c1|ana|uno  dos")

    def test_contar_duplicados(self):
        self.assertEqual(contar_duplicados(["a", "b", "a", "c"]), (4, 3))  # 1 dup
        self.assertEqual(contar_duplicados(["x", "y"]), (2, 2))            # sin dup
        self.assertEqual(contar_duplicados([" a ", "a"]), (2, 1))          # strip


if __name__ == "__main__":
    unittest.main()
