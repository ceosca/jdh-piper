import unittest
from dataset_builder import fila_multi, plan_clips


class TestFilaMulti(unittest.TestCase):
    def test_fila_multi_formato(self):
        self.assertEqual(fila_multi("clip_0", "silvio", "hola mundo"),
                         "clip_0|silvio|hola mundo")

    def test_fila_multi_limpia_pipes_del_texto(self):
        # el texto no debe romper el CSV con '|'
        self.assertEqual(fila_multi("c1", "ana", "uno | dos"), "c1|ana|uno  dos")


class TestPlanClips(unittest.TestCase):
    def test_archivo_corto_se_conserva_entero(self):
        # un audio suelto de 10s (<= max_clip) queda como UN clip, sin re-segmentar
        self.assertEqual(plan_clips(10.0, [], max_clip=15.0), [(0.0, 10.0)])

    def test_clip_muy_corto_no_se_descarta(self):
        # un clip de 0.8s (antes se tiraba por < 1s) ahora se conserva entero
        self.assertEqual(plan_clips(0.8, [], max_clip=15.0), [(0.0, 0.8)])

    def test_archivo_largo_si_se_segmenta(self):
        # un audio largo (> max_clip) SÍ se parte por silencios (comportamiento previo)
        segs = plan_clips(40.0, [(15.0, 16.0), (30.0, 31.0)], max_clip=15.0)
        self.assertGreater(len(segs), 1)
        self.assertNotEqual(segs, [(0.0, 40.0)])  # no devuelve el archivo entero

    def test_tope_chico_hace_mas_clips_y_mas_cortos(self):
        # la palanca "muy chiquitos": bajar max_clip parte las tiradas largas en
        # más clips y más cortos, aunque no haya silencios internos.
        chicos = plan_clips(20.0, [], min_clip=1.5, max_clip=5.0)
        grandes = plan_clips(20.0, [], min_clip=1.5, max_clip=15.0)
        self.assertGreater(len(chicos), len(grandes))
        self.assertTrue(all((b - a) <= 5.2 for a, b in chicos))  # todos ~<= 5 s


if __name__ == "__main__":
    unittest.main()
