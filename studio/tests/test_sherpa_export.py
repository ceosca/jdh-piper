import json, unittest, tempfile
from pathlib import Path
from studio.sherpa_export import tokens_txt, meta_data


class TestPuros(unittest.TestCase):
    def test_tokens_txt_formato_y_primer_id(self):
        m = {"a": [5], "b": [6, 99]}  # toma SOLO el primer id
        self.assertEqual(tokens_txt(m), "a 5\nb 6\n")

    def test_meta_data_toma_voice_del_config(self):
        cfg = {"espeak": {"voice": "es"}, "language": {"name_english": "Spanish"},
               "num_speakers": 1, "audio": {"sample_rate": 22050}}
        md = meta_data(cfg)
        self.assertEqual(md["voice"], "es")
        self.assertEqual(md["model_type"], "vits")
        self.assertEqual(md["comment"], "piper")
        self.assertEqual(md["has_espeak"], 1)
        self.assertEqual(md["sample_rate"], 22050)

    def test_meta_data_defaults_si_faltan(self):
        md = meta_data({})  # config vacío
        self.assertEqual(md["voice"], "es")
        self.assertEqual(md["language"], "Spanish")
        self.assertEqual(md["n_speakers"], 1)
        self.assertEqual(md["sample_rate"], 22050)


if __name__ == "__main__":
    unittest.main()
