import json, unittest, tempfile
from pathlib import Path
from studio.sherpa_export import tokens_txt, meta_data, espeak_data_dir, empaquetar


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


class TestIO(unittest.TestCase):
    def test_espeak_data_dir_encuentra_y_none(self):
        tmp = Path(tempfile.mkdtemp())
        d = tmp / "Lib" / "site-packages" / "piper" / "espeak-ng-data"
        d.mkdir(parents=True)
        self.assertEqual(espeak_data_dir(tmp), d)
        self.assertIsNone(espeak_data_dir(Path(tempfile.mkdtemp())))

    def test_empaquetar_arma_carpeta_completa(self):
        tmp = Path(tempfile.mkdtemp())
        onnx = tmp / "mario.onnx"; onnx.write_bytes(b"ONNXFAKE")
        cfg = tmp / "mario.onnx.json"
        cfg.write_text(json.dumps({"phoneme_id_map": {"a": [5], "b": [6]},
                                   "espeak": {"voice": "es"}}), encoding="utf-8")
        espeak = tmp / "espeak-ng-data"; espeak.mkdir()
        (espeak / "phontab").write_text("x", encoding="utf-8")
        out = tmp / "out"
        calls = []
        def fake_add(p, meta): calls.append((Path(p).name, meta["voice"]))
        res = empaquetar(onnx, cfg, out, espeak, add_meta=fake_add)
        self.assertEqual(res, out)
        self.assertTrue((out / "mario.onnx").exists())
        self.assertEqual((out / "tokens.txt").read_text(encoding="utf-8"), "a 5\nb 6\n")
        self.assertTrue((out / "espeak-ng-data" / "phontab").exists())
        self.assertTrue((out / "LEEME.txt").exists())
        self.assertEqual(calls, [("mario.onnx", "es")])  # add_meta recibió el onnx copiado

    def test_empaquetar_sin_phoneme_id_map_falla(self):
        tmp = Path(tempfile.mkdtemp())
        onnx = tmp / "v.onnx"; onnx.write_bytes(b"x")
        cfg = tmp / "v.onnx.json"; cfg.write_text("{}", encoding="utf-8")
        espeak = tmp / "e"; espeak.mkdir()
        with self.assertRaises(ValueError):
            empaquetar(onnx, cfg, tmp / "out", espeak, add_meta=lambda p, m: None)


if __name__ == "__main__":
    unittest.main()
