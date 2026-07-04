# studio/tests/test_runs.py
import json, unittest, tempfile, os
from pathlib import Path
from studio.runs import (RunState, save_run, load_run, list_runs, latest_epoch, run_dir,
                          pid_alive, pick_resume_ckpt)


class TestRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _mk(self, nombre="silvio", **kw):
        base = dict(nombre=nombre, modo="finetune", dataset="datasets/silvio",
                    base_ckpt="base.ckpt", resume_ckpt=None, max_epochs=800,
                    auto_stop=True, paciencia=12, cada=10, pid=None,
                    started_at="2026-07-02T00:00:00", estado="pausado", last_event="")
        base.update(kw)
        return RunState(**base)

    def test_save_and_load_roundtrip(self):
        st = self._mk()
        save_run(self.tmp, st)
        rj = run_dir(self.tmp, "silvio") / "run.json"
        self.assertTrue(rj.exists())
        got = load_run(rj)
        self.assertEqual(got.nombre, "silvio")
        self.assertEqual(got.max_epochs, 800)
        self.assertTrue(got.auto_stop)

    def test_latest_epoch_reads_highest(self):
        ck = run_dir(self.tmp, "silvio") / "ckpts"
        ck.mkdir(parents=True)
        for name in ("silvio-epoch=99.ckpt", "silvio-epoch=1499.ckpt", "last.ckpt"):
            (ck / name).write_text("x")
        self.assertEqual(latest_epoch(run_dir(self.tmp, "silvio")), 1499)

    def test_latest_epoch_none_when_empty(self):
        (run_dir(self.tmp, "silvio") / "ckpts").mkdir(parents=True)
        self.assertIsNone(latest_epoch(run_dir(self.tmp, "silvio")))

    def test_pid_alive_false_for_none_and_dead(self):
        self.assertFalse(pid_alive(None))
        self.assertFalse(pid_alive(999999))  # PID improbable

    def test_pid_alive_true_for_self(self):
        self.assertTrue(pid_alive(os.getpid()))

    def test_list_runs_refreshes_estado(self):
        st = self._mk(pid=999999, estado="entrenando")
        save_run(self.tmp, st)
        runs = list_runs(self.tmp)
        self.assertEqual(len(runs), 1)
        # PID muerto => estado deja de ser "entrenando"
        self.assertNotEqual(runs[0].estado, "entrenando")


class TestPickResumeCkpt(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_prefers_last_ckpt_when_it_exists(self):
        rd = run_dir(self.tmp, "silvio")
        ck = rd / "ckpts"
        ck.mkdir(parents=True)
        (ck / "last.ckpt").write_text("x")
        (ck / "silvio-epoch=99.ckpt").write_text("x")
        self.assertEqual(pick_resume_ckpt(rd, "base.ckpt"), str(ck / "last.ckpt"))

    def test_falls_back_to_highest_epoch_when_no_last_ckpt(self):
        rd = run_dir(self.tmp, "silvio")
        ck = rd / "ckpts"
        ck.mkdir(parents=True)
        (ck / "voz-epoch=99.ckpt").write_text("x")
        (ck / "voz-epoch=299.ckpt").write_text("x")
        self.assertEqual(pick_resume_ckpt(rd, "base.ckpt"), str(ck / "voz-epoch=299.ckpt"))

    def test_falls_back_to_base_ckpt_when_ckpts_dir_empty(self):
        rd = run_dir(self.tmp, "silvio")
        (rd / "ckpts").mkdir(parents=True)
        self.assertEqual(pick_resume_ckpt(rd, "base.ckpt"), "base.ckpt")


from studio.runs import build_train_argv

class TestArgv(unittest.TestCase):
    def _mk(self, **kw):
        from studio.runs import RunState
        base = dict(nombre="silvio", modo="finetune", dataset="datasets/silvio",
                    base_ckpt="base.ckpt", max_epochs=1500, auto_stop=True,
                    paciencia=20, cada=10)
        base.update(kw); return RunState(**base)

    def test_autostop_usa_entrenar(self):
        argv = build_train_argv("py.exe", Path("."), self._mk(auto_stop=True))
        self.assertIn("entrenar.py", " ".join(argv))
        self.assertIn("--paciencia", argv)
        self.assertIn("20", argv)

    def test_autostop_pasa_dataset_explicito(self):
        # el nombre de la voz puede no coincidir con la carpeta del dataset
        argv = build_train_argv("py.exe", Path("."),
                                self._mk(nombre="mario", dataset="datasets/mario-castanieda"))
        self.assertIn("--dataset", argv)
        self.assertIn("datasets/mario-castanieda", argv)

    def test_manual_usa_train_run_sin_earlystop(self):
        argv = build_train_argv("py.exe", Path("."), self._mk(auto_stop=False, max_epochs=2000))
        joined = " ".join(argv)
        self.assertIn("train_run.py", joined)
        self.assertIn("fit", argv)
        self.assertIn("--trainer.max_epochs", argv)
        self.assertIn("2000", argv)
        self.assertNotIn("EarlyStopping", joined)

    def test_resume_pasa_ckpt(self):
        st = self._mk(auto_stop=False, resume_ckpt="training/silvio/ckpts/last.ckpt")
        argv = build_train_argv("py.exe", Path("."), st)
        self.assertIn("--ckpt_path", argv)
        self.assertIn("training/silvio/ckpts/last.ckpt", argv)


class TestArgvBase(unittest.TestCase):
    def test_modo_base_usa_entrenar_base(self):
        from studio.runs import RunState, build_train_argv
        from pathlib import Path
        st = RunState(nombre="base_latino", modo="base", dataset="datasets/base_latino",
                      base_ckpt="base_ckpt/silvio_base_clean.ckpt", max_epochs=4000,
                      num_speakers=12)
        argv = build_train_argv("py.exe", Path("."), st)
        j = " ".join(argv)
        self.assertIn("entrenar_base.py", j)
        self.assertIn("--num-speakers", argv)
        self.assertIn("12", argv)
        self.assertIn("--base-mono", argv)
        self.assertIn("--voz", argv)
        self.assertIn(st.nombre, argv)

    def test_modo_base_con_resume_pasa_resume_ckpt(self):
        from studio.runs import RunState, build_train_argv
        from pathlib import Path
        resume_path = "training/base_latino/ckpts/last.ckpt"
        st = RunState(nombre="base_latino", modo="base", dataset="datasets/base_latino",
                      base_ckpt="base_ckpt/silvio_base_clean.ckpt", max_epochs=4000,
                      num_speakers=12, resume_ckpt=resume_path)
        argv = build_train_argv("py.exe", Path("."), st)
        self.assertIn("--resume", argv)
        self.assertIn(resume_path, argv)


class TestLeerEpoca(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_epoch_txt_gana_sobre_nombre(self):
        from studio.runs import leer_epoca, run_dir
        rd = run_dir(self.tmp, "v"); (rd / "ckpts").mkdir(parents=True)
        (rd / "ckpts" / "v-epoch=50.ckpt").write_text("x")
        (rd / "epoch.txt").write_text("123")
        self.assertEqual(leer_epoca(rd), 123)

    def test_fallback_al_nombre_sin_epoch_txt(self):
        from studio.runs import leer_epoca, run_dir
        rd = run_dir(self.tmp, "v2"); (rd / "ckpts").mkdir(parents=True)
        (rd / "ckpts" / "v2-epoch=77.ckpt").write_text("x")
        self.assertEqual(leer_epoca(rd), 77)

    def test_none_si_no_hay_nada(self):
        from studio.runs import leer_epoca, run_dir
        rd = run_dir(self.tmp, "v3"); (rd / "ckpts").mkdir(parents=True)
        self.assertIsNone(leer_epoca(rd))

    def test_leer_mejor(self):
        from studio.runs import leer_mejor, run_dir
        rd = run_dir(self.tmp, "v4"); rd.mkdir(parents=True)
        (rd / "mejor.txt").write_text("549 19.4200")
        self.assertEqual(leer_mejor(rd), (549, 19.42))
        rd2 = run_dir(self.tmp, "v5"); rd2.mkdir(parents=True)
        self.assertIsNone(leer_mejor(rd2))

    def test_leer_progreso(self):
        from studio.runs import leer_progreso, run_dir
        rd = run_dir(self.tmp, "v6"); rd.mkdir(parents=True)
        (rd / "progreso.log").write_text("linea 1\nlinea 2\nlinea 3\n")
        self.assertEqual(leer_progreso(rd), ["linea 1", "linea 2", "linea 3"])
        self.assertEqual(leer_progreso(rd, max_lineas=2), ["linea 2", "linea 3"])
        self.assertEqual(leer_progreso(run_dir(self.tmp, "v7")), [])

    def test_config_de_voz_usa_dataset_del_run(self):
        # voz "mario" con dataset "mario-castanieda" (nombre != carpeta)
        from studio.runs import config_de_voz, save_run, RunState
        ds = self.tmp / "datasets" / "mario-castanieda"; ds.mkdir(parents=True)
        (ds / "config.json").write_text("{}")
        save_run(self.tmp / "training", RunState(nombre="mario", dataset=str(ds)))
        self.assertEqual(config_de_voz(self.tmp, "mario"), ds / "config.json")

    def test_config_de_voz_fallback_y_none(self):
        from studio.runs import config_de_voz
        d = self.tmp / "datasets" / "pedro"; d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        self.assertEqual(config_de_voz(self.tmp, "pedro"), d / "config.json")
        self.assertIsNone(config_de_voz(self.tmp, "noexiste"))


if __name__ == "__main__":
    unittest.main()
