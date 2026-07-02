# studio/tests/test_runs.py
import json, unittest, tempfile, os
from pathlib import Path
from studio.runs import RunState, save_run, load_run, list_runs, latest_epoch, run_dir, pid_alive


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


if __name__ == "__main__":
    unittest.main()
