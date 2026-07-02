# Piper Studio GUI — Plan de Implementación (Plan 1: cáscara + Entrenar/Comparar/Exportar)

> **Para quien ejecute:** SUB-SKILL REQUERIDA: usar superpowers:subagent-driven-development (recomendado) o superpowers:executing-plans para implementar tarea por tarea. Los pasos usan checkbox (`- [ ]`).

**Goal:** Una app accesible (wxPython + NVDA) que centraliza el flujo de Piper Studio, empezando por entrenar un fine-tune de voz **desprendido** (sobrevive a cerrar la ventana, re-enganchable), con parada automática (early-stop) o épocas manuales, avisos NVDA, más las secciones Comparar por oído y Exportar/Instalar.

**Architecture:** Un paquete `studio/` con una ventana wxPython de secciones. La lógica de corridas (estado en `run.json`, liveness de PID, estimación de época, lanzar/pausar/reanudar como proceso desprendido) vive en `studio/runs.py` — puro y testeable con `unittest`. Las secciones envuelven los scripts ya probados (`entrenar.py`, `train_run.py`, `comparar_checkpoints.py`, `export_run.py`) sin reemplazarlos.

**Tech Stack:** Python 3.11 (env `C:\ia\piper-studio\env`), wxPython 4.2.x, `unittest` (stdlib) para tests, `subprocess` con detached flags, `ctypes` para liveness en Windows. NVDA vía `nvdaControllerClient*.dll`.

## Global Constraints

- **Windows nativo**, sin WSL. Rutas con `pathlib`. El python del proyecto es `C:\ia\piper-studio\env\python.exe`.
- **Accesibilidad NVDA obligatoria**: foco anunciado, aceleradores en botones, nombres accesibles; `stdout/stderr`→archivo log (evita `WinError 1` de tqdm en GUI).
- **No romper lo existente**: todo envuelve scripts que ya funcionan; no se editan `train_run.py`, `entrenar.py`, `export_run.py`, `comparar_checkpoints.py` salvo que una tarea lo diga.
- **Desprendido de verdad**: el proceso de entrenamiento NO debe morir al cerrar la GUI (`DETACHED_PROCESS` en Windows).
- **Stack estándar** (Piper1-gpl 1.4.2 + espeak). Nada de piper-plus.
- Idioma de la UI y de los avisos: **español**.

## Estructura de archivos

- `studio/__init__.py` — paquete.
- `studio/runs.py` — estado de corridas + liveness + estimación de época + lanzar/pausar/reanudar. (Núcleo testeable.)
- `studio/nvda.py` — `NVDAController` compartido.
- `studio/app.py` — app wxPython, ventana principal, contenedor de secciones (Notebook).
- `studio/section_train.py` — sección Entrenar (fine-tune).
- `studio/section_compare.py` — sección Comparar por oído.
- `studio/section_export.py` — sección Exportar / Instalar.
- `studio/tests/test_runs.py` — tests de `runs.py`.
- `Piper Studio.bat` — lanzador.

Cada corrida: `training/<nombre>/` con `ckpts/`, `run.json`, `train.log`.

---

### Task 1: `runs.py` — estado de corrida + liveness de PID + estimación de época

**Files:**
- Create: `studio/__init__.py` (vacío)
- Create: `studio/runs.py`
- Test: `studio/tests/test_runs.py`, `studio/tests/__init__.py` (vacío)

**Interfaces:**
- Produces:
  - `RunState` dataclass: `nombre:str, modo:str, dataset:str, base_ckpt:str, resume_ckpt:str|None, max_epochs:int, auto_stop:bool, paciencia:int, cada:int, pid:int|None, started_at:str, estado:str, last_event:str`
  - `run_dir(root:Path, nombre:str) -> Path`
  - `save_run(root:Path, st:RunState) -> None` (escribe `training/<nombre>/run.json`)
  - `load_run(run_json:Path) -> RunState`
  - `list_runs(root:Path) -> list[RunState]` (escanea `training/*/run.json`, refresca `estado`)
  - `pid_alive(pid:int|None) -> bool`
  - `latest_epoch(run_dir:Path) -> int|None` (mayor `epoch=` en `ckpts/*.ckpt`)

- [ ] **Step 1: Escribir el test que falla**

```python
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
```

- [ ] **Step 2: Correr el test para verlo fallar**

Run: `C:\ia\piper-studio\env\python.exe -m unittest studio.tests.test_runs -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'studio.runs'`.

- [ ] **Step 3: Implementar `runs.py` (núcleo, sin lanzar procesos todavía)**

```python
# studio/runs.py
"""Estado y control de corridas de entrenamiento (desprendidas, re-enganchables)."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunState:
    nombre: str
    modo: str = "finetune"          # finetune | base
    dataset: str = ""
    base_ckpt: str = ""
    resume_ckpt: str | None = None
    max_epochs: int = 800
    auto_stop: bool = True
    paciencia: int = 12
    cada: int = 10
    pid: int | None = None
    started_at: str = ""
    estado: str = "pausado"         # entrenando | pausado | terminado | fallo
    last_event: str = ""


def run_dir(root: Path, nombre: str) -> Path:
    return Path(root) / nombre


def _run_json(root: Path, nombre: str) -> Path:
    return run_dir(root, nombre) / "run.json"


def save_run(root: Path, st: RunState) -> None:
    d = run_dir(root, st.nombre)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps(asdict(st), ensure_ascii=False, indent=2),
                                encoding="utf-8")


def load_run(run_json: Path) -> RunState:
    data = json.loads(Path(run_json).read_text(encoding="utf-8"))
    return RunState(**data)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k = ctypes.windll.kernel32
        h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if not k.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            k.CloseHandle(h)
    else:
        import os
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def latest_epoch(rd: Path) -> int | None:
    ck = Path(rd) / "ckpts"
    if not ck.is_dir():
        return None
    epochs = []
    for f in ck.glob("*.ckpt"):
        m = re.search(r"epoch=(\d+)", f.stem)
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


def list_runs(root: Path) -> list[RunState]:
    root = Path(root)
    out: list[RunState] = []
    if not root.is_dir():
        return out
    for rj in sorted(root.glob("*/run.json")):
        try:
            st = load_run(rj)
        except Exception:
            continue
        # refrescar estado según liveness real del proceso
        if st.estado == "entrenando" and not pid_alive(st.pid):
            # se murió sin marcar; asumimos terminado (o pausa por kill)
            st.estado = "terminado"
        out.append(st)
    return out
```

- [ ] **Step 4: Correr los tests y verlos pasar**

Run: `C:\ia\piper-studio\env\python.exe -m unittest studio.tests.test_runs -v`
Expected: PASS (6 tests).

---

### Task 2: `runs.py` — lanzar / pausar / reanudar (proceso desprendido)

**Files:**
- Modify: `studio/runs.py`
- Test: `studio/tests/test_runs.py` (agregar)

**Interfaces:**
- Produces:
  - `build_train_argv(py:str, root_proj:Path, st:RunState) -> list[str]` — arma el comando (usa `entrenar.py` si `auto_stop`, si no `train_run.py fit` con `max_epochs` y sin early-stop). PURA → testeable.
  - `launch(root_runs:Path, root_proj:Path, st:RunState, py:str) -> RunState` — lanza desprendido, guarda pid+estado.
  - `pause(root_runs:Path, st:RunState) -> RunState` — mata el proceso, estado `pausado`.
  - `resume(root_runs:Path, root_proj:Path, st:RunState, py:str) -> RunState` — relanza desde `ckpts/last.ckpt` (o `st.resume_ckpt`).

- [ ] **Step 1: Escribir el test del constructor de comando (falla)**

```python
# añadir a studio/tests/test_runs.py
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
```

- [ ] **Step 2: Correr y ver fallar**

Run: `C:\ia\piper-studio\env\python.exe -m unittest studio.tests.test_runs -v`
Expected: FAIL con `ImportError: cannot import name 'build_train_argv'`.

- [ ] **Step 3: Implementar constructor + lanzar/pausar/reanudar**

```python
# añadir a studio/runs.py
import datetime as _dt
import subprocess


def build_train_argv(py: str, root_proj: Path, st: RunState) -> list[str]:
    rp = Path(root_proj)
    ds = st.dataset
    ckpt = st.resume_ckpt or st.base_ckpt
    if st.auto_stop:
        # entrenar.py trae EarlyStopping(val_mel) + checkpoints + best
        argv = [py, str(rp / "entrenar.py"),
                "--voz", st.nombre,
                "--base", ckpt,
                "--max-epochs", str(st.max_epochs),
                "--paciencia", str(st.paciencia),
                "--cada", str(st.cada)]
    else:
        # épocas manuales, sin early-stop: train_run.py fit
        ck = rp / "training" / st.nombre / "ckpts"
        cb = ('{"class_path":"lightning.pytorch.callbacks.ModelCheckpoint",'
              '"init_args":{"dirpath":"%s","every_n_epochs":100,'
              '"save_top_k":-1,"save_last":true,"filename":"%s-{epoch}"}}'
              % (ck.as_posix(), st.nombre))
        argv = [py, str(rp / "train_run.py"), "fit",
                "--data.voice_name", st.nombre,
                "--data.csv_path", f"{ds}/metadata.csv",
                "--data.audio_dir", f"{ds}/wavs",
                "--model.sample_rate", "22050",
                "--data.espeak_voice", "es",
                "--data.cache_dir", f"{ds}/cache",
                "--data.config_path", f"{ds}/config.json",
                "--data.batch_size", "8", "--data.num_workers", "0",
                "--ckpt_path", ckpt,
                "--trainer.max_epochs", str(st.max_epochs),
                "--trainer.accelerator", "gpu", "--trainer.devices", "1",
                "--trainer.default_root_dir", str(rp / "training" / st.nombre),
                "--trainer.callbacks+", cb]
    return argv


def _spawn_detached(argv: list[str], cwd: Path, log_path: Path) -> int:
    log = open(log_path, "a", encoding="utf-8", buffering=1)
    kwargs = dict(cwd=str(cwd), stdout=log, stderr=log, stdin=subprocess.DEVNULL)
    if sys.platform == "win32":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    p = subprocess.Popen(argv, **kwargs)
    return p.pid


def launch(root_runs: Path, root_proj: Path, st: RunState, py: str) -> RunState:
    rd = run_dir(root_runs, st.nombre)
    (rd / "ckpts").mkdir(parents=True, exist_ok=True)
    argv = build_train_argv(py, root_proj, st)
    st.pid = _spawn_detached(argv, Path(root_proj), rd / "train.log")
    st.started_at = _dt.datetime.now().isoformat(timespec="seconds")
    st.estado = "entrenando"
    st.last_event = "lanzado"
    save_run(root_runs, st)
    return st


def _kill(pid: int | None) -> None:
    if not pid or not pid_alive(pid):
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True)
    else:
        import os, signal
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def pause(root_runs: Path, st: RunState) -> RunState:
    _kill(st.pid)
    st.pid = None
    st.estado = "pausado"
    st.last_event = "pausado"
    save_run(root_runs, st)
    return st


def resume(root_runs: Path, root_proj: Path, st: RunState, py: str) -> RunState:
    last = run_dir(root_runs, st.nombre) / "ckpts" / "last.ckpt"
    st.resume_ckpt = st.resume_ckpt or (str(last) if last.exists() else st.base_ckpt)
    return launch(root_runs, root_proj, st, py)
```

- [ ] **Step 4: Correr tests y ver pasar**

Run: `C:\ia\piper-studio\env\python.exe -m unittest studio.tests.test_runs -v`
Expected: PASS (todos, incl. los 3 nuevos de argv).

---

### Task 3: Cáscara de la app + NVDA

**Files:**
- Create: `studio/nvda.py`
- Create: `studio/app.py`
- Create: `Piper Studio.bat`

**Interfaces:**
- Consumes: nada (shell).
- Produces: `NVDAController` (`.speak(text, interrupt=True)`); `StudioFrame` (wx.Frame con `wx.Notebook` y placeholders de secciones); `main()`.

- [ ] **Step 1: `studio/nvda.py`** (copiar el patrón ya usado en `gui_dataset.py`)

```python
# studio/nvda.py
import ctypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class NVDAController:
    def __init__(self, base_dir: Path = ROOT):
        self.dll = None
        self.ready = False
        for name in ("nvdaControllerClient64.dll", "nvdaControllerClient.dll"):
            p = base_dir / name
            if p.exists():
                try:
                    self.dll = ctypes.WinDLL(str(p)); self.ready = True; break
                except Exception:
                    self.dll = None

    def speak(self, text, interrupt=True):
        if not self.ready or self.dll is None:
            return
        try:
            if interrupt and hasattr(self.dll, "nvdaController_cancelSpeech"):
                self.dll.nvdaController_cancelSpeech()
            self.dll.nvdaController_speakText(ctypes.c_wchar_p(str(text)))
        except Exception:
            pass
```

- [ ] **Step 2: `studio/app.py`** (ventana + Notebook + redirección de log)

```python
# studio/app.py
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    _log = open(ROOT / "studio.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = _log; sys.stderr = _log
except Exception:
    pass

import wx  # noqa: E402
from studio.nvda import NVDAController  # noqa: E402


class StudioFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Piper Studio", size=(820, 620))
        self.nvda = NVDAController()
        self.nb = wx.Notebook(self)
        # Las secciones reales se agregan en tareas siguientes:
        self._add_placeholder("Entrenar")
        self._add_placeholder("Comparar")
        self._add_placeholder("Exportar")
        self.Centre(); self.Show()
        self.nvda.speak("Piper Studio abierto", True)

    def _add_placeholder(self, label):
        p = wx.Panel(self.nb)
        s = wx.BoxSizer(wx.VERTICAL)
        s.Add(wx.StaticText(p, label=f"Sección {label} (en construcción)"), 0, wx.ALL, 10)
        p.SetSizer(s)
        self.nb.AddPage(p, label)


def main():
    app = wx.App(False)
    StudioFrame()
    app.MainLoop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: `Piper Studio.bat`**

```bat
@echo off
cd /d "%~dp0"
"%~dp0env\python.exe" -m studio.app
```

- [ ] **Step 4: Verificar que abre**

Run: `C:\ia\piper-studio\env\python.exe -c "import wx, studio.app; print('import OK')"`
Expected: `import OK` (sin errores de import).
Luego lanzar `Piper Studio.bat` a mano: la ventana abre con 3 pestañas y NVDA dice "Piper Studio abierto". (Verificación visual/auditiva; cerrar la ventana.)

---

### Task 4: Sección Entrenar (fine-tune) — controles + lista de corridas + start/pause/resume/stop

**Files:**
- Create: `studio/section_train.py`
- Modify: `studio/app.py` (usar la sección real en vez del placeholder "Entrenar")

**Interfaces:**
- Consumes: `studio.runs` (RunState, list_runs, launch, pause, resume, save_run, latest_epoch, pid_alive), `NVDAController`.
- Produces: `TrainPanel(wx.Panel)` con: elegir dataset (DirDialog), campo **Épocas**, checkbox **Parar automático**, campos paciencia/cada (habilitados si auto), botones **Entrenar/Pausar/Reanudar/Detener**, **lista de corridas** (ListBox), botón **¿Cómo va?**.

- [ ] **Step 1: Implementar `TrainPanel`**

```python
# studio/section_train.py
from __future__ import annotations
import datetime as dt
from pathlib import Path

import wx

from studio import runs

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / "env" / "python.exe")
TRAIN_ROOT = ROOT / "training"
DEFAULT_BASE = ROOT / "base_ckpt" / "silvio_base_clean.ckpt"


class TrainPanel(wx.Panel):
    def __init__(self, parent, nvda):
        super().__init__(parent)
        self.nvda = nvda
        self.dataset = ""
        self._build()
        self.refresh_runs()

    def _build(self):
        s = wx.BoxSizer(wx.VERTICAL)
        self.status = wx.StaticText(self, label="Listo.")
        f = self.status.GetFont(); f.MakeBold(); self.status.SetFont(f)
        s.Add(self.status, 0, wx.ALL | wx.EXPAND, 6)

        row = wx.BoxSizer(wx.HORIZONTAL)
        self.name_ctrl = wx.TextCtrl(self, value="mivoz", name="Nombre de la voz")
        self.ds_btn = wx.Button(self, label="Elegir &dataset…")
        row.Add(wx.StaticText(self, label="Voz:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row.Add(self.name_ctrl, 1, wx.ALL, 4)
        row.Add(self.ds_btn, 0, wx.ALL, 4)
        s.Add(row, 0, wx.EXPAND)

        row2 = wx.BoxSizer(wx.HORIZONTAL)
        self.epochs = wx.SpinCtrl(self, min=1, max=100000, initial=800, name="Épocas")
        self.auto = wx.CheckBox(self, label="&Parar automático (cuando deje de mejorar)")
        self.auto.SetValue(True)
        row2.Add(wx.StaticText(self, label="Épocas:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row2.Add(self.epochs, 0, wx.ALL, 4)
        row2.Add(self.auto, 0, wx.ALL, 4)
        s.Add(row2, 0, wx.EXPAND)

        row3 = wx.BoxSizer(wx.HORIZONTAL)
        self.paciencia = wx.SpinCtrl(self, min=1, max=200, initial=12, name="Paciencia")
        self.cada = wx.SpinCtrl(self, min=1, max=100, initial=10, name="Validar cada")
        row3.Add(wx.StaticText(self, label="Paciencia:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row3.Add(self.paciencia, 0, wx.ALL, 4)
        row3.Add(wx.StaticText(self, label="Validar cada:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row3.Add(self.cada, 0, wx.ALL, 4)
        s.Add(row3, 0, wx.EXPAND)
        self.auto.Bind(wx.EVT_CHECKBOX, self._toggle_auto)

        btns = wx.BoxSizer(wx.HORIZONTAL)
        self.start_btn = wx.Button(self, label="&Entrenar")
        self.pause_btn = wx.Button(self, label="Pausar")
        self.resume_btn = wx.Button(self, label="Reanudar")
        self.stop_btn = wx.Button(self, label="Detener")
        for b in (self.start_btn, self.pause_btn, self.resume_btn, self.stop_btn):
            btns.Add(b, 0, wx.ALL, 4)
        s.Add(btns, 0, wx.ALL, 4)

        s.Add(wx.StaticText(self, label="Corridas:"), 0, wx.LEFT | wx.TOP, 6)
        self.runs_list = wx.ListBox(self, size=(-1, 160), name="Corridas")
        s.Add(self.runs_list, 1, wx.ALL | wx.EXPAND, 6)
        self.howto_btn = wx.Button(self, label="¿&Cómo va?")
        s.Add(self.howto_btn, 0, wx.ALL, 4)
        self.SetSizer(s)

        self.ds_btn.Bind(wx.EVT_BUTTON, self._pick_dataset)
        self.start_btn.Bind(wx.EVT_BUTTON, self._on_start)
        self.pause_btn.Bind(wx.EVT_BUTTON, self._on_pause)
        self.resume_btn.Bind(wx.EVT_BUTTON, self._on_resume)
        self.stop_btn.Bind(wx.EVT_BUTTON, self._on_stop)
        self.howto_btn.Bind(wx.EVT_BUTTON, self._on_howto)

    def _toggle_auto(self, e):
        on = self.auto.GetValue()
        self.paciencia.Enable(on); self.cada.Enable(on)

    def set_status(self, t):
        self.status.SetLabel(str(t)); self.nvda.speak(str(t), True)

    def _pick_dataset(self, e):
        with wx.DirDialog(self, "Elegí la carpeta del dataset (con metadata.csv y wavs)") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.dataset = dlg.GetPath()
                self.set_status(f"Dataset: {Path(self.dataset).name}")

    def _current_state(self) -> runs.RunState:
        return runs.RunState(
            nombre=self.name_ctrl.GetValue().strip() or "mivoz",
            modo="finetune",
            dataset=self.dataset or str(ROOT / "datasets" / (self.name_ctrl.GetValue().strip() or "mivoz")),
            base_ckpt=str(DEFAULT_BASE),
            max_epochs=self.epochs.GetValue(),
            auto_stop=self.auto.GetValue(),
            paciencia=self.paciencia.GetValue(),
            cada=self.cada.GetValue(),
            started_at=dt.datetime.now().isoformat(timespec="seconds"),
        )

    def _selected_run(self) -> runs.RunState | None:
        i = self.runs_list.GetSelection()
        if i == wx.NOT_FOUND:
            return None
        return self._runs[i]

    def _on_start(self, e):
        st = self._current_state()
        if not (Path(st.dataset) / "metadata.csv").exists():
            self.set_status("No encuentro metadata.csv en el dataset."); return
        runs.launch(TRAIN_ROOT, ROOT, st, PY)
        self.set_status(f"Entrenando «{st.nombre}».")
        self.refresh_runs()

    def _on_pause(self, e):
        st = self._selected_run()
        if not st:
            self.set_status("Elegí una corrida de la lista."); return
        runs.pause(TRAIN_ROOT, st); self.set_status(f"Pausada «{st.nombre}».")
        self.refresh_runs()

    def _on_resume(self, e):
        st = self._selected_run()
        if not st:
            self.set_status("Elegí una corrida de la lista."); return
        st.resume_ckpt = None
        runs.resume(TRAIN_ROOT, ROOT, st, PY)
        self.set_status(f"Reanudada «{st.nombre}».")
        self.refresh_runs()

    def _on_stop(self, e):
        st = self._selected_run()
        if not st:
            self.set_status("Elegí una corrida de la lista."); return
        runs.pause(TRAIN_ROOT, st)
        st.estado = "terminado"; runs.save_run(TRAIN_ROOT, st)
        self.set_status(f"Detenida «{st.nombre}»."); self.refresh_runs()

    def _describe(self, st: runs.RunState) -> str:
        ep = runs.latest_epoch(runs.run_dir(TRAIN_ROOT, st.nombre))
        ep_s = f"época {ep}" if ep is not None else "sin checkpoints"
        return f"{st.nombre} — {st.estado} — {ep_s}"

    def refresh_runs(self):
        self._runs = runs.list_runs(TRAIN_ROOT)
        self.runs_list.Set([self._describe(s) for s in self._runs])

    def _on_howto(self, e):
        self.refresh_runs()
        st = self._selected_run() or (self._runs[0] if self._runs else None)
        if not st:
            self.set_status("No hay corridas."); return
        self.set_status(self._describe(st))
```

- [ ] **Step 2: Enganchar la sección real en `app.py`**

Reemplazar en `studio/app.py` el `self._add_placeholder("Entrenar")` por:

```python
from studio.section_train import TrainPanel  # arriba con los imports
# ...
self.nb.AddPage(TrainPanel(self.nb, self.nvda), "Entrenar")
```

- [ ] **Step 3: Verificar import + smoke**

Run: `C:\ia\piper-studio\env\python.exe -c "import studio.section_train, studio.app; print('OK')"`
Expected: `OK`.
Lanzar `Piper Studio.bat`: la pestaña Entrenar muestra los controles; el checkbox habilita/deshabilita paciencia/cada; la lista de corridas muestra las existentes (silvio). (Verificación auditiva NVDA + cerrar.)

- [ ] **Step 4: Prueba funcional corta de lanzar/re-enganche**

Con un dataset chico existente (silvio), poner Épocas=5, **destildar** "Parar automático", Entrenar. Cerrar la GUI. Reabrir → la corrida silvio aparece "entrenando" o "terminado" con su época. Confirma el desprendido + re-enganche real.

---

### Task 5: Avisos automáticos NVDA (watcher de hitos)

**Files:**
- Modify: `studio/section_train.py`

**Interfaces:**
- Consumes: `wx.Timer`, `runs.list_runs`, `runs.latest_epoch`, `runs.pid_alive`.
- Produces: en `TrainPanel`, un `wx.Timer` que cada ~15 s revisa las corridas y **habla por NVDA** los hitos: cruce de cada N épocas (default 100) y cambios de estado (terminó / se pausó / falló).

- [ ] **Step 1: Añadir el watcher al `TrainPanel.__init__`**

```python
# en TrainPanel.__init__, después de refresh_runs():
        self._seen = {}   # nombre -> (estado, hito_epoca)
        self._every = 100
        self._timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._tick, self._timer)
        self._timer.Start(15000)
```

- [ ] **Step 2: Implementar `_tick`**

```python
    def _tick(self, e):
        changed = False
        for st in runs.list_runs(TRAIN_ROOT):
            ep = runs.latest_epoch(runs.run_dir(TRAIN_ROOT, st.nombre)) or 0
            hito = ep // self._every
            prev_estado, prev_hito = self._seen.get(st.nombre, (None, None))
            if prev_estado is None:
                self._seen[st.nombre] = (st.estado, hito); continue
            if st.estado != prev_estado:
                msg = {"terminado": "terminó", "pausado": "se pausó",
                       "fallo": "falló", "entrenando": "entrenando"}.get(st.estado, st.estado)
                self.nvda.speak(f"{st.nombre}: {msg}", False); changed = True
            elif hito != prev_hito and st.estado == "entrenando":
                self.nvda.speak(f"{st.nombre}: época {hito * self._every}", False)
            self._seen[st.nombre] = (st.estado, hito)
        if changed:
            self.refresh_runs()
```

- [ ] **Step 3: Verificar**

Lanzar una corrida corta (Épocas=250, manual) y confirmar que NVDA anuncia "época 100", "época 200" y luego "terminó", sin intervención. (Verificación auditiva.)

- [ ] **Step 4: Detener el timer al cerrar** — agregar en `app.py` `StudioFrame` un `EVT_CLOSE` que pare timers y `Destroy()`; o en `TrainPanel` bindear `wx.EVT_WINDOW_DESTROY` para `self._timer.Stop()`.

```python
# en TrainPanel._build (al final):
        self.Bind(wx.EVT_WINDOW_DESTROY, lambda e: (self._timer.Stop(), e.Skip()))
```

---

### Task 6: Sección Comparar por oído

**Files:**
- Create: `studio/section_compare.py`
- Modify: `studio/app.py`

**Interfaces:**
- Consumes: `comparar_checkpoints.py` (subprocess), `NVDAController`.
- Produces: `ComparePanel(wx.Panel)`: elegir corrida (o carpeta `ckpts`), campo de frase, botón **Generar WAVs**, botón **Abrir carpeta de comparación**. Corre en hilo, avisa al terminar.

- [ ] **Step 1: Implementar `ComparePanel`** (patrón de `gui_dataset.py`: hilo + `wx.CallAfter`)

```python
# studio/section_compare.py
from __future__ import annotations
import subprocess, threading
from pathlib import Path
import wx

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / "env" / "python.exe")


class ComparePanel(wx.Panel):
    def __init__(self, parent, nvda):
        super().__init__(parent); self.nvda = nvda; self._build()

    def _build(self):
        s = wx.BoxSizer(wx.VERTICAL)
        self.status = wx.StaticText(self, label="Elegí una voz y generá los WAVs para comparar.")
        s.Add(self.status, 0, wx.ALL, 6)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.voz = wx.TextCtrl(self, value="silvio", name="Voz")
        row.Add(wx.StaticText(self, label="Voz:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row.Add(self.voz, 1, wx.ALL, 4)
        s.Add(row, 0, wx.EXPAND)
        self.text = wx.TextCtrl(self, value="Hola, esta es una prueba de la voz.",
                                style=wx.TE_MULTILINE, size=(-1, 60), name="Frase")
        s.Add(self.text, 0, wx.ALL | wx.EXPAND, 6)
        self.gen_btn = wx.Button(self, label="&Generar WAVs")
        self.open_btn = wx.Button(self, label="Abrir carpeta")
        s.Add(self.gen_btn, 0, wx.ALL, 4); s.Add(self.open_btn, 0, wx.ALL, 4)
        self.SetSizer(s)
        self.gen_btn.Bind(wx.EVT_BUTTON, self._on_gen)
        self.open_btn.Bind(wx.EVT_BUTTON, self._on_open)

    def _out(self, voz): return ROOT / "training" / voz / "comparar"

    def _on_gen(self, e):
        voz = self.voz.GetValue().strip()
        ck = ROOT / "training" / voz / "ckpts"
        cfg = ROOT / "datasets" / voz / "config.json"
        if not ck.is_dir() or not cfg.exists():
            self.status.SetLabel("No encuentro ckpts o config.json de esa voz."); return
        self.status.SetLabel("Generando WAVs…"); self.nvda.speak("Generando WAVs", True)
        argv = [PY, str(ROOT / "comparar_checkpoints.py"),
                "--ckpts", str(ck), "--config", str(cfg),
                "--out", str(self._out(voz)), "--text", self.text.GetValue()]
        threading.Thread(target=self._worker, args=(argv, voz), daemon=True).start()

    def _worker(self, argv, voz):
        subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
        wx.CallAfter(self._done, voz)

    def _done(self, voz):
        self.status.SetLabel(f"Listo. WAVs en {self._out(voz)}")
        self.nvda.speak("WAVs listos para comparar", True)

    def _on_open(self, e):
        import os
        try:
            os.startfile(str(self._out(self.voz.GetValue().strip())))  # noqa: S606
        except Exception:
            pass
```

- [ ] **Step 2: Enganchar en `app.py`** (reemplazar placeholder "Comparar"):

```python
from studio.section_compare import ComparePanel
self.nb.AddPage(ComparePanel(self.nb, self.nvda), "Comparar")
```

- [ ] **Step 3: Verificar** — import OK, y en la GUI generar WAVs de silvio; confirmar que aparecen en `training/silvio/comparar` y NVDA avisa.

---

### Task 7: Sección Exportar / Instalar

**Files:**
- Create: `studio/section_export.py`
- Modify: `studio/app.py`

**Interfaces:**
- Consumes: `export_run.py` (subprocess), `datasets/<voz>/config.json`, carpeta del reproductor.
- Produces: `ExportPanel(wx.Panel)`: elegir voz + checkpoint (last/best/época), botón **Exportar a ONNX**, botón **Instalar en el reproductor** (copia `.onnx` + `.onnx.json` a `C:\ia\modelos pc\piper\voces\<voz>\`).

- [ ] **Step 1: Implementar `ExportPanel`**

```python
# studio/section_export.py
from __future__ import annotations
import shutil, subprocess, threading
from pathlib import Path
import wx

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / "env" / "python.exe")
PLAYER_VOICES = Path(r"C:\ia\modelos pc\piper\voces")


class ExportPanel(wx.Panel):
    def __init__(self, parent, nvda):
        super().__init__(parent); self.nvda = nvda; self._build()

    def _build(self):
        s = wx.BoxSizer(wx.VERTICAL)
        self.status = wx.StaticText(self, label="Exportá un checkpoint a ONNX e instalalo.")
        s.Add(self.status, 0, wx.ALL, 6)
        row = wx.BoxSizer(wx.HORIZONTAL)
        self.voz = wx.TextCtrl(self, value="silvio", name="Voz")
        self.ckpt_btn = wx.Button(self, label="Elegir &checkpoint…")
        row.Add(wx.StaticText(self, label="Voz:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row.Add(self.voz, 1, wx.ALL, 4); row.Add(self.ckpt_btn, 0, wx.ALL, 4)
        s.Add(row, 0, wx.EXPAND)
        self.ckpt = ""
        self.exp_btn = wx.Button(self, label="&Exportar a ONNX")
        self.inst_btn = wx.Button(self, label="&Instalar en el reproductor")
        s.Add(self.exp_btn, 0, wx.ALL, 4); s.Add(self.inst_btn, 0, wx.ALL, 4)
        self.SetSizer(s)
        self.ckpt_btn.Bind(wx.EVT_BUTTON, self._pick)
        self.exp_btn.Bind(wx.EVT_BUTTON, self._on_export)
        self.inst_btn.Bind(wx.EVT_BUTTON, self._on_install)

    def _pick(self, e):
        voz = self.voz.GetValue().strip()
        with wx.FileDialog(self, "Elegí el .ckpt", defaultDir=str(ROOT / "training" / voz / "ckpts"),
                           wildcard="Checkpoint|*.ckpt",
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.ckpt = dlg.GetPath(); self.status.SetLabel(f"Checkpoint: {Path(self.ckpt).name}")

    def _onnx_path(self, voz): return ROOT / "training" / voz / f"{voz}.onnx"

    def _on_export(self, e):
        voz = self.voz.GetValue().strip()
        if not self.ckpt:
            self.status.SetLabel("Elegí un checkpoint."); return
        onnx = self._onnx_path(voz)
        self.status.SetLabel("Exportando…"); self.nvda.speak("Exportando a ONNX", True)
        argv = [PY, str(ROOT / "export_run.py"), "--checkpoint", self.ckpt, "--output-file", str(onnx)]
        threading.Thread(target=self._worker, args=(argv, voz), daemon=True).start()

    def _worker(self, argv, voz):
        r = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True)
        wx.CallAfter(self._exported, voz, r.returncode)

    def _exported(self, voz, code):
        if code == 0:
            cfg = ROOT / "datasets" / voz / "config.json"
            shutil.copyfile(cfg, self._onnx_path(voz).with_suffix(".onnx.json"))
            self.status.SetLabel("Exportado. Ya podés instalar.")
            self.nvda.speak("Exportado", True)
        else:
            self.status.SetLabel("Error exportando (ver studio.log).")

    def _on_install(self, e):
        voz = self.voz.GetValue().strip()
        onnx = self._onnx_path(voz)
        if not onnx.exists():
            self.status.SetLabel("Primero exportá."); return
        dst = PLAYER_VOICES / voz
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(onnx, dst / f"{voz}.onnx")
        shutil.copyfile(onnx.with_suffix(".onnx.json"), dst / f"{voz}.onnx.json")
        self.status.SetLabel(f"Instalada en el reproductor: {dst}")
        self.nvda.speak("Voz instalada en el reproductor", True)
```

- [ ] **Step 2: Enganchar en `app.py`** (reemplazar placeholder "Exportar").

```python
from studio.section_export import ExportPanel
self.nb.AddPage(ExportPanel(self.nb, self.nvda), "Exportar")
```

- [ ] **Step 3: Verificar** — exportar `silvio-ep719.ckpt` a ONNX desde la GUI e instalar; confirmar que `C:\ia\modelos pc\piper\voces\silvio\silvio.onnx` se actualiza y NVDA avisa.

---

### Task 8: Integración final + smoke test

**Files:**
- Modify: `studio/app.py` (asegurar cierre limpio: parar timers/procesos hijos NO — los hijos son desprendidos y deben seguir).

- [ ] **Step 1:** Confirmar los 4 imports: `python -c "import studio.app, studio.section_train, studio.section_compare, studio.section_export; print('OK')"`.
- [ ] **Step 2:** Correr toda la suite: `env\python.exe -m unittest discover studio/tests -v` → PASS.
- [ ] **Step 3:** Lanzar `Piper Studio.bat`, recorrer las 3 pestañas con teclado confirmando que NVDA lee cada control.
- [ ] **Step 4:** Flujo end-to-end: Entrenar (5 épocas, manual) → cerrar GUI → reabrir (re-enganche) → Comparar → Exportar → Instalar. Verificar avisos NVDA en cada hito.

---

## Auto-revisión del plan

- **Cobertura de la spec**: cáscara ✓ (T3), desprendido+re-enganche ✓ (T1/T2/T4), parar-auto vs épocas-manual ✓ (T2/T4), avisos NVDA ✓ (T5), Comparar ✓ (T6), Exportar/Instalar ✓ (T7). **Multi-hablante queda para el Plan 2** (decomposición declarada).
- **Sin placeholders**: cada paso tiene código real o comando con salida esperada.
- **Consistencia de tipos**: `RunState` y las firmas de `runs.py` (T1/T2) se usan igual en `section_train.py` (T4/T5).
- **Riesgo abierto**: PID reuse (bajo; se puede endurecer luego guardando el nombre del proceso). Detached en Windows validado por la prueba funcional de T4.

## Handoff de ejecución

Al terminar de aprobar el plan, elegir modo de ejecución (subagente por tarea vs inline).
