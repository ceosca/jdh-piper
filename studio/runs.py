# studio/runs.py
"""Estado y control de corridas de entrenamiento (desprendidas, re-enganchables)."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
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
    num_speakers: int = 1
    pid: int | None = None
    started_at: str = ""
    estado: str = "pausado"         # entrenando | pausado | terminado | fallo
    last_event: str = ""


def run_dir(root: Path, nombre: str) -> Path:
    return Path(root) / nombre


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


def leer_epoca(rd: Path) -> int | None:
    """Época en vivo, barata: epoch.txt (lo escribe el entrenamiento cada época) ->
    si no, el nombre de checkpoint más alto (epoch=N) -> None."""
    try:
        p = Path(rd) / "epoch.txt"
        if p.exists():
            return int(p.read_text(encoding="utf-8").strip())
    except Exception:
        pass
    return latest_epoch(rd)


def leer_mejor(rd: Path):
    """(época, val_mel) del mejor punto, desde mejor.txt. None si no hay."""
    try:
        p = Path(rd) / "mejor.txt"
        if p.exists():
            ep, v = p.read_text(encoding="utf-8").split()
            return int(ep), float(v)
    except Exception:
        pass
    return None


def leer_progreso(rd: Path, max_lineas: int = 300) -> list[str]:
    """Últimas líneas de progreso.log (el log en vivo del entrenamiento). [] si no hay."""
    try:
        p = Path(rd) / "progreso.log"
        if p.exists():
            return p.read_text(encoding="utf-8").splitlines()[-max_lineas:]
    except Exception:
        pass
    return []


def config_de_voz(root: Path, voz: str) -> Path | None:
    """config.json de la voz: el del dataset de su run.json (voz puede != carpeta);
    fallback a datasets/<voz>/config.json. None si no se encuentra."""
    root = Path(root)
    rj = root / "training" / voz / "run.json"
    try:
        if rj.exists():
            ds = load_run(rj).dataset
            if ds:
                c = Path(ds) / "config.json"
                if c.exists():
                    return c
    except Exception:
        pass
    c = root / "datasets" / voz / "config.json"
    return c if c.exists() else None


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
        # refrescar estado según liveness real del proceso: solo si estaba
        # "entrenando" y el proceso murió (una pausa/detención explícita se respeta).
        if st.estado == "entrenando" and not pid_alive(st.pid):
            final = run_dir(root, st.nombre) / "estado_final.txt"  # "terminado"|"fallo"
            try:
                st.estado = (final.read_text(encoding="utf-8").strip() or "terminado"
                             if final.exists() else "terminado")
            except Exception:
                st.estado = "terminado"
        out.append(st)
    return out


import datetime as _dt
import subprocess


def build_train_argv(py: str, root_proj: Path, st: RunState) -> list[str]:
    rp = Path(root_proj)
    ds = st.dataset
    if st.modo == "base":
        argv = [py, str(rp / "entrenar_base.py"),
                "--dataset", st.dataset,
                "--base-mono", st.base_ckpt,
                "--voz", st.nombre,
                "--num-speakers", str(st.num_speakers),
                "--max-epochs", str(st.max_epochs)]
        if st.resume_ckpt:
            argv += ["--resume", str(st.resume_ckpt)]
        return argv
    ckpt = st.resume_ckpt or st.base_ckpt
    if st.auto_stop:
        # entrenar.py trae EarlyStopping(val_mel) + checkpoints + best.
        # Pasar --dataset explícito: el nombre de la voz puede no coincidir con
        # la carpeta del dataset (ej. voz "mario", dataset "mario-castanieda").
        argv = [py, str(rp / "entrenar.py"),
                "--voz", st.nombre,
                "--dataset", ds,
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
        ep_txt = (rp / "training" / st.nombre / "epoch.txt").as_posix()
        ep_cb = ('{"class_path":"studio.progress.EscritorEpoca",'
                 '"init_args":{"path":"%s"}}' % ep_txt)
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
                "--trainer.callbacks+", cb,
                "--trainer.callbacks+", ep_cb]
    return argv


def _spawn_detached(argv: list[str], cwd: Path, log_path: Path) -> int:
    with open(log_path, "a", encoding="utf-8", buffering=1) as log:
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
    (rd / "estado_final.txt").unlink(missing_ok=True)  # marca fresca para esta corrida
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


def pick_resume_ckpt(rd: Path, base_ckpt: str) -> str:
    rd = Path(rd)
    last = rd / "ckpts" / "last.ckpt"
    if last.exists():
        return str(last)
    ck = rd / "ckpts"
    best_path = None
    best_epoch = -1
    if ck.is_dir():
        for f in ck.glob("*.ckpt"):
            m = re.search(r"epoch=(\d+)", f.stem)
            if m:
                epoch = int(m.group(1))
                if epoch > best_epoch:
                    best_epoch = epoch
                    best_path = f
    if best_path is not None:
        return str(best_path)
    return base_ckpt


def resume(root_runs: Path, root_proj: Path, st: RunState, py: str) -> RunState:
    st.resume_ckpt = st.resume_ckpt or pick_resume_ckpt(run_dir(root_runs, st.nombre), st.base_ckpt)
    return launch(root_runs, root_proj, st, py)
