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
