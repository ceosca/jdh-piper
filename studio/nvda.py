# studio/nvda.py
import ctypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def app_is_foreground() -> bool:
    """True solo si la ventana en primer plano pertenece a ESTE proceso.

    El cliente de NVDA habla global: sin este chequeo, la app sigue hablando (y con
    interrupt corta a NVDA en la app a la que te cambiaste) aunque no tenga el foco.
    Ante cualquier error o fuera de Windows, devuelve True (no silencia de más)."""
    try:
        u = ctypes.windll.user32
        pid = ctypes.c_ulong()
        u.GetWindowThreadProcessId(u.GetForegroundWindow(), ctypes.byref(pid))
        return pid.value == ctypes.windll.kernel32.GetCurrentProcessId()
    except Exception:
        return True


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
        if not app_is_foreground():   # no hablar (ni interrumpir) si no tenés el foco
            return
        try:
            if interrupt and hasattr(self.dll, "nvdaController_cancelSpeech"):
                self.dll.nvdaController_cancelSpeech()
            self.dll.nvdaController_speakText(ctypes.c_wchar_p(str(text)))
        except Exception:
            pass
