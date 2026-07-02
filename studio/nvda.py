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
