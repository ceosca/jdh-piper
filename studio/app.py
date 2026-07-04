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
from studio.section_train import TrainPanel  # noqa: E402
from studio.section_compare import ComparePanel  # noqa: E402
from studio.section_export import ExportPanel  # noqa: E402
from studio.section_player import PlayerPanel  # noqa: E402


class StudioFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Piper Studio", size=(820, 620))
        self.nvda = NVDAController()
        self.nb = wx.Notebook(self)
        # Las secciones reales se agregan en tareas siguientes:
        self.nb.AddPage(TrainPanel(self.nb, self.nvda), "Entrenar")
        self.nb.AddPage(ComparePanel(self.nb, self.nvda), "Comparar")
        self.nb.AddPage(ExportPanel(self.nb, self.nvda), "Exportar")
        self.nb.AddPage(PlayerPanel(self.nb, self.nvda), "Reproductor")
        self.Centre(); self.Show()
        self.nvda.speak("Piper Studio abierto", True)


def main():
    app = wx.App(False)
    StudioFrame()
    app.MainLoop()


if __name__ == "__main__":
    main()
