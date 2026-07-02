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
        self.open_btn = wx.Button(self, label="Abrir &carpeta")
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
            self.status.SetLabel("Todavía no hay carpeta para abrir. Generá los WAVs primero.")
            self.nvda.speak("Todavía no hay carpeta para abrir", True)
