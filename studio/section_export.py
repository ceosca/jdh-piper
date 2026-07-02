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
        # NVDA usa como etiqueta el StaticText creado JUSTO ANTES del control
        # (orden de creación). La etiqueta se instancia antes que su campo.
        row = wx.BoxSizer(wx.HORIZONTAL)
        lbl_voz = wx.StaticText(self, label="Voz:")
        self.voz = wx.TextCtrl(self, value="silvio", name="Voz")
        self.ckpt_btn = wx.Button(self, label="Elegir &checkpoint…")
        row.Add(lbl_voz, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
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
            try:
                shutil.copyfile(cfg, self._onnx_path(voz).with_suffix(".onnx.json"))
                self.status.SetLabel("Exportado. Ya podés instalar.")
                self.nvda.speak("Exportado", True)
            except (FileNotFoundError, OSError):
                self.status.SetLabel("Exportado, pero falta config.json del dataset.")
                self.nvda.speak("Exportado, pero falta el config del dataset", True)
        else:
            self.status.SetLabel("Error exportando (ver studio.log).")
            self.nvda.speak("Error exportando", True)

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
