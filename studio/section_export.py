# studio/section_export.py
from __future__ import annotations
import os
import shutil, subprocess, threading
from pathlib import Path
import wx

from studio import runs, sherpa_export

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / "env" / "python.exe")
# Carpeta de voces del reproductor CPU. Configurable por env para portabilidad.
PLAYER_VOICES = Path(os.environ.get("PIPER_PLAYER_VOICES", r"C:\ia\modelos pc\piper\voces"))
# Carpeta de voces del add-on Piper/Sonata para NVDA.
NVDA_VOICES = Path(os.environ.get("APPDATA", "")) / "nvda" / "piper" / "voices" / "v1.0"
# Códigos de dialecto para el nombre de la voz en NVDA (solo etiqueta; la
# fonemización real ya está horneada en el .onnx). Ampliable a gusto.
REGIONES = ["es", "es_419", "es_AR", "es_MX", "es_CL", "es_CO", "es_PE", "es_VE",
            "es_EC", "es_CU", "es_UY", "es_BO", "es_PY", "es_CR", "es_GT", "es_DO",
            "es_PA", "es_HN", "es_NI", "es_SV", "es_PR", "es_ES"]


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
        # Región para el nombre de la voz en NVDA (etiqueta antes del control).
        row_reg = wx.BoxSizer(wx.HORIZONTAL)
        reg_lbl = wx.StaticText(self, label="Región (para NVDA):")
        self.region = wx.Choice(self, choices=REGIONES, name="Región de la voz")
        self.region.SetSelection(0)
        row_reg.Add(reg_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row_reg.Add(self.region, 0, wx.ALL, 4)
        s.Add(row_reg, 0, wx.EXPAND)

        self.exp_btn = wx.Button(self, label="&Exportar a ONNX")
        self.inst_btn = wx.Button(self, label="&Instalar en el reproductor")
        self.nvda_btn = wx.Button(self, label="Instalar para &NVDA (Sonata)")
        s.Add(self.exp_btn, 0, wx.ALL, 4)
        s.Add(self.inst_btn, 0, wx.ALL, 4)
        s.Add(self.nvda_btn, 0, wx.ALL, 4)
        self.sherpa_btn = wx.Button(self, label="Exportar para &sherpa-onnx")
        s.Add(self.sherpa_btn, 0, wx.ALL, 4)
        self.SetSizer(s)
        self.ckpt_btn.Bind(wx.EVT_BUTTON, self._pick)
        self.exp_btn.Bind(wx.EVT_BUTTON, self._on_export)
        self.inst_btn.Bind(wx.EVT_BUTTON, self._on_install)
        self.nvda_btn.Bind(wx.EVT_BUTTON, self._on_install_nvda)
        self.sherpa_btn.Bind(wx.EVT_BUTTON, self._on_sherpa)

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
            cfg = runs.config_de_voz(ROOT, voz)  # del dataset del run.json (voz puede != carpeta)
            try:
                if cfg is None:
                    raise FileNotFoundError
                shutil.copyfile(cfg, self._onnx_path(voz).with_suffix(".onnx.json"))
                self.status.SetLabel("Exportado. Ya podés instalar.")
                self.nvda.speak("Exportado", True)
            except (FileNotFoundError, OSError):
                self.status.SetLabel("Exportado, pero no encuentro el config.json del dataset.")
                self.nvda.speak("Exportado, pero falta el config del dataset", True)
        else:
            self.status.SetLabel("Error exportando (ver studio.log).")
            self.nvda.speak("Error exportando", True)

    def _on_install(self, e):
        voz = self.voz.GetValue().strip()
        if not voz:
            self.status.SetLabel("Escribí el nombre de la voz."); return
        onnx = self._onnx_path(voz)
        if not onnx.exists() or not onnx.with_suffix(".onnx.json").exists():
            self.status.SetLabel("Primero exportá a ONNX (falta el .onnx o su .json)."); return
        dst = PLAYER_VOICES / voz
        try:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(onnx, dst / f"{voz}.onnx")
            shutil.copyfile(onnx.with_suffix(".onnx.json"), dst / f"{voz}.onnx.json")
        except OSError as ex:
            self.status.SetLabel(f"No se pudo instalar en el reproductor: {ex}"); return
        self.status.SetLabel(f"Instalada en el reproductor: {dst}")
        self.nvda.speak("Voz instalada en el reproductor", True)

    def _on_install_nvda(self, e):
        voz = self.voz.GetValue().strip()
        if not voz:
            self.status.SetLabel("Escribí el nombre de la voz."); return
        onnx = self._onnx_path(voz)
        if not onnx.exists() or not onnx.with_suffix(".onnx.json").exists():
            self.status.SetLabel("Primero exportá a ONNX."); return
        if not NVDA_VOICES.parent.parent.exists():  # …\nvda\piper
            self.status.SetLabel("No encuentro Piper para NVDA (¿está instalado el add-on?)."); return
        region = self.region.GetStringSelection() or "es"
        vid = f"{region}-{voz}-medium"   # convención de Sonata: idioma-nombre-calidad
        dst = NVDA_VOICES / vid
        try:
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(onnx, dst / f"{vid}.onnx")
            shutil.copyfile(onnx.with_suffix(".onnx.json"), dst / f"{vid}.onnx.json")
        except OSError as ex:
            self.status.SetLabel(f"No se pudo instalar para NVDA: {ex}"); return
        self.status.SetLabel(f"Instalada para NVDA como «{vid}». Reiniciá NVDA y elegí la voz.")
        self.nvda.speak(f"Voz {voz} instalada para NVDA. Reiniciá NVDA para usarla.", True)

    def _on_sherpa(self, e):
        voz = self.voz.GetValue().strip()
        if not voz:
            self.status.SetLabel("Escribí el nombre de la voz."); return
        onnx = self._onnx_path(voz)
        if not onnx.exists() or not onnx.with_suffix(".onnx.json").exists():
            self.status.SetLabel("Primero exportá a ONNX."); return
        espeak = sherpa_export.espeak_data_dir(ROOT / "env")
        if espeak is None:
            self.status.SetLabel("No encuentro espeak-ng-data en el env."); return
        out = ROOT / "sherpa" / voz
        self.status.SetLabel("Empaquetando para sherpa-onnx…")
        self.nvda.speak("Empaquetando para sherpa onnx", True)
        threading.Thread(target=self._sherpa_worker,
                         args=(onnx, onnx.with_suffix(".onnx.json"), out, espeak),
                         daemon=True).start()

    def _sherpa_worker(self, onnx, cfg, out, espeak):
        try:
            sherpa_export.empaquetar(onnx, cfg, out, espeak)
            wx.CallAfter(self._sherpa_done, out, None)
        except Exception as ex:
            wx.CallAfter(self._sherpa_done, out, ex)

    def _sherpa_done(self, out, err):
        if err:
            self.status.SetLabel(f"Error empaquetando: {err}")
            self.nvda.speak("Error empaquetando para sherpa", True)
        else:
            self.status.SetLabel(f"Listo para sherpa-onnx en {out}")
            self.nvda.speak("Voz empaquetada para sherpa onnx", True)
