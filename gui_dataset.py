"""
GUI accesible del armador de dataset (Fase 2 de Piper Studio).

Elegís uno o más audios (archivos y/o carpetas), un nombre de dataset y el modelo
de whisper. "Armar dataset" corre todo solo (silencios -> clips -> transcripción)
y deja un dataset LJSpeech en datasets/<nombre>/. Accesible con NVDA.
"""
from __future__ import annotations

import ctypes
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
try:
    _log = open(ROOT / "gui_dataset.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = _log
    sys.stderr = _log
except Exception:
    pass

import wx  # noqa: E402

from dataset_builder import build_dataset, build_multispeaker_dataset  # noqa: E402

DATASETS_DIR = ROOT / "datasets"

# Presets de tamaño de clip (etiqueta, perillas de segmentación). "Muy chiquitos"
# primero = default: más clips y más cortos, cortando en más silencios.
CLIP_PRESETS = [
    ("Muy chiquitos (2 a 4 s)", dict(min_sil=0.25, min_clip=1.5, max_clip=5.0)),
    ("Chicos (3 a 6 s)", dict(min_sil=0.30, min_clip=1.5, max_clip=7.0)),
    ("Medianos (5 a 10 s)", dict(min_sil=0.35, min_clip=2.0, max_clip=10.0)),
    ("Largos (hasta 15 s)", dict(min_sil=0.40, min_clip=2.0, max_clip=15.0)),
]


def _app_is_foreground() -> bool:
    """True solo si la ventana en primer plano es de ESTE proceso. Sin esto, el
    lector sigue hablando (e interrumpiendo otras apps) aunque cambies de ventana."""
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
        if not _app_is_foreground():   # no hablar (ni interrumpir) si no tenés el foco
            return
        try:
            if interrupt and hasattr(self.dll, "nvdaController_cancelSpeech"):
                self.dll.nvdaController_cancelSpeech()
            self.dll.nvdaController_speakText(ctypes.c_wchar_p(str(text)))
        except Exception:
            pass


class DatasetFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Piper Studio — Armar dataset", size=(760, 560))
        self.nvda = NVDAController()
        self.inputs: list[str] = []
        self._busy = False
        self._stop = threading.Event()
        DATASETS_DIR.mkdir(exist_ok=True)
        self._build_ui()
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.Centre(); self.Show()

    def _build_ui(self):
        p = wx.Panel(self); s = wx.BoxSizer(wx.VERTICAL)
        self.status = wx.StaticText(p, label="Listo. Agregá audios y armá el dataset.")
        f = self.status.GetFont(); f.MakeBold(); self.status.SetFont(f)
        s.Add(self.status, 0, wx.ALL | wx.EXPAND, 6)

        s.Add(wx.StaticText(p, label="Audios de entrada (archivos y/o carpetas):"), 0, wx.ALL, 4)
        self.inp_list = wx.ListBox(p, size=(-1, 130), name="Audios de entrada")
        s.Add(self.inp_list, 1, wx.ALL | wx.EXPAND, 4)
        r = wx.BoxSizer(wx.HORIZONTAL)
        self.add_file_btn = wx.Button(p, label="Agregar archivo…")
        self.add_dir_btn = wx.Button(p, label="Agregar carpeta…")
        self.rm_btn = wx.Button(p, label="Quitar seleccionado")
        for b in (self.add_file_btn, self.add_dir_btn, self.rm_btn):
            r.Add(b, 0, wx.ALL, 4)
        s.Add(r, 0, wx.ALL, 2)

        # Modo: una voz (LJSpeech id|text) o multi-hablante para el base
        # (cada carpeta/archivo agregado = un hablante; CSV id|speaker|text).
        r_modo = wx.BoxSizer(wx.HORIZONTAL)
        modo_lbl = wx.StaticText(p, label="Modo:")  # etiqueta ANTES del control (NVDA)
        self.modo = wx.Choice(p, choices=["Una voz", "Multi-hablante (base)"], name="Modo del dataset")
        self.modo.SetSelection(0)
        r_modo.Add(modo_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        r_modo.Add(self.modo, 0, wx.ALL, 4)
        s.Add(r_modo, 0, wx.ALL, 2)

        # Tamaño de los clips (etiqueta ANTES del control, por NVDA).
        r_clip = wx.BoxSizer(wx.HORIZONTAL)
        clip_lbl = wx.StaticText(p, label="Tamaño de los clips (audios largos):")
        self.clip_choice = wx.Choice(p, choices=[c[0] for c in CLIP_PRESETS],
                                     name="Tamaño de los clips")
        self.clip_choice.SetSelection(0)
        r_clip.Add(clip_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        r_clip.Add(self.clip_choice, 0, wx.ALL, 4)
        s.Add(r_clip, 0, wx.ALL, 2)

        r2 = wx.BoxSizer(wx.HORIZONTAL)
        self.name_ctrl = wx.TextCtrl(p, value="mivoz", name="Nombre del dataset")
        self.model_choice = wx.Choice(p, choices=["large-v3", "medium", "small"], name="Modelo whisper")
        self.model_choice.SetSelection(0)
        r2.Add(wx.StaticText(p, label="Nombre del dataset:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        r2.Add(self.name_ctrl, 1, wx.ALL, 4)
        r2.Add(wx.StaticText(p, label="Whisper:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        r2.Add(self.model_choice, 0, wx.ALL, 4)
        s.Add(r2, 0, wx.ALL | wx.EXPAND, 4)

        r3 = wx.BoxSizer(wx.HORIZONTAL)
        self.build_btn = wx.Button(p, label="&Armar dataset")
        self.stop_btn = wx.Button(p, label="Cancelar")
        self.open_btn = wx.Button(p, label="Abrir carpeta de datasets")
        self.stop_btn.Enable(False)
        for b in (self.build_btn, self.stop_btn, self.open_btn):
            r3.Add(b, 0, wx.ALL, 4)
        s.Add(r3, 0, wx.ALL, 4)
        p.SetSizer(s)

        for c, l in [(self.inp_list, "Audios"), (self.modo, "Modo"),
                     (self.clip_choice, "Tamaño de los clips"),
                     (self.name_ctrl, "Nombre del dataset"),
                     (self.model_choice, "Modelo whisper")]:
            c.Bind(wx.EVT_SET_FOCUS, lambda e, c=c, l=l: self._focus(e, c, l))
        self.modo.Bind(wx.EVT_CHOICE, self._on_modo)
        self.add_file_btn.Bind(wx.EVT_BUTTON, self._on_add_file)
        self.add_dir_btn.Bind(wx.EVT_BUTTON, self._on_add_dir)
        self.rm_btn.Bind(wx.EVT_BUTTON, self._on_remove)
        self.build_btn.Bind(wx.EVT_BUTTON, self._on_build)
        self.stop_btn.Bind(wx.EVT_BUTTON, lambda e: self._stop.set())
        self.open_btn.Bind(wx.EVT_BUTTON, self._on_open)

    def _focus(self, e, c, l):
        v = c.GetStringSelection() if isinstance(c, (wx.ListBox, wx.Choice)) else c.GetValue()
        self.nvda.speak(f"{l} {v}".strip(), True); e.Skip()

    def set_status(self, t):
        self.status.SetLabel(str(t)); self.nvda.speak(str(t), True)

    def set_status_ts(self, t):
        wx.CallAfter(self.set_status, t)

    def _on_modo(self, e):
        if self.modo.GetSelection() == 1:
            self.set_status("Modo multi-hablante: cada carpeta/archivo que agregues es un hablante.")
        else:
            self.set_status("Modo una voz: todos los audios son de la misma voz.")

    def _refresh_inputs(self):
        self.inp_list.Set([Path(i).name + ("  [carpeta]" if Path(i).is_dir() else "") for i in self.inputs])

    def _on_add_file(self, e):
        with wx.FileDialog(self, "Elegí audio(s)", wildcard="Audio|*.wav;*.mp3;*.flac;*.ogg;*.m4a;*.opus",
                           style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.inputs += dlg.GetPaths(); self._refresh_inputs()
                self.set_status(f"{len(self.inputs)} entradas.")

    def _on_add_dir(self, e):
        with wx.DirDialog(self, "Elegí una carpeta de audios") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.inputs.append(dlg.GetPath()); self._refresh_inputs()
                self.set_status(f"{len(self.inputs)} entradas.")

    def _on_remove(self, e):
        i = self.inp_list.GetSelection()
        if i != wx.NOT_FOUND:
            del self.inputs[i]; self._refresh_inputs()

    def _on_open(self, e):
        try:
            import os
            os.startfile(str(DATASETS_DIR))  # noqa: S606
        except Exception:
            self.set_status(f"Carpeta: {DATASETS_DIR}")

    def _on_build(self, e):
        if self._busy:
            self.set_status("Ya hay un armado en curso."); return
        if not self.inputs:
            self.set_status("Agregá al menos un audio o carpeta."); return
        name = (self.name_ctrl.GetValue().strip() or "mivoz")
        out = str(DATASETS_DIR / name)
        model = self.model_choice.GetStringSelection()
        multi = self.modo.GetSelection() == 1
        clip = CLIP_PRESETS[max(0, self.clip_choice.GetSelection())][1]
        self._busy = True; self._stop.clear()
        self.build_btn.Enable(False); self.stop_btn.Enable(True)
        self.set_status("Armando dataset… (whisper puede tardar)")
        threading.Thread(target=self._worker, args=(list(self.inputs), out, model, multi, clip), daemon=True).start()

    def _worker(self, inputs, out, model, multi, clip):
        try:
            if multi:
                # Cada entrada = un hablante (carpeta -> su nombre; archivo -> su stem).
                speakers: dict[str, list[str]] = {}
                for inp in inputs:
                    pth = Path(inp)
                    key = pth.name if pth.is_dir() else pth.stem
                    speakers.setdefault(key, []).append(inp)
                n = build_multispeaker_dataset(speakers, out, model_size=model,
                                               progress=self.set_status_ts, stop_flag=self._stop, **clip)
                wx.CallAfter(self._done_multi, out, n)
            else:
                build_dataset(inputs, out, model_size=model, progress=self.set_status_ts, stop_flag=self._stop, **clip)
                wx.CallAfter(self._done, out)
        except Exception as ex:
            traceback.print_exc()
            wx.CallAfter(self._done_err, ex)

    def _done(self, out):
        self._busy = False; self.build_btn.Enable(True); self.stop_btn.Enable(False)
        self.set_status(f"Dataset listo en {out}. Revisá metadata.csv y a entrenar.")

    def _done_multi(self, out, n):
        self._busy = False; self.build_btn.Enable(True); self.stop_btn.Enable(False)
        self.set_status(f"Dataset multi-hablante listo en {out}: {n} hablantes. A entrenar el base.")

    def _done_err(self, ex):
        self._busy = False; self.build_btn.Enable(True); self.stop_btn.Enable(False)
        self.set_status(f"Error: {ex}")

    def _on_close(self, e):
        self._stop.set(); self.Destroy()


def main():
    app = wx.App(False)
    DatasetFrame()
    app.MainLoop()


if __name__ == "__main__":
    main()
