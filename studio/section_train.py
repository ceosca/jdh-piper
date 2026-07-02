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
        self._seen = {}   # nombre -> (estado, hito_epoca)
        self._every = 100
        self._timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._tick, self._timer)
        self._timer.Start(15000)

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

        self.Bind(wx.EVT_WINDOW_DESTROY, lambda e: (self._timer.Stop(), e.Skip()))

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
        st = runs.resume(TRAIN_ROOT, ROOT, st, PY)
        self.set_status(f"Reanudada «{st.nombre}» desde {Path(st.resume_ckpt).name}.")
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

    def _tick(self, e):
        try:
            changed = False
            for st in runs.list_runs(TRAIN_ROOT):
                try:
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
                except Exception:
                    continue
            if changed:
                self.refresh_runs()
        except Exception:
            pass

    def _on_howto(self, e):
        self.refresh_runs()
        st = self._selected_run() or (self._runs[0] if self._runs else None)
        if not st:
            self.set_status("No hay corridas."); return
        self.set_status(self._describe(st))
