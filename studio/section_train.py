# studio/section_train.py
from __future__ import annotations
import datetime as dt
import subprocess
import threading
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

        # NVDA toma como etiqueta el StaticText creado JUSTO ANTES del control
        # (orden de creación, no de layout). Por eso cada etiqueta se instancia
        # antes que su campo.
        row = wx.BoxSizer(wx.HORIZONTAL)
        lbl_voz = wx.StaticText(self, label="Voz:")
        self.name_ctrl = wx.TextCtrl(self, value="mivoz", name="Nombre de la voz")
        self.ds_btn = wx.Button(self, label="Elegir &dataset…")
        row.Add(lbl_voz, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row.Add(self.name_ctrl, 1, wx.ALL, 4)
        row.Add(self.ds_btn, 0, wx.ALL, 4)
        s.Add(row, 0, wx.EXPAND)

        row_modo = wx.BoxSizer(wx.HORIZONTAL)
        modo_lbl = wx.StaticText(self, label="Modo:")
        self.modo = wx.Choice(self, choices=["Fine-tune de una voz", "Base multi-hablante"],
                              name="Modo de entrenamiento")
        self.modo.SetSelection(0)
        spk_lbl = wx.StaticText(self, label="Nº de hablantes:")
        self.num_spk = wx.SpinCtrl(self, min=1, max=1000, initial=10, name="Número de hablantes")
        self.num_spk.Enable(False)
        row_modo.Add(modo_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row_modo.Add(self.modo, 0, wx.ALL, 4)
        row_modo.Add(spk_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row_modo.Add(self.num_spk, 0, wx.ALL, 4)
        s.Add(row_modo, 0, wx.EXPAND)
        self.modo.Bind(wx.EVT_CHOICE, self._toggle_modo)

        row2 = wx.BoxSizer(wx.HORIZONTAL)
        lbl_ep = wx.StaticText(self, label="Épocas:")
        self.epochs = wx.SpinCtrl(self, min=1, max=100000, initial=800, name="Épocas")
        self.auto = wx.CheckBox(self, label="&Parar automático (cuando deje de mejorar)")
        self.auto.SetValue(True)
        row2.Add(lbl_ep, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row2.Add(self.epochs, 0, wx.ALL, 4)
        row2.Add(self.auto, 0, wx.ALL, 4)
        s.Add(row2, 0, wx.EXPAND)

        row3 = wx.BoxSizer(wx.HORIZONTAL)
        lbl_pac = wx.StaticText(self, label="Paciencia (épocas):")
        self.paciencia = wx.SpinCtrl(self, min=1, max=100000, initial=200, name="Paciencia en épocas")
        row3.Add(lbl_pac, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row3.Add(self.paciencia, 0, wx.ALL, 4)
        lbl_cada = wx.StaticText(self, label="Validar cada:")
        self.cada = wx.SpinCtrl(self, min=1, max=100, initial=10, name="Validar cada")
        row3.Add(lbl_cada, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
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

        # Selector de corrida (para los botones y para elegir qué log ver).
        row_sel = wx.BoxSizer(wx.HORIZONTAL)
        sel_lbl = wx.StaticText(self, label="Corrida:")
        self.run_sel = wx.Choice(self, name="Corrida")
        row_sel.Add(sel_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        row_sel.Add(self.run_sel, 1, wx.ALL, 4)
        s.Add(row_sel, 0, wx.EXPAND)

        # Log EN VIVO del entrenamiento (una línea por validación).
        log_lbl = wx.StaticText(self, label="Progreso (en vivo):")
        s.Add(log_lbl, 0, wx.LEFT | wx.TOP, 6)
        self.log_view = wx.ListBox(self, size=(-1, 200), name="Progreso del entrenamiento")
        s.Add(self.log_view, 1, wx.ALL | wx.EXPAND, 6)
        self.howto_btn = wx.Button(self, label="¿&Cómo va?")
        s.Add(self.howto_btn, 0, wx.ALL, 4)
        self.SetSizer(s)
        self._log_cache = None
        self.run_sel.Bind(wx.EVT_CHOICE, lambda e: self._refresh_log())

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

    def _toggle_modo(self, e):
        es_base = self.modo.GetSelection() == 1
        self.num_spk.Enable(es_base)

    def set_status(self, t):
        self.status.SetLabel(str(t)); self.nvda.speak(str(t), True)

    def _pick_dataset(self, e):
        with wx.DirDialog(self, "Elegí la carpeta del dataset (con metadata.csv y wavs)") as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.dataset = dlg.GetPath()
                self.set_status(f"Dataset: {Path(self.dataset).name}")

    def _current_state(self) -> runs.RunState:
        es_base = self.modo.GetSelection() == 1
        return runs.RunState(
            nombre=self.name_ctrl.GetValue().strip() or "mivoz",
            modo="base" if es_base else "finetune",
            num_speakers=self.num_spk.GetValue(),
            dataset=self.dataset or str(ROOT / "datasets" / (self.name_ctrl.GetValue().strip() or "mivoz")),
            base_ckpt=str(DEFAULT_BASE),
            max_epochs=self.epochs.GetValue(),
            auto_stop=self.auto.GetValue(),
            paciencia=self.paciencia.GetValue(),
            cada=self.cada.GetValue(),
            started_at=dt.datetime.now().isoformat(timespec="seconds"),
        )

    def _selected_run(self) -> runs.RunState | None:
        i = self.run_sel.GetSelection()
        if i == wx.NOT_FOUND or not self._runs or i >= len(self._runs):
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
        ep = runs.leer_epoca(runs.run_dir(TRAIN_ROOT, st.nombre))
        ep_s = f"época {ep}" if ep is not None else "sin checkpoints"
        return f"{st.nombre} — {st.estado} — {ep_s}"

    def refresh_runs(self):
        self._runs = runs.list_runs(TRAIN_ROOT)
        sel = self.run_sel.GetSelection()
        self.run_sel.Set([self._describe(s) for s in self._runs])
        if self._runs:
            self.run_sel.SetSelection(sel if 0 <= sel < len(self._runs) else 0)
        self._refresh_log()

    def _refresh_log(self):
        """Muestra el progreso.log de la corrida seleccionada (sin robar el foco)."""
        st = self._selected_run()
        if not st:
            if self._log_cache is not None:
                self._log_cache = None; self.log_view.Set([])
            return
        rd = runs.run_dir(TRAIN_ROOT, st.nombre)
        lineas = runs.leer_progreso(rd)
        if not lineas:
            ep = runs.leer_epoca(rd)
            hint = f"; época {ep}" if ep is not None else ""
            lineas = [f"(sin progreso registrado todavía{hint})"]
        if self._log_cache != lineas:
            self._log_cache = lineas
            prev = self.log_view.GetSelection()
            self.log_view.Set(lineas)
            if 0 <= prev < len(lineas):
                self.log_view.SetSelection(prev)  # preservar dónde estaba leyendo

    def _tick(self, e):
        try:
            changed = False
            for st in runs.list_runs(TRAIN_ROOT):
                try:
                    ep = runs.leer_epoca(runs.run_dir(TRAIN_ROOT, st.nombre)) or 0
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
            self._refresh_log()  # el log en vivo se actualiza cada tick
        except Exception:
            pass

    def _decir_estado(self, st, ep, mejor):
        msg = f"{st.nombre} — {st.estado} — época {ep}"
        if mejor:
            msg += f" — mejor: época {mejor[0]} (val_mel {float(mejor[1]):.2f})"
        self.set_status(msg)

    def _on_howto(self, e):
        self.refresh_runs()
        st = self._selected_run() or (self._runs[0] if self._runs else None)
        if not st:
            self.set_status("No hay corridas."); return
        rd = runs.run_dir(TRAIN_ROOT, st.nombre)
        ep = runs.leer_epoca(rd)
        mejor = runs.leer_mejor(rd)
        if ep is not None and mejor is not None:
            self._decir_estado(st, ep, mejor); return
        # falta época o mejor (corrida vieja): leer de los checkpoints (subproceso).
        self.set_status(f"{st.nombre} — {st.estado} — consultando…")
        threading.Thread(target=self._howto_ckpt,
                         args=(st.nombre, st.estado, str(rd), ep), daemon=True).start()

    def _howto_ckpt(self, nombre, estado, rd, ep_vivo=None):
        ep = None; mejor = None
        try:
            r = subprocess.run([PY, str(ROOT / "epoca_ckpt.py"), rd],
                               capture_output=True, text=True, timeout=120)
            for ln in (r.stdout or "").splitlines():
                p = ln.split()
                if len(p) >= 2 and p[0] == "epoca":
                    ep = p[1]
                elif len(p) >= 3 and p[0] == "mejor":
                    mejor = (p[1], p[2])
        except Exception:
            pass
        if ep_vivo is not None:       # preferir la época en vivo (epoch.txt) si la hay
            ep = str(ep_vivo)
        if ep:
            msg = f"{nombre} — {estado} — época {ep}"
            if mejor:
                msg += f" — mejor: época {mejor[0]} (val_mel {mejor[1]})"
        else:
            msg = f"{nombre} — {estado} — sin checkpoints todavía"
        wx.CallAfter(self.set_status, msg)
