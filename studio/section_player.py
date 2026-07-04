"""Pestaña Reproductor de Piper Studio (portada del reproductor CPU, con GPU).

Multi-voz: cuadro de texto + menú contextual (tecla Aplicaciones / Shift+F10) que
inserta #nombredevoz; "Generar" parte en fragmentos; reproducir/regenerar/editar
por fragmento, reproducir todo, guardar WAV. Comparte voces con el reproductor CPU.
"""
from __future__ import annotations

import os
import re
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import wx

from studio.player_engine import TTSEngine, VoiceLibrary, VOICES_DIR

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "salidas"
WXK_APPS = getattr(wx, "WXK_APPS", 0x5D)


class PlayerPanel(wx.Panel):
    def __init__(self, parent, nvda):
        super().__init__(parent)
        self.nvda = nvda
        self.engine = TTSEngine()
        self.library = VoiceLibrary()
        self.fragments = []
        self._busy = False
        self._playing = False
        self._play_stop = threading.Event()
        try:
            OUTPUT_DIR.mkdir(exist_ok=True)
        except Exception:
            pass
        self._build_ui()
        self._bind_accessibility()
        self.Bind(wx.EVT_WINDOW_DESTROY, lambda e: (self._stop(), e.Skip()))
        self.refresh_voices()
        self._start_load()

    def _build_ui(self):
        s = wx.BoxSizer(wx.VERTICAL)
        self.status_label = wx.StaticText(self, label="Iniciando…")
        f = self.status_label.GetFont(); f.MakeBold(); self.status_label.SetFont(f)
        s.Add(self.status_label, 0, wx.ALL | wx.EXPAND, 6)

        vr = wx.BoxSizer(wx.HORIZONTAL)
        # etiqueta ANTES del control (regla NVDA)
        voz_lbl = wx.StaticText(self, label="Voz:")
        self.voice_choice = wx.Choice(self, name="Voz")
        self.refresh_btn = wx.Button(self, label="Actualizar voces")
        self.open_btn = wx.Button(self, label="Abrir carpeta de voces")
        vr.Add(voz_lbl, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 4)
        vr.Add(self.voice_choice, 1, wx.ALL | wx.EXPAND, 4)
        vr.Add(self.refresh_btn, 0, wx.ALL, 4)
        vr.Add(self.open_btn, 0, wx.ALL, 4)
        s.Add(vr, 0, wx.ALL | wx.EXPAND, 6)

        txt_lbl = wx.StaticText(self, label="Texto (tecla Aplicaciones o Shift+F10 para insertar #voz):")
        s.Add(txt_lbl, 0, wx.ALL, 4)
        self.text_ctrl = wx.TextCtrl(self, style=wx.TE_MULTILINE, size=(-1, 170), name="Texto")
        s.Add(self.text_ctrl, 1, wx.ALL | wx.EXPAND, 4)

        br = wx.BoxSizer(wx.HORIZONTAL)
        self.generate_btn = wx.Button(self, label="&Generar (Alt+G)")
        self.play_btn = wx.Button(self, label="&Reproducir todo / Detener (Alt+R)")
        self.save_btn = wx.Button(self, label="Guardar WAV… (Alt+S)")
        self.play_btn.Enable(False); self.save_btn.Enable(False)
        for b in (self.generate_btn, self.play_btn, self.save_btn):
            br.Add(b, 0, wx.ALL, 4)
        s.Add(br, 0, wx.ALL, 4)

        frag_lbl = wx.StaticText(self, label="Fragmentos:")
        s.Add(frag_lbl, 0, wx.ALL, 4)
        self.frag_list = wx.ListBox(self, size=(-1, 130), name="Fragmentos")
        s.Add(self.frag_list, 1, wx.ALL | wx.EXPAND, 4)
        fr = wx.BoxSizer(wx.HORIZONTAL)
        self.frag_play_btn = wx.Button(self, label="Reproducir fragmento")
        self.frag_regen_btn = wx.Button(self, label="Regenerar fragmento")
        self.frag_edit_btn = wx.Button(self, label="Editar texto del fragmento")
        for b in (self.frag_play_btn, self.frag_regen_btn, self.frag_edit_btn):
            fr.Add(b, 0, wx.ALL, 4)
        s.Add(fr, 0, wx.ALL, 4)
        self.SetSizer(s)

        self.voice_choice.Bind(wx.EVT_CHOICE,
                               lambda e: self.nvda.speak(f"Voz {self.voice_choice.GetStringSelection()}", True))
        self.refresh_btn.Bind(wx.EVT_BUTTON, lambda e: self.refresh_voices(announce=True))
        self.open_btn.Bind(wx.EVT_BUTTON, lambda e: self._open_voices_folder())
        self.text_ctrl.Bind(wx.EVT_CONTEXT_MENU, lambda e: self._show_menu())
        self.text_ctrl.Bind(wx.EVT_KEY_DOWN, self._on_text_key)
        self.generate_btn.Bind(wx.EVT_BUTTON, self._on_generate)
        self.play_btn.Bind(wx.EVT_BUTTON, self._on_play_all)
        self.save_btn.Bind(wx.EVT_BUTTON, lambda e: self._save(self._concat()))
        self.frag_play_btn.Bind(wx.EVT_BUTTON, self._on_play_fragment)
        self.frag_regen_btn.Bind(wx.EVT_BUTTON, self._on_regen_fragment)
        self.frag_edit_btn.Bind(wx.EVT_BUTTON, self._on_edit_fragment)
        self.frag_list.Bind(wx.EVT_LISTBOX_DCLICK, self._on_play_fragment)
        self.SetAcceleratorTable(wx.AcceleratorTable([
            (wx.ACCEL_ALT, ord('G'), self.generate_btn.GetId()),
            (wx.ACCEL_ALT, ord('R'), self.play_btn.GetId()),
            (wx.ACCEL_ALT, ord('S'), self.save_btn.GetId()),
        ]))

    def _bind_accessibility(self):
        for c, l in [(self.voice_choice, "Voz"), (self.text_ctrl, "Texto"), (self.frag_list, "Fragmentos")]:
            c.Bind(wx.EVT_SET_FOCUS, lambda e, c=c, l=l: self._on_focus(e, c, l))

    def _on_focus(self, event, ctrl, label):
        val = ctrl.GetStringSelection() if isinstance(ctrl, (wx.ListBox, wx.Choice)) else ""
        self.nvda.speak(f"{label} {val}".strip(), True)
        event.Skip()

    def set_status(self, text):
        self.status_label.SetLabel(str(text)); self.nvda.speak(str(text), True)

    def set_status_threadsafe(self, text):
        wx.CallAfter(self.set_status, text)

    def _start_load(self):
        self.set_status("Cargando…")
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        try:
            self.engine.load(progress=self.set_status_threadsafe)
            wx.CallAfter(self.set_status,
                         f"Listo ({self.engine.device}). Elegí una voz, escribí y Generá.")
        except Exception as e:
            traceback.print_exc(); wx.CallAfter(self.set_status, f"Error: {e}")

    def refresh_voices(self, select=None, announce=False):
        voices = self.library.voices()
        self.voice_choice.Set(voices)
        if voices:
            idx = voices.index(select) if select in voices else 0
            self.voice_choice.SetSelection(idx)
        if announce:
            self.set_status(f"{len(voices)} voces." if voices else "No hay voces (.onnx) en la carpeta.")

    def _default_voice(self):
        return self.voice_choice.GetStringSelection()

    def _open_voices_folder(self):
        try:
            os.startfile(str(VOICES_DIR))  # noqa: S606
        except Exception:
            self.set_status(f"Carpeta: {VOICES_DIR}")

    def _on_text_key(self, event):
        kc = event.GetKeyCode()
        if kc == WXK_APPS or (event.ShiftDown() and kc == wx.WXK_F10):
            self._show_menu()
        else:
            event.Skip()

    def _show_menu(self):
        menu = wx.Menu()
        sub = wx.Menu()
        voices = self.library.voices()
        if voices:
            for vn in voices:
                iid = wx.NewIdRef()
                sub.Append(iid, vn)
                self.Bind(wx.EVT_MENU, lambda e, v=vn: self._insert_voice(v), id=iid)
        else:
            it = sub.Append(wx.ID_ANY, "No hay voces"); it.Enable(False)
        menu.AppendSubMenu(sub, "Voces")
        self.PopupMenu(menu)
        menu.Destroy()

    def _insert_voice(self, vn):
        self.text_ctrl.WriteText(f"#{vn} ")
        self.nvda.speak(f"Insertada voz {vn}", True)

    def _parse_segments(self, text):
        default = self._default_voice()
        parts = re.split(r"#(\w+)", text)
        segs = []
        if parts and parts[0].strip():
            segs.append((default, parts[0].strip()))
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts) and parts[i + 1].strip():
                segs.append((parts[i], parts[i + 1].strip()))
        return segs

    @staticmethod
    def _split(text, max_chars=350):
        out = []
        for para in re.split(r"\n\s*\n", text.strip()):
            for spart in re.findall(r".*?(?:[.!?…]+(?:\s|$)|\n+|$)", para, re.S):
                sp = spart.strip()
                if not sp:
                    continue
                if out and len(out[-1]) + 1 + len(sp) <= max_chars:
                    out[-1] = out[-1] + " " + sp
                else:
                    out.append(sp)
        return out

    def _on_generate(self, event):
        if self._guard():
            return
        text = self.text_ctrl.GetValue().strip()
        if not text:
            self.set_status("El texto está vacío."); return
        if not self.library.voices():
            self.set_status("No hay voces. Poné un .onnx en voces\\<nombre>\\ y Actualizá."); return
        frags = []
        for voice, seg in self._parse_segments(text):
            for piece in self._split(seg):
                frags.append({"voice": voice, "text": piece, "audio": None})
        if not frags:
            self.set_status("No hay texto para generar."); return
        self.fragments = frags
        self._refresh_frag_list()
        self._set_busy(True, f"Generando {len(frags)} fragmentos…")
        threading.Thread(target=self._gen_worker, daemon=True).start()

    def _gen_worker(self):
        try:
            for i, fr in enumerate(self.fragments):
                self.set_status_threadsafe(f"Fragmento {i + 1}/{len(self.fragments)} — {fr['voice'] or 'voz'}…")
                sr, audio = self.engine.generate(fr["text"], fr["voice"])
                fr["audio"] = (sr, audio)
                wx.CallAfter(self._refresh_frag_list)
            wx.CallAfter(self._after_gen)
        except Exception as e:
            traceback.print_exc(); wx.CallAfter(self._gen_error, e)

    def _after_gen(self):
        self.play_btn.Enable(True); self.save_btn.Enable(True)
        self._set_busy(False, f"{len(self.fragments)} fragmentos listos.")
        if self.fragments:
            self.frag_list.SetSelection(0); self.frag_list.SetFocus()

    def _refresh_frag_list(self):
        items = []
        for i, fr in enumerate(self.fragments, 1):
            mark = "✓" if fr["audio"] else "…"
            items.append(f"{i}. [{mark}] ({fr['voice'] or 'voz'}) {fr['text'][:55]}")
        sel = self.frag_list.GetSelection()
        self.frag_list.Set(items)
        if 0 <= sel < len(items):
            self.frag_list.SetSelection(sel)
        elif items:
            self.frag_list.SetSelection(0)

    def _sel(self):
        i = self.frag_list.GetSelection()
        if i == wx.NOT_FOUND or not self.fragments:
            self.set_status("Seleccioná un fragmento."); return None
        return i

    def _on_play_fragment(self, event):
        if self._playing:
            self._stop(); return
        i = self._sel()
        if i is None:
            return
        au = self.fragments[i]["audio"]
        if not au:
            self.set_status("Ese fragmento no tiene audio."); return
        self._play([(i, au)])

    def _on_regen_fragment(self, event):
        if self._guard():
            return
        i = self._sel()
        if i is None:
            return
        self._set_busy(True, f"Regenerando fragmento {i + 1}…")
        threading.Thread(target=self._regen_worker, args=(i,), daemon=True).start()

    def _regen_worker(self, i):
        try:
            fr = self.fragments[i]
            sr, audio = self.engine.generate(fr["text"], fr["voice"])
            fr["audio"] = (sr, audio)
            wx.CallAfter(self._refresh_frag_list)
            wx.CallAfter(self._set_busy, False, f"Fragmento {i + 1} regenerado.")
        except Exception as e:
            traceback.print_exc(); wx.CallAfter(self._gen_error, e)

    def _on_edit_fragment(self, event):
        i = self._sel()
        if i is None:
            return
        fr = self.fragments[i]
        dlg = wx.TextEntryDialog(self, "Editá el texto:", "Editar fragmento", fr["text"])
        if dlg.ShowModal() == wx.ID_OK:
            fr["text"] = dlg.GetValue().strip(); fr["audio"] = None
            self._refresh_frag_list(); self.frag_list.SetSelection(i)
            self.set_status("Texto cambiado. Regeneralo.")
        dlg.Destroy()

    def _concat(self):
        pieces, sr = [], self.engine.sample_rate
        for fr in self.fragments:
            if fr["audio"]:
                sr, data = fr["audio"]
                pieces.append(data)
                pieces.append(np.zeros(int(sr * 0.3), dtype=np.float32))
        return (sr, np.concatenate(pieces)) if pieces else None

    def _on_play_all(self, event):
        if self._playing:
            self._stop(); return
        indexed = [(i, fr["audio"]) for i, fr in enumerate(self.fragments) if fr["audio"]]
        if not indexed:
            self.set_status("No hay fragmentos generados."); return
        self._play(indexed, highlight=True)

    def _play(self, indexed, highlight=False):
        self._play_stop.clear(); self._playing = True
        self.set_status("Reproduciendo…")
        threading.Thread(target=self._play_worker, args=(indexed, highlight), daemon=True).start()

    def _play_worker(self, indexed, highlight):
        import sounddevice as sd
        try:
            for idx, (sr, data) in indexed:
                if self._play_stop.is_set():
                    break
                if highlight:
                    wx.CallAfter(self.frag_list.SetSelection, idx)
                sd.play(np.concatenate([data, np.zeros(int(sr * 0.12), dtype=np.float32)]), sr)
                while True:
                    try:
                        if not sd.get_stream().active:
                            break
                    except Exception:
                        break
                    if self._play_stop.is_set():
                        sd.stop(); break
                    time.sleep(0.05)
        except Exception as e:
            self.set_status_threadsafe(f"No se pudo reproducir: {e}")
        finally:
            self._playing = False
            self.set_status_threadsafe("Reproducción terminada.")

    def _stop(self):
        self._play_stop.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._playing = False

    def _save(self, audio):
        if not audio:
            self.set_status("No hay audio. Generá primero."); return
        import soundfile as sf
        with wx.FileDialog(self, "Guardar audio", defaultDir=str(OUTPUT_DIR), defaultFile="audio.wav",
                           wildcard="WAV (*.wav)|*.wav", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            path = dlg.GetPath()
        sr, data = audio
        try:
            sf.write(path, data, sr); self.set_status(f"Guardado en {path}")
        except Exception as e:
            self.set_status(f"No se pudo guardar: {e}")

    def _guard(self):
        if self._busy:
            self.set_status("Esperá: hay una tarea en curso."); return True
        if not self.engine.loaded:
            self.set_status("Cargando todavía."); return True
        return False

    def _set_busy(self, busy, status=None):
        self._busy = busy
        for b in (self.generate_btn, self.frag_regen_btn):
            b.Enable(not busy)
        if status:
            self.set_status(status)

    def _gen_error(self, e):
        self._set_busy(False, f"Error al generar: {e}")
