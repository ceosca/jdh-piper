"""Callback de progreso: escribe la época y el log legible del entrenamiento.

El train.log queda vacío al correr desprendido (Lightning no vuelca la barra de
progreso a un archivo no-tty). Este callback escribe, en `training/<voz>/`:
  - `epoch.txt`   : época actual (cada época; fuente barata para la GUI).
  - `progreso.log`: una línea por validación ("Época N — val_mel V — mejor: …").
  - `mejor.txt`   : "<época> <val_mel>" del mejor punto hasta ahora.
La GUI muestra `progreso.log` en vivo y lee `epoch.txt`/`mejor.txt`.
"""
from pathlib import Path

from lightning.pytorch.callbacks import Callback


class EscritorEpoca(Callback):
    def __init__(self, path: str):
        super().__init__()
        self.epoch_txt = Path(path)               # .../epoch.txt
        rd = self.epoch_txt.parent
        self.prog_log = rd / "progreso.log"
        self.mejor_txt = rd / "mejor.txt"
        self._best = None                          # (época, val_mel)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        try:
            self.epoch_txt.write_text(str(int(trainer.current_epoch)), encoding="utf-8")
        except Exception:
            pass

    def on_validation_end(self, trainer, pl_module) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        try:
            m = trainer.callback_metrics.get("val_mel")
            if m is None:
                return
            v = float(m)
            ep = int(trainer.current_epoch)
            if self._best is None or v < self._best[1]:
                self._best = (ep, v)
                self.mejor_txt.write_text(f"{ep} {v:.4f}", encoding="utf-8")
            be, bv = self._best
            linea = f"Época {ep} — val_mel {v:.3f} — mejor: época {be} ({bv:.3f})"
            with open(self.prog_log, "a", encoding="utf-8") as f:
                f.write(linea + "\n")
        except Exception:
            pass
