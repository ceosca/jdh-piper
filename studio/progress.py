"""Callback liviano: escribe la época actual en un archivo de texto.

Fuente BARATA de progreso en vivo para la GUI (el train.log queda vacío cuando el
entrenamiento corre desprendido, y cargar un checkpoint de 846MB por tick es caro).
La GUI lee `training/<voz>/epoch.txt` con `runs.leer_epoca`.
"""
from pathlib import Path

from lightning.pytorch.callbacks import Callback


class EscritorEpoca(Callback):
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def _escribir(self, trainer) -> None:
        try:
            Path(self.path).write_text(str(int(trainer.current_epoch)), encoding="utf-8")
        except Exception:
            pass

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._escribir(trainer)
