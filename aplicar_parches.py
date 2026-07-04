# -*- coding: utf-8 -*-
"""Aplica los parches de entrenamiento a la instalación de piper-tts.

El wheel de piper-tts 1.4.2 no trae el código de entrenamiento (`piper.train.vits`),
y además necesita dos parches para funcionar en Windows nativo:
  - `monotonic_align/__init__.py` en versión numba (sin necesidad de Visual C++).
  - `lightning.py` que loguea `val_mel` (para el early-stop por calidad).
Todo eso está en `setup/piper_train/` (código v1.4.2 ya parcheado). Este script lo
copia dentro del paquete piper instalado.

Uso (después de `pip install -r requirements.txt`):  python aplicar_parches.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "setup" / "piper_train"


def main() -> None:
    if not SRC.exists():
        sys.exit(f"Falta {SRC} (¿clonaste el repo completo?).")
    try:
        import piper
    except ImportError:
        sys.exit("Instalá las dependencias primero: pip install -r requirements.txt")
    dst = Path(piper.__file__).resolve().parent / "train"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(SRC, dst)
    print(f"Parches aplicados: {SRC}  ->  {dst}")
    print("Incluye: monotonic_align (numba, sin MSVC) + lightning.py (val_mel para el early-stop).")


if __name__ == "__main__":
    main()
