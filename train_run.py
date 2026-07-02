"""Wrapper de entrenamiento Piper.

Parchea torch.load para usar weights_only=False (los checkpoints base de Piper
traen objetos como pathlib.Path que torch 2.6+ bloquea por defecto). El
checkpoint es de fuente confiable (rhasspy/piper-checkpoints). Después ejecuta
`piper.train` con los argumentos que le pases.

Uso: env\\python.exe train_run.py fit --data.voice_name ... (mismos args que piper.train)
"""
import pathlib
import runpy
import sys

import torch

# El checkpoint base se guardó en Linux y trae objetos pathlib.PosixPath, que
# Windows no puede instanciar al deserializar. Mapear PosixPath -> WindowsPath.
pathlib.PosixPath = pathlib.WindowsPath

_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

# Ejecuta piper.train como si fuera `python -m piper.train <args>`
runpy.run_module("piper.train", run_name="__main__")
