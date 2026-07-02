"""Wrapper de exportación a ONNX de Piper.

Igual que train_run.py: parchea torch.load (weights_only=False) y mapea
PosixPath->WindowsPath por si el checkpoint trae rutas Linux, y después ejecuta
`piper.train.export_onnx` con los argumentos que le pases.

Uso: env\\python.exe export_run.py --checkpoint <ckpt> --output-file <onnx>
"""
import pathlib
import runpy

import torch

pathlib.PosixPath = pathlib.WindowsPath

_orig_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_load(*args, **kwargs)


torch.load = _patched_load

runpy.run_module("piper.train.export_onnx", run_name="__main__")
