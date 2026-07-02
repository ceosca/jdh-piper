"""Comparador de checkpoints por oído (Piper Studio, Fase 3).

Para cada checkpoint .ckpt de una carpeta, exporta el .onnx y genera LA MISMA
frase en un WAV numerado por época. Escuchás los WAV y te quedás con el que mejor
suena: así elegís el punto justo de entrenamiento (evita el sobreentrenamiento
sin depender de la pérdida, que en una GAN como VITS no sigue a la calidad).

Uso:
  env\\python.exe comparar_checkpoints.py --ckpts training/silvio/ckpts \\
      --config datasets/silvio/config.json --out training/silvio/comparar \\
      --text "Hola, esta es una prueba de la voz."
"""
import argparse
import pathlib
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

# Parches para cargar checkpoints (torch 2.8 + rutas Linux) al sintetizar.
pathlib.PosixPath = pathlib.WindowsPath
import torch  # noqa: E402

_orig_load = torch.load


def _patched_load(*a, **k):
    k["weights_only"] = False
    return _orig_load(*a, **k)


torch.load = _patched_load

from piper import PiperVoice  # noqa: E402

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def epoch_of(ckpt: Path) -> str:
    """Devuelve la época como texto de 4 dígitos para ordenar/nombrar."""
    m = re.search(r"epoch=(\d+)", ckpt.stem)
    if m:
        return m.group(1).zfill(4)
    return ckpt.stem  # p.ej. "last"


def export(ckpt: Path, onnx_out: Path) -> bool:
    """Exporta un .ckpt a .onnx usando export_run.py (aísla el proceso)."""
    r = subprocess.run(
        [PY, str(ROOT / "export_run.py"), "--checkpoint", str(ckpt),
         "--output-file", str(onnx_out)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not onnx_out.exists():
        print(f"  ERROR exportando {ckpt.name}:\n{r.stderr[-500:]}")
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="Carpeta con los .ckpt")
    ap.add_argument("--config", required=True, help="config.json (sidecar de la voz)")
    ap.add_argument("--out", required=True, help="Carpeta de salida para .onnx y .wav")
    ap.add_argument("--text", default="Hola, esta es una prueba de la voz clonada con Piper.",
                    help="Frase a sintetizar en todos los checkpoints")
    ap.add_argument("--keep-onnx", action="store_true",
                    help="Conservar los .onnx (default: sí, para instalar el elegido)")
    args = ap.parse_args()

    ckpts_dir = Path(args.ckpts)
    config = Path(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(ckpts_dir.glob("*.ckpt"), key=lambda p: epoch_of(p))
    if not ckpts:
        print(f"No hay checkpoints en {ckpts_dir}")
        return
    print(f"{len(ckpts)} checkpoints. Frase: {args.text!r}\n")

    for ckpt in ckpts:
        tag = epoch_of(ckpt)
        onnx_out = out / f"ep{tag}.onnx"
        wav_out = out / f"ep{tag}.wav"
        print(f"[ep{tag}] {ckpt.name} -> exportando…")
        if not export(ckpt, onnx_out):
            continue
        shutil.copyfile(config, onnx_out.with_suffix(".onnx.json"))
        voice = PiperVoice.load(str(onnx_out))
        with wave.open(str(wav_out), "wb") as w:
            voice.synthesize_wav(args.text, w)
        print(f"        WAV: {wav_out.name}")

    print(f"\nListo. Escuchá los WAV en {out} y quedate con la época que mejor suene.")
    print("Para instalar la elegida en el reproductor, copiá epNNNN.onnx y "
          "epNNNN.onnx.json a 'modelos pc/piper/voces/silvio/'.")


if __name__ == "__main__":
    main()
