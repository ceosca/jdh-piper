"""Lanzador de fine-tuning Piper con early-stopping (Piper Studio, Fase 3).

Entrena una voz desde el checkpoint base y PARA SOLO cuando la pérdida mel sobre
el set de validación (datos no vistos) deja de mejorar — así evita el
sobreentrenamiento sin que tengas que adivinar cuántas épocas. Igual guarda un
checkpoint cada 100 épocas por si querés comparar por oído con
comparar_checkpoints.py.

Uso mínimo (dataset en datasets/<voz>/):
  env\\python.exe entrenar.py --voz silvio

Opciones: --max-epochs, --paciencia, --cada (validar cada N épocas),
--batch-size, --base <ckpt limpio>.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
DEFAULT_BASE = ROOT / "base_ckpt" / "silvio_base_clean.ckpt"  # base es_MX ya saneado (genérico)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voz", required=True, help="Nombre de la voz (= carpeta en datasets/)")
    ap.add_argument("--dataset", default=None, help="Carpeta del dataset (default datasets/<voz>)")
    ap.add_argument("--base", default=str(DEFAULT_BASE), help="Checkpoint base saneado")
    ap.add_argument("--max-epochs", type=int, default=2000,
                    help="Tope duro; el early-stop normalmente corta antes")
    ap.add_argument("--paciencia", type=int, default=12,
                    help="Validaciones sin mejora de val_mel antes de parar")
    ap.add_argument("--cada", type=int, default=10, help="Validar cada N épocas")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--sample-rate", type=int, default=22050)
    ap.add_argument("--espeak", default="es", help="Voz de espeak-ng (idioma)")
    args = ap.parse_args()

    ds = Path(args.dataset) if args.dataset else ROOT / "datasets" / args.voz
    csv = ds / "metadata.csv"
    wavs = ds / "wavs"
    if not csv.exists() or not wavs.exists():
        sys.exit(f"Falta {csv} o {wavs}. Armá el dataset primero.")
    from preparar_base import asegurar_base
    asegurar_base(args.base)  # regenera el base saneado del crudo si falta
    if not Path(args.base).exists():
        sys.exit(f"No existe el checkpoint base: {args.base}")

    ckpts_dir = ROOT / "training" / args.voz / "ckpts"
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks: guardar cada N épocas (comparar por oído) + early-stop por val_mel.
    ckpt_cb = {
        "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
        "init_args": {
            "dirpath": str(ckpts_dir), "every_n_epochs": 100,
            "save_top_k": -1, "save_last": True, "filename": f"{args.voz}-{{epoch}}",
        },
    }
    # Guarda además el MEJOR por val_mel (el punto que el propio entreno "cree" óptimo).
    best_cb = {
        "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
        "init_args": {
            "dirpath": str(ckpts_dir), "monitor": "val_mel", "mode": "min",
            "save_top_k": 1, "filename": f"{args.voz}-best",
        },
    }
    early_cb = {
        "class_path": "lightning.pytorch.callbacks.EarlyStopping",
        "init_args": {
            "monitor": "val_mel", "mode": "min",
            "patience": args.paciencia, "min_delta": 0.0,
        },
    }

    cmd = [
        PY, str(ROOT / "train_run.py"), "fit",
        "--data.voice_name", args.voz,
        "--data.csv_path", str(csv),
        "--data.audio_dir", str(wavs),
        "--model.sample_rate", str(args.sample_rate),
        "--data.espeak_voice", args.espeak,
        "--data.cache_dir", str(ds / "cache"),
        "--data.config_path", str(ds / "config.json"),
        "--data.batch_size", str(args.batch_size),
        "--data.num_workers", "0",  # OBLIGATORIO en Windows (workers cuelgan)
        "--ckpt_path", str(args.base),
        "--trainer.max_epochs", str(args.max_epochs),
        "--trainer.check_val_every_n_epoch", str(args.cada),
        "--trainer.accelerator", "gpu", "--trainer.devices", "1",
        "--trainer.default_root_dir", str(ROOT / "training" / args.voz),
        "--trainer.callbacks+", json.dumps(ckpt_cb),
        "--trainer.callbacks+", json.dumps(best_cb),
        "--trainer.callbacks+", json.dumps(early_cb),
    ]
    print(f"Entrenando «{args.voz}»: early-stop en val_mel "
          f"(paciencia {args.paciencia} × cada {args.cada} épocas), "
          f"tope {args.max_epochs}.\n")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
