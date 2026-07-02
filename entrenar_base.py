# entrenar_base.py
"""Entrena un base multi-hablante desde un checkpoint mono, por cirugía de pesos.

Construye un VitsModel multi-hablante, le inyecta los ~784 pesos del mono
(fusionar_pesos deja random solo emb_g + capas cond), y entrena de cero (fit).
Uso:
  env\\python.exe entrenar_base.py --dataset datasets/base_latino --base-mono base_ckpt/silvio_base_clean.ckpt --num-speakers 12
"""
import argparse
import inspect
import pathlib
from pathlib import Path

pathlib.PosixPath = pathlib.WindowsPath  # checkpoints guardados en Linux
import torch  # noqa: E402

_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})

import lightning as L  # noqa: E402
from lightning.pytorch.callbacks import ModelCheckpoint  # noqa: E402

from piper.train.vits.dataset import VitsDataModule  # noqa: E402
from piper.train.vits.lightning import VitsModel  # noqa: E402
from studio.base_multi import fusionar_pesos, hparams_multi  # noqa: E402

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Carpeta con metadata.csv (wav|speaker|text) y wavs/")
    ap.add_argument("--base-mono", required=True, help="Checkpoint mono saneado")
    ap.add_argument("--num-speakers", type=int, required=True)
    ap.add_argument("--gin-channels", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--espeak", default="es")  # España: preserva c/z (θ) y ll/y;
    # cada dialecto (AR/MX/neutro asesean, España cecea) lo aprende su embedding.
    ap.add_argument("--sample-rate", type=int, default=22050)
    ap.add_argument("--voz", default=None, help="Nombre de la corrida (clave del dir de checkpoints en la GUI)")
    ap.add_argument("--resume", default=None, help="Checkpoint del que reanudar (salta la cirugía)")
    args = ap.parse_args()

    ds = Path(args.dataset)
    nombre = args.voz or ds.name
    ckpts = ROOT / "training" / nombre / "ckpts"
    ckpts.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckr = torch.load(args.resume, map_location="cpu")
        validos = set(inspect.signature(VitsModel.__init__).parameters) - {"self", "kwargs"}
        hp = {k: v for k, v in ckr["hyper_parameters"].items() if k in validos}
        hp["batch_size"] = args.batch_size
        model = VitsModel(**hp)
        resume_ckpt = args.resume
    else:
        # hparams multi a partir del mono
        ck = torch.load(args.base_mono, map_location="cpu")
        hp = hparams_multi(ck["hyper_parameters"], args.num_speakers, args.gin_channels)
        hp["batch_size"] = args.batch_size
        model = VitsModel(**hp)
        merged, n_cop, n_new = fusionar_pesos(ck["state_dict"], model.state_dict())
        model.load_state_dict(merged)
        print(f"[cirugía] copiadas={n_cop} nuevas(random)={n_new}")
        resume_ckpt = None

    data = VitsDataModule(
        csv_path=str(ds / "metadata.csv"),
        audio_dir=str(ds / "wavs"),
        cache_dir=str(ds / "cache"),
        config_path=str(ds / "config.json"),
        voice_name=nombre,
        sample_rate=args.sample_rate,
        espeak_voice=args.espeak,
        num_speakers=args.num_speakers,
        batch_size=args.batch_size,
        num_workers=0,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs, accelerator="gpu", devices=1,
        default_root_dir=str(ROOT / "training" / nombre),
        callbacks=[ModelCheckpoint(dirpath=str(ckpts), every_n_epochs=50,
                                   save_top_k=-1, save_last=True,
                                   filename=nombre + "-{epoch}")],
    )
    trainer.fit(model, data, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
