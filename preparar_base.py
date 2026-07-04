# -*- coding: utf-8 -*-
"""Regenera los checkpoints base saneados (base_ckpt/*_base_clean.ckpt).

El saneado se deriva de un crudo (rhasspy/piper-checkpoints): filtra los
hyper_parameters a los args de VitsModel (descarta claves de Trainer viejo que la
CLI rechaza), quita loops/callbacks y resetea epoch/global_step a 0 (para
fine-tunear desde época 0). Conserva optimizer_states + lr_schedulers (LR ~5.7e-5,
ideal para fine-tune suave). SOLO toca metadata; los pesos (state_dict) quedan
idénticos — no afecta la calidad.

Hay dos bases (mismo tamaño «medium», 22050 Hz, num_symbols 256):
  - davefx (es_ES / España): fonemizado con espeak `es` (θ de c/z de fábrica) — el
    RECOMENDADO, alineado con nuestra fonemización `es`. ~1,6 M steps.
  - ald    (es_MX / México): fonemizado con espeak `es-419` (seseo) — probado con
    silvio/mario/pedro. Es, de hecho, un fine-tune de davefx. ~1,75 M steps.

Son artefactos DERIVADOS y gitignoreados. `asegurar_base()` regenera el que falte
si su crudo está presente (lo llaman entrenar.py / entrenar_base.py). Uso manual:
  env\\python.exe preparar_base.py       # regenera toda base cuyo crudo esté bajado
"""
import inspect
import os
import pathlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT / "base_ckpt"

# saneado (nombre de archivo) -> checkpoint crudo del que se deriva
CRUDOS = {
    "davefx_base_clean.ckpt": BASE_DIR / "es" / "es_ES" / "davefx" / "medium"
    / "epoch=5629-step=1605020.ckpt",
    "silvio_base_clean.ckpt": BASE_DIR / "es" / "es_MX" / "ald" / "medium"
    / "epoch=9999-step=1753600.ckpt",
}


def regenerar(salida: Path) -> Path:
    salida = Path(salida)
    crudo = CRUDOS.get(salida.name)
    if crudo is None:
        raise SystemExit(f"No sé de qué crudo derivar {salida.name} "
                         f"(conocidos: {', '.join(CRUDOS)}).")
    if not crudo.exists():
        raise SystemExit(f"Falta el checkpoint crudo: {crudo}\n"
                         "Bajalo de rhasspy/piper-checkpoints.")
    pathlib.PosixPath = pathlib.WindowsPath  # el crudo se guardó en Linux
    import torch
    from piper.train.vits.lightning import VitsModel
    valid = set(inspect.signature(VitsModel.__init__).parameters) - {"self", "kwargs"}
    ck = torch.load(str(crudo), map_location="cpu", weights_only=False)
    ck["hyper_parameters"] = {k: v for k, v in ck["hyper_parameters"].items() if k in valid}
    ck.pop("loops", None)
    ck.pop("callbacks", None)
    ck["epoch"] = 0
    ck["global_step"] = 0
    salida.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, str(salida))
    print(f"Base saneado listo: {salida} ({os.path.getsize(salida) / 1e6:.0f} MB)")
    return salida


def asegurar_base(path) -> None:
    """Si falta un base saneado conocido, lo regenera de su crudo (automático)."""
    p = Path(path)
    if p.exists():
        return
    crudo = CRUDOS.get(p.name)
    if crudo is not None and crudo.exists():
        print(f"[base] falta {p.name}; regenerando del crudo…", flush=True)
        regenerar(p)


def main() -> None:
    hechos = 0
    for nombre, crudo in CRUDOS.items():
        if crudo.exists():
            regenerar(BASE_DIR / nombre)
            hechos += 1
    if not hechos:
        raise SystemExit("No hay ningún crudo en base_ckpt/ para sanear "
                         "(bajá al menos davefx o ald de rhasspy/piper-checkpoints).")


if __name__ == "__main__":
    main()
