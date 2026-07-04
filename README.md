# Piper Studio

App accesible (wxPython + NVDA) para **entrenar, comparar, exportar e instalar
voces TTS de [Piper](https://github.com/OHF-Voice/piper1-gpl)** en Windows nativo,
con GPU. Pensada para un flujo de una sola ventana: armar dataset → entrenar
(fine-tune de una voz o base multi-hablante) → comparar por oído → exportar e
instalar (en el reproductor local y/o como voz de NVDA vía Sonata/Piper-for-NVDA).

Todo el entrenamiento corre **desprendido** de la ventana (sobrevive a cerrarla) y
la GUI se re-engancha al reabrir. Accesible por teclado y lector de pantalla.

## Requisitos

- **Windows** + **GPU NVIDIA** (probado en RTX 5070, CUDA 12.8).
- Un entorno Python en `env/` (micromamba/venv) con: `torch` (cu128), `lightning`,
  `piper-tts` **1.4.2** (piper1-gpl), `faster-whisper`, `wxPython`, `num2words`,
  `soundfile`, `sounddevice`, `numba`. La única puerta de tipos es
  `env\python.exe -m ... tsc`… (no aplica; es Python).
- Binarios de sistema en `PATH`: **ffmpeg** (recorte/silencios), **espeak-ng**
  (lo trae piper-tts). **yt-dlp** opcional.
- **Checkpoint base** (para fine-tune): descargar de
  [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints)
  el `es_MX/ald/medium` a `base_ckpt/es/es_MX/ald/medium/`. Piper Studio lo
  **sanea solo** a `base_ckpt/silvio_base_clean.ckpt` la primera vez
  (`preparar_base.py` / `asegurar_base`).
- Parche necesario del entrenador: `vits/monotonic_align/__init__.py` reemplazado
  por una versión **numba** (sin MSVC). Ver notas en `docs/`.

## Uso

```
Piper Studio.bat          # abre la app (pestañas Entrenar / Comparar / Exportar)
Armar dataset.bat         # armador de dataset (whisper + silencios)
```

### Pestañas

- **Entrenar** — fine-tune de una voz o **base multi-hablante** (cirugía de pesos).
  Épocas manuales o "parar automático" (early-stop por `val_mel`, en **épocas**).
  Lanzar/Pausar/Reanudar/Detener; log en vivo; botón "¿Cómo va?".
- **Comparar** — genera la misma frase desde cada checkpoint → WAVs por época para
  elegir el mejor **por oído** (VITS es GAN: la pérdida no sigue linealmente a la
  calidad; el oído es el juez).
- **Exportar** — `.ckpt` → `.onnx`, e instalar en el reproductor local o como voz
  de **NVDA (Sonata)**.

### Variables de entorno

- `PIPER_PLAYER_VOICES` — carpeta de voces del reproductor CPU (default
  `C:\ia\modelos pc\piper\voces`).

## Conceptos clave

- **Fonemización `es` (España) para todo**: preserva la distinción c/z (θ) y ll/y;
  cada dialecto (rioplatense/mexicano/etc. aseseando, España ceceando) lo aprende el
  *speaker embedding*. Base y fine-tunes deben usar el **mismo** espeak.
- **Base multi-hablante por cirugía**: se copian ~784 pesos acústicos de un
  checkpoint de un hablante a un modelo multi-hablante nuevo; solo las ~20 perillas
  de hablante (`emb_g` + `cond`) arrancan de cero (piper1-gpl 1.4.2 no trae
  `resume_from_single_speaker_checkpoint`).
- **Números en TTS**: los expande espeak, no el modelo; hay una capa de
  normalización de texto (`normalizar_es.py`) en el reproductor.

## Tests

```
env\python.exe -m unittest discover studio/tests
```

Cubren la lógica pura (estado de corridas, cirugía de pesos, lectura de época/mejor,
resolución de config, formato multi-hablante). La GUI se verifica lanzándola.

## Estructura

- `studio/` — la app: `app.py` (ventana), `runs.py` (corridas desprendidas),
  `section_*.py` (pestañas), `base_multi.py` (cirugía), `progress.py`, `nvda.py`.
- `entrenar.py` / `entrenar_base.py` / `train_run.py` — entradas de entrenamiento.
- `comparar_checkpoints.py`, `export_run.py`, `preparar_base.py`, `epoca_ckpt.py`.
- `dataset_builder.py` + `gui_dataset.py` — armador de dataset.
- `docs/` — spec y planes.
