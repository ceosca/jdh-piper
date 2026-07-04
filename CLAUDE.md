# CLAUDE.md

Guía para Claude Code (claude.ai/code) al trabajar en este repo. Estas
instrucciones MANDAN sobre el comportamiento por defecto. El `README.md` tiene el
detalle para humanos; este archivo es el manual operativo para vos (Claude).

## Qué es

**Piper Studio** — app accesible (wxPython + NVDA) para **entrenar, comparar,
exportar e instalar voces TTS de [Piper](https://github.com/OHF-Voice/piper1-gpl)**
en **Windows nativo con GPU**. Flujo de una ventana: armar dataset → entrenar
(fine-tune de una voz o base multi-hablante) → comparar por oído → exportar/instalar
(reproductor local y/o voz de NVDA vía Sonata). El entrenamiento corre
**desprendido** (sobrevive a cerrar la GUI) y la app se re-engancha al reabrir.

## Cómo trabajar acá (reglas de la casa)

1. **Respondé en español** (rioplatense/neutro). Todo el proyecto y sus usuarios
   son en español.
2. **La accesibilidad es un requisito, no un extra.** La app se maneja con lector
   de pantalla (NVDA) y teclado. Regla de oro de las GUIs wxPython: **creá el
   `wx.StaticText` (etiqueta) ANTES que su control** — NVDA toma la etiqueta por
   orden de creación, no de layout. Si creás el control primero, corre las
   etiquetas un lugar. No rompas foco ni navegación por teclado.
3. **Windows nativo, sin WSL.** Usá SIEMPRE `env\python.exe` (el entorno portable
   del repo) para correr cualquier cosa. No asumas Linux ni rutas POSIX.
4. Es **Python**, no Node — no hay pnpm/npm/tsc acá.
5. **Verificá antes de cantar victoria**: corré los tests después de tocar lógica.
   La GUI se verifica lanzándola (no hay tests de UI).

## Comandos

```
Piper Studio.bat                                  # abre la app
Armar dataset.bat                                 # armador de dataset (whisper + silencios)
env\python.exe -m unittest discover studio/tests  # tests (correr tras cada cambio de lógica)
env\python.exe -m unittest studio.tests.test_runs # un solo módulo de test
env\python.exe entrenar.py --voz <voz>            # fine-tune por CLI (default base davefx)
env\python.exe preparar_base.py                   # (re)genera los base saneados presentes
```

## Decisiones que NO se cambian (y por qué)

- **Fonemización espeak `es` (España) para TODO** (base y fine-tunes, mismo espeak).
  Preserva la distinción c/z (θ) y ll/y; cada dialecto (rioplatense/mexicano/neutro
  asesean, España cecea) lo aprende el **speaker embedding**. **No** pasar a `es-419`
  aunque la voz sea latina — perderías la θ y no se recupera.
- **Base por defecto: `davefx` (es_ES/España).** Selector **Base** en Entrenar:
  *España (davefx) — recomendado* / *México (ald) — probado*. El **dialecto del
  base es irrelevante**: el fine-tune reescribe la voz. Lo que importa es que el
  espeak sea `es`, y davefx ya fonemiza `es` de fábrica (ald terminó en `es-419`).
  ald sigue válido; es, de hecho, un fine-tune de davefx.
- **Paciencia = ÉPOCAS por encima del mejor**, no validaciones. El early-stop
  convierte internamente: `exámenes = paciencia // cada`. Si lo tocás, mantené la
  semántica de "N épocas sin mejorar el `val_mel`".
- **La calidad se juzga por OÍDO, no por la loss.** VITS es GAN: la pérdida no
  sigue linealmente a la calidad. Por eso existe la pestaña Comparar. `val_mel`
  solo compara puntos DENTRO de una misma voz (no entre voces/hablantes).
- **Entrenamiento desprendido** (DETACHED + grupo nuevo), re-enganchable por
  `training/<voz>/run.json` + pid vivo. No lo vuelvas "attached" a la GUI.
- **Números en TTS = capa de normalización de TEXTO** (`normalizar_es.py`,
  num2words + regex), no el modelo. Si un número suena mal, se arregla ahí.

## Gotchas de Windows / entrenamiento (ya resueltos — no los re-rompas)

- `pathlib.PosixPath = pathlib.WindowsPath` **antes** de `torch.load` (los ckpt de
  rhasspy se guardaron en Linux).
- `torch.load(..., weights_only=False)`.
- `--data.num_workers 0` SIEMPRE (los DataLoader workers cuelgan en Windows).
- El código de entrenamiento es **piper1-gpl v1.4.2 parcheado**, no HEAD (HEAD pasa
  `vowel_clusters` y rompe). Vive en `setup/piper_train/` y se copia al paquete
  instalado con `aplicar_parches.py` (monotonic_align en **numba** sin MSVC +
  logging de `val_mel`).
- **Saneo del base** (`preparar_base.py`): filtrar `hyper_parameters` a los args de
  `VitsModel`, sacar `loops`/`callbacks`, `epoch`/`global_step = 0`, **conservar**
  `optimizer_states`/`lr_schedulers` (LR ~5.7e-5 = fine-tune suave). Solo toca
  metadata; el `state_dict` (pesos) queda idéntico.
- **Base multi-hablante por cirugía** (`studio/base_multi.py`): de un checkpoint de
  1 hablante se copian ~784 pesos por nombre+forma; las ~20 perillas de hablante
  (`emb_g` + capas `cond`) arrancan de cero. piper1-gpl 1.4.2 **no** trae
  `resume_from_single_speaker_checkpoint`.

## Estructura (dónde está cada cosa)

- `studio/` — la app: `app.py` (ventana), `runs.py` (corridas desprendidas +
  `build_train_argv`), `section_*.py` (pestañas Entrenar/Comparar/Exportar/Reproductor),
  `base_multi.py` (cirugía), `progress.py` (época en vivo), `nvda.py`, `tests/`.
- `entrenar.py` / `entrenar_base.py` / `train_run.py` — entradas de entrenamiento.
- `preparar_base.py` (saneo de bases), `comparar_checkpoints.py`, `export_run.py`,
  `epoca_ckpt.py`, `normalizar_es.py`.
- `dataset_builder.py` + `gui_dataset.py` — armador de dataset.
- `base_ckpt/` (gitignored) — crudos + saneados. `training/`, `datasets/` (gitignored).
- `docs/` — spec y planes. `SETUP.md` — instalación (Camino A env-zip / B desde cero).

## Al terminar

- Corré `env\python.exe -m unittest discover studio/tests`.
- No commitees pesos ni datos: `*.ckpt`, `*.onnx`, `*.wav`, `env/`, `training/`,
  `datasets/`, `base_ckpt/` están gitignored (se pasan por zip, ver `SETUP.md`).
- Commit/push **solo si te lo piden**; primero rama si estás en `main`.
