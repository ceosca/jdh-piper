# Instalación de Piper Studio

Windows + GPU NVIDIA. Hay dos caminos: el **fácil** (con el `env/` ya armado en un
zip) y el **desde cero**.

---

## Camino A — Fácil (con el zip del entorno)

Es lo más rápido: el `env/` portable ya trae Python, todas las dependencias y los
parches aplicados.

1. **Cloná el repo.**
   ```
   git clone https://github.com/ceosca/jdh-piper
   ```
2. **Descomprimí el `env/`** que te pasaron **dentro de la carpeta del repo**
   (queda `jdh-piper/env/`).
3. **Poné el checkpoint base** en `base_ckpt/es/es_MX/ald/medium/` (te lo pasan en
   el zip, o bajalo — ver "Checkpoint base" abajo).
4. **Instalá los binarios de sistema** (ver abajo).
5. Listo: doble clic en **`Piper Studio.bat`** (o `Armar dataset.bat`).

> Con este camino NO hace falta `requirements.txt` ni `aplicar_parches.py`: ya está
> todo dentro del `env/`.

---

## Camino B — Desde cero (sin el zip)

1. **Cloná el repo** (igual que arriba).
2. **Creá un entorno Python 3.11** en `env/` (venv o micromamba) dentro del repo.
   El resto asume que `env/python.exe` existe.
3. **Instalá las dependencias** (ojo con torch — usa el índice CUDA):
   ```
   env\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
   env\python.exe -m pip install -r requirements.txt
   ```
4. **Aplicá los parches de entrenamiento** (código v1.4.2 + numba + val_mel):
   ```
   env\python.exe aplicar_parches.py
   ```
5. **Bajá el checkpoint base** (ver abajo).
6. **Binarios de sistema** (ver abajo).
7. Verificá:
   ```
   env\python.exe -m unittest discover studio/tests
   ```

---

## Checkpoint base (para fine-tune)

Descargá de [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints)
el **`es/es_MX/ald/medium/epoch=9999-step=1753600.ckpt`** y ponelo en
`base_ckpt/es/es_MX/ald/medium/`. Piper Studio lo **sanea solo** la primera vez
(`preparar_base.py` / auto). También podés recibirlo en el zip.

## Binarios de sistema (en el PATH)

- **ffmpeg** — recorte de audio y silencios (obligatorio).
- **yt-dlp** — opcional (no lo usa el flujo principal).
- **espeak-ng** — lo trae `piper-tts`, no hace falta instalarlo aparte.

## Voces

El repo NO incluye voces entrenadas (`.onnx`). Opciones:
- Entrená las tuyas (pestaña Entrenar).
- Copiá `.onnx` + `.onnx.json` a la carpeta de voces (por defecto
  `C:\ia\modelos pc\piper\voces\<nombre>\`, configurable con la variable de
  entorno `PIPER_PLAYER_VOICES`).

## Nota de licencia

El código de entrenamiento en `setup/piper_train/` deriva de
[piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) **v1.4.2** (GPL-3.0) con dos
parches (monotonic_align numba + logging de `val_mel`).
