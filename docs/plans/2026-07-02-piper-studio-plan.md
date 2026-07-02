# Piper Studio — Plan de implementación

> Ejecución INLINE en la sesión (proyecto exploratorio en la máquina del user;
> mucho debug específico de Windows). Cada tarea termina en un deliverable
> verificable en vivo, no en tests unitarios ficticios.

**Goal:** Dos herramientas Piper en Windows nativo — un Reproductor portable
multi-voz (para Franco, CPU) y un Entrenador (dataset auto + fine-tuning en GPU).

**Arquitectura:** Reproductor en `C:\ia\modelos pc\piper\` (liviano, piper CPU,
GUI accesible estilo xtts/f5/qwen). Entrenador en `C:\ia\piper-studio\` (env con
GPU, whisper, piper1-gpl). Se comunican por archivos `.onnx` (voz entrenada →
carpeta de voces del reproductor).

**Tech:** micromamba py3.11, piper-tts (infer CPU), piper1-gpl (train GPU),
faster-whisper, ffmpeg, espeak-ng (Windows), wxPython, sounddevice.

## Constraintes globales
- Windows nativo (WSL solo como fallback del paso de entrenamiento).
- Reproductor: liviano, arranque/gen instantáneos, portable a otra PC Windows x64.
- Accesibilidad: NVDA (nvdaControllerClient.dll), foco→anuncio, teclado.
- Multi-voz inline `#nombredevoz` + menú tecla Aplicaciones, como las otras tools.

---

## FASE 1 — Reproductor Piper (portable, Franco)  [BAJO riesgo, valor inmediato]

Independiente: se construye y prueba con una voz Piper en español ya entrenada
(descargada), sin depender del entrenador. Le da a Franco algo usable YA.

### Tarea 1.1 — Entorno + Piper inferencia
- Crear `C:\ia\modelos pc\piper\env` (micromamba py3.11).
- `pip install piper-tts sounddevice`. Copiar `nvdaControllerClient.dll`.
- **Verificar:** `python -c "import piper; import sounddevice"` OK.

### Tarea 1.2 — Descargar una voz base en español (.onnx)
- Bajar una voz Piper es_* (ej. es_AR o es_ES) de HuggingFace rhasspy/piper-voices
  a `piper/voces/<nombre>/` (archivos `.onnx` + `.onnx.json`).
- **Verificar:** `piper` genera un WAV de prueba desde CLI/python con esa voz.

### Tarea 1.3 — Motor `engine.py` (interfaz común)
- `TTSEngine` con `load(progress)` + `generate(text, voice_onnx)->(sr, np.wav)`.
  Carga cada `.onnx` bajo demanda (cachea los cargados). Piper síntesis → wav 22050.
- Biblioteca de voces = subcarpetas en `voces/` con un `.onnx` cada una.
- **Verificar:** script corto genera audio con la voz descargada; medir latencia
  (debe ser ~instantáneo).

### Tarea 1.4 — GUI accesible multi-voz
- Adaptar la `gui.py` de las otras tools: una pantalla, NVDA, elegir voz,
  cuadro de texto, menú Aplicaciones → submenú "Voces" que inserta `#voz`,
  parseo `#voz` → fragmentos, reproducir/regenerar/editar, "Reproducir todo",
  "Guardar WAV". Sin "Agregar voz por audio" (Piper no clona; las voces son .onnx).
- Lanzador `Piper.bat`. LEEME.
- **Verificar:** la GUI arranca, carga la voz, genera y reproduce; navegación con
  lector OK (smoke test: proceso vivo + genera sin error).

**Deliverable Fase 1:** carpeta `modelos pc\piper` portable, con TTS multi-voz
accesible funcionando con voces es_* descargadas. Copiable a Franco.

---

## FASE 2 — Armador de dataset (automático)  [BAJO-MEDIO riesgo]

En `C:\ia\piper-studio\`. Nativo (ffmpeg + faster-whisper GPU).

### Tarea 2.1 — Entorno entrenador (base)
- Crear `C:\ia\piper-studio\env` (micromamba py3.11) + torch cu128 (GPU) +
  `faster-whisper`. Verificar `torch.cuda.is_available()` y whisper carga.

### Tarea 2.2 — Corte por silencios (ffmpeg)
- `dataset_builder.py`: función que toma un audio y usa ffmpeg `silencedetect`
  para obtener tiempos de silencio; recorta en frases de 2-15s (junta cortas,
  parte largas por el silencio más cercano); exporta cada frase a WAV 22050 mono.
- **Verificar:** con un audio de prueba, genera N clips en rango 2-15s.

### Tarea 2.3 — Transcripción whisper + LJSpeech
- Transcribir cada clip con faster-whisper (large-v3, es) en GPU.
- Escribir `metadata.csv` (`id|texto`) + `wavs/`. Aceptar 1 archivo O carpeta.
- Normalización básica de texto (minúsculas/espacios, opcional).
- **Verificar:** dataset LJSpeech válido; revisar 3-4 transcripciones a mano.

### Tarea 2.4 — GUI/CLI del armador
- Pestaña o script "Dataset": elegir entrada (archivo/carpeta) + salida, correr,
  mostrar progreso. Accesible.
- **Verificar:** flujo entrada→dataset completo desde la interfaz.

**Deliverable Fase 2:** de audio(s) a dataset LJSpeech listo para entrenar, solo.

---

## FASE 3 — Entrenador (fine-tuning)  [ALTO riesgo — exploratorio]

El paso frágil en Windows nativo. Verificación gradual; WSL de fallback.

### Tarea 3.1 — Instalar piper1-gpl (train) + probar imports
- Instalar piper1-gpl con extras de training en el env. Ver qué rompe.
- **Verificar:** importar el módulo de training; listar los blockers reales.

### Tarea 3.2 — monotonic_align (Cython) en Windows
- Instalar un compilador en el env (conda-forge compilers / m2w64) y compilar
  `monotonic_align` (`build_ext --inplace`).
- **Verificar:** `import monotonic_align` OK.

### Tarea 3.3 — Fonemización con espeak-ng (binario Windows)
- Instalar/ubicar espeak-ng para Windows. Adaptar el preprocess de piper para
  fonemizar el metadata con espeak-ng (subproceso o wrapper), produciendo el
  config/dataset que espera el training.
- **Verificar:** preprocess genera el dataset fonemizado sin la extensión C.

### Tarea 3.4 — Fine-tuning en GPU + export ONNX
- Bajar checkpoint base es_*. Correr fine-tuning (pocas épocas) sobre el dataset.
- Exportar a `.onnx` + `.onnx.json`.
- **Verificar:** entrena algunas iteraciones sin OOM; exporta un `.onnx` cargable.
- **Fallback:** si 3.1-3.4 se traban irremediablemente en nativo, hacer SOLO este
  paso en WSL2 (documentarlo); el resto queda nativo.

### Tarea 3.5 — Integración
- Copiar el `.onnx` entrenado a `modelos pc\piper\voces\` y probarlo en el
  Reproductor (Fase 1).
- **Verificar:** el reproductor genera con la voz recién entrenada.

**Deliverable Fase 3:** voz propia entrenada, usable en el reproductor de Franco.

---

## Orden recomendado
Fase 1 (player, valor ya) → Fase 2 (dataset) → Fase 3 (train, riesgoso).
Cada fase es útil por sí sola.
