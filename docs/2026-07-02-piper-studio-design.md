# Piper Studio — Diseño

Fecha: 2026-07-02
Estado: aprobado (brainstorming)

## Objetivo

Dos herramientas Piper, **separadas**, en Windows nativo:

1. **Entrenador** (en la máquina del user, RTX 5070): arma el dataset solo y hace
   fine-tuning de una voz Piper con la GPU.
2. **Reproductor** (portable, para la PC de Franco, RX 580 + i5): TTS multi-voz
   accesible, corre en CPU, instantáneo. Liviano — no carga nada de entrenamiento.

Motivación: Piper es rápido de arrancar y generar (sin esperas de carga como fish),
ideal para el uso diario y para el equipo de Franco.

## Deliverable A — Entrenador  (`C:\ia\piper-studio\`)

Un entorno micromamba (Python 3.11) + GPU (torch cu128), con una GUI accesible de
DOS pestañas (o dos scripts, a definir en el plan). Solo en la máquina del user.

### A.1 Armador de dataset (automático)
- Entrada: **un audio largo O una carpeta de audios** (ambos soportados).
- Corte por **silencios con ffmpeg** (`silencedetect`) → frases de ~2 a 15 s.
- Normaliza a **22050 Hz mono**.
- Transcribe cada frase con **faster-whisper (large-v3, español)** en la GPU.
- Salida en **formato LJSpeech**: `wavs/` + `metadata.csv` (`id|texto`).
- 100% automático ("se hace solito"). Revisión opcional del metadata antes de entrenar.

### A.2 Entrenamiento (fine-tuning)
- Parte de un **checkpoint base de voz Piper en español** (se descarga).
- Fonemiza el dataset con el **binario espeak-ng de Windows** (evita la extensión C
  con CMake, que es inviable en Windows nativo).
- Compila **monotonic_align** (Cython) una vez, con un compilador instalado en el env
  conda (sin depender de Visual Studio del sistema).
- Entrena VITS (piper1-gpl, PyTorch, GPU) → **exporta a `.onnx` + `.onnx.json`**.
- Progreso/log visible en la GUI (entrena horas).

### Riesgos (nativo) y red de contención
Puntos frágiles: build de `monotonic_align`, plumbing de espeak-ng, adaptar los
scripts Linux de piper1-gpl a Windows. Si un paso se traba de forma irresoluble,
**WSL2 queda como fallback** para SOLO el paso de entrenamiento (el resto —dataset
y reproductor— es nativo igual).

## Deliverable B — Reproductor  (`C:\ia\modelos pc\piper\`, portable)

Herramienta separada y liviana, misma línea que xtts-argentino / f5-tts / qwen-tts.

- Entorno micromamba propio (Python 3.11) + Piper de **inferencia** (CPU, sin torch
  de entrenamiento pesado).
- **GUI accesible idéntica en filosofía** a las otras: una pantalla, NVDA, biblioteca
  de voces, **multi-voz inline con `#nombredevoz`** + menú contextual (tecla
  Aplicaciones) submenú "Voces", fragmentos (reproducir/regenerar/editar), guardar WAV.
- Diferencia clave con las otras: Piper **no clona por referencia** — usa **voces
  entrenadas** (archivos `.onnx`). La "biblioteca de voces" son los `.onnx` (cada voz
  = un modelo entrenado). `#voz` cambia de modelo `.onnx` por tramo.
- Arranque y generación **instantáneos** (CPU). Copiable tal cual a la PC de Franco.

## Flujo end-to-end
audio(s) → [A.1 dataset builder] → dataset LJSpeech → [A.2 fine-tune] → voz `.onnx`
→ se copia el `.onnx` a la carpeta de voces del **Reproductor** → Franco la usa.

## Stack
- micromamba, Python 3.11.
- Entrenador: torch cu128 (GPU), faster-whisper, piper1-gpl (train), espeak-ng
  (binario Windows), compilador conda para monotonic_align, ffmpeg, wxPython.
- Reproductor: piper-tts (inferencia CPU), sounddevice, wxPython, nvdaControllerClient.dll.

## No incluye (YAGNI)
- Entrenar desde cero (solo fine-tuning).
- Clonación por referencia (Piper no la hace; para eso están xtts/fish).
- Nada de entrenamiento en el reproductor (liviano).
