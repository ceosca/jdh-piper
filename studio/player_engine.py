"""Motor Piper para el reproductor interno de Piper Studio.

Igual que el reproductor CPU (voces .onnx en carpetas, multi-voz), pero usa GPU
si onnxruntime tiene CUDAExecutionProvider (si no, cae a CPU). Comparte la carpeta
de voces con el reproductor CPU (env PIPER_PLAYER_VOICES).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from piper import PiperVoice

VOICES_DIR = Path(os.environ.get("PIPER_PLAYER_VOICES", r"C:\ia\modelos pc\piper\voces"))

try:
    from normalizar_es import normalizar
except Exception:  # pragma: no cover
    def normalizar(t):
        return t


class VoiceLibrary:
    def __init__(self, root: Path = VOICES_DIR):
        self.root = Path(root)

    def voices(self) -> list[str]:
        if not self.root.exists():
            return []
        return [p.name for p in sorted(self.root.iterdir())
                if p.is_dir() and any(p.glob("*.onnx"))]

    def onnx_for(self, name: str) -> Path | None:
        folder = self.root / name
        if not folder.exists():
            return None
        onnxs = sorted(folder.glob("*.onnx"))
        return onnxs[0] if onnxs else None


class TTSEngine:
    NAME = "Piper (multi-voz, GPU)"

    def __init__(self):
        self._library = VoiceLibrary()
        self._voices: dict[str, PiperVoice] = {}
        self.loaded = False
        self._sr = 22050
        self._lock = threading.Lock()
        self._cuda = False
        self.device = "?"

    def load(self, progress=None):
        try:
            import onnxruntime as ort
            self._cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            self._cuda = False
        self.device = "GPU" if self._cuda else "CPU"
        if progress:
            progress(f"Piper listo ({self.device}).")
        self.loaded = True

    def _get_voice(self, name: str) -> PiperVoice:
        with self._lock:
            if name not in self._voices:
                onnx = self._library.onnx_for(name)
                if not onnx:
                    raise RuntimeError(f"No se encontró el .onnx de la voz «{name}».")
                try:
                    self._voices[name] = PiperVoice.load(str(onnx), use_cuda=self._cuda)
                except Exception:
                    self._voices[name] = PiperVoice.load(str(onnx))  # fallback CPU
            return self._voices[name]

    def generate(self, text: str, voice_name: str = ""):
        if not self.loaded:
            raise RuntimeError("El motor no está listo.")
        if not voice_name:
            raise RuntimeError("Elegí una voz (o usá #nombredevoz en el texto).")
        v = self._get_voice(voice_name)
        text = normalizar(text)  # números/ordinales/unidades -> palabras
        chunks = list(v.synthesize(text))
        if not chunks:
            raise RuntimeError("No se generó audio (¿texto vacío?).")
        audio = np.concatenate(
            [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks]
        ).astype(np.float32) / 32768.0
        self._sr = chunks[0].sample_rate
        return self._sr, audio

    @property
    def sample_rate(self) -> int:
        return self._sr
