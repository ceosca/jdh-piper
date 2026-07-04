"""
Armador de dataset automático para entrenar Piper.

Toma un audio largo O una carpeta de audios -> corta por silencios (ffmpeg) en
frases de 2-15s -> normaliza a 22050 Hz mono -> transcribe con faster-whisper
(GPU si hay, si no CPU) -> escribe formato LJSpeech (wavs/ + metadata.csv).

Uso (programático):
    from dataset_builder import build_dataset
    build_dataset(["voz.wav"], "dataset_mivoz", model_size="large-v3",
                  progress=print)
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from pathlib import Path

# --- CUDA DLLs para faster-whisper en Windows (cublas/cudnn de los pip nvidia) ---
def _setup_cuda_dlls():
    try:
        import nvidia
        dirs = []
        for base in nvidia.__path__:
            dirs += glob.glob(os.path.join(base, "*", "bin"))
        for d in dirs:
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
        if dirs:
            os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


_setup_cuda_dlls()

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus"}
SAMPLE_RATE = 22050


def _ffmpeg() -> str:
    return os.environ.get("FFMPEG_PATH") or "ffmpeg"


def _ffprobe_duration(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def detect_silences(path: str, noise_db: int = -30, min_sil: float = 0.4):
    """Devuelve lista de (inicio, fin) de silencios usando ffmpeg silencedetect."""
    proc = subprocess.run(
        [_ffmpeg(), "-i", path, "-af", f"silencedetect=noise={noise_db}dB:d={min_sil}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    log = proc.stderr
    starts = [float(m) for m in re.findall(r"silence_start:\s*([0-9.]+)", log)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([0-9.]+)", log)]
    sils = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else s + min_sil
        sils.append((s, e))
    return sils


def speech_segments(total_dur, silences, min_clip=2.0, max_clip=15.0, pad=0.15):
    """Regiones de voz (entre silencios), fusionando cortas y partiendo largas."""
    regions = []
    prev = 0.0
    for s, e in silences:
        if s > prev:
            regions.append((prev, s))
        prev = e
    if prev < total_dur:
        regions.append((prev, total_dur))

    segs = []
    for a, b in regions:
        a2 = max(0.0, a - pad)
        b2 = min(total_dur, b + pad)
        dur = b2 - a2
        if dur < min_clip:
            if segs and (a2 - segs[-1][1]) < 1.0 and (b2 - segs[-1][0]) <= max_clip:
                segs[-1] = (segs[-1][0], b2)   # fusionar con la anterior
            elif dur >= 1.0:
                segs.append((a2, b2))          # muy corta pero usable
        elif dur > max_clip:
            n = int(dur // max_clip) + 1
            step = dur / n
            for k in range(n):
                segs.append((a2 + k * step, min(b2, a2 + (k + 1) * step)))
        else:
            segs.append((a2, b2))
    return segs


def extract_clip(src: str, start: float, end: float, out_wav: str):
    """Extrae [start,end] a WAV 22050 mono."""
    subprocess.run(
        [_ffmpeg(), "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src,
         "-ac", "1", "-ar", str(SAMPLE_RATE), out_wav],
        capture_output=True,
    )


def _iter_inputs(inputs):
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
                    yield str(f)
        elif p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            yield str(p)


def build_dataset(inputs, out_dir, model_size="large-v3", language="es",
                  noise_db=-30, min_sil=0.4, min_clip=2.0, max_clip=15.0,
                  progress=None, stop_flag=None):
    """Arma el dataset LJSpeech. `inputs`: lista de archivos y/o carpetas."""
    def say(m):
        if progress:
            progress(m)

    out = Path(out_dir)
    wavs = out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)

    say(f"Cargando whisper {model_size}…")
    from faster_whisper import WhisperModel
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
        say("whisper en GPU.")
    except Exception:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        say("whisper en CPU (sin GPU).")

    rows = []
    idx = 0
    sources = list(_iter_inputs(inputs))
    if not sources:
        raise RuntimeError("No se encontraron audios en la entrada.")

    for si, src in enumerate(sources, 1):
        say(f"[{si}/{len(sources)}] Analizando silencios: {Path(src).name}")
        dur = _ffprobe_duration(src)
        sils = detect_silences(src, noise_db, min_sil)
        segs = speech_segments(dur, sils, min_clip, max_clip)
        say(f"  {len(segs)} frases detectadas.")
        for (a, b) in segs:
            if stop_flag is not None and stop_flag.is_set():
                say("Cancelado.")
                return str(out)
            idx += 1
            clip_id = f"clip_{idx:05d}"
            out_wav = str(wavs / f"{clip_id}.wav")
            extract_clip(src, a, b, out_wav)
            segments, _info = model.transcribe(out_wav, language=language, beam_size=5)
            text = " ".join(s.text for s in segments).strip()
            text = re.sub(r"\s+", " ", text)
            if not text:
                os.remove(out_wav)
                continue
            rows.append((clip_id, text))
            if idx % 10 == 0:
                say(f"  transcritas {idx} frases…")

    meta = out / "metadata.csv"
    with open(meta, "w", encoding="utf-8") as f:
        for clip_id, text in rows:
            f.write(f"{clip_id}|{text}\n")
    total, unicos = contar_duplicados([t for _, t in rows])
    if total > unicos:
        say(f"OJO: {total - unicos} de {total} clips están DUPLICADOS (mismo texto). "
            "Los duplicados pueden filtrar a validación y engañar el val_mel; "
            "conviene usar audio ÚNICO, no copiado.")
    say(f"LISTO: {len(rows)} frases -> {meta}")
    return str(out)


def contar_duplicados(textos) -> tuple[int, int]:
    """Devuelve (total, únicos) de una lista de textos. total>únicos ⇒ hay duplicados."""
    textos = [t.strip() for t in textos]
    return len(textos), len(set(textos))


def fila_multi(wav_id: str, speaker: str, text: str) -> str:
    """Fila de metadata multi-hablante: id|speaker|text (sin '|' en el texto)."""
    return f"{wav_id}|{speaker}|{text.replace('|', '').strip()}"


def build_multispeaker_dataset(speakers, out_dir, model_size="large-v3",
                               progress=None, stop_flag=None):
    """speakers: {nombre_hablante: [audios/carpetas]}. Arma wavs/ + metadata.csv
    (id|speaker|text). La fonemización la fija el entrenamiento, no el armado.
    Devuelve la cantidad de hablantes que quedaron con clips."""
    import csv as _csv

    def say(m):
        if progress:
            progress(m)

    out = Path(out_dir)
    wavs = out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    filas = []
    for speaker, entradas in speakers.items():
        if stop_flag is not None and stop_flag.is_set():
            break
        say(f"Procesando hablante: {speaker}")
        tmp = out / f"_tmp_{speaker}"          # carpeta temporal por hablante
        build_dataset(entradas, str(tmp), model_size=model_size,
                      progress=progress, stop_flag=stop_flag)
        meta = tmp / "metadata.csv"
        if meta.exists():
            with open(meta, "r", encoding="utf-8") as f:
                for row in _csv.reader(f, delimiter="|"):
                    if len(row) < 2:
                        continue
                    wid, text = row[0], row[-1]
                    new_id = f"{speaker}_{wid}"
                    src = tmp / "wavs" / f"{wid}.wav"
                    if src.exists():
                        src.replace(wavs / f"{new_id}.wav")
                        filas.append(fila_multi(new_id, speaker, text))
        shutil.rmtree(tmp, ignore_errors=True)  # limpiar la temporal
    (out / "metadata.csv").write_text("\n".join(filas) + "\n", encoding="utf-8")
    total, unicos = contar_duplicados([f.split("|", 2)[-1] for f in filas])
    if total > unicos:
        say(f"OJO: {total - unicos} de {total} clips duplicados (mismo texto) — "
            "conviene audio único.")
    n_hablantes = len({f.split("|")[1] for f in filas})  # los que quedaron con clips
    return n_hablantes
