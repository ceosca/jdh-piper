# Plan 3 — Infraestructura del base multi-hablante (Opción 1: cirugía / warm-start parcial)

> **Para quien ejecute:** SUB-SKILL REQUERIDA: superpowers:subagent-driven-development. Vive en `C:\ia\piper-studio` (repo git, rama nueva). Los pasos usan checkbox (`- [ ]`).

**Goal:** poder entrenar un **base multi-hablante** en español latino (`es-419`) partiendo de un checkpoint de un solo hablante mediante **cirugía de pesos**: se copian los ~784 pesos acústicos/fonéticos del modelo mono a un modelo multi-hablante nuevo, y solo las ~20 "perillas de hablante" (`emb_g` + capas `cond`) arrancan de cero. Incluye armar el dataset multi-hablante, el script de entrenamiento y el modo en la GUI. **El entrenamiento real lo corre el usuario cuando tenga el corpus**; este plan construye y testea la infraestructura.

**Architecture:** un módulo puro `base_multi.py` (fusión de pesos + hparams multi, testeable) usado por `entrenar_base.py` (construye `VitsModel` multi + `VitsDataModule`, inyecta los pesos del mono con `load_state_dict`, y hace `trainer.fit` fresco con checkpoints). El armador de dataset gana un modo multi-hablante (CSV `wav|speaker|text`). La GUI (`section_train`) gana el modo "Base multi-hablante", que despacha `entrenar_base.py` vía `runs.build_train_argv`.

**Tech Stack:** Python (env `C:\ia\piper-studio\env\python.exe`), PyTorch + Lightning, piper1-gpl 1.4.2 (`piper.train.vits`), `unittest`. GPU RTX 5070.

## Global Constraints

- Windows nativo, sin WSL. Rutas con `pathlib`. `--data.num_workers 0` obligatorio (workers cuelgan en Windows).
- Stack estándar (piper1-gpl 1.4.2 + espeak). Fonemización **`es-419`** para unificar acentos.
- Parches de carga ya conocidos: `torch.load(weights_only=False)` + `pathlib.PosixPath = pathlib.WindowsPath` (usar el patrón de `train_run.py`).
- **Hecho verificado (grounding):** un `VitsModel` con `num_speakers=N, gin_channels=256` tiene 804 claves; **784 coinciden en nombre+forma con el checkpoint mono** (`base_ckpt/silvio_base_clean.ckpt`), 0 mismatches; las 20 nuevas son `model_g.emb_g.weight`, `model_g.dec.cond.*`, `model_g.dp.cond.*`, `model_g.enc_q.enc.cond_layer.*`, `model_g.flow.flows.N.enc.cond_layer.*`.
- CSV multi-hablante = `wav|speaker|text` (delimitado por `|`). El modelo arma el `speaker_id_map` solo.
- Código y UI en español.

## Estructura de archivos

- Create: `studio/base_multi.py` — fusión de pesos + hparams multi (puro, testeable).
- Create: `studio/tests/test_base_multi.py`.
- Create: `entrenar_base.py` — entry de entrenamiento del base (cirugía en memoria + fit).
- Modify: `dataset_builder.py` — modo multi-hablante (`build_multispeaker_dataset`).
- Create: `studio/tests/test_dataset_multi.py`.
- Modify: `studio/runs.py` — rama `modo=="base"` en `build_train_argv`.
- Modify: `studio/section_train.py` — selector de modo "Base multi-hablante".

---

### Task 1: `base_multi.py` — fusión de pesos e hparams multi (puro + tests)

**Files:**
- Create: `studio/base_multi.py`
- Test: `studio/tests/test_base_multi.py`

**Interfaces:**
- Produces:
  - `hparams_multi(hp_mono: dict, num_speakers: int, gin_channels: int = 256) -> dict` — copia los hparams del mono válidos para `VitsModel.__init__` y fija `num_speakers`/`gin_channels`.
  - `fusionar_pesos(mono_sd: dict, multi_sd: dict) -> tuple[dict, int, int]` — devuelve `(merged, n_copiadas, n_nuevas)`: parte de `multi_sd` (init aleatorio) y copia de `mono_sd` toda clave que exista en ambos con **misma forma**; el resto queda como estaba. `n_nuevas` = claves de `multi_sd` no copiadas.

- [ ] **Step 1: escribir el test (falla)**

```python
# studio/tests/test_base_multi.py
import unittest
import torch
from studio.base_multi import hparams_multi, fusionar_pesos


class TestBaseMulti(unittest.TestCase):
    def test_hparams_multi_fija_speakers_y_gin(self):
        hp = {"num_symbols": 256, "num_speakers": 1, "gin_channels": 0,
              "inter_channels": 192, "no_valido_xyz": 1}
        out = hparams_multi(hp, num_speakers=8, gin_channels=256)
        self.assertEqual(out["num_speakers"], 8)
        self.assertEqual(out["gin_channels"], 256)
        self.assertEqual(out["inter_channels"], 192)
        self.assertNotIn("no_valido_xyz", out)  # se filtra a args de VitsModel

    def test_fusionar_copia_coincidentes_y_conserva_nuevas(self):
        mono = {"a": torch.ones(3), "b": torch.ones(2, 2)}
        multi = {"a": torch.zeros(3), "b": torch.zeros(2, 2),
                 "emb_g": torch.zeros(5), "cond": torch.zeros(4)}
        merged, n_cop, n_new = fusionar_pesos(mono, multi)
        self.assertTrue(torch.equal(merged["a"], torch.ones(3)))      # copiada
        self.assertTrue(torch.equal(merged["b"], torch.ones(2, 2)))   # copiada
        self.assertTrue(torch.equal(merged["emb_g"], torch.zeros(5)))  # nueva (init)
        self.assertTrue(torch.equal(merged["cond"], torch.zeros(4)))   # nueva (init)
        self.assertEqual(n_cop, 2)
        self.assertEqual(n_new, 2)

    def test_fusionar_ignora_shape_distinta(self):
        mono = {"a": torch.ones(3)}
        multi = {"a": torch.zeros(9)}          # misma clave, otra forma
        merged, n_cop, n_new = fusionar_pesos(mono, multi)
        self.assertTrue(torch.equal(merged["a"], torch.zeros(9)))  # NO copia
        self.assertEqual(n_cop, 0)
        self.assertEqual(n_new, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: correr y ver fallar**

Run: `cd /c/ia/piper-studio && "C:/ia/piper-studio/env/python.exe" -m unittest studio.tests.test_base_multi -v`
Expected: FAIL (`No module named 'studio.base_multi'`).

- [ ] **Step 3: implementar `base_multi.py`**

```python
# studio/base_multi.py
"""Cirugía de pesos mono -> multi-hablante para VITS (Piper)."""
from __future__ import annotations

import inspect


def hparams_multi(hp_mono: dict, num_speakers: int, gin_channels: int = 256) -> dict:
    """hparams del base mono, filtrados a args de VitsModel, con speakers/gin fijados."""
    from piper.train.vits.lightning import VitsModel
    validos = set(inspect.signature(VitsModel.__init__).parameters) - {"self", "kwargs"}
    hp = {k: v for k, v in hp_mono.items() if k in validos}
    hp["num_speakers"] = int(num_speakers)
    hp["gin_channels"] = int(gin_channels)
    return hp


def fusionar_pesos(mono_sd: dict, multi_sd: dict):
    """Parte del state_dict multi (init) y copia de mono toda clave con misma forma.

    Devuelve (merged, n_copiadas, n_nuevas). n_nuevas = claves de multi que quedaron
    con su valor inicial (p.ej. emb_g y las capas cond de condicionamiento por hablante).
    """
    merged = dict(multi_sd)
    copiadas = 0
    for k, v in mono_sd.items():
        if k in merged and hasattr(v, "shape") and merged[k].shape == v.shape:
            merged[k] = v
            copiadas += 1
    nuevas = len(multi_sd) - copiadas
    return merged, copiadas, nuevas
```

- [ ] **Step 4: correr y ver pasar**

Run: `cd /c/ia/piper-studio && "C:/ia/piper-studio/env/python.exe" -m unittest studio.tests.test_base_multi -v`
Expected: PASS (3 tests).

---

### Task 2: `entrenar_base.py` — entrenamiento del base con cirugía

**Files:**
- Create: `entrenar_base.py`

**Interfaces:**
- Consumes: `studio.base_multi.hparams_multi`/`fusionar_pesos`, `piper.train.vits.lightning.VitsModel`, `piper.train.vits.dataset.VitsDataModule`, `lightning`.
- Produces: CLI `entrenar_base.py --dataset <dir> --base-mono <ckpt> --num-speakers N [--gin-channels 256 --max-epochs 4000 --batch-size 12 --espeak es-419]` que entrena el base y guarda checkpoints en `training/<nombre_dataset>/ckpts`.

- [ ] **Step 1: implementar `entrenar_base.py`**

```python
# entrenar_base.py
"""Entrena un base multi-hablante desde un checkpoint mono, por cirugía de pesos.

Construye un VitsModel multi-hablante, le inyecta los ~784 pesos del mono
(fusionar_pesos deja random solo emb_g + capas cond), y entrena de cero (fit).
Uso:
  env\\python.exe entrenar_base.py --dataset datasets/base_latino --base-mono base_ckpt/silvio_base_clean.ckpt --num-speakers 12
"""
import argparse
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
    ap.add_argument("--espeak", default="es-419")
    ap.add_argument("--sample-rate", type=int, default=22050)
    args = ap.parse_args()

    ds = Path(args.dataset)
    nombre = ds.name
    ckpts = ROOT / "training" / nombre / "ckpts"
    ckpts.mkdir(parents=True, exist_ok=True)

    # hparams multi a partir del mono
    ck = torch.load(args.base_mono, map_location="cpu")
    hp = hparams_multi(ck["hyper_parameters"], args.num_speakers, args.gin_channels)
    hp["batch_size"] = args.batch_size

    model = VitsModel(**hp)
    merged, n_cop, n_new = fusionar_pesos(ck["state_dict"], model.state_dict())
    model.load_state_dict(merged)
    print(f"[cirugía] copiadas={n_cop} nuevas(random)={n_new}")

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
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: verificación de construcción (sin entrenar, sin dataset real)**

Run:
```
cd /c/ia/piper-studio && "C:/ia/piper-studio/env/python.exe" -c "import pathlib; pathlib.PosixPath=pathlib.WindowsPath; import torch; _l=torch.load; torch.load=lambda *a,**k:_l(*a,**{**k,'weights_only':False}); from studio.base_multi import hparams_multi, fusionar_pesos; from piper.train.vits.lightning import VitsModel; ck=torch.load('base_ckpt/silvio_base_clean.ckpt',map_location='cpu'); hp=hparams_multi(ck['hyper_parameters'],12,256); m=VitsModel(**hp); merged,c,n=fusionar_pesos(ck['state_dict'],m.state_dict()); m.load_state_dict(merged); print('OK copiadas',c,'nuevas',n)"
```
Expected: `OK copiadas 784 nuevas 20` (confirma la cirugía + carga sin error). NO entrenar (requiere corpus + GPU largo; lo corre el humano).

---

### Task 3: `dataset_builder.py` — modo multi-hablante

**Files:**
- Modify: `dataset_builder.py`
- Test: `studio/tests/test_dataset_multi.py`

**Interfaces:**
- Consumes: las funciones existentes de `dataset_builder` (silencios + whisper) por carpeta.
- Produces: `build_multispeaker_dataset(speakers: dict[str, list[str]], out_dir, model_size="large-v3", espeak="es-419", progress=None, stop_flag=None) -> int` — `speakers` mapea nombre_de_hablante → lista de audios/carpetas; arma `out_dir/wavs/` y `out_dir/metadata.csv` con filas `id|speaker|text`; devuelve la **cantidad de hablantes**. Además una pura `fila_multi(wav_id, speaker, text) -> str` que produce `"wav_id|speaker|text"`.

- [ ] **Step 1: test de la fila multi (falla)**

```python
# studio/tests/test_dataset_multi.py
import unittest
from dataset_builder import fila_multi


class TestDatasetMulti(unittest.TestCase):
    def test_fila_multi_formato(self):
        self.assertEqual(fila_multi("clip_0", "silvio", "hola mundo"),
                         "clip_0|silvio|hola mundo")

    def test_fila_multi_limpia_pipes_del_texto(self):
        # el texto no debe romper el CSV con '|'
        self.assertEqual(fila_multi("c1", "ana", "uno | dos"), "c1|ana|uno  dos")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: correr y ver fallar**

Run: `cd /c/ia/piper-studio && "C:/ia/piper-studio/env/python.exe" -m unittest studio.tests.test_dataset_multi -v`
Expected: FAIL (`cannot import name 'fila_multi'`).

- [ ] **Step 3: implementar en `dataset_builder.py`**

Agregar la helper pura y la función de armado (reusa el pipeline por-carpeta existente; `build_dataset` ya produce clips + transcripción — se envuelve por hablante):

```python
def fila_multi(wav_id: str, speaker: str, text: str) -> str:
    """Fila de metadata multi-hablante: id|speaker|text (sin '|' en el texto)."""
    return f"{wav_id}|{speaker}|{text.replace('|', ' ').strip()}"


def build_multispeaker_dataset(speakers, out_dir, model_size="large-v3",
                               espeak="es-419", progress=None, stop_flag=None):
    """speakers: {nombre_hablante: [audios/carpetas]}. Arma wavs/ + metadata.csv
    (id|speaker|text). Devuelve la cantidad de hablantes."""
    from pathlib import Path
    out = Path(out_dir)
    wavs = out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    filas = []
    for speaker, entradas in speakers.items():
        if stop_flag is not None and stop_flag.is_set():
            break
        if progress:
            progress(f"Procesando hablante: {speaker}")
        # build_dataset arma clips+texto en una carpeta temporal por hablante
        tmp = out / f"_tmp_{speaker}"
        build_dataset(entradas, str(tmp), model_size=model_size,
                      progress=progress, stop_flag=stop_flag)
        # mover wavs con prefijo de hablante y volcar filas con la columna speaker
        import csv as _csv
        with open(tmp / "metadata.csv", "r", encoding="utf-8") as f:
            for row in _csv.reader(f, delimiter="|"):
                if len(row) < 2:
                    continue
                wid, text = row[0], row[-1]
                new_id = f"{speaker}_{wid}"
                (tmp / "wavs" / f"{wid}.wav").replace(wavs / f"{new_id}.wav")
                filas.append(fila_multi(new_id, speaker, text))
    (out / "metadata.csv").write_text("\n".join(filas) + "\n", encoding="utf-8")
    return len(speakers)
```

(Nota: `build_dataset` ya existe y acepta `espeak`/idioma vía whisper; la fonemización `es-419` la fija el entrenamiento, no el armado. Si `build_dataset` no expone `espeak`, omitir ese kwarg — el armado solo transcribe.)

- [ ] **Step 4: correr y ver pasar**

Run: `cd /c/ia/piper-studio && "C:/ia/piper-studio/env/python.exe" -m unittest studio.tests.test_dataset_multi -v`
Expected: PASS (2 tests). (El armado real multi-hablante lo prueba el humano con audios; el test cubre el formato de fila.)

---

### Task 4: `runs.py` — rama `modo=="base"` en `build_train_argv`

**Files:**
- Modify: `studio/runs.py`
- Test: `studio/tests/test_runs.py` (agregar)

**Interfaces:**
- Consumes: `RunState` (ya tiene `modo`, `dataset`, `base_ckpt`, `max_epochs`; se reutilizan `paciencia`/`cada` NO aplican al base).
- Produces: cuando `st.modo == "base"`, `build_train_argv` arma `entrenar_base.py --dataset <dir> --base-mono <ckpt> --num-speakers <n> ...`. El número de hablantes se guarda en `RunState` (nuevo campo `num_speakers: int = 1`).

- [ ] **Step 1: test (falla)**

```python
# agregar a studio/tests/test_runs.py
class TestArgvBase(unittest.TestCase):
    def test_modo_base_usa_entrenar_base(self):
        from studio.runs import RunState, build_train_argv
        from pathlib import Path
        st = RunState(nombre="base_latino", modo="base", dataset="datasets/base_latino",
                      base_ckpt="base_ckpt/silvio_base_clean.ckpt", max_epochs=4000,
                      num_speakers=12)
        argv = build_train_argv("py.exe", Path("."), st)
        j = " ".join(argv)
        self.assertIn("entrenar_base.py", j)
        self.assertIn("--num-speakers", argv)
        self.assertIn("12", argv)
        self.assertIn("--base-mono", argv)
```

- [ ] **Step 2: correr y ver fallar** — Run el módulo de tests; FAIL por `num_speakers` inexistente / rama faltante.

- [ ] **Step 3: implementar** — en `studio/runs.py`:
  a. Añadir a `RunState` el campo `num_speakers: int = 1`.
  b. Al principio de `build_train_argv`, antes de la lógica actual:

```python
    if st.modo == "base":
        return [py, str(Path(root_proj) / "entrenar_base.py"),
                "--dataset", st.dataset,
                "--base-mono", st.base_ckpt,
                "--num-speakers", str(st.num_speakers),
                "--max-epochs", str(st.max_epochs)]
```

- [ ] **Step 4: correr y ver pasar** — toda la suite `test_runs` PASA (incluida la nueva).

---

### Task 5: GUI — modo "Base multi-hablante" en la sección Entrenar

**Files:**
- Modify: `studio/section_train.py`

**Interfaces:**
- Consumes: `runs.RunState` (con `modo`/`num_speakers`).
- Produces: un selector de modo (radio: "Fine-tune de una voz" | "Base multi-hablante") y un campo `Nº de hablantes` (visible en modo base). `_current_state` setea `modo` y `num_speakers` según la UI.

- [ ] **Step 1:** agregar el selector y el campo (etiquetas creadas ANTES de su control, por NVDA):

```python
        modo_lbl = wx.StaticText(self, label="Modo:")
        self.modo = wx.Choice(self, choices=["Fine-tune de una voz", "Base multi-hablante"],
                              name="Modo de entrenamiento")
        self.modo.SetSelection(0)
        spk_lbl = wx.StaticText(self, label="Nº de hablantes:")
        self.num_spk = wx.SpinCtrl(self, min=1, max=1000, initial=10, name="Número de hablantes")
        self.num_spk.Enable(False)
        # ...añadir modo_lbl+self.modo y spk_lbl+self.num_spk a un sizer horizontal
        self.modo.Bind(wx.EVT_CHOICE, self._toggle_modo)
```
```python
    def _toggle_modo(self, e):
        es_base = self.modo.GetSelection() == 1
        self.num_spk.Enable(es_base)
```

- [ ] **Step 2:** en `_current_state`, reflejar el modo:

```python
        es_base = self.modo.GetSelection() == 1
        return runs.RunState(
            nombre=self.name_ctrl.GetValue().strip() or "mivoz",
            modo="base" if es_base else "finetune",
            num_speakers=self.num_spk.GetValue(),
            dataset=...,  # igual que antes
            base_ckpt=str(DEFAULT_BASE),
            max_epochs=self.epochs.GetValue(),
            auto_stop=self.auto.GetValue(),
            paciencia=self.paciencia.GetValue(),
            cada=self.cada.GetValue(),
            started_at=dt.datetime.now().isoformat(timespec="seconds"),
        )
```

- [ ] **Step 3:** verificación de import: `import studio.section_train, studio.app` exit 0. (La prueba con NVDA/lanzamiento la hace el humano.)

- [ ] **Step 4:** commit.

---

## Auto-revisión del plan

- **Cobertura de la spec (backend multi-hablante):** cirugía de pesos (T1, grounded 784/20) ✓; entrenamiento con inyección (T2) ✓; dataset `wav|speaker|text` (T3) ✓; despacho desde runs (T4) ✓; modo en la GUI (T5) ✓.
- **Sin placeholders:** código real + tests + comandos con salida esperada (incl. la verificación `OK copiadas 784 nuevas 20`).
- **Consistencia de tipos:** `RunState.num_speakers` (T4) lo usan `build_train_argv` (T4) y `section_train._current_state` (T5); `entrenar_base.py` (T2) consume `hparams_multi`/`fusionar_pesos` (T1) con las firmas definidas.
- **Fuera de alcance (del usuario):** juntar el corpus multi-acento y correr el entrenamiento real (GPU, días, pausable con la GUI ya existente); elegir `gin_channels` (default 256). La fonemización `es-419` la fija `entrenar_base.py`.
- **Riesgo:** `VitsDataModule`/`VitsModel` instanciados directo (no vía LightningCLI) — la verificación de construcción de T2 lo cubre; el `fit` real puede requerir ajustes menores de args del DataModule (validarlos en la primera corrida del humano).
```
