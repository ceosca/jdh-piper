# Exportar para sherpa-onnx — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agregar a la pestaña Exportar un botón que empaquete una voz Piper (`.onnx` + `.onnx.json`) en una carpeta autocontenida lista para sherpa-onnx.

**Architecture:** Un módulo nuevo `studio/sherpa_export.py` separa lógica pura (arma `tokens.txt` y el dict de metadatos) de la I/O (copia archivos, embebe metadatos en el onnx, copia `espeak-ng-data`). La I/O recibe un `add_meta` inyectable para testearse sin onnx real. La pestaña Exportar (`studio/section_export.py`) suma un botón que llama a `empaquetar` en un hilo.

**Tech Stack:** Python 3.11, wxPython, `onnx` (ya instalado, 1.22.0), unittest. Todo se corre con `env\python.exe`.

## Global Constraints

- Windows nativo; correr todo con `env\python.exe` (nunca `python` a secas).
- Tests con `unittest` (NO pytest): `.\env\python.exe -m unittest ...`.
- Sin dependencias nuevas (`onnx` ya está; sherpa-onnx NO se instala — solo se empaqueta).
- espeak-ng-data se reusa del env: `env/Lib/site-packages/piper/espeak-ng-data`.
- Accesibilidad NVDA: en la GUI, el `wx.StaticText` (etiqueta) se crea ANTES de su control. (Este plan agrega un botón sin etiqueta previa, así que no aplica, pero no romper el patrón existente.)
- Metadatos que sherpa lee del onnx (claves exactas): `model_type="vits"`, `comment="piper"`, `language`, `voice`, `has_espeak=1`, `n_speakers`, `sample_rate`.
- `tokens.txt`: por cada `(símbolo, ids)` de `config["phoneme_id_map"]`, línea `"{símbolo} {ids[0]}"`.
- Salida en `ROOT/sherpa/<voz>/` (gitignored).

---

### Task 1: Helpers puros (`tokens_txt`, `meta_data`)

**Files:**
- Create: `studio/sherpa_export.py`
- Test: `studio/tests/test_sherpa_export.py`

**Interfaces:**
- Produces:
  - `tokens_txt(phoneme_id_map: dict) -> str` — contenido completo de tokens.txt (líneas `"{s} {ids[0]}"`, `\n` final).
  - `meta_data(config: dict) -> dict` — dict de metadatos con defaults (`voice→"es"`, `language→"Spanish"`, `n_speakers→1`, `sample_rate→22050`).

- [ ] **Step 1: Escribir el test que falla**

Crear `studio/tests/test_sherpa_export.py`:

```python
import json, unittest, tempfile
from pathlib import Path
from studio.sherpa_export import tokens_txt, meta_data


class TestPuros(unittest.TestCase):
    def test_tokens_txt_formato_y_primer_id(self):
        m = {"a": [5], "b": [6, 99]}  # toma SOLO el primer id
        self.assertEqual(tokens_txt(m), "a 5\nb 6\n")

    def test_meta_data_toma_voice_del_config(self):
        cfg = {"espeak": {"voice": "es"}, "language": {"name_english": "Spanish"},
               "num_speakers": 1, "audio": {"sample_rate": 22050}}
        md = meta_data(cfg)
        self.assertEqual(md["voice"], "es")
        self.assertEqual(md["model_type"], "vits")
        self.assertEqual(md["comment"], "piper")
        self.assertEqual(md["has_espeak"], 1)
        self.assertEqual(md["sample_rate"], 22050)

    def test_meta_data_defaults_si_faltan(self):
        md = meta_data({})  # config vacío
        self.assertEqual(md["voice"], "es")
        self.assertEqual(md["language"], "Spanish")
        self.assertEqual(md["n_speakers"], 1)
        self.assertEqual(md["sample_rate"], 22050)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `.\env\python.exe -m unittest studio.tests.test_sherpa_export -v`
Expected: FAIL — `ImportError: cannot import name 'tokens_txt' from 'studio.sherpa_export'` (o módulo inexistente).

- [ ] **Step 3: Implementación mínima**

Crear `studio/sherpa_export.py`:

```python
# studio/sherpa_export.py
"""Empaqueta una voz Piper (.onnx + .onnx.json) para sherpa-onnx.

Produce una carpeta autocontenida <voz>/ con el .onnx (metadatos embebidos),
tokens.txt y espeak-ng-data — lista para sherpa-onnx (CLI/Android/navegador).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def tokens_txt(phoneme_id_map: dict) -> str:
    """Contenido de tokens.txt: por símbolo, su PRIMER id -> línea '<símbolo> <id>'."""
    lineas = [f"{s} {ids[0]}" for s, ids in phoneme_id_map.items()]
    return "\n".join(lineas) + "\n"


def meta_data(config: dict) -> dict:
    """Metadatos que sherpa-onnx lee del .onnx (defaults ante config incompleto)."""
    espeak = config.get("espeak") or {}
    lang = config.get("language") or {}
    audio = config.get("audio") or {}
    return {
        "model_type": "vits",
        "comment": "piper",
        "language": lang.get("name_english", "Spanish"),
        "voice": espeak.get("voice", "es"),
        "has_espeak": 1,
        "n_speakers": config.get("num_speakers", 1),
        "sample_rate": audio.get("sample_rate", 22050),
    }
```

- [ ] **Step 4: Correr el test para verificar que pasa**

Run: `.\env\python.exe -m unittest studio.tests.test_sherpa_export -v`
Expected: PASS (3 tests OK).

- [ ] **Step 5: Commit**

```
git add studio/sherpa_export.py studio/tests/test_sherpa_export.py
git commit -m "feat(sherpa): helpers puros tokens_txt + meta_data"
```

---

### Task 2: I/O del empaquetado (`espeak_data_dir`, `_add_meta_data`, `empaquetar`)

**Files:**
- Modify: `studio/sherpa_export.py` (agrega funciones al final)
- Test: `studio/tests/test_sherpa_export.py` (agrega clase `TestIO`)

**Interfaces:**
- Consumes: `tokens_txt`, `meta_data` (Task 1).
- Produces:
  - `espeak_data_dir(env_root: Path) -> Path | None` — ubica `Lib/site-packages/piper/espeak-ng-data` bajo `env_root`; `None` si no está.
  - `_add_meta_data(onnx_path: Path, meta: dict) -> None` — embebe metadatos en el onnx (in-place, usa `onnx.load`/`save`).
  - `empaquetar(onnx: Path, config_json: Path, out_dir: Path, espeak_dir: Path, add_meta=_add_meta_data) -> Path` — orquesta el empaquetado; `add_meta` inyectable para tests. Devuelve `out_dir`.

- [ ] **Step 1: Escribir el test que falla**

Agregar a `studio/tests/test_sherpa_export.py` (después de imports, sumar `espeak_data_dir, empaquetar`):

```python
from studio.sherpa_export import espeak_data_dir, empaquetar


class TestIO(unittest.TestCase):
    def test_espeak_data_dir_encuentra_y_none(self):
        tmp = Path(tempfile.mkdtemp())
        d = tmp / "Lib" / "site-packages" / "piper" / "espeak-ng-data"
        d.mkdir(parents=True)
        self.assertEqual(espeak_data_dir(tmp), d)
        self.assertIsNone(espeak_data_dir(Path(tempfile.mkdtemp())))

    def test_empaquetar_arma_carpeta_completa(self):
        tmp = Path(tempfile.mkdtemp())
        onnx = tmp / "mario.onnx"; onnx.write_bytes(b"ONNXFAKE")
        cfg = tmp / "mario.onnx.json"
        cfg.write_text(json.dumps({"phoneme_id_map": {"a": [5], "b": [6]},
                                   "espeak": {"voice": "es"}}), encoding="utf-8")
        espeak = tmp / "espeak-ng-data"; espeak.mkdir()
        (espeak / "phontab").write_text("x", encoding="utf-8")
        out = tmp / "out"
        calls = []
        def fake_add(p, meta): calls.append((Path(p).name, meta["voice"]))
        res = empaquetar(onnx, cfg, out, espeak, add_meta=fake_add)
        self.assertEqual(res, out)
        self.assertTrue((out / "mario.onnx").exists())
        self.assertEqual((out / "tokens.txt").read_text(encoding="utf-8"), "a 5\nb 6\n")
        self.assertTrue((out / "espeak-ng-data" / "phontab").exists())
        self.assertTrue((out / "LEEME.txt").exists())
        self.assertEqual(calls, [("mario.onnx", "es")])  # add_meta recibió el onnx copiado

    def test_empaquetar_sin_phoneme_id_map_falla(self):
        tmp = Path(tempfile.mkdtemp())
        onnx = tmp / "v.onnx"; onnx.write_bytes(b"x")
        cfg = tmp / "v.onnx.json"; cfg.write_text("{}", encoding="utf-8")
        espeak = tmp / "e"; espeak.mkdir()
        with self.assertRaises(ValueError):
            empaquetar(onnx, cfg, tmp / "out", espeak, add_meta=lambda p, m: None)
```

- [ ] **Step 2: Correr el test para verificar que falla**

Run: `.\env\python.exe -m unittest studio.tests.test_sherpa_export -v`
Expected: FAIL — `ImportError: cannot import name 'espeak_data_dir'`.

- [ ] **Step 3: Implementación mínima**

Agregar al final de `studio/sherpa_export.py`:

```python
def espeak_data_dir(env_root: Path) -> Path | None:
    """Ubica el espeak-ng-data que trae piper en el env. None si no está."""
    p = Path(env_root) / "Lib" / "site-packages" / "piper" / "espeak-ng-data"
    return p if p.is_dir() else None


def _add_meta_data(onnx_path: Path, meta: dict) -> None:
    """Embebe metadatos en el .onnx (in-place)."""
    import onnx
    model = onnx.load(str(onnx_path))
    for k, v in meta.items():
        m = model.metadata_props.add()
        m.key = k
        m.value = str(v)
    onnx.save(model, str(onnx_path))


def _leeme(voz: str) -> str:
    return (
        "Voz Piper empaquetada para sherpa-onnx.\n\n"
        "Probar (con sherpa-onnx instalado, desde esta carpeta):\n\n"
        "  sherpa-onnx-offline-tts \\\n"
        f"    --vits-model={voz}.onnx \\\n"
        "    --vits-tokens=tokens.txt \\\n"
        "    --vits-data-dir=espeak-ng-data \\\n"
        "    --output-filename=prueba.wav \\\n"
        "    \"Hola, esto es una prueba.\"\n"
    )


def empaquetar(onnx: Path, config_json: Path, out_dir: Path, espeak_dir: Path,
               add_meta=_add_meta_data) -> Path:
    """Arma la carpeta autocontenida para sherpa-onnx. Devuelve out_dir."""
    onnx, config_json = Path(onnx), Path(config_json)
    out_dir, espeak_dir = Path(out_dir), Path(espeak_dir)
    config = json.loads(config_json.read_text(encoding="utf-8"))
    if "phoneme_id_map" not in config:
        raise ValueError("El config no tiene 'phoneme_id_map'.")
    voz = onnx.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_onnx = out_dir / f"{voz}.onnx"
    shutil.copyfile(onnx, dst_onnx)
    (out_dir / "tokens.txt").write_text(tokens_txt(config["phoneme_id_map"]),
                                        encoding="utf-8")
    add_meta(dst_onnx, meta_data(config))
    dst_espeak = out_dir / "espeak-ng-data"
    if dst_espeak.exists():
        shutil.rmtree(dst_espeak)
    shutil.copytree(espeak_dir, dst_espeak)
    (out_dir / "LEEME.txt").write_text(_leeme(voz), encoding="utf-8")
    return out_dir
```

- [ ] **Step 4: Correr los tests para verificar que pasan**

Run: `.\env\python.exe -m unittest studio.tests.test_sherpa_export -v`
Expected: PASS (6 tests OK).

- [ ] **Step 5: Commit**

```
git add studio/sherpa_export.py studio/tests/test_sherpa_export.py
git commit -m "feat(sherpa): empaquetar (I/O) con add_meta inyectable + espeak_data_dir"
```

---

### Task 3: Botón en la pestaña Exportar + gitignore + verificación real

**Files:**
- Modify: `studio/section_export.py` (import, botón, handler)
- Modify: `.gitignore` (agregar `sherpa/`)

**Interfaces:**
- Consumes: `sherpa_export.espeak_data_dir`, `sherpa_export.empaquetar` (Task 2).

- [ ] **Step 1: Ignorar la carpeta de salida**

Agregar `sherpa/` a `.gitignore` (junto a `training/`, `salidas/`). Editar la sección de artefactos:

```
# Salidas del reproductor
salidas/
sherpa/
```

- [ ] **Step 2: Importar el módulo en section_export.py**

En `studio/section_export.py`, junto a `from studio import runs`, agregar:

```python
from studio import runs, sherpa_export
```

- [ ] **Step 3: Agregar el botón (en `_build`, después de `self.nvda_btn`)**

En `studio/section_export.py`, dentro de `_build`, después de la línea
`s.Add(self.nvda_btn, 0, wx.ALL, 4)` y antes de `self.SetSizer(s)`:

```python
        self.sherpa_btn = wx.Button(self, label="Exportar para &sherpa-onnx")
        s.Add(self.sherpa_btn, 0, wx.ALL, 4)
```

Y en la zona de binds (después de `self.nvda_btn.Bind(...)`):

```python
        self.sherpa_btn.Bind(wx.EVT_BUTTON, self._on_sherpa)
```

- [ ] **Step 4: Agregar el handler (al final de la clase `ExportPanel`)**

En `studio/section_export.py`, agregar estos métodos al final de la clase:

```python
    def _on_sherpa(self, e):
        voz = self.voz.GetValue().strip()
        if not voz:
            self.status.SetLabel("Escribí el nombre de la voz."); return
        onnx = self._onnx_path(voz)
        if not onnx.exists() or not onnx.with_suffix(".onnx.json").exists():
            self.status.SetLabel("Primero exportá a ONNX."); return
        espeak = sherpa_export.espeak_data_dir(ROOT / "env")
        if espeak is None:
            self.status.SetLabel("No encuentro espeak-ng-data en el env."); return
        out = ROOT / "sherpa" / voz
        self.status.SetLabel("Empaquetando para sherpa-onnx…")
        self.nvda.speak("Empaquetando para sherpa onnx", True)
        threading.Thread(target=self._sherpa_worker,
                         args=(onnx, onnx.with_suffix(".onnx.json"), out, espeak),
                         daemon=True).start()

    def _sherpa_worker(self, onnx, cfg, out, espeak):
        try:
            sherpa_export.empaquetar(onnx, cfg, out, espeak)
            wx.CallAfter(self._sherpa_done, out, None)
        except Exception as ex:
            wx.CallAfter(self._sherpa_done, out, ex)

    def _sherpa_done(self, out, err):
        if err:
            self.status.SetLabel(f"Error empaquetando: {err}")
            self.nvda.speak("Error empaquetando para sherpa", True)
        else:
            self.status.SetLabel(f"Listo para sherpa-onnx en {out}")
            self.nvda.speak("Voz empaquetada para sherpa onnx", True)
```

- [ ] **Step 5: Compilar/importar para verificar que no hay errores de sintaxis**

Run:
```
.\env\python.exe -c "import studio.section_export, studio.sherpa_export; import py_compile; py_compile.compile('studio/section_export.py', doraise=True); print('OK')"
```
Expected: `OK`.

- [ ] **Step 6: Verificación REAL contra una voz instalada (end-to-end del empaquetado)**

Empaquetar una voz `.onnx` real (usá una instalada en el reproductor, p.ej. `mario`
o `pedro` en `C:\ia\modelos pc\piper\voces\<voz>\`). Correr:

```
.\env\python.exe -c "from pathlib import Path; from studio import sherpa_export as se; v='mario'; base=Path(r'C:\ia\modelos pc\piper\voces')/v; se.empaquetar(base/(v+'.onnx'), base/(v+'.onnx.json'), Path('sherpa')/v, se.espeak_data_dir(Path('env'))); import onnx; m=onnx.load(str(Path('sherpa')/v/(v+'.onnx'))); print({p.key:p.value for p in m.metadata_props}); print(open(Path('sherpa')/v/'tokens.txt',encoding='utf-8').read()[:40])"
```

Expected: imprime un dict con `model_type=vits`, `comment=piper`, `voice=es` (o el
espeak del config), y las primeras líneas de `tokens.txt` con formato `<símbolo> <id>`.
Confirmar que existe `sherpa/mario/espeak-ng-data/` y `sherpa/mario/LEEME.txt`.
(Si `mario` no existe, usar cualquier voz instalada; si no hay ninguna, exportar una
primero desde la pestaña Exportar.)

- [ ] **Step 7: Correr la suite completa**

Run: `.\env\python.exe -m unittest discover studio/tests`
Expected: OK (todos los tests, incluidos los 6 nuevos).

- [ ] **Step 8: Commit**

```
git add studio/section_export.py .gitignore
git commit -m "feat(sherpa): boton 'Exportar para sherpa-onnx' en la pestana Exportar"
```

---

### Task 4 (OPCIONAL): Probar la voz en sherpa-onnx de verdad

Solo si querés la prueba de fuego end-to-end (síntesis real). Instala sherpa-onnx
(CPU) y sintetiza un wav con la carpeta empaquetada. **No** es requisito del feature
(el destino real es navegador/Android), por eso es opcional y va aparte.

- [ ] **Step 1: Instalar sherpa-onnx en el env**

Run: `.\env\python.exe -m pip install sherpa-onnx`
Expected: instala el wheel CPU.

- [ ] **Step 2: Sintetizar un wav desde la carpeta empaquetada**

Run (ajustar `<voz>`):
```
.\env\python.exe -m sherpa_onnx.cli offline-tts --vits-model=sherpa/<voz>/<voz>.onnx --vits-tokens=sherpa/<voz>/tokens.txt --vits-data-dir=sherpa/<voz>/espeak-ng-data --output-filename=sherpa/<voz>/prueba.wav "Hola, esto es una prueba con sherpa onnx."
```
Expected: crea `sherpa/<voz>/prueba.wav`. Escucharlo: debe sonar como la voz (con la
θ/seseo correctos según su dialecto). Si el binario CLI difiere, usar la API Python
`sherpa_onnx.OfflineTts` con `OfflineTtsVitsModelConfig(model=..., tokens=..., data_dir=...)`.

- [ ] **Step 3: (No commitear el wav — está gitignored.)**

---

## Notas de verificación

- El módulo `sherpa_export.py` es puro+I/O testeable: los 6 unit tests cubren
  `tokens_txt`, `meta_data` (incl. defaults) y `empaquetar` (estructura + inyección
  de `add_meta` + error sin `phoneme_id_map`). El botón GUI se verifica compilando +
  la corrida real del Step 6 de Task 3.
- No se toca el reproductor ni el flujo de exportar/instalar existente.
