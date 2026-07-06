# Exportar voces Piper para sherpa-onnx — Diseño

**Fecha:** 2026-07-04
**Estado:** aprobado (a implementar)

## Objetivo

Agregar a la pestaña **Exportar** de Piper Studio un botón **"Exportar para
sherpa-onnx"** que empaquete una voz ya entrenada (su `.onnx` + `.onnx.json`) en
una carpeta autocontenida lista para correr con [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
en cualquier plataforma (CLI, **Android, navegador/WASM**, embebidos).

**Fuera de alcance (etapa 2):** cambiar el reproductor interno para usar
sherpa-onnx. Esto es solo el empaquetado.

## Qué produce

Por voz, una carpeta **autocontenida** `sherpa/<voz>/`:

- **`<voz>.onnx`** — copia del `.onnx` exportado, con los metadatos que sherpa-onnx
  lee de la propia red (ver abajo).
- **`tokens.txt`** — mapa símbolo→id derivado del `phoneme_id_map` del `.onnx.json`.
- **`espeak-ng-data/`** — copia de la que ya trae el env
  (`env/Lib/site-packages/piper/espeak-ng-data`), es decir **la misma que se usa al
  entrenar** → fonemas idénticos, sin descargas.
- **`LEEME.txt`** — el comando `sherpa-onnx-offline-tts` ya armado para probar.

Se zipea esa carpeta y corre tal cual en sherpa-onnx.

## Hechos técnicos verificados

Del conversor oficial de sherpa (`sherpa/docs/source/onnx/tts/code/piper.py`):

- **tokens.txt**: por cada `(s, ids)` en `config["phoneme_id_map"]`, se escribe
  `f"{s} {ids[0]}\n"`.
- **Metadatos** que sherpa espera embebidos en el `.onnx` (vía
  `onnx.metadata_props`):
  - `model_type = "vits"`
  - `comment = "piper"`
  - `language = config["language"]["name_english"]`
  - **`voice = config["espeak"]["voice"]`** ← acá viaja **nuestro `es`**; por eso la
    voz suena igual que en Piper (sherpa fonemiza con ese espeak).
  - `has_espeak = 1`
  - `n_speakers = config["num_speakers"]`
  - `sample_rate = config["audio"]["sample_rate"]`
- **espeak-ng-data**: es común a todos los modelos Piper; reusamos la local.
- Dependencias: solo `onnx` (ya está, 1.22.0). **No hace falta instalar
  sherpa-onnx para empaquetar** (solo para reproducir, que es la etapa 2).

## Arquitectura

Módulo nuevo **`studio/sherpa_export.py`**, separando lógica pura de I/O (patrón
del repo: puro testeable, I/O fino).

### Funciones puras (unit-testeables, sin archivos)

- `tokens_txt(phoneme_id_map: dict) -> str`
  Devuelve el contenido completo de `tokens.txt` (líneas `"{símbolo} {id}"`,
  una por entrada, en el orden del dict). Toma `ids[0]` de cada valor.

- `meta_data(config: dict) -> dict`
  Arma el dict de metadatos con **defaults robustos** ante un config incompleto:
  - `voice` = `config["espeak"]["voice"]` si está, si no `"es"`.
  - `language` = `config["language"]["name_english"]` si está, si no `"Spanish"`.
  - `n_speakers` = `config.get("num_speakers", 1)`.
  - `sample_rate` = `config["audio"]["sample_rate"]` si está, si no `22050`.
  - `model_type="vits"`, `comment="piper"`, `has_espeak=1` fijos.

### Funciones de I/O

- `espeak_data_dir(env_root: Path) -> Path | None`
  Localiza `…/site-packages/piper/espeak-ng-data` dentro del env. `None` si no está.

- `empaquetar(onnx: Path, config_json: Path, out_dir: Path, espeak_dir: Path) -> Path`
  Orquesta: crea `out_dir`, copia `onnx`→`out_dir/<voz>.onnx`, escribe `tokens.txt`
  (de `tokens_txt`), agrega metadatos al `.onnx` copiado (`onnx.load`/`save` con
  `meta_data`), copia `espeak_dir`→`out_dir/espeak-ng-data`, escribe `LEEME.txt`.
  Devuelve `out_dir`.

### UI (`studio/section_export.py`)

- Botón **"Exportar para &sherpa-onnx"** junto a los de instalar.
- Guardas idénticas a "Instalar": exige que existan `<voz>.onnx` y `<voz>.onnx.json`
  (si no, avisa "Primero exportá a ONNX").
- Corre en un hilo (`threading.Thread`), avisa el resultado por `status` + NVDA.
- Salida en `ROOT/"sherpa"/<voz>/` (agregar `sherpa/` al `.gitignore`).
- Si `espeak_data_dir()` devuelve `None`: avisa que no encuentra `espeak-ng-data`.

## Manejo de errores / bordes

- Falta `.onnx` o `.onnx.json` → mensaje "Primero exportá a ONNX" (no hace nada).
- `config.json` sin `phoneme_id_map` → error claro ("el config no tiene
  phoneme_id_map"); no se genera tokens.txt vacío.
- Sin `espeak-ng-data` local → error claro.
- Campos opcionales del config faltantes → defaults (ver `meta_data`).

## Tests (`studio/tests/test_sherpa_export.py`)

Solo lo puro (el I/O se verifica corriendo la exportación):

- `tokens_txt`: un `phoneme_id_map` chico produce el texto esperado
  (`"a 5\nb 6\n"`, toma `ids[0]`, respeta el orden).
- `meta_data`: toma `voice` del config; y aplica los defaults cuando faltan
  `espeak`, `language`, `num_speakers`, `audio`.

## Sin cambios en

- El reproductor (etapa 2).
- El flujo de exportar a ONNX / instalar en reproductor / instalar para NVDA.
- Dependencias (no se agrega nada a `requirements.txt`).
