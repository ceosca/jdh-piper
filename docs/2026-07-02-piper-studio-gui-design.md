# Piper Studio GUI — Diseño (spec)

Fecha: 2026-07-02
Estado: aprobado el diseño en conversación; pendiente de revisión del spec escrito.

## Objetivo

Una **única app accesible (wxPython + NVDA)** que centralice todo el flujo de Piper
Studio: armar dataset → entrenar → pausar/reanudar → comparar por oído →
exportar/instalar. Debe soportar **dos tipos de entrenamiento**: fine-tune de una
voz y **entrenamiento del base multi-hablante** (multi-acento, con speaker IDs). El
entrenamiento corre **desprendido** de la ventana (sobrevive a cerrar la GUI) y la
GUI se **re-engancha** al reabrir.

Usuario: desarrollador **ciego**, lector de pantalla **NVDA**. Todo control debe ser
accesible por teclado y hablado.

## No-objetivos (por ahora)

- No migrar a piper-plus ni a otro fork: nos quedamos en el **stack estándar**
  (Piper1-gpl 1.4.2 + espeak), para que los `.onnx` corran en cualquier Piper,
  NVDA (Sonata) y Android.
- No entrenar desde cero el base (usamos bootstrap desde checkpoint de un hablante).
- No descargar/gestionar el corpus: el usuario provee las carpetas de audio.

## Arquitectura: entrenamiento desprendido + re-enganche

Cada entrenamiento vive en una **carpeta de corrida** bajo `training/<nombre>/` con:

- `ckpts/` — checkpoints (ya existe: `silvio-{epoch}.ckpt`, `last.ckpt`, `*-best.ckpt`).
- `run.json` — **estado de la corrida**: `{ nombre, modo (finetune|base), dataset,
  base_ckpt, resume_ckpt, max_epochs, auto_stop (bool), paciencia, cada, pid,
  started_at, last_event }`.
- `train.log` — salida del proceso.

Mecánica:

- **Lanzar**: la GUI arranca el entrenamiento como **proceso independiente**
  (detached, no hijo de la ventana) y escribe el `pid` en `run.json`.
- **Cerrar la GUI**: el proceso sigue.
- **Reabrir**: la GUI escanea `training/*/run.json`, verifica qué `pid` siguen vivos
  (multiplataforma), y arma la **lista de corridas** con su estado
  (entrenando/pausado/terminado/falló + época estimada leída de los checkpoints/log).
- **Pausar** = terminar el proceso (checkpoint ya en disco) y marcar `pausado`.
- **Reanudar** = relanzar desde `last.ckpt` (o el que se elija), continuando época.
- **Detener** = terminar y marcar `terminado`.

La época "en vivo" se estima del último checkpoint + parseo del `train.log`
(la barra tqdm no se captura bien; los checkpoints son la señal fiable).

## La app (una sola ventana, secciones)

Ventana única wxPython, secciones navegables por teclado (pestañas o lista):

1. **Entrenar** — se construye primero.
2. **Dataset** — absorbe `gui_dataset.py` (armador whisper + silencios).
3. **Comparar por oído** — envuelve `comparar_checkpoints.py`.
4. **Exportar / Instalar** — envuelve `export_run.py` + copiar la voz al reproductor.

Convenciones de accesibilidad (como las GUIs existentes): `NVDAController`,
`stdout/stderr`→log, foco anunciado, botones con aceleradores, nombres accesibles.

## Sección "Entrenar"

Selector de **modo**: **Fine-tune de una voz** | **Base multi-hablante**.

Controles:

- **Dataset**: elegir la carpeta del dataset (o, en modo base, varias — ver abajo).
- **Checkpoint base / desde dónde**: base a usar; en reanudar, elegir `last` / `best`
  / una época puntual.
- **Cantidad de épocas (manual)**: campo numérico. Es el tope de la corrida.
- **Parar automático (early-stop)**: checkbox. Si está **prendido**, entrena hasta
  que `val_mel` deja de mejorar (paciencia configurable) usando el tope como techo.
  Si está **apagado**, entrena exactamente hasta la cantidad de épocas manual.
  → Los dos modos conviven: el usuario elige. (Backend: prendido = `entrenar.py`;
  apagado = `train_run.py fit` sin EarlyStopping.)
- **Perillas del early-stop** (visibles solo si está prendido): `paciencia`, `cada`.
- Botones: **Entrenar / Pausar / Reanudar / Detener**.
- **"¿Cómo va?"**: lee por NVDA estado + época + ETA + última pérdida.
- **Lista de corridas activas**: para el re-enganche; cada una con su estado.

## Backend multi-hablante (nuevo)

- **Modelo mental: una carpeta/audiolibro = un hablante.** El usuario agrega varias
  carpetas; se asigna un `speaker_id` incremental por carpeta y se arma el dataset
  **multi-hablante** (metadata con columna de hablante, formato que pide Piper para
  `num_speakers > 1`).
- **Acentos**: fonemización unificada **`es`** (España — preserva c/z y ll/y; cada dialecto lo aprende su embedding); chileno/
  argentino/cubano/etc. conviven, y los `speaker embeddings` absorben la variación.
- **Bootstrap**: el base arranca desde un checkpoint de **un solo hablante** vía
  `--resume_from_single_speaker_checkpoint` (mucho más rápido que desde cero).
- Requiere **extender el armador de dataset** (`dataset_builder.py`) al modo
  multi-hablante (emitir columna de hablante, recorrer varias carpetas).

Notas técnicas a validar en el plan: formato exacto del CSV multi-hablante que
espera Piper1-gpl 1.4.2 (`dataset.py` tiene manejo de `is_multispeaker` /
`num_speakers` / `speaker_id_map`); confirmar el flag
`--resume_from_single_speaker_checkpoint` en 1.4.2 y su comportamiento con el
checkpoint base ya saneado.

## Monitoreo / avisos (para NVDA)

- **Avisos automáticos** por NVDA en los hitos, sin tener que preguntar:
  - cada **N épocas** (configurable, default 100): "época NNN".
  - eventos: **terminó** / **se pausó** / **falló** / **nuevo mejor (val_mel)**.
- Además el botón **"¿Cómo va?"** para consultar on-demand.
- Implementación: un "watcher" liviano en la GUI que sondea los checkpoints/`run.json`
  del proceso desprendido y habla los cambios (no necesita que el proceso hijo hable).

## Reutilización de lo existente

- `entrenar.py` — modo con early-stop (parar automático).
- `train_run.py` — modo épocas manual (sin early-stop); base de todos los comandos
  (parches torch.load + PosixPath).
- `comparar_checkpoints.py` — sección Comparar.
- `export_run.py` — sección Exportar.
- `dataset_builder.py` + `gui_dataset.py` — sección Dataset (extender a multi-hablante).

## Orden de construcción (por partes)

1. **Cáscara de la app** + sección **Entrenar** (fine-tune, desprendido, re-enganche,
   avisos NVDA, parar-auto/épocas-manual). ← primero, es lo que más urge.
2. Sección **Comparar por oído**.
3. Sección **Exportar / Instalar**.
4. Sección **Dataset** (absorbe la GUI actual) + **modo multi-hablante**.
5. Modo **Base multi-hablante** en Entrenar (usa el dataset multi-hablante del paso 4).

## Riesgos / a resolver en el plan

- Detached real en Windows (que sobreviva a cerrar la GUI) + chequeo de PID vivo
  multiplataforma.
- Formato multi-hablante exacto y `--resume_from_single_speaker_checkpoint` en 1.4.2.
- Estimar época "en vivo" de forma fiable sin la barra tqdm.
- No romper el flujo probado hoy (todo lo nuevo envuelve scripts ya funcionando).
