# Plan 2 — Capa de normalización de números (español)

> **Para quien ejecute:** módulo puro y muy testeable → ideal TDD. Los pasos usan checkbox (`- [ ]`).

**Goal:** que las voces Piper lean bien números, ordinales, negativos, decimales, porcentajes y unidades, con un módulo que expande texto a palabras en español ANTES de sintetizar. Independiente del modelo/voz.

**Architecture:** un módulo puro `normalizar_es.py` (regex + `num2words`) con `normalizar(texto) -> texto`. Se integra en el motor del reproductor (`engine.generate`) para que corra antes de que Piper/espeak vean el texto. Cubre los agujeros reales de espeak (ordinales `1º`, negativos `-5`, teléfonos, unidades `km`, fracciones), y por consistencia expande también los cardinales.

**Tech Stack:** Python (env del reproductor `C:\ia\modelos pc\piper\env\python.exe`), `num2words` (pip, puro Python), `unittest` (stdlib).

## Global Constraints

- Salida en **español** (`num2words(..., lang="es")`).
- **Portable**: `num2words` es puro Python; se instala en el env del reproductor.
- Convención española de separadores: `.` = miles, `,` = decimal (`1.000.000` → "un millón"; `3,5` → "tres coma cinco").
- No romper el flujo actual del reproductor: `engine.generate` normaliza y sigue igual.
- El módulo es **puro y testeable** (sin tocar audio ni Piper en los tests).

## Estructura de archivos

- Crear: `C:\ia\modelos pc\piper\normalizar_es.py`
- Test: `C:\ia\modelos pc\piper\test_normalizar.py`
- Modificar: `C:\ia\modelos pc\piper\engine.py` (llamar `normalizar` en `generate`)

---

### Task 1: `normalizar_es.py` + tests (TDD)

**Files:**
- Create: `C:\ia\modelos pc\piper\normalizar_es.py`
- Test: `C:\ia\modelos pc\piper\test_normalizar.py`

**Interfaces:**
- Produces: `normalizar(texto: str) -> str` — expande números/ordinales/negativos/decimales/porcentajes/unidades a palabras en español; deja el resto igual.

- [ ] **Step 1: instalar num2words en el env del reproductor**

Run: `"C:/ia/modelos pc/piper/env/python.exe" -m pip install num2words`
Expected: instala sin error.

- [ ] **Step 2: escribir el test (falla)**

```python
# C:\ia\modelos pc\piper\test_normalizar.py
# -*- coding: utf-8 -*-
import unittest
from normalizar_es import normalizar


class TestNormalizar(unittest.TestCase):
    def test_cardinales(self):
        self.assertEqual(normalizar("Tengo 20 años"), "Tengo veinte años")
        self.assertEqual(normalizar("son 1234 casos"), "son mil doscientos treinta y cuatro casos")

    def test_miles_millones(self):
        self.assertEqual(normalizar("1.000.000 de pesos"), "un millón de pesos")
        self.assertEqual(normalizar("el año 2026"), "el año dos mil veintiséis")

    def test_decimales_coma(self):
        self.assertEqual(normalizar("mide 3,5 metros"), "mide tres coma cinco metros")

    def test_negativos(self):
        self.assertEqual(normalizar("hace -5 grados"), "hace menos cinco grados")

    def test_porcentaje(self):
        self.assertEqual(normalizar("subió 50%"), "subió cincuenta por ciento")

    def test_ordinales_simbolo(self):
        self.assertEqual(normalizar("el 2º piso"), "el segundo piso")
        self.assertEqual(normalizar("la 3ª fila"), "la tercera fila")

    def test_ordinales_apocope(self):
        self.assertEqual(normalizar("1er premio"), "primer premio")
        self.assertEqual(normalizar("3er lugar"), "tercer lugar")

    def test_unidades(self):
        self.assertEqual(normalizar("corrí 10 km"), "corrí diez kilómetros")
        self.assertEqual(normalizar("pesa 2 kg"), "pesa dos kilogramos")

    def test_sin_numeros_no_cambia(self):
        self.assertEqual(normalizar("hola qué tal"), "hola qué tal")

    def test_texto_mixto(self):
        self.assertEqual(
            normalizar("El 2º día recorrí 10 km a -5 grados"),
            "El segundo día recorrí diez kilómetros a menos cinco grados",
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: correr y ver fallar**

Run: `cd /c/ia/modelos\ pc/piper && "C:/ia/modelos pc/piper/env/python.exe" -m unittest test_normalizar -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'normalizar_es'`).

- [ ] **Step 4: implementar `normalizar_es.py`**

```python
# C:\ia\modelos pc\piper\normalizar_es.py
# -*- coding: utf-8 -*-
"""Normaliza números/ordinales/unidades a palabras en español, antes de TTS.

Cubre los agujeros de espeak (ordinales, negativos, unidades) y por consistencia
expande también cardinales/decimales. Convención: '.' miles, ',' decimal.
"""
import re

from num2words import num2words

UNIDADES = {
    "km/h": "kilómetros por hora", "km": "kilómetros", "kg": "kilogramos",
    "cm": "centímetros", "mm": "milímetros", "mg": "miligramos",
    "ml": "mililitros", "kb": "kilobytes", "mb": "megabytes", "gb": "gigabytes",
    "kg.": "kilogramos", "g": "gramos", "l": "litros", "m": "metros",
    "h": "horas", "min": "minutos", "seg": "segundos",
}
# unidades ordenadas por longitud desc para que "km/h" gane a "km", "km" a "m", etc.
_UNID_RE = "|".join(sorted((re.escape(u) for u in UNIDADES), key=len, reverse=True))

APOCOPE = {1: "primer", 3: "tercer"}


def _card(n: int) -> str:
    return num2words(int(n), lang="es")


def _ord(n: int) -> str:
    return num2words(int(n), lang="es", to="ordinal")


def normalizar(texto: str) -> str:
    s = texto
    # 1) Ordinales con símbolo: 2º 2° 2ª
    s = re.sub(r"\b(\d+)\s*[º°ª]", lambda m: _ord(m.group(1)), s)
    # 2) Ordinales apocopados: 1er 3er (primer, tercer)
    s = re.sub(r"\b(\d+)\s*er\b",
               lambda m: APOCOPE.get(int(m.group(1)), _ord(m.group(1))), s)
    # 3) Porcentaje: 50%
    s = re.sub(r"\b(\d+)\s*%", lambda m: _card(m.group(1)) + " por ciento", s)
    # 4) Unidades: 10 km, 2kg
    s = re.sub(r"\b(\d+)\s*(" + _UNID_RE + r")\b",
               lambda m: _card(m.group(1)) + " " + UNIDADES[m.group(2)], s)
    # 5) Miles con punto: 1.000.000 -> 1000000 (colapsar separador de miles)
    s = re.sub(r"\b(\d{1,3}(?:\.\d{3})+)\b", lambda m: m.group(1).replace(".", ""), s)
    # 6) Decimales con coma: 3,5 -> tres coma cinco
    s = re.sub(r"\b(\d+),(\d+)\b",
               lambda m: _card(m.group(1)) + " coma " + _card(m.group(2)), s)
    # 7) Negativos: -5 (precedido por inicio o espacio)
    s = re.sub(r"(^|\s)-(\d+)\b", lambda m: m.group(1) + "menos " + _card(m.group(2)), s)
    # 8) Cardinales restantes
    s = re.sub(r"\b(\d+)\b", lambda m: _card(m.group(1)), s)
    return s
```

- [ ] **Step 5: correr y ver pasar**

Run: `cd /c/ia/modelos\ pc/piper && "C:/ia/modelos pc/piper/env/python.exe" -m unittest test_normalizar -v`
Expected: PASS (10 tests). Si algún caso falla, ajustar la regex/orden y re-correr (no cambiar los tests salvo error real en el test).

---

### Task 2: integrar en el motor del reproductor

**Files:**
- Modify: `C:\ia\modelos pc\piper\engine.py`

**Interfaces:**
- Consumes: `normalizar_es.normalizar`.
- Effect: `engine.generate(text, voice_name)` normaliza `text` antes de sintetizar.

- [ ] **Step 1: importar y aplicar en `generate`**

En `engine.py`, importar arriba: `from normalizar_es import normalizar`. En `generate`, como primera línea del cuerpo, envolver el texto:
```python
text = normalizar(text)
```
(justo antes de fonemizar/sintetizar). No cambiar nada más.

- [ ] **Step 2: verificación de humo (no interactiva)**

Run: `cd /c/ia/modelos\ pc/piper && "C:/ia/modelos pc/piper/env/python.exe" -c "from engine import TTSEngine; from normalizar_es import normalizar; print(normalizar('El 2º día a -5 grados, 10 km'))"`
Expected: imprime `El segundo día a menos cinco grados, diez kilómetros` (imports del engine OK + normalizador correcto).

- [ ] **Step 3 (interactivo, para el humano):** en la GUI del reproductor, generar audio de "El 20 de marzo, 2º piso, -5 grados, 10 km" y confirmar por oído que lee todo bien.

---

## Auto-revisión del plan

- **Cobertura:** cardinales, miles/millones, decimales, negativos, porcentaje, ordinales (símbolo + apócope), unidades, texto mixto, y "sin números no cambia" — todos con test. ✓
- **Sin placeholders:** código real + tests reales + comandos con salida esperada. ✓
- **Riesgo/limitaciones documentadas:** `3.14` (punto como decimal estilo inglés) se tratará como miles y puede leerse mal — en español se escribe `3,14`; teléfonos/fracciones quedan fuera de v1 (ampliables luego). El orden de las regex (ordinal/%/unidad/miles/decimal/negativo/cardinal) importa y está fijado.
