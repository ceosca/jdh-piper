# -*- coding: utf-8 -*-
"""Normaliza números/ordinales/unidades a palabras en español, antes de TTS.

Cubre los agujeros de espeak (ordinales, negativos, unidades) y por consistencia
expande también cardinales/decimales. Convención española: '.' miles, ',' decimal.
"""
import re

from num2words import num2words

UNIDADES = {
    "km/h": "kilómetros por hora", "km": "kilómetros", "kg": "kilogramos",
    "cm": "centímetros", "mm": "milímetros", "mg": "miligramos",
    "ml": "mililitros", "kb": "kilobytes", "mb": "megabytes", "gb": "gigabytes",
    "g": "gramos", "l": "litros", "m": "metros",
    "h": "horas", "min": "minutos", "seg": "segundos",
}
# Alternación ordenada por longitud desc: "km/h" antes que "km", "km" antes que "m".
_UNID_RE = "|".join(sorted((re.escape(u) for u in UNIDADES), key=len, reverse=True))

APOCOPE = {1: "primer", 3: "tercer"}


def _card(n) -> str:
    return num2words(int(n), lang="es")


def _ord(n, fem: bool = False) -> str:
    w = num2words(int(n), lang="es", to="ordinal")
    if fem and w.endswith("o"):
        w = w[:-1] + "a"
    return w


def normalizar(texto: str) -> str:
    s = texto
    # 1) Ordinales con símbolo: 2º 2° (masc) / 3ª (fem)
    s = re.sub(r"\b(\d+)\s*([º°ª])",
               lambda m: _ord(m.group(1), fem=(m.group(2) == "ª")), s)
    # 2) Ordinales apocopados: 1er 3er -> primer, tercer
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
