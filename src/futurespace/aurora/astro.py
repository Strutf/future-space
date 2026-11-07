"""Posição do Sol e da Lua no céu de um lugar, com precisão suficiente para dizer se está escuro.

Fórmulas de baixa precisão do Astronomical Almanac (Sol ~0,01°, Lua ~0,3°), sem dependências
além do numpy. Os testes comparam com o astropy. A refração atmosférica é ignorada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAD = np.pi / 180
J2000 = pd.Timestamp("2000-01-01T12:00:00Z")


def _days(times) -> np.ndarray:
    """Dias desde J2000.0 (UTC; a diferença para TT, ~1 min, não importa aqui)."""
    t = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    return ((t - J2000) / pd.Timedelta(days=1)).to_numpy(dtype=float)


def _obliquity(d: np.ndarray) -> np.ndarray:
    return (23.439 - 0.0000004 * d) * RAD


def _sun_ecliptic(d: np.ndarray) -> np.ndarray:
    g = (357.529 + 0.98560028 * d) * RAD
    q = 280.459 + 0.98564736 * d
    return (q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g)) * RAD


def _moon_ecliptic(d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Longitude, latitude eclíptica e paralaxe horizontal da Lua (radianos)."""
    t = d / 36525

    def s(a, b):
        return np.sin((a + b * t) * RAD)

    def c(a, b):
        return np.cos((a + b * t) * RAD)

    lon = (
        218.32 + 481267.881 * t
        + 6.29 * s(135.0, 477198.87) - 1.27 * s(259.3, -413335.36) + 0.66 * s(235.7, 890534.22)
        + 0.21 * s(269.9, 954397.74) - 0.19 * s(357.5, 35999.05) - 0.11 * s(186.5, 966404.03)
    )
    lat = 5.13 * s(93.3, 483202.02) + 0.28 * s(228.2, 960400.89) - 0.28 * s(318.3, 6003.15) - 0.17 * s(217.6, -407332.21)
    parallax = (
        0.9508 + 0.0518 * c(135.0, 477198.87) + 0.0095 * c(259.3, -413335.36)
        + 0.0078 * c(235.7, 890534.22) + 0.0028 * c(269.9, 954397.74)
    )
    return lon * RAD, lat * RAD, parallax * RAD


def _altitude(d, ra, dec, lat, lon) -> np.ndarray:
    gmst = (280.46061837 + 360.98564736629 * d) * RAD
    hour_angle = gmst + np.asarray(lon) * RAD - ra
    lat = np.asarray(lat) * RAD
    s = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(hour_angle)
    return np.arcsin(np.clip(s, -1, 1))


def _equatorial(lon_ecl, lat_ecl, eps) -> tuple[np.ndarray, np.ndarray]:
    ra = np.arctan2(np.sin(lon_ecl) * np.cos(eps) - np.tan(lat_ecl) * np.sin(eps), np.cos(lon_ecl))
    dec = np.arcsin(np.sin(lat_ecl) * np.cos(eps) + np.cos(lat_ecl) * np.sin(eps) * np.sin(lon_ecl))
    return ra, dec


def sun_altitude(times, lat, lon) -> np.ndarray:
    """Altura do centro do Sol acima do horizonte, em graus."""
    d = _days(times)
    ra, dec = _equatorial(_sun_ecliptic(d), 0.0, _obliquity(d))
    return _altitude(d, ra, dec, lat, lon) / RAD


def moon_altitude(times, lat, lon) -> np.ndarray:
    """Altura da Lua vista do local (com a correção de paralaxe), em graus."""
    d = _days(times)
    lon_ecl, lat_ecl, parallax = _moon_ecliptic(d)
    ra, dec = _equatorial(lon_ecl, lat_ecl, _obliquity(d))
    geocentric = _altitude(d, ra, dec, lat, lon)
    return (geocentric - parallax * np.cos(geocentric)) / RAD


def moon_illumination(times) -> np.ndarray:
    """Fração iluminada do disco da Lua (0 = nova, 1 = cheia)."""
    d = _days(times)
    lon_ecl, lat_ecl, _ = _moon_ecliptic(d)
    elongation = np.arccos(np.cos(lat_ecl) * np.cos(lon_ecl - _sun_ecliptic(d)))
    return (1 - np.cos(elongation)) / 2
