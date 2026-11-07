"""Cidades acompanhadas na previsão de visibilidade da aurora.

Inclui cidades das zonas de aurora, cidades de latitude média (onde só tempestades fortes
trazem a aurora), referências do Hemisfério Sul e do Brasil, e locais com câmeras all-sky da
rede THEMIS (NASA), que poderão servir para verificar se a aurora prevista apareceu.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Site:
    id: str
    name: str
    lat: float
    lon: float
    themis: str | None = None  # código da câmera all-sky THEMIS mais próxima


SITES: tuple[Site, ...] = (
    # Zona de aurora e arredores, Europa
    Site("tromso", "Tromsø", 69.65, 18.96),
    Site("kiruna", "Kiruna", 67.86, 20.23),
    Site("rovaniemi", "Rovaniemi", 66.50, 25.73),
    Site("reykjavik", "Reykjavík", 64.15, -21.94),
    Site("helsinki", "Helsinki", 60.17, 24.94),
    Site("oslo", "Oslo", 59.91, 10.75),
    Site("stockholm", "Stockholm", 59.33, 18.07),
    Site("edinburgh", "Edinburgh", 55.95, -3.19),
    Site("berlin", "Berlin", 52.52, 13.40),
    Site("london", "London", 51.51, -0.13),
    Site("paris", "Paris", 48.86, 2.35),
    Site("madrid", "Madrid", 40.42, -3.70),
    Site("lisboa", "Lisboa", 38.72, -9.14),
    # América do Norte (com câmeras THEMIS)
    Site("inuvik", "Inuvik", 68.36, -133.72, "inuv"),
    Site("fairbanks", "Fairbanks", 64.84, -147.72),
    Site("yellowknife", "Yellowknife", 62.45, -114.37, "yknf"),
    Site("anchorage", "Anchorage", 61.22, -149.90),
    Site("whitehorse", "Whitehorse", 60.72, -135.06, "whit"),
    Site("fort_smith", "Fort Smith", 60.01, -111.89, "fsmi"),
    Site("gillam", "Gillam", 56.35, -94.71, "gill"),
    Site("athabasca", "Athabasca", 54.72, -113.29, "atha"),
    Site("seattle", "Seattle", 47.61, -122.33),
    Site("minneapolis", "Minneapolis", 44.98, -93.27),
    Site("toronto", "Toronto", 43.65, -79.38),
    Site("new_york", "New York", 40.71, -74.01),
    Site("mexico_city", "Ciudad de México", 19.43, -99.13),
    # Ásia
    Site("tokyo", "Tokyo", 35.68, 139.69),
    # Hemisfério Sul
    Site("ushuaia", "Ushuaia", -54.80, -68.30),
    Site("punta_arenas", "Punta Arenas", -53.16, -70.91),
    Site("dunedin", "Dunedin", -45.87, 170.50),
    Site("hobart", "Hobart", -42.88, 147.33),
    Site("melbourne", "Melbourne", -37.81, 144.96),
    Site("buenos_aires", "Buenos Aires", -34.60, -58.38),
    # Brasil
    Site("porto_alegre", "Porto Alegre", -30.03, -51.23),
    Site("sao_paulo", "São Paulo", -23.55, -46.63),
    Site("rio_de_janeiro", "Rio de Janeiro", -22.91, -43.17),
    Site("manaus", "Manaus", -3.12, -60.02),
)

BY_ID = {s.id: s for s in SITES}
