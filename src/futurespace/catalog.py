"""Catálogo de datasets: metadados compartilhados pelo pipeline e pela API.

Este módulo não depende de pandas para que a API continue leve.
Toda tabela tem uma coluna ``time`` (UTC) que define a partição e a ordenação.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

FieldType = Literal["float", "int", "str", "bool", "time"]
Partition = Literal["day", "month", "year"]
Stage = Literal["sun", "l1", "orbit", "ground", "multi"]
# "derived": produzido pelo próprio projeto (ex.: previsões), não por um conector.
Schedule = Literal["realtime", "daily", "derived"]

SWPC = "NOAA Space Weather Prediction Center"
SWPC_LICENSE = "Domínio público (governo dos EUA). Cite: NOAA SWPC."


@dataclass(frozen=True)
class Field:
    name: str
    type: FieldType
    unit: str | None
    description: str


@dataclass(frozen=True)
class Dataset:
    id: str
    title: str
    description: str
    stage: Stage
    provider: str
    source_url: str
    license: str
    cadence: str
    partition: Partition
    key: tuple[str, ...]
    fields: tuple[Field, ...]
    schedule: Schedule
    # Quantos dias para trás o comando `update` busca a cada execução.
    update_days: int = 3
    max_query_days: int = 31
    default_days: int = 1
    # "last": última linha de cada grupo em `latest_group`; "snapshot": todas as linhas do último `time`.
    latest_mode: Literal["last", "snapshot"] = "last"
    latest_group: tuple[str, ...] = ()
    # True quando a fonte oferece histórico longo e `futurespace backfill` faz sentido.
    historical: bool = False
    # False para datasets grandes demais para a API: só Parquet.
    json_partitions: bool = True
    # False quando nem o latest.json cabe: regravá-lo a cada hora incharia o histórico do repositório.
    latest_json: bool = True
    # False para fontes de componentes experimentais: se falharem, a coleta segue sem marcar erro.
    critical: bool = True
    notes: str = ""

    @property
    def columns(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def time_fields(self) -> list[str]:
        return [f.name for f in self.fields if f.type == "time"]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["key"] = list(self.key)
        d["latest_group"] = list(self.latest_group)
        return d


def _time(desc: str = "Instante da medição (UTC)") -> Field:
    return Field("time", "time", None, desc)


DATASETS: tuple[Dataset, ...] = (
    Dataset(
        id="solar-wind-plasma",
        title="Vento solar: plasma (L1, 1 min)",
        description="Velocidade, densidade e temperatura dos prótons do vento solar medidos "
        "pelas sondas no ponto L1 (SOLAR-1/SWFO-L1, IMAP, ACE, DSCOVR).",
        stage="l1",
        provider=SWPC,
        source_url="https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
        license=SWPC_LICENSE,
        cadence="1min",
        partition="day",
        key=("time", "source"),
        fields=(
            _time(),
            Field("source", "str", None, "Sonda que fez a medição"),
            Field("active", "bool", None, "True se é a sonda operacional da NOAA naquele momento"),
            Field("speed", "float", "km/s", "Velocidade dos prótons"),
            Field("density", "float", "cm^-3", "Densidade dos prótons"),
            Field("temperature", "float", "K", "Temperatura dos prótons"),
            Field("vx_gsm", "float", "km/s", "Componente X da velocidade (GSM)"),
            Field("vy_gsm", "float", "km/s", "Componente Y da velocidade (GSM)"),
            Field("vz_gsm", "float", "km/s", "Componente Z da velocidade (GSM)"),
        ),
        schedule="realtime",
        max_query_days=3,
        latest_group=("source",),
        notes="A fonte só mantém ~24h; a coleta precisa rodar pelo menos a cada poucas horas.",
    ),
    Dataset(
        id="solar-wind-mag",
        title="Vento solar: campo magnético (L1, 1 min)",
        description="Campo magnético interplanetário (IMF) no ponto L1. Bz negativo (sul) "
        "é o principal gatilho de tempestades geomagnéticas.",
        stage="l1",
        provider=SWPC,
        source_url="https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
        license=SWPC_LICENSE,
        cadence="1min",
        partition="day",
        key=("time", "source"),
        fields=(
            _time(),
            Field("source", "str", None, "Sonda que fez a medição"),
            Field("active", "bool", None, "True se é a sonda operacional da NOAA naquele momento"),
            Field("bt", "float", "nT", "Intensidade total do campo"),
            Field("bx_gsm", "float", "nT", "Componente X (GSM)"),
            Field("by_gsm", "float", "nT", "Componente Y (GSM)"),
            Field("bz_gsm", "float", "nT", "Componente Z (GSM)"),
            Field("theta_gsm", "float", "deg", "Latitude do vetor de campo (GSM)"),
            Field("phi_gsm", "float", "deg", "Longitude do vetor de campo (GSM)"),
            Field("quality", "int", None, "Flag de qualidade geral (0 = boa)"),
        ),
        schedule="realtime",
        max_query_days=3,
        latest_group=("source",),
        notes="A fonte só mantém ~24h; a coleta precisa rodar pelo menos a cada poucas horas.",
    ),
    Dataset(
        id="goes-xray",
        title="GOES: fluxo de raios-X (1 min)",
        description="Fluxo de raios-X do Sol medido pelos satélites GOES. A banda longa "
        "(0.1-0.8 nm) define a classe dos flares (A, B, C, M, X).",
        stage="orbit",
        provider=SWPC,
        source_url="https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
        license=SWPC_LICENSE,
        cadence="1min",
        partition="day",
        key=("time", "satellite"),
        fields=(
            _time(),
            Field("satellite", "int", None, "Número do satélite GOES"),
            Field("flux_short", "float", "W/m^2", "Fluxo na banda 0.05-0.4 nm"),
            Field("flux_long", "float", "W/m^2", "Fluxo na banda 0.1-0.8 nm"),
        ),
        schedule="realtime",
        max_query_days=7,
        latest_group=("satellite",),
    ),
    Dataset(
        id="goes-protons",
        title="GOES: prótons energéticos (5 min)",
        description="Fluxo integral de prótons por faixa de energia. >=10 MeV acima de 10 pfu "
        "caracteriza uma tempestade de radiação (escala S da NOAA).",
        stage="orbit",
        provider=SWPC,
        source_url="https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json",
        license=SWPC_LICENSE,
        cadence="5min",
        partition="day",
        key=("time", "satellite"),
        fields=(
            _time(),
            Field("satellite", "int", None, "Número do satélite GOES"),
            *(
                Field(f"flux_ge_{e}mev", "float", "pfu", f"Fluxo integral de prótons >= {e} MeV")
                for e in (1, 5, 10, 30, 50, 60, 100, 500)
            ),
        ),
        schedule="realtime",
        max_query_days=31,
        latest_group=("satellite",),
    ),
    Dataset(
        id="goes-flares",
        title="GOES: flares solares (eventos)",
        description="Flares detectados no fluxo de raios-X do GOES, com início, pico, fim e classe.",
        stage="sun",
        provider=SWPC,
        source_url="https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json",
        license=SWPC_LICENSE,
        cadence="event",
        partition="month",
        key=("time", "satellite"),
        fields=(
            _time("Início do flare (UTC)"),
            Field("satellite", "int", None, "Número do satélite GOES"),
            Field("max_time", "time", None, "Pico do flare (UTC)"),
            Field("end_time", "time", None, "Fim do flare (UTC); vazio se ainda em andamento"),
            Field("begin_class", "str", None, "Classe no início"),
            Field("max_class", "str", None, "Classe no pico (ex.: M1.2)"),
            Field("end_class", "str", None, "Classe no fim"),
            Field("max_flux", "float", "W/m^2", "Fluxo 0.1-0.8 nm no pico"),
            Field("integrated_flux", "float", "J/m^2", "Fluxo integrado 0.1-0.8 nm"),
        ),
        schedule="realtime",
        max_query_days=366,
        default_days=30,
    ),
    Dataset(
        id="solar-regions",
        title="Regiões ativas do Sol (diário)",
        description="Regiões ativas numeradas pela NOAA, com classificação magnética e "
        "probabilidades oficiais de flare.",
        stage="sun",
        provider=SWPC,
        source_url="https://services.swpc.noaa.gov/json/solar_regions.json",
        license=SWPC_LICENSE,
        cadence="1d",
        partition="month",
        key=("time", "region"),
        fields=(
            _time("Data da observação (UTC, meia-noite)"),
            Field("region", "int", None, "Número NOAA da região"),
            Field("location", "str", None, "Posição heliográfica (ex.: N13W18)"),
            Field("latitude", "int", "deg", "Latitude heliográfica"),
            Field("longitude", "int", "deg", "Longitude heliográfica"),
            Field("area", "int", "MSH", "Área das manchas (milionésimos do hemisfério)"),
            Field("spot_class", "str", None, "Classificação McIntosh"),
            Field("mag_class", "str", None, "Classificação magnética de Mount Wilson"),
            Field("number_spots", "int", None, "Número de manchas"),
            Field("c_flare_probability", "int", "%", "Probabilidade NOAA de flare classe C"),
            Field("m_flare_probability", "int", "%", "Probabilidade NOAA de flare classe M"),
            Field("x_flare_probability", "int", "%", "Probabilidade NOAA de flare classe X"),
            Field("c_xray_events", "int", None, "Flares C no dia"),
            Field("m_xray_events", "int", None, "Flares M no dia"),
            Field("x_xray_events", "int", None, "Flares X no dia"),
        ),
        schedule="realtime",
        max_query_days=366,
        default_days=7,
        latest_mode="snapshot",
    ),
    Dataset(
        id="cme-cactus",
        title="Ejeções de massa coronal (CACTus, desde 1997)",
        description="CMEs detectadas automaticamente pelo CACTus nas imagens do coronógrafo "
        "LASCO/SOHO. CMEs largas (halo) e rápidas são as que podem atingir a Terra em 1 a 4 dias.",
        stage="sun",
        provider="SIDC / Observatório Real da Bélgica (CACTus)",
        source_url="https://www.sidc.be/cactus/",
        license="Uso livre com citação: Robbrecht & Berghmans (2004), doi:10.1051/0004-6361:20041302",
        cadence="event",
        partition="year",
        key=("time", "pa", "width"),
        fields=(
            _time("Início estimado da CME (t0, UTC)"),
            Field("duration", "int", "h", "Duração da liberação"),
            Field("pa", "int", "deg", "Ângulo principal, anti-horário a partir do norte solar"),
            Field("width", "int", "deg", "Largura angular"),
            Field("speed", "int", "km/s", "Velocidade mediana"),
            Field("speed_sigma", "int", "km/s", "Variação da velocidade ao longo da CME"),
            Field("speed_min", "int", "km/s", "Menor velocidade detectada"),
            Field("speed_max", "int", "km/s", "Maior velocidade detectada"),
            Field("halo", "int", None, "0 = não; 2 = largura >90°; 3 = >180°; 4 = >270° (halo)"),
            Field("catalog", "str", None, "science (1997-2015), quicklook ou realtime"),
        ),
        historical=True,
        schedule="realtime",
        update_days=40,
        max_query_days=3660,
        default_days=30,
    ),
    Dataset(
        id="predictions",
        title="Previsões de Dst e Kp (modelos do projeto)",
        description="Previsões emitidas de hora em hora pelos modelos de IA do projeto, a partir "
        "do vento solar no L1, dos índices atuais e das CMEs recentes.",
        stage="ground",
        provider="Future Space",
        source_url="https://github.com/Strutf/future-space",
        license="MIT. Previsões experimentais: não use para decisões operacionais.",
        cadence="1h",
        partition="month",
        key=("time", "target", "horizon_hours"),
        fields=(
            _time("Momento da emissão: fim da última hora completa de dados (UTC)"),
            Field("target", "str", None, "Índice previsto: dst ou kp"),
            Field("horizon_hours", "int", "h", "Antecedência da previsão"),
            Field("target_time", "time", None, "Início da hora (Dst) ou do bloco de 3 h (Kp) previsto"),
            Field("value", "float", None, "Valor previsto (nT para Dst)"),
            Field("storm_probability", "float", None, "Probabilidade de tempestade (Kp>=5 ou Dst<=-50)"),
            Field("storm_alert", "bool", None, "True se a probabilidade passou do limiar do modelo"),
            Field("model_version", "str", None, "Versão do conjunto de modelos"),
        ),
        schedule="derived",
        max_query_days=93,
        default_days=2,
        latest_mode="snapshot",
    ),
    Dataset(
        id="verification",
        title="Verificação das previsões",
        description="Cada previsão do projeto comparada com o valor observado depois, junto da "
        "persistência e, para o Kp, da previsão oficial da NOAA emitida até o mesmo instante.",
        stage="ground",
        provider="Future Space",
        source_url="https://github.com/Strutf/future-space",
        license="MIT",
        cadence="1h",
        partition="month",
        key=("time", "target", "horizon_hours"),
        fields=(
            _time("Momento em que a previsão foi emitida (UTC)"),
            Field("target", "str", None, "Índice previsto: dst ou kp"),
            Field("horizon_hours", "int", "h", "Antecedência da previsão"),
            Field("target_time", "time", None, "Hora (Dst) ou bloco de 3 h (Kp) previsto"),
            Field("predicted", "float", None, "Valor previsto pelo modelo"),
            Field("observed", "float", None, "Valor observado"),
            Field("error", "float", None, "Previsto menos observado"),
            Field("persistence", "float", None, "Último valor conhecido na emissão (baseline)"),
            Field("noaa", "float", None, "Kp previsto pela NOAA até o mesmo instante (só Kp)"),
            Field("storm_probability", "float", None, "Probabilidade de tempestade prevista"),
            Field("storm_alert", "bool", None, "Se o modelo emitiu alerta"),
            Field("storm_observed", "bool", None, "Se houve tempestade (Kp>=5 ou Dst<=-50)"),
            Field("model_version", "str", None, "Versão dos modelos"),
        ),
        schedule="derived",
        max_query_days=93,
        default_days=7,
    ),
    Dataset(
        id="kp",
        title="Índice Kp planetário (NOAA, 3 h)",
        description="Estimativa em tempo quase real do índice Kp (0 a 9) pela NOAA. "
        "Kp >= 5 corresponde a tempestade geomagnética (G1 ou mais).",
        stage="ground",
        provider=SWPC,
        source_url="https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
        license=SWPC_LICENSE,
        cadence="3h",
        partition="year",
        key=("time",),
        fields=(
            _time("Início do intervalo de 3 h (UTC)"),
            Field("kp", "float", None, "Índice Kp"),
            Field("a_running", "int", "nT", "Índice a equivalente"),
            Field("station_count", "int", None, "Estações usadas na estimativa"),
        ),
        schedule="realtime",
        max_query_days=3660,
        default_days=7,
    ),
    Dataset(
        id="kp-forecast",
        title="Previsão oficial de Kp (NOAA)",
        description="Previsões de Kp emitidas pela NOAA, guardadas a cada coleta. Serve de "
        "baseline para comparar com os modelos de IA do projeto.",
        stage="ground",
        provider=SWPC,
        source_url="https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json",
        license=SWPC_LICENSE,
        cadence="3h",
        partition="month",
        key=("time", "target_time"),
        fields=(
            _time("Momento em que a previsão foi coletada (UTC, hora cheia)"),
            Field("target_time", "time", None, "Intervalo de 3 h previsto (UTC)"),
            Field("kp", "float", None, "Kp previsto"),
            Field("noaa_scale", "str", None, "Escala G prevista (G1-G5), se houver"),
        ),
        schedule="realtime",
        max_query_days=93,
        default_days=1,
        latest_mode="snapshot",
    ),
    Dataset(
        id="kp-forecast-3day",
        title="Previsão oficial de Kp: arquivo do 3-Day Forecast (NOAA, desde 2022)",
        description="Todas as edições do 3-Day Forecast da NOAA (00:30 e 12:30 UTC), com o Kp "
        "previsto por bloco de 3 h. Permite comparar a NOAA com o Kp observado no histórico.",
        stage="ground",
        provider="NOAA SWPC, arquivo da NCEI",
        source_url="https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/3day_forecast/",
        license=SWPC_LICENSE,
        cadence="12h",
        partition="month",
        key=("time", "target_time"),
        fields=(
            _time("Publicação da previsão (UTC)"),
            Field("target_time", "time", None, "Bloco de 3 h previsto (UTC)"),
            Field("kp", "float", None, "Kp previsto"),
            Field("noaa_scale", "str", None, "Escala G prevista (G1-G5), se houver"),
        ),
        schedule="daily",
        update_days=5,
        historical=True,
        max_query_days=93,
        default_days=3,
        latest_mode="snapshot",
        notes="A NCEI publica com cerca de um dia de atraso; as previsões do dia vêm de kp-forecast.",
    ),
    Dataset(
        id="dst",
        title="Índice Dst (Kyoto, 1 h)",
        description="Índice Dst em tempo quase real (WDC Kyoto, via NOAA). Mede a corrente de "
        "anel; valores abaixo de -50 nT indicam tempestade. Mais sensível a latitudes baixas.",
        stage="ground",
        provider="WDC for Geomagnetism, Kyoto (via NOAA SWPC)",
        source_url="https://services.swpc.noaa.gov/products/kyoto-dst.json",
        license="Uso científico livre com citação: WDC for Geomagnetism, Kyoto.",
        cadence="1h",
        partition="year",
        key=("time",),
        fields=(
            _time("Início da hora (UTC)"),
            Field("dst", "float", "nT", "Índice Dst (tempo real, provisório)"),
        ),
        schedule="realtime",
        max_query_days=1830,
        default_days=7,
    ),
    Dataset(
        id="kp-gfz",
        title="Índice Kp oficial (GFZ Potsdam, desde 1932)",
        description="Série oficial do índice Kp mantida pelo GFZ, desde 1932. A coluna status "
        "indica se o valor é preliminar (pre), nowcast (now) ou definitivo (def).",
        stage="ground",
        provider="GFZ German Research Centre for Geosciences",
        source_url="https://kp.gfz.de/",
        license="CC BY 4.0. Cite: Matzka et al. (2021), doi:10.5880/Kp.0001",
        cadence="3h",
        partition="year",
        key=("time",),
        fields=(
            _time("Início do intervalo de 3 h (UTC)"),
            Field("kp", "float", None, "Índice Kp"),
            Field("status", "str", None, "pre, now ou def"),
        ),
        historical=True,
        schedule="daily",
        update_days=60,
        max_query_days=3660,
        default_days=30,
    ),
    Dataset(
        id="omni-hourly",
        title="OMNI: vento solar + índices (1 h, desde 1963)",
        description="Base OMNI2 da NASA: vento solar propagado até a Terra e índices "
        "geomagnéticos, alinhados hora a hora. É a principal base de treino dos modelos.",
        stage="multi",
        provider="NASA GSFC SPDF (OMNIWeb), via CDAWeb HAPI",
        source_url="https://cdaweb.gsfc.nasa.gov/hapi/info?id=OMNI2_H0_MRG1HR",
        license="Domínio público (NASA). Cite: King & Papitashvili (2005), doi:10.1029/2004JA010649",
        cadence="1h",
        partition="month",
        key=("time",),
        fields=(
            _time("Início da hora (UTC)"),
            Field("b_abs", "float", "nT", "Média da intensidade do IMF"),
            Field("bx_gse", "float", "nT", "IMF Bx (GSE)"),
            Field("by_gsm", "float", "nT", "IMF By (GSM)"),
            Field("bz_gsm", "float", "nT", "IMF Bz (GSM)"),
            Field("temperature", "float", "K", "Temperatura dos prótons"),
            Field("density", "float", "cm^-3", "Densidade dos prótons"),
            Field("speed", "float", "km/s", "Velocidade do vento solar"),
            Field("pressure", "float", "nPa", "Pressão dinâmica"),
            Field("e_field", "float", "mV/m", "Campo elétrico"),
            Field("beta", "float", None, "Beta do plasma"),
            Field("sunspot_number", "float", None, "Número de manchas solares (diário)"),
            Field("f107", "float", "sfu", "Fluxo de rádio solar F10.7"),
            Field("kp", "float", None, "Índice Kp"),
            Field("dst", "float", "nT", "Índice Dst"),
            Field("ae", "float", "nT", "Índice AE"),
            Field("ap", "float", "nT", "Índice ap"),
            Field("al", "float", "nT", "Índice AL"),
            Field("au", "float", "nT", "Índice AU"),
        ),
        historical=True,
        schedule="daily",
        update_days=120,
        max_query_days=366,
        default_days=30,
        notes="O OMNI é publicado com atraso de algumas semanas em relação ao tempo real.",
    ),
    Dataset(
        id="aurora-ovation",
        title="Mapa de probabilidade de aurora (NOAA OVATION, 1°)",
        description="Probabilidade de aurora visível em cada ponto de uma grade global de 1°, "
        "calculada pelo modelo OVATION Prime da NOAA a partir do vento solar no L1. Válida para "
        "cerca de 30 a 90 minutos à frente. Só os pontos com probabilidade acima de zero.",
        stage="multi",
        provider=SWPC,
        source_url="https://services.swpc.noaa.gov/json/ovation_aurora_latest.json",
        license=SWPC_LICENSE,
        cadence="1h (coleta); a fonte atualiza a cada ~5 min",
        partition="day",
        key=("time", "lon", "lat"),
        fields=(
            _time("Instante das observações de vento solar usadas (UTC)"),
            Field("forecast_time", "time", None, "Instante para o qual o mapa vale (UTC)"),
            Field("lon", "int", "deg", "Longitude geográfica, 0 a 359 (convenção da fonte)"),
            Field("lat", "int", "deg", "Latitude geográfica, -90 a 90"),
            Field("aurora", "int", "%", "Probabilidade de aurora visível"),
        ),
        schedule="realtime",
        update_days=1,
        max_query_days=1,
        latest_mode="snapshot",
        json_partitions=False,
        latest_json=False,
        critical=False,
        notes="A NOAA não mantém arquivo público deste produto; o histórico existe a partir do "
        "início da coleta. Por ser grande (~20 mil pontos por mapa), fica só em Parquet.",
    ),
    Dataset(
        id="aurora-weather",
        title="Nuvens e visibilidade nas cidades da previsão de aurora (Open-Meteo)",
        description="Cobertura de nuvens (total, baixa, média e alta) e visibilidade horizontal "
        "previstas hora a hora para as cidades acompanhadas. Cada hora guarda a previsão mais "
        "recente; `issued` diz quando ela foi obtida.",
        stage="ground",
        provider="Open-Meteo (modelos meteorológicos nacionais, best match)",
        source_url="https://open-meteo.com/en/docs",
        license="CC BY 4.0. Cite: Weather data by Open-Meteo.com. Uso gratuito não comercial.",
        cadence="1h",
        partition="day",
        key=("time", "site"),
        fields=(
            _time("Hora prevista (UTC)"),
            Field("site", "str", None, "Cidade (id em futurespace/aurora/sites.py)"),
            Field("issued", "time", None, "Hora da coleta da previsão (UTC)"),
            Field("cloud_cover", "float", "%", "Cobertura total de nuvens"),
            Field("cloud_cover_low", "float", "%", "Nuvens baixas"),
            Field("cloud_cover_mid", "float", "%", "Nuvens médias"),
            Field("cloud_cover_high", "float", "%", "Nuvens altas"),
            Field("visibility", "float", "m", "Visibilidade horizontal"),
        ),
        schedule="realtime",
        update_days=2,
        max_query_days=7,
        default_days=2,
        latest_group=("site",),
        critical=False,
    ),
    Dataset(
        id="aurora-visibility",
        title="Previsão de visibilidade da aurora por cidade (experimental)",
        description="Pontuação de 0 a 1 das condições para ver a aurora em cada cidade nas "
        "próximas 24 h, emitida de hora em hora. Não é uma probabilidade: os pesos ainda não "
        "foram calibrados com auroras observadas. Combina a aurora esperada pelo Kp previsto pelos modelos do projeto com nuvens, "
        "escuridão e Lua; nas horas 0 e 1 guarda também a versão pelo mapa OVATION. Cada "
        "componente fica numa coluna, para a fórmula poder ser recalibrada.",
        stage="ground",
        provider="Future Space",
        source_url="https://github.com/Strutf/future-space",
        license="MIT. Previsões experimentais.",
        cadence="1h",
        partition="day",
        key=("time", "site", "target_time"),
        fields=(
            _time("Momento da emissão (UTC)"),
            Field("site", "str", None, "Cidade (id em futurespace/aurora/sites.py)"),
            Field("target_time", "time", None, "Hora prevista (UTC)"),
            Field("lead_hours", "int", "h", "Antecedência"),
            Field("latitude", "float", "deg", "Latitude geográfica da cidade"),
            Field("longitude", "float", "deg", "Longitude geográfica da cidade"),
            Field("mag_lat", "float", "deg", "Latitude magnética (dipolo)"),
            Field("kp_expected", "float", None, "Kp esperado na hora prevista"),
            Field("kp_source", "str", None, "observed (último Kp medido) ou model (previsão do projeto)"),
            Field("kp_needed", "float", None, "Menor Kp que leva a aurora ao horizonte da cidade"),
            Field("ovation_nearby", "float", "%", "Maior probabilidade OVATION num raio de 500 km"),
            Field("aurora_score_ovation", "float", None, "Pontuação de aurora pelo OVATION (só próxima hora)"),
            Field("aurora_score_kp", "float", None, "Pontuação de aurora pelo Kp esperado"),
            Field("aurora_score", "float", None, "Pontuação de aurora usada (pelo Kp, em todas as horas)"),
            Field("cloud_cover", "float", "%", "Cobertura total de nuvens prevista"),
            Field("sun_altitude", "float", "deg", "Altura do Sol"),
            Field("moon_altitude", "float", "deg", "Altura da Lua"),
            Field("moon_illumination", "float", None, "Fração iluminada da Lua"),
            Field("darkness", "float", None, "0 (Sol acima de -6°) a 1 (abaixo de -15°)"),
            Field("visibility_score", "float", None, "Pontuação de 0 a 1 para ver a aurora (não calibrada)"),
            Field("visibility_score_ovation", "float", None, "Pontuação para ver a aurora pelo OVATION (só próxima hora)"),
            Field("method", "str", None, "Versão da fórmula"),
        ),
        schedule="derived",
        max_query_days=1,
        latest_mode="snapshot",
        json_partitions=False,
        notes="Fórmula inicial (v0), ainda não calibrada com observações. Ver o README.",
    ),
)

BY_ID: dict[str, Dataset] = {d.id: d for d in DATASETS}


def get(dataset_id: str) -> Dataset:
    try:
        return BY_ID[dataset_id]
    except KeyError:
        raise KeyError(f"dataset desconhecido: {dataset_id}") from None


__all__ = ["BY_ID", "DATASETS", "Dataset", "Field", "get"]
