# Future Space

Dados abertos de **clima espacial** reunidos de várias fontes (sondas, satélites, observatórios
e redes de solo), padronizados num formato único e servidos por uma **API pública e gratuita**.
O objetivo de longo prazo é aplicar IA para encontrar padrões entre as fontes e prever
tempestades solares e geomagnéticas, com atenção especial aos efeitos sobre o Brasil.

> **Status:** coleta de 12 fontes de hora em hora, API pública e modelos de IA que preveem
> Dst e Kp de 1 a 24 h à frente, publicando previsões a cada hora. A interface web ainda não
> começou (veja o [roadmap](#roadmap)).
>
> - **API:** https://future-space-six.vercel.app/docs
> - **Dados (Parquet/JSON):** https://huggingface.co/datasets/Strutf/future-space

## Por que clima espacial?

Uma tempestade solar é um único fenômeno que atravessa várias etapas, e cada etapa é
observada por instrumentos diferentes:

```
   SOL                 ESPAÇO (L1)               ÓRBITA DA TERRA            SOLO
 flares, regiões  ──▶  vento solar e campo  ──▶  raios-X, prótons     ──▶  índices Kp e Dst,
 ativas, ejeções       magnético (~1 h antes     energéticos              magnetômetros,
 de massa (CMEs)       de chegar à Terra)                                  ionosfera/GPS
```

Os padrões interessantes estão nas **relações entre as fontes**: o que acontece no Sol hoje
prevê o que os magnetômetros vão medir em dois dias? Por isso o projeto junta tudo num
formato comum: toda tabela tem uma coluna `time` em UTC.

## Como funciona

Não há servidor nem banco de dados rodando o tempo todo. Tudo é pré-calculado e salvo em
arquivos, o que permite manter o projeto **100% gratuito**.

```
 GitHub Actions (de hora em hora)
   ├─ futurespace update
   │    ├─ conectores: baixam cada fonte e padronizam as colunas
   │    └─ writer: mescla com o que já existe, remove duplicatas e grava
   │          ├─ Parquet  → para treino de IA e análise (pandas, DuckDB, Polars)
   │          └─ JSON     → para a API e o futuro frontend
   └─ futurespace predict
        └─ aplica os modelos de IA aos dados mais recentes e grava as previsões
                    │
                    ▼
 Hugging Face Datasets (dados + modelos, versionados e gratuitos)
                    │
                    ▼
 API FastAPI na Vercel (só lê os JSON, com cache)  ──▶  usuários / frontend

 Treino (manual, no seu computador): futurespace train --publish
```

- **Idempotente:** rodar a coleta duas vezes não duplica nada. Valores revisados pela fonte
  (ex.: Kp preliminar → definitivo) substituem os antigos.
- **Tolerante a falhas:** se uma fonte estiver fora do ar, as outras continuam, e o erro
  fica registrado no log da execução.

---

## Fontes de dados

Todas as fontes abaixo são **públicas e gratuitas**, e nenhuma exige cadastro.
Organizadas pela etapa da tempestade.

### 🌞 Etapa 1: No Sol

#### `goes-flares`: Flares solares
- **De onde vem:** NOAA Space Weather Prediction Center (SWPC), o centro oficial de previsão
  de clima espacial dos EUA, a partir dos satélites **GOES**.
- **Endpoint:** `https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json`
- **O que traz:** cada flare detectado, com horário de início, pico e fim, a classe
  (A, B, C, M ou X, em escala logarítmica: um X é 10× mais intenso que um M) e o fluxo de
  raios-X no pico.
- **Para que serve:** flares classe M e X causam apagões de rádio (escala R da NOAA) e
  costumam acompanhar ejeções de massa coronal.
- **Cadência / histórico na fonte:** por evento / últimos 7 dias.

#### `solar-regions`: Regiões ativas
- **De onde vem:** NOAA SWPC.
- **Endpoint:** `https://services.swpc.noaa.gov/json/solar_regions.json`
- **O que traz:** as regiões de manchas solares numeradas pela NOAA, com posição no disco
  solar, área, classificação McIntosh, classificação magnética de Mount Wilson
  (ex.: `BGD` = Beta-Gamma-Delta, alta complexidade) e as **probabilidades oficiais de flare**
  C, M e X.
- **Para que serve:** regiões magneticamente complexas são as que produzem os grandes flares.
  As probabilidades da NOAA servem de baseline para um futuro modelo de previsão de flares.
- **Cadência / histórico na fonte:** diária / cerca de 30 dias.

#### `cme-cactus`: Ejeções de massa coronal (CMEs), desde 1997
- **De onde vem:** **CACTus**, do SIDC (Observatório Real da Bélgica). Um algoritmo detecta
  CMEs automaticamente nas imagens do coronógrafo **LASCO**, a bordo da sonda SOHO.
- **Endpoints:** três produtos do mesmo algoritmo, combinados pelo conector:
  - catálogo científico (1997–2015): `https://www.sidc.be/cactus/catalog/LASCO/2_5_0/cme_lz.txt`
  - quicklook mensal (2015 em diante): `…/2_5_0/qkl/<ano>/<mês>/cmecat.txt`
  - tempo real (última semana): `https://www.sidc.be/cactus/out/cmecat.txt`
- **O que traz:** cada CME com horário de início, direção (`pa`), largura angular (`width`),
  velocidade e indicação de halo. A coluna `catalog` diz de qual produto veio.
- **Para que serve:** CMEs **largas e rápidas** são a principal causa das grandes tempestades
  geomagnéticas e chegam à Terra em 1 a 4 dias. São as variáveis que dão antecedência aos
  modelos. Como o algoritmo é o mesmo desde 1997, o modelo treina e roda ao vivo com dados
  equivalentes.
- **Detalhe:** quando o quicklook passa a cobrir um período, as detecções de tempo real
  desse período são descartadas, porque o mesmo evento aparece nos dois com valores um pouco
  diferentes.
- **Licença:** uso livre com citação. Cite Robbrecht & Berghmans (2004), doi:10.1051/0004-6361:20041302.
- **Por que não a NASA DONKI?** A DONKI é a referência para CMEs, mas estava fora do ar
  durante a implementação e só cobre desde 2010. Ela está no roadmap como fonte complementar.

### 🛰️ Etapa 2: No espaço, no ponto L1

O ponto de Lagrange L1 fica a cerca de 1,5 milhão de km da Terra, na direção do Sol. As
sondas ali medem o vento solar **30 a 60 minutos antes** de ele chegar à Terra.

#### `solar-wind-plasma`: Plasma do vento solar
- **De onde vem:** NOAA SWPC, Real-Time Solar Wind (RTSW).
- **Endpoint:** `https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json`
- **O que traz:** velocidade, densidade e temperatura dos prótons, e as componentes da
  velocidade. **Inclui várias sondas ao mesmo tempo** (hoje SOLAR-1/SWFO-L1, IMAP e ACE).
  A coluna `source` diz qual sonda mediu, e `active` indica a operacional da NOAA.
- **Para que serve:** velocidade e densidade altas indicam a chegada de uma CME ou de um
  feixe rápido de vento solar.
- **Cadência / histórico na fonte:** 1 minuto / **só cerca de 24 h**. Por isso a coleta
  precisa ser frequente: o que não for coletado se perde.

#### `solar-wind-mag`: Campo magnético interplanetário (IMF)
- **De onde vem:** NOAA SWPC, RTSW.
- **Endpoint:** `https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json`
- **O que traz:** intensidade total do campo (`bt`) e componentes no sistema GSM,
  também por sonda.
- **Para que serve:** a componente **`bz_gsm` negativa (apontando para o sul)** é o principal
  gatilho de tempestades geomagnéticas: é quando o campo do Sol se "conecta" ao da Terra.
- **Cadência / histórico na fonte:** 1 minuto / cerca de 24 h.

### 🌍 Etapa 3: Na órbita da Terra

#### `goes-xray`: Fluxo de raios-X
- **De onde vem:** NOAA SWPC, satélites geoestacionários **GOES** (hoje o GOES-18).
- **Endpoint:** `https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json`
- **O que traz:** fluxo de raios-X do Sol em duas bandas: curta (0,05–0,4 nm) e longa
  (0,1–0,8 nm). A banda longa define a classe dos flares.
- **Para que serve:** série contínua para detectar flares e para treinar modelos de previsão.
- **Cadência / histórico na fonte:** 1 minuto / 7 dias.

#### `goes-protons`: Prótons energéticos
- **De onde vem:** NOAA SWPC, satélites GOES.
- **Endpoint:** `https://services.swpc.noaa.gov/json/goes/primary/integral-protons-7-day.json`
- **O que traz:** fluxo integral de prótons em 8 faixas de energia (≥1 a ≥500 MeV).
- **Para que serve:** ≥10 MeV acima de 10 pfu define uma **tempestade de radiação**
  (escala S), que é perigosa para astronautas e satélites e afeta voos polares.
- **Cadência / histórico na fonte:** 5 minutos / 7 dias.

### 🧲 Etapa 4: No solo, onde o impacto é medido

#### `kp`: Índice Kp (NOAA)
- **De onde vem:** NOAA SWPC, a partir de uma rede de magnetômetros em solo.
- **Endpoint:** `https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json`
- **O que traz:** o índice Kp (0 a 9) a cada 3 horas, a medida global mais usada de
  atividade geomagnética. Kp ≥ 5 equivale a tempestade G1; Kp = 9 é G5 (extrema).
- **Para que serve:** é o **alvo principal** dos modelos de previsão do projeto.
- **Cadência / histórico na fonte:** 3 horas / 7 dias.

#### `kp-forecast`: Previsão oficial de Kp
- **De onde vem:** NOAA SWPC.
- **Endpoint:** `https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json`
- **O que traz:** o Kp previsto pela NOAA para os próximos 3 dias. Guardamos **cada
  previsão emitida** (coluna `time` = hora da coleta, `target_time` = intervalo previsto).
- **Para que serve:** a fonte só mostra a previsão atual. Arquivando o histórico, conseguimos
  comparar de forma justa os modelos do projeto com a previsão oficial.

#### `kp-forecast-3day`: Arquivo das previsões oficiais, desde 2022
- **De onde vem:** o produto **3-Day Forecast** da NOAA SWPC, guardado pela NCEI.
- **Endpoint:** `https://www.ngdc.noaa.gov/stp/space-weather/swpc-products/daily_reports/3day_forecast/<ano>/<mês>/`
- **O que traz:** todas as edições da previsão oficial (00:30 e 12:30 UTC), com o Kp esperado
  em cada bloco de 3 h dos três dias seguintes e a escala G prevista. Só guardamos os blocos
  que ainda não tinham começado na hora da publicação.
- **Para que serve:** medir a NOAA contra o Kp que aconteceu e comparar com a IA no histórico.
  Também completa o painel nos dias antes de a coleta horária existir.
- **Cadência / histórico:** duas edições por dia / desde março de 2022, com cerca de 1 dia de
  atraso. Setembro e outubro de 2022 não existem no arquivo.

#### `dst`: Índice Dst
- **De onde vem:** World Data Center for Geomagnetism, Kyoto (Japão), redistribuído pela
  NOAA SWPC.
- **Endpoint:** `https://services.swpc.noaa.gov/products/kyoto-dst.json`
- **O que traz:** o índice Dst em nT, de hora em hora, calculado com magnetômetros próximos
  do equador. Valores abaixo de −50 nT indicam tempestade, e abaixo de −250 nT, uma
  supertempestade.
- **Para que serve:** é mais representativo das **latitudes baixas, como as do Brasil**, do que o Kp.
- **Cadência / histórico na fonte:** 1 hora / cerca de 7 dias (valores de tempo real, provisórios).

#### `kp-gfz`: Índice Kp oficial, desde 1932
- **De onde vem:** **GFZ Potsdam** (Alemanha), o responsável oficial pelo índice Kp desde 1997.
- **Endpoint:** `https://kp.gfz.de/app/json/?start=…&end=…&index=Kp`
- **O que traz:** a série histórica completa de Kp desde 1932 (~277 mil valores). A coluna
  `status` indica se o valor é preliminar (`pre`), nowcast (`now`) ou definitivo (`def`).
- **Para que serve:** quase um século de rótulos para treinar e validar modelos, cobrindo
  vários ciclos solares de ~11 anos.
- **Licença:** CC BY 4.0. Cite Matzka et al. (2021), doi:10.5880/Kp.0001.

### 🔗 Multi-etapa

#### `omni-hourly`: NASA OMNI, desde 1963
- **De onde vem:** **NASA Goddard (SPDF/OMNIWeb)**, acessado pelo servidor
  [HAPI](https://hapi-server.org/) do CDAWeb. O HAPI é uma API padronizada da heliofísica,
  usada por vários provedores.
- **Endpoint:** `https://cdaweb.gsfc.nasa.gov/hapi/data?id=OMNI2_H0_MRG1HR&…`
- **O que traz:** a base mais usada em pesquisa de clima espacial. Ela combina dados de
  **dezenas de sondas** desde 1963 (IMP, ISEE, Wind, ACE, DSCOVR...), já intercalibrados e
  propagados até a Terra, junto com índices geomagnéticos na mesma linha do tempo: campo
  magnético, velocidade, densidade, pressão, F10.7, manchas solares, Kp, Dst, AE, AL e AU.
- **Para que serve:** é a **principal base de treino** do projeto (~560 mil horas). Aparece ali,
  por exemplo, a supertempestade de maio de 2024, com Dst = −406 nT e Bz = −35 nT.
- **Atraso:** publicado com algumas semanas de defasagem. Para o tempo real, usamos as fontes da NOAA.
- **Convenções aplicadas:** `time` = início da hora (o HAPI carimba o centro, HH:30), Kp
  dividido por 10 e valores de preenchimento (`9999.9`...) convertidos em vazio.

### Resumo

| Dataset | Etapa | Fonte | Cadência | Histórico disponível |
|---|---|---|---|---|
| `goes-flares` | Sol | NOAA SWPC / GOES | evento | a partir do início da coleta |
| `solar-regions` | Sol | NOAA SWPC | 1 dia | a partir do início da coleta |
| `cme-cactus` | Sol | SIDC / CACTus (LASCO) | evento | **desde 1997** |
| `solar-wind-plasma` | L1 | NOAA SWPC / SOLAR-1, IMAP, ACE | 1 min | a partir do início da coleta |
| `solar-wind-mag` | L1 | NOAA SWPC / SOLAR-1, IMAP, ACE | 1 min | a partir do início da coleta |
| `goes-xray` | Órbita | NOAA SWPC / GOES | 1 min | a partir do início da coleta |
| `goes-protons` | Órbita | NOAA SWPC / GOES | 5 min | a partir do início da coleta |
| `kp` | Solo | NOAA SWPC | 3 h | a partir do início da coleta |
| `kp-forecast` | Solo | NOAA SWPC | 3 h | a partir do início da coleta |
| `kp-forecast-3day` | Solo | NOAA SWPC / NCEI | 12 h | **desde 2022** |
| `dst` | Solo | WDC Kyoto | 1 h | a partir do início da coleta |
| `kp-gfz` | Solo | GFZ Potsdam | 3 h | **desde 1932** |
| `omni-hourly` | Todas | NASA OMNI | 1 h | **desde 1963** |
| `predictions` | — | Modelos deste projeto | 1 h | a partir da publicação dos modelos |

Os dados de tempo real da NOAA são de **domínio público** (governo dos EUA). Mesmo assim,
cite sempre a fonte original: o campo `license` de cada dataset na API diz como.

---

## Usando a API

Documentação interativa (Swagger) em `/docs`.

| Endpoint | O que retorna |
|---|---|
| `GET /v1/datasets` | Lista de datasets e a cobertura temporal de cada um |
| `GET /v1/datasets/{id}` | Metadados, campos, unidades, licença e cobertura |
| `GET /v1/datasets/{id}/latest` | Valores mais recentes |
| `GET /v1/datasets/{id}/data` | Série temporal num intervalo |
| `GET /v1/datasets/predictions/latest` | Previsões mais recentes da IA |
| `GET /v1/compare/kp?date=AAAA-MM-DD` | IA × NOAA × observado, bloco a bloco (dia e anterior) |
| `GET /v1/verification/summary` | Erros das previsões nos últimos 7 e 30 dias |
| `GET /`, `/dashboard`, `/sobre`, `/interativo` (e `/en/...`) | Páginas do site em português e inglês |

Parâmetros de `/data`:
- `start`, `end`: ISO 8601 em UTC (`2024-05-10` ou `2024-05-10T18:00Z`). Sem eles, a
  resposta traz os últimos dias disponíveis.
- `fields`: campos separados por vírgula (`time` e a chave sempre vêm).
- `format`: `json` (padrão) ou `csv`.

```bash
# Supertempestade de maio de 2024 vista pelo OMNI
curl "http://localhost:8000/v1/datasets/omni-hourly/data?start=2024-05-10&end=2024-05-12&fields=bz_gsm,speed,kp,dst"

# Vento solar agora, por sonda
curl "http://localhost:8000/v1/datasets/solar-wind-mag/latest"

# Kp do último mês em CSV
curl "http://localhost:8000/v1/datasets/kp-gfz/data?format=csv"
```

Cada dataset tem uma janela máxima por consulta (`max_query_days`, ex.: 3 dias para os de
1 minuto), para manter as respostas leves. **Para volumes grandes, como treino de IA, use os
arquivos Parquet diretamente** (próxima seção).

Limites de uso:
- **60 requisições por minuto por IP.** Acima disso, a resposta é `429` com `Retry-After`.
- Parâmetros fora da lista (`start`, `end`, `fields`, `format`, `date`) são recusados com `400`.

Detalhes em [SECURITY.md](SECURITY.md).

## Usando os dados para treinar IA

Os arquivos Parquet ficam num repositório de dataset no Hugging Face e podem ser lidos
direto, sem passar pela API:

```python
import pandas as pd

# Requer: pip install pandas pyarrow huggingface_hub
omni = pd.read_parquet("hf://datasets/Strutf/future-space/parquet/omni-hourly/")
kp = pd.read_parquet("hf://datasets/Strutf/future-space/parquet/kp-gfz/")
cmes = pd.read_parquet("hf://datasets/Strutf/future-space/parquet/cme-cactus/")
```

Funciona igual no Kaggle, no Colab ou localmente. Com DuckDB, também é possível consultar
com SQL sem baixar tudo.

### Layout dos arquivos

```
parquet/<dataset>/<partição>.parquet   ← treino e análise
json/<dataset>/<partição>.json         ← lido pela API
json/<dataset>/manifest.json           ← partições existentes e cobertura
json/<dataset>/latest.json             ← últimos valores
json/catalog.json                      ← catálogo completo
models/current.json                    ← versão dos modelos em uso
models/<versão>/bundle.joblib          ← modelos treinados
models/<versão>/report.json            ← métricas de avaliação daquela versão
```

Partições por **dia** (dados de 1 a 5 min), **mês** (OMNI, flares, regiões, previsões) ou
**ano** (Kp, Dst, CMEs).

### Tamanho dos dados (medido)

| | Parquet | CSV equivalente | JSON |
|---|---|---|---|
| OMNI completo (1963–hoje, 560 mil linhas, 768 partições mensais) | 25 MB | 60 MB | 148 MB |
| Kp GFZ completo (1932–hoje, 277 mil linhas) | 2,6 MB | — | 15 MB |
| Campo magnético L1 (1 dia, 3 sondas) | ~50 KB | — | ~340 KB |

Estimativa para os dados de tempo real: algumas dezenas de MB por ano em Parquet.

---

## IA: previsão de tempestades geomagnéticas

### Que perguntas os modelos respondem

| Pergunta | Índice | Horizontes |
|---|---|---|
| Vai ter tempestade geomagnética? Quão forte, na escala G da NOAA? | **Kp** | 3, 6, 12 e 24 h |
| Quão intenso vai ser o efeito em latitudes baixas, como no Brasil? | **Dst** | 1, 3, 6, 12 e 24 h |

Para cada índice e horizonte existem **dois modelos**:
- **Valor:** o número previsto (ex.: "Dst de −80 nT daqui a 6 h").
- **Probabilidade de tempestade:** a chance de Kp ≥ 5 (G1 ou mais) ou de Dst ≤ −50 nT,
  mais um **alerta** que dispara quando a probabilidade passa de um limiar calibrado.

### Que tipo de IA

**Gradient boosting** (`HistGradientBoostingRegressor` e `HistGradientBoostingClassifier`,
do scikit-learn): centenas de árvores de decisão pequenas, cada uma corrigindo o erro das
anteriores. É aprendizado de máquina clássico, treinado do zero nos dados do projeto, e não
um modelo de linguagem. Foi escolhido porque costuma ter o melhor desempenho em dados
tabulares, treina em minutos sem GPU, lida sozinho com dados faltantes e é explicável.

### Variáveis usadas

Regra de ouro: **só entram variáveis que também existem em tempo real.** Assim, o mesmo
modelo treinado no histórico roda ao vivo com os dados coletados de hora em hora.

| Grupo | Variáveis | Por quê |
|---|---|---|
| Vento solar agora | Bz, By, \|B\|, velocidade, densidade, temperatura, pressão | O que está chegando à Terra |
| Acoplamento | Função de Newell, VBs, ângulo do campo | Quanta energia do vento solar entra na magnetosfera |
| Histórico recente | Valores de 1–6 h atrás, médias de 3–24 h, Bz mínimo | Se a perturbação está crescendo ou passando |
| Estado da Terra | Dst atual e passado, último Kp encerrado, Kp máximo em 24 h | Se já há uma tempestade em andamento |
| CMEs (Sol) | CME larga mais rápida em 24/48/72/96 h, nº de CMEs largas, chegada estimada | Antecedência de horas a dias |
| Recorrência (Sol) | Velocidade, Kp e Dst de 27 dias atrás (uma rotação solar) | Feixes rápidos de vento solar tendem a voltar |
| Calendário | Dia do ano, hora UTC | Época do ano e hora do dia também influenciam |

Cuidados contra resultados otimistas demais:
- Cada linha usa só dados **até aquela hora**, e há um teste automático que garante isso.
- O Kp só entra depois que o bloco de 3 h termina.
- Uma CME só "existe" para o modelo **6 h depois** do início, que é o atraso típico do
  catálogo em tempo real.

### Como os modelos são avaliados

- **Divisão temporal, nunca aleatória:** treino de 1995 a 2014, validação de 2015 a 2018
  (para escolher hiperparâmetros e o limiar de alerta) e **teste de 2019 até hoje**. O teste
  inclui a supertempestade de maio de 2024.
- **Comparação com a persistência** ("vai continuar igual"). Um modelo que não vence isso
  não serve.
- **Métricas de evento**, porque tempestades são raras e a acurácia engana:
  - **TSS:** 1 é perfeito, 0 é sem habilidade.
  - **POD:** fração das tempestades que foram detectadas.
  - **FAR:** fração dos alertas que eram falsos.
  - **BSS:** qualidade da probabilidade em relação à frequência histórica.

### Resultados no período de teste

Versão `v3` (82 variáveis), avaliada de jan/2019 a nov/2026, com cerca de 2.200 horas de
tempestade pelo Dst e 1.600 pelo Kp:

| Previsão | RMSE IA | RMSE persistência | Ganho | POD | FAR | TSS IA | TSS persistência | BSS | BSS sem CMEs |
|---|---|---|---|---|---|---|---|---|---|
| Dst +1 h | 3.83 | 4.40 | 13% | 91% | 10% | 0.90 | 0.87 | 0.85 | 0.85 |
| Dst +3 h | 7.49 | 8.88 | 16% | 83% | 27% | 0.82 | 0.73 | 0.67 | 0.68 |
| Dst +6 h | 10.55 | 12.58 | 16% | 72% | 38% | 0.71 | 0.61 | 0.53 | 0.53 |
| Dst +12 h | 14.10 | 16.79 | 16% | 50% | 51% | 0.49 | 0.45 | 0.33 | 0.34 |
| Dst +24 h | 16.76 | 20.59 | 19% | 39% | 72% | 0.36 | 0.28 | 0.16 | 0.15 |
| Kp +3 h | 0.76 | 1.01 | 25% | 61% | 55% | 0.60 | 0.40 | 0.36 | 0.36 |
| Kp +6 h | 0.91 | 1.17 | 23% | 55% | 71% | 0.52 | 0.29 | 0.21 | 0.22 |
| Kp +12 h | 1.04 | 1.34 | 22% | 39% | 84% | 0.34 | 0.19 | 0.08 | 0.09 |
| Kp +24 h | 1.15 | 1.50 | 23% | 45% | 92% | 0.32 | 0.10 | 0.03 | 0.02 |

RMSE em nT para o Dst e em unidades de Kp. Ganho = redução do RMSE em relação à persistência.
POD, FAR e TSS são do alerta de tempestade, com o limiar escolhido pelo HSS na validação.

O que esses números dizem:
- **O valor previsto é sempre melhor que a persistência**: de 13% a 25% menos erro.
- **O alerta de tempestade é útil até cerca de 6 h.** Para o Dst +3 h, ele detecta 83% das
  tempestades com 27% de alarmes falsos. Para 12 h e 24 h, a maioria dos alertas é falsa.
- **As CMEs do CACTus quase não mudaram o resultado** (compare as duas colunas de BSS).
  O catálogo não diz se a CME vem na direção da Terra, e muitas tempestades vêm de feixes
  rápidos de vento solar. Integrar a direção das CMEs (NASA DONKI) é o próximo passo para
  ganhar antecedência.
- Escolher o limiar pelo TSS daria TSS maiores (até 0,98), mas com 60% a 95% de alarmes
  falsos. Por isso o projeto usa o HSS.

### IA × NOAA no histórico (2022–2026)

Usando o arquivo do 3-Day Forecast, comparamos as duas previsões com o Kp observado (GFZ) nos
**mesmos blocos e com a mesma antecedência**. Para cada edição da NOAA, a IA usa só dados até
a hora cheia anterior, com o modelo treinado até 2018, que nunca viu esse período.

| Antecedência | Comparações | Erro médio IA | Erro médio NOAA | HSS do alerta IA | HSS NOAA |
|---|---|---|---|---|---|
| ~6 h | 2.534 | **0,72** | 1,16 | **0,40** | 0,24 |
| ~12 h | 2.534 | **0,80** | 1,18 | **0,21** | 0,16 |
| ~24 h | 2.534 | **0,90** | 1,16 | 0,08 | **0,16** |

- No valor do Kp, a IA erra menos em todos os horizontes.
- No alerta de tempestade, a IA ganha em 6 h e 12 h, e a NOAA em 24 h. A NOAA analisa imagens
  do Sol e a direção das CMEs, o que dá vantagem nas antecedências maiores.
- Sozinha, a NOAA erra ~1,15 de Kp em média, com 6 h ou 3 dias de antecedência, e cerca de 82%
  dos alertas de G1 dela não se confirmaram.
- **Ressalva:** nessa simulação, a IA usa o OMNI e o Kp definitivo, um pouco mais limpos que os
  dados de tempo real. A verificação diária ao vivo confirma (ou não) essa vantagem.

### Limitações conhecidas

- **O início de uma tempestade é difícil de prever.** Sem CMEs, o modelo só "vê" o vento
  solar no L1, cerca de 1 h antes de ele chegar à Terra.
- **Tempestades extremas são subestimadas.** Modelos de árvore não preveem valores além dos
  que viram no treino, e eventos como maio de 2024 (Dst −406 nT) são raríssimos no histórico.
- **Diferença entre treino e operação:** o treino usa o OMNI, que propaga o vento solar até
  a Terra, e a operação usa as medidas brutas no L1, que chegam cerca de 45 min antes. Isso
  torna a operação um pouco conservadora.
- **As previsões são experimentais.** Não use para decisões operacionais. Para isso,
  consulte a NOAA SWPC.

### Previsões ao vivo

A cada coleta de hora em hora, o workflow roda `futurespace predict`:
1. Monta a tabela horária com os dados de tempo real (sonda operacional no L1, Dst de Kyoto,
   Kp da NOAA, CMEs do CACTus).
2. Aplica os modelos publicados.
3. Grava no dataset `predictions`.

```bash
curl "https://future-space-six.vercel.app/v1/datasets/predictions/latest"
```

Cada previsão traz `target` (dst ou kp), `horizon_hours`, `target_time`, `value`,
`storm_probability`, `storm_alert` e `model_version`.

### Verificação: quanto as previsões erraram

Uma vez por dia, `futurespace verify` junta cada previsão com o valor observado depois e
grava o dataset `verification`. Para cada previsão, ele guarda:
- o erro;
- a persistência no momento da emissão;
- no caso do Kp, a previsão da NOAA emitida até o mesmo instante.

O resumo dos últimos 7 e 30 dias fica em `GET /v1/verification/summary`.

O modelo **não se ajusta sozinho com os erros**. Isso é proposital: um modelo que muda a cada
erro pode piorar sem ninguém perceber. A verificação serve para acompanhar o desempenho e
decidir quando treinar de novo.

### Treinar e publicar novos modelos

O treino roda no GitHub Actions (workflow `train`), uma vez por mês ou pelo botão
**Run workflow**:
1. Atualiza o histórico de CMEs.
2. Avalia com a divisão temporal fixa, para os números serem comparáveis entre versões.
3. Re-treina o modelo publicado com **todos os dados até hoje**.
4. **Só publica se a avaliação não piorar** em relação à versão atual. A opção `force`
   ignora essa trava.

O token do Hugging Face fica só nos secrets do GitHub.

Para experimentar localmente:

```bash
futurespace backfill cme-cactus --start 1997-01-01   # se ainda não tiver as CMEs
futurespace train                                    # treina e avalia (~10 min), salva em ./models
futurespace train --no-cme --out models-sem-cme      # mesmo treino sem CMEs, para comparar
```

`models/current.json` aponta para a versão em uso. Para voltar atrás, aponte-o para a
versão anterior, que continua guardada em `models/<versão>/`.

### Site

O site fica em https://future-space-six.vercel.app, com visual de quadro-negro e desenhos em
giz, **em português e inglês**. Ele é servido pela própria API, sem build: cada página é um
molde em `src/futurespace/api/web/pages`, encaixado em `web/base.html`, e o tema está em
`web/static/chalk.css`.

| Página | Português | Inglês | O que mostra |
|---|---|---|---|
| Home | `/` | `/en` | O que o projeto faz, como funciona e a previsão do momento |
| Previsões | `/dashboard` | `/en/dashboard` | Kp previsto pela IA e pela NOAA para ontem, hoje e amanhã, com o observado e quem chegou mais perto |
| Sobre | `/sobre` | `/en/about` | Como acontece uma tempestade solar, o que são Kp, Dst e CME, e todas as fórmulas |
| Interativo | `/interativo` | `/en/interactive` | Em breve |

**Traduções:** todo o texto fica em `web/locales/pt.json` e `web/locales/en.json`.
- Nas páginas, os textos aparecem como `{{t:secao.chave}}` e os links internos como
  `{{url:pagina}}`. O servidor monta as duas versões ao iniciar.
- Os textos usados pelo JavaScript ficam na seção `js` e são acessados com `t("chave", {variavel})`.
- O seletor de idioma do menu (globinho) leva à mesma página na outra língua e lembra a escolha. Na
  primeira visita, navegadores que não estão em português vão para a versão em inglês.
- Os testes garantem que os dois arquivos têm as mesmas chaves e as mesmas variáveis.

**Para adicionar um idioma:** crie `web/locales/<idioma>.json` com as mesmas chaves, inclua o
idioma em `LANGS` e os endereços em `PAGE_URLS`, ambos em `api/app.py`.

Os dados do painel vêm de `GET /v1/compare/kp`.

---

## Rodando localmente

Requer Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,hf]"

futurespace datasets                                     # lista o catálogo
futurespace update                                       # coleta o tempo real → ./data
futurespace update --schedule daily                      # OMNI e Kp GFZ recentes
futurespace backfill omni-hourly --start 1963-01-01     # histórico completo (~4 min)
futurespace backfill kp-gfz --start 1932-01-01          # histórico completo (~30 s)
futurespace backfill cme-cactus --start 1997-01-01      # CMEs desde 1997 (~40 s)
futurespace train                                        # treina os modelos (~10 min)
futurespace predict                                      # emite previsões com os dados atuais
futurespace serve --reload                               # API em http://localhost:8000/docs

pytest -q                                                # testes
```

### Variáveis de ambiente

| Variável | Uso | Padrão |
|---|---|---|
| `FS_STORAGE` | Onde o pipeline grava: `local`, `hf` ou `s3` | `local` |
| `FS_DATA_DIR` | Pasta para `local` | `data` |
| `FS_HF_REPO` | Repositório de dataset no HF, ex.: `Strutf/future-space` | — |
| `HF_TOKEN` | Token do Hugging Face com permissão de escrita (só no pipeline) | — |
| `FS_DATA_URL` | URL base alternativa para a API ler os JSON | — |
| `FS_RATE_LIMIT_PER_MINUTE` | Requisições por minuto por IP na API (`0` desliga) | `60` |
| `FS_S3_BUCKET`, `FS_S3_ENDPOINT_URL` | Para `s3` (Cloudflare R2, Backblaze B2...) | — |

A API lê de `FS_DATA_URL`; se não houver, de `FS_HF_REPO`; se não houver, da pasta local.

## Publicando de graça

1. **Hugging Face:** crie uma conta e um repositório do tipo *Dataset* **público**. Gere um
   token *fine-grained* com escrita só nesse repositório, em *Settings → Access Tokens*.
2. **GitHub:** suba este repositório como **público** (Actions é ilimitado em repositórios
   públicos). Em *Settings → Secrets and variables → Actions*:
   - *Secret* `HF_TOKEN` = o token do passo 1
   - *Variable* (não *Secret*) `FS_HF_REPO` = `usuario/repositorio`
3. **Carga inicial:** rode os backfills localmente e envie a pasta `data/` com
   `hf upload <repo> data . --repo-type dataset`, ou rode os backfills com `FS_STORAGE=hf`.
4. **Modelos:** rode o workflow `train` pela aba Actions (Run workflow). Ele treina e publica
   usando o token guardado nos secrets.
5. **Vercel:** importe o repositório (não clone) e defina a variável `FS_HF_REPO`. O token
   não é necessário ali, porque a API só lê dados públicos.

Recomendações de segurança para o repositório estão em [SECURITY.md](SECURITY.md).

### Disparo das coletas (importante)

O agendamento nativo do GitHub Actions (`on.schedule`) **nunca disparou neste repositório**.
Ele foi testado até com intervalo de 5 minutos, enquanto as execuções manuais funcionam
normalmente. Por isso, o disparo principal é externo, pelo [cron-job.org](https://cron-job.org)
(gratuito), chamando a API `workflow_dispatch` do GitHub:

| Job | Horário (Brasília) | URL (`.../actions/workflows/…/dispatches`) | Body |
|---|---|---|---|
| ingest realtime | `23 * * * *` | `ingest.yml` | `{"ref":"main","inputs":{"modo":"realtime"}}` |
| ingest daily | `47 0 * * *` | `ingest.yml` | `{"ref":"main","inputs":{"modo":"daily"}}` |
| train mensal | `41 1 1 * *` | `train.yml` | `{"ref":"main"}` |

Configuração de cada job:
- **Método:** `POST`
- **Cabeçalhos:** `Authorization: Bearer <token>`, `Accept: application/vnd.github+json` e
  `Content-Type: application/json`
- **Token:** *fine-grained*, só com **Actions: Read and write** neste repositório
- **Resposta esperada:** `204`
- **Alerta:** ative o aviso por e-mail em caso de falha. Lembre de renovar o token antes de
  ele expirar.

**Cargas históricas em produção:** em Actions → ingest → Run workflow, preencha `backfill`
com o dataset (ex.: `kp-forecast-3day`) e `desde` com a data inicial. O workflow roda só a
carga, com o token que já está nos secrets.

O `on.schedule` continua no workflow, no minuto 53, como reserva. Se ele voltar a funcionar,
os dois disparos convivem: um espera o outro, e os dados não duplicam.

### Cuidados com planos gratuitos
- O GitHub **desativa workflows agendados após 60 dias sem atividade** no repositório.
  Faça pelo menos um commit nesse período, ou reative pela aba Actions.
- O cron do GitHub pode atrasar, pular ou nem disparar, e neste repositório nunca disparou.
  Por isso, o disparo principal é externo (veja "Disparo das coletas"). O vento solar só
  fica ~24 h na fonte: mais de um dia sem coleta perde dados.
- O plano Hobby da Vercel é para uso não comercial e limita respostas a ~4,5 MB, por isso
  existe a janela máxima por consulta.
- Os limites dos serviços mudam com o tempo. Confira antes de depender deles.

## Estrutura do código

```
src/futurespace/
├── catalog.py              # metadados de todos os datasets (campos, unidades, licenças)
├── partitions.py           # layout dos arquivos e utilidades de tempo
├── storage.py              # backends: local, Hugging Face, S3, HTTP
├── cli.py                  # comando `futurespace`
├── pipeline/
│   ├── http.py             # cliente HTTP com novas tentativas
│   ├── writer.py           # mescla, deduplica e grava Parquet + JSON
│   └── connectors/         # um conector por fonte (swpc, gfz, omni, cactus)
├── ml/
│   ├── features.py         # tabela horária, variáveis e alvos (sem vazamento do futuro)
│   ├── train.py            # treino, validação temporal e avaliação
│   ├── metrics.py          # RMSE, TSS, POD, FAR, Brier
│   ├── serve.py            # previsões ao vivo e publicação dos modelos
│   └── verify.py           # comparação das previsões com o observado
└── api/
    ├── app.py              # API FastAPI (sem pandas, para caber em serverless)
    ├── compare.py          # comparação IA x NOAA usada pelo dashboard
    └── dashboard.html      # painel web
api/index.py + vercel.json  # entrada da Vercel
.github/workflows/          # coleta + previsão agendadas, testes
requirements-pipeline.txt   # versões fixas usadas no workflow que tem o token
```

**Para adicionar uma fonte:** descreva o dataset em `catalog.py`, escreva uma função
`(client, start, end) -> DataFrame` em `pipeline/connectors/` e registre-a em
`connectors/__init__.py`. O writer e a API passam a atendê-la automaticamente.

## Roadmap

- [x] Pipeline de coleta de 12 fontes (NOAA, NASA, GFZ, Kyoto, SIDC)
- [x] API pública com JSON/CSV, cache e documentação automática
- [x] **IA, fase 1:** previsão de Kp e Dst de 1 a 24 h, com valor, probabilidade de
      tempestade e alerta, comparada à persistência
- [x] CMEs do CACTus como variáveis de antecedência
- [x] Previsões ao vivo de hora em hora, servidas pela API
- [x] Verificação diária e comparação com a previsão oficial da NOAA
- [x] Re-treino mensal no GitHub Actions, com trava de publicação
- [x] Dashboard simples de acompanhamento
- [ ] Integrar a direção das CMEs (NASA DONKI) para ganhar antecedência
- [ ] Novas fontes: NASA DONKI, magnetômetros INTERMAGNET (Vassouras e Tatuoca, no Brasil),
      INPE/EMBRACE, TEC ionosférico
- [ ] **IA, fase 2:** previsão de flares (parâmetros SHARP do SDO/HMI), ligação automática
      de eventos entre fontes (qual CME causou qual tempestade), detecção de anomalias
- [ ] Interface web interativa (Next.js na Vercel)

## Licença

Código sob licença MIT. Cada dataset mantém a licença da fonte original, indicada no
catálogo (`/v1/datasets/{id}`).
