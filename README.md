# Future Space

Dados abertos de **clima espacial** reunidos de várias fontes (sondas, satélites, observatórios
e redes de solo), padronizados num formato único e servidos por uma **API pública e gratuita**.
O objetivo de longo prazo é aplicar IA para encontrar padrões entre as fontes e prever
tempestades solares e geomagnéticas, com atenção especial aos efeitos sobre o Brasil.

> **Status:** v0.1. Pipeline de coleta e API funcionando para 11 datasets. Modelos de IA e
> interface web ainda não começaram (veja o [roadmap](#roadmap)).

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
   └─ futurespace update
        ├─ conectores: baixam cada fonte e padronizam as colunas
        └─ writer: mescla com o que já existe, remove duplicatas e grava
              ├─ Parquet  → para treino de IA e análise (pandas, DuckDB, Polars)
              └─ JSON     → para a API e o futuro frontend
                    │
                    ▼
 Hugging Face Datasets (armazenamento versionado e gratuito)
                    │
                    ▼
 API FastAPI na Vercel (só lê os JSON, com cache)  ──▶  usuários / frontend
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
| `solar-wind-plasma` | L1 | NOAA SWPC / SOLAR-1, IMAP, ACE | 1 min | a partir do início da coleta |
| `solar-wind-mag` | L1 | NOAA SWPC / SOLAR-1, IMAP, ACE | 1 min | a partir do início da coleta |
| `goes-xray` | Órbita | NOAA SWPC / GOES | 1 min | a partir do início da coleta |
| `goes-protons` | Órbita | NOAA SWPC / GOES | 5 min | a partir do início da coleta |
| `kp` | Solo | NOAA SWPC | 3 h | a partir do início da coleta |
| `kp-forecast` | Solo | NOAA SWPC | 3 h | a partir do início da coleta |
| `dst` | Solo | WDC Kyoto | 1 h | a partir do início da coleta |
| `kp-gfz` | Solo | GFZ Potsdam | 3 h | **desde 1932** |
| `omni-hourly` | Todas | NASA OMNI | 1 h | **desde 1963** |

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

## Usando os dados para treinar IA

Os arquivos Parquet ficam num repositório de dataset no Hugging Face e podem ser lidos
direto, sem passar pela API:

```python
import pandas as pd

# Requer: pip install pandas pyarrow huggingface_hub
omni = pd.read_parquet("hf://datasets/<usuario>/future-space-data/parquet/omni-hourly/")
kp = pd.read_parquet("hf://datasets/<usuario>/future-space-data/parquet/kp-gfz/")
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
```

Partições por **dia** (dados de 1 a 5 min), **mês** (OMNI, flares, regiões, previsões) ou
**ano** (Kp, Dst).

### Tamanho dos dados (medido)

| | Parquet | CSV equivalente | JSON |
|---|---|---|---|
| OMNI completo (1963–hoje, 560 mil linhas, 768 partições mensais) | 25 MB | 60 MB | 148 MB |
| Kp GFZ completo (1932–hoje, 277 mil linhas) | 2,6 MB | — | 15 MB |
| Campo magnético L1 (1 dia, 3 sondas) | ~50 KB | — | ~340 KB |

Estimativa para os dados de tempo real: algumas dezenas de MB por ano em Parquet.

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
futurespace serve --reload                               # API em http://localhost:8000/docs

pytest -q                                                # testes
```

### Variáveis de ambiente

| Variável | Uso | Padrão |
|---|---|---|
| `FS_STORAGE` | Onde o pipeline grava: `local`, `hf` ou `s3` | `local` |
| `FS_DATA_DIR` | Pasta para `local` | `data` |
| `FS_HF_REPO` | Repositório de dataset no HF, ex.: `usuario/future-space-data` | — |
| `HF_TOKEN` | Token do Hugging Face com permissão de escrita (só no pipeline) | — |
| `FS_DATA_URL` | URL base alternativa para a API ler os JSON | — |
| `FS_S3_BUCKET`, `FS_S3_ENDPOINT_URL` | Para `s3` (Cloudflare R2, Backblaze B2...) | — |

A API lê de `FS_DATA_URL`; se não houver, de `FS_HF_REPO`; se não houver, da pasta local.

## Publicando de graça

1. **Hugging Face:** crie uma conta e um repositório do tipo *Dataset* **público**
   (ex.: `future-space-data`). Gere um token com permissão de escrita em
   *Settings → Access Tokens*.
2. **GitHub:** suba este repositório como **público** (Actions é ilimitado em repositórios
   públicos). Em *Settings → Secrets and variables → Actions*:
   - *Secret* `HF_TOKEN` = o token do passo 1
   - *Variable* `FS_HF_REPO` = `usuario/future-space-data`
3. **Carga inicial:** rode os backfills localmente com `FS_STORAGE=hf` (ou pela aba Actions
   → *ingest* → *Run workflow* com `all`).
4. **Vercel:** importe o repositório e defina a variável `FS_HF_REPO`. O `vercel.json`
   direciona todas as rotas para a API.

### Cuidados com planos gratuitos
- O GitHub **desativa workflows agendados após 60 dias sem atividade** no repositório.
  Faça pelo menos um commit nesse período, ou reative pela aba Actions.
- O cron do GitHub pode atrasar alguns minutos em horários de pico. Isso não é problema,
  porque os endpoints guardam de 24 h a 7 dias.
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
│   └── connectors/         # um conector por fonte (swpc.py, gfz.py, omni.py)
└── api/app.py              # API FastAPI (sem pandas, para caber em serverless)
api/index.py + vercel.json  # entrada da Vercel
.github/workflows/          # coleta agendada e testes
```

**Para adicionar uma fonte:** descreva o dataset em `catalog.py`, escreva uma função
`(client, start, end) -> DataFrame` em `pipeline/connectors/` e registre-a em
`connectors/__init__.py`. O writer e a API passam a atendê-la automaticamente.

## Roadmap

- [x] Pipeline de coleta de 11 datasets (NOAA, NASA, GFZ, Kyoto)
- [x] API pública com JSON/CSV, cache e documentação automática
- [ ] **IA, fase 1:** previsão de Kp e Dst de 1 a 6 h à frente com OMNI + vento solar,
      comparada à persistência e à previsão oficial da NOAA
- [ ] Novas fontes: NASA DONKI (catálogo de CMEs), magnetômetros INTERMAGNET (Vassouras e
      Tatuoca, no Brasil), INPE/EMBRACE, TEC ionosférico
- [ ] **IA, fase 2:** previsão de flares (parâmetros SHARP do SDO/HMI), ligação automática
      de eventos entre fontes (qual CME causou qual tempestade), detecção de anomalias
- [ ] Endpoints de previsão na API (`/v1/predictions/...`)
- [ ] Interface web (Next.js na Vercel)

## Licença

Código sob licença MIT. Cada dataset mantém a licença da fonte original, indicada no
catálogo (`/v1/datasets/{id}`).
