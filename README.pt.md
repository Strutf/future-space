# Future Space

O Future Space coleta dados públicos de clima espacial de vários provedores e os padroniza
num único conjunto de dados indexado no tempo. Com esse conjunto, prevê tempestades
geomagnéticas usando aprendizado de máquina. Dados, modelos, previsões e a verificação das
previsões são publicados abertamente, sem custo de infraestrutura.

> Versão em inglês: [README.md](README.md)

| Recurso | Endereço |
|---|---|
| Site (PT, EN, ES, FR) | https://future-space-six.vercel.app |
| Documentação da API (OpenAPI) | https://future-space-six.vercel.app/docs |
| Dados e modelos | https://huggingface.co/datasets/Strutf/future-space |
| Histórico dos modelos | https://future-space-six.vercel.app/v1/models |

As previsões são experimentais e não devem ser usadas em decisões operacionais. As previsões
oficiais são emitidas pela [NOAA SWPC](https://www.swpc.noaa.gov/).

## Visão geral

Uma tempestade solar se desenvolve em etapas, e cada etapa é observada por instrumentos
diferentes:

```
   SOL                 ESPAÇO (L1)               ÓRBITA DA TERRA            SOLO
 flares, regiões  ──▶  vento solar e campo  ──▶  raios-X, prótons     ──▶  índices Kp e Dst,
 ativas, ejeções       magnético (~1 h antes     energéticos              magnetômetros,
 de massa (CMEs)       de chegar à Terra)                                  ionosfera/GPS
```

O projeto tem três componentes:

1. **Coleta de dados.** Quinze datasets são coletados a cada hora, padronizados (coluna `time`
   em UTC, unidades e valores ausentes consistentes) e arquivados. Várias fontes guardam
   apenas 1 a 7 dias de dados de tempo real, e é a coleta contínua que torna o arquivo
   possível.
2. **Previsão.** Modelos de árvores com gradient boosting preveem o índice Kp (3 a 24 h à
   frente) e o índice Dst (1 a 24 h à frente), cada um com valor, probabilidade de tempestade
   e alerta. As previsões são emitidas a cada hora e verificadas contra o observado
   diariamente.
3. **Distribuição.** Uma API pública, arquivos Parquet/JSON e um site.

Uma previsão experimental de visibilidade da aurora roda no backend (ver "Visibilidade da
aurora"). Ela ainda não aparece no site.

O índice Dst complementa o Kp porque representa melhor as latitudes baixas.

### Arquitetura

Nenhum servidor ou banco de dados roda continuamente. Todos os resultados são pré-calculados
e guardados em arquivos.

```
 GitHub Actions (de hora em hora, com disparo externo; ver "Agendamento")
   ├─ futurespace update     conectores baixam e padronizam cada fonte;
   │                         o writer mescla, remove duplicatas e grava Parquet + JSON
   ├─ futurespace predict    aplica os modelos publicados aos dados mais recentes
   ├─ futurespace aurora     previsão de visibilidade da aurora por cidade
   └─ futurespace verify     (diário) compara previsões passadas com o observado
                    │
                    ▼
 Dataset no Hugging Face (dados e modelos, versionados por commit)
                    │
                    ▼
 FastAPI na Vercel (só lê JSON, com cache)  ──▶  clientes da API / site

 GitHub Actions (mensal): futurespace train --publish-if-better
```

- **Escrita idempotente:** repetir uma coleta não cria duplicatas. Valores revisados pela
  fonte (ex.: Kp preliminar → definitivo) substituem os anteriores.
- **Isolamento de falhas:** uma fonte indisponível não interrompe as outras; a falha é
  registrada.
- **Particionamento:** por dia (dados de 1 a 5 min), mês (OMNI, flares, regiões, previsões) ou
  ano (Kp, Dst, CMEs). Cada dataset tem um `manifest.json` com suas partições e cobertura.

## Fontes de dados

Todas as fontes são públicas e gratuitas, e nenhuma exige cadastro.

| Dataset | Etapa | Provedor | Cadência | Cobertura | Uso |
|---|---|---|---|---|---|
| `omni-hourly` | todas | NASA GSFC OMNI2 (CDAWeb HAPI) | 1 h | 1963– | treino (vento solar e índices) |
| `kp-gfz` | solo | GFZ Potsdam | 3 h | 1932– | rótulos de treino (Kp definitivo) |
| `cme-cactus` | Sol | SIDC CACTus (SOHO/LASCO) | evento | 1997– | variáveis de CME |
| `kp-forecast-3day` | solo | NOAA SWPC 3-Day Forecast (arquivo NCEI) | 12 h | 2022– | referência de comparação |
| `solar-wind-plasma` | L1 | NOAA SWPC RTSW (SOLAR-1, IMAP, ACE) | 1 min | desde a coleta | entradas ao vivo |
| `solar-wind-mag` | L1 | NOAA SWPC RTSW | 1 min | desde a coleta | entradas ao vivo |
| `kp` | solo | NOAA SWPC | 3 h | desde a coleta | entradas ao vivo, verificação |
| `kp-forecast` | solo | NOAA SWPC | 3 h | desde a coleta | referência ao vivo |
| `dst` | solo | WDC Kyoto (via NOAA SWPC) | 1 h | desde a coleta | entradas ao vivo, verificação |
| `goes-xray` | órbita | NOAA SWPC / GOES | 1 min | desde a coleta | arquivado |
| `goes-protons` | órbita | NOAA SWPC / GOES | 5 min | desde a coleta | arquivado |
| `goes-flares` | Sol | NOAA SWPC / GOES | evento | desde a coleta | arquivado |
| `solar-regions` | Sol | NOAA SWPC | 1 dia | desde a coleta | arquivado |
| `aurora-ovation` | todas | NOAA SWPC OVATION Prime | 1 h | desde a coleta | mapa de probabilidade de aurora (só Parquet) |
| `aurora-weather` | solo | Open-Meteo | 1 h | desde a coleta | nuvens nas cidades da previsão de aurora |
| `predictions` | — | este projeto | 1 h | desde a publicação dos modelos | previsões ao vivo |
| `verification` | — | este projeto | 1 h | desde a publicação dos modelos | previsto × observado |
| `aurora-visibility` | — | este projeto | 1 h | desde a coleta | previsão de visibilidade da aurora (só Parquet) |

<details>
<summary>Detalhes das fontes (endpoints, convenções, ressalvas)</summary>

- **NASA OMNI2** (`OMNI2_H0_MRG1HR`, https://cdaweb.gsfc.nasa.gov/hapi): vento solar de várias
  sondas desde 1963 (IMP, ISEE, Wind, ACE, DSCOVR), intercalibrado e deslocado no tempo até a
  frente de choque, com Kp, Dst, AE e F10.7 no mesmo eixo de tempo. Publicado com algumas
  semanas de atraso. Convenções aplicadas: `time` é o início da hora (o HAPI carimba HH:30),
  o Kp é dividido por 10 e os valores de preenchimento (`9999.9` etc.) viram ausentes.
- **Kp do GFZ** (`https://kp.gfz.de/app/json/`): Kp definitivo desde 1932, com o campo
  `status` (`def`, `pre`, `now`). CC BY 4.0.
- **CACTus:** detecção automática de CMEs nas imagens LASCO C2/C3. O conector combina três
  produtos do mesmo algoritmo:
  - o catálogo científico (1997–2015);
  - o catálogo quicklook mensal (2015–);
  - o feed de tempo real (última semana).

  As detecções de tempo real são descartadas quando o quicklook passa a cobrir o período,
  porque o mesmo evento aparece nos dois com valores um pouco diferentes. A NASA DONKI estava
  indisponível durante o desenvolvimento e só começa em 2010; ela está prevista como fonte
  complementar.
- **NOAA RTSW** (`/json/rtsw/rtsw_{wind,mag}_1m.json`): várias sondas são reportadas ao mesmo
  tempo. `source` identifica a sonda e `active` marca a operacional da NOAA, que é a usada nas
  previsões. A fonte guarda apenas cerca de 24 h de dados.
- **Previsão de Kp da NOAA** (`noaa-planetary-k-index-forecast.json`): a fonte mostra apenas a
  previsão atual. Cada versão emitida é arquivada (`time` = hora da coleta,
  `target_time` = bloco previsto), incluindo os blocos futuros `estimated`.
- **Arquivo do 3-Day Forecast da NOAA** (NCEI, `…/swpc-products/daily_reports/3day_forecast/`):
  todas as edições (00:30 e 12:30 UTC) desde março de 2022. Só são mantidos os blocos que
  ainda não tinham começado no momento da emissão. Setembro e outubro de 2022 não existem no
  arquivo.
- **Dst de Kyoto** (`kyoto-dst.json`): valores provisórios de tempo real.
- **OVATION Prime** (`ovation_aurora_latest.json`): probabilidade de aurora visível numa grade
  global de 1°, calculada pela NOAA a partir do vento solar no L1 e válida para cerca de 30 a
  90 minutos à frente. Só as células diferentes de zero são guardadas (~20 mil por mapa). A
  NOAA não mantém arquivo público deste produto, então o histórico começa com esta coleta. As
  longitudes seguem a convenção da fonte (0 a 359).
- **Open-Meteo** (`api.open-meteo.com/v1/forecast`): cobertura de nuvens total, baixa, média e
  alta e visibilidade horizontal, hora a hora, para as próximas 25 horas em cada cidade da
  previsão de aurora, numa única requisição. Cada hora guarda a previsão mais recente; `issued`
  registra quando ela foi obtida. CC BY 4.0, gratuito para uso não comercial.
- **GOES e regiões solares:**
  - fluxo de raios-X em duas bandas;
  - fluxo integral de prótons (≥1 a ≥500 MeV);
  - a lista de flares;
  - regiões ativas com as classes McIntosh e Mount Wilson e as probabilidades de flare C/M/X
    da NOAA.

</details>

Os produtos de tempo real da NOAA são de domínio público. O campo `license` de cada dataset na
API indica como creditar a fonte original.

## Modelos de previsão

### Alvos

| Índice | Horizontes | Definição de tempestade |
|---|---|---|
| Kp | 3, 6, 12, 24 h | Kp ≥ 5− (4,67), G1 ou mais na escala da NOAA |
| Dst | 1, 3, 6, 12, 24 h | Dst ≤ −50 nT |

Cada índice e horizonte tem dois modelos:

- um **regressor**, que prevê o valor em *t + h*;
- um **classificador**, que dá a probabilidade de tempestade em *t + h*. Um alerta é emitido
  quando a probabilidade passa de um limiar escolhido no conjunto de validação.

Os alvos de Kp se referem ao bloco de 3 horas que contém *t + h*.

### Variáveis

Só são usadas variáveis que também existem em tempo real. Isso permite que um modelo treinado
no OMNI rode com os dados ao vivo. A versão 3 das variáveis tem 82 variáveis em cinco grupos:

| Grupo | Variáveis | Justificativa |
|---|---|---|
| Vento solar no L1 (55) | Bz, By, \|B\|, velocidade, densidade, temperatura e pressão dinâmica; acoplamento de Newell `v^(4/3) B_T^(2/3) sin^(8/3)(θ/2)`; VBs = v·max(−Bz, 0); ângulo do relógio (sen, cos); para Bz, v, n, Newell e VBs, defasagens de 1 a 6 h e médias de 3 a 24 h; Bz mínimo em 3 a 12 h | energia que entra na magnetosfera e sua tendência |
| Estado geomagnético atual (11) | Dst, suas defasagens, diferenças de 1 h e 3 h e mínimo em 24 h; último bloco de Kp encerrado, o bloco anterior e o máximo em 24 h | tempestade em andamento |
| Recorrência de 27 dias (4) | velocidade do vento solar uma rotação solar antes; velocidade, Kp e Dst nas 24 h seguintes correspondentes, uma rotação antes | feixes rápidos recorrentes |
| CMEs (8) | CME larga (largura ≥ 90°) mais rápida nas últimas 24/48/72/96 h; número de CMEs largas em 72 h; estimativa balística de chegada (tempo até a chegada, velocidade e largura da CME mais próxima) | antecedência de horas a dias |
| Calendário (4) | dia do ano e hora UT (sen, cos) | efeitos sazonal (Russell–McPherron) e diurno |

Controles contra vazamento de informação do futuro, garantidos por testes automáticos:

- a linha da hora *t* usa apenas dados até o fim da hora *t*;
- um valor de Kp só é usado depois que seu bloco de 3 horas termina;
- uma CME fica disponível para o modelo 6 h depois do seu início, o atraso típico do catálogo
  de tempo real;
- antes do início do catálogo de CMEs, as variáveis de CME ficam ausentes, e não zeradas.

### Modelo e hiperparâmetros

Os modelos são `HistGradientBoostingRegressor` e `HistGradientBoostingClassifier` do
scikit-learn (versão fixada em 1.9.1). O algoritmo foi escolhido porque:

- tem bom desempenho em dados tabulares;
- treina em minutos numa CPU;
- trata valores ausentes nativamente;
- pode ser inspecionado.

| Hiperparâmetro | Valor |
|---|---|
| taxa de aprendizado | 0,05 |
| máximo de folhas por árvore | 31 |
| mínimo de amostras por folha | 40 |
| regularização L2 | 1,0 |
| parada antecipada | desativada |
| semente aleatória | 0 |
| iterações do regressor | {200, 500, 1000}, escolhidas pelo RMSE na validação |
| iterações do classificador | 400 |
| limiar de alerta | 0,02 a 0,90 (passo 0,01), escolhido pelo HSS na validação |

A versão publicada, `v3-20261106`, contém 9 pares de regressor e classificador, com 8.400
árvores e 512.400 nós no total.

### Treino e avaliação

1. **Divisão temporal:**
   - treino: 01/01/1995 a 31/12/2014;
   - validação: 01/01/2015 a 31/12/2018;
   - teste: de 01/01/2019 em diante.

   A divisão nunca é aleatória.
2. **Iterações:** um regressor é ajustado no treino para cada número de iterações, e fica o
   de menor RMSE na validação.
3. **Limiar de alerta:** o classificador é ajustado no treino, e o limiar que maximiza o HSS
   na validação é escolhido. Maximizar o TSS levava a 60–95% de alarmes falsos, porque o TSS
   quase não penaliza alarmes falsos em eventos raros.
4. **Avaliação:** os modelos são reajustados em treino + validação e avaliados uma única vez
   no teste. Essas são as métricas apresentadas abaixo.
5. **Modelo publicado:** antes da publicação, o modelo é reajustado com todos os dados,
   inclusive o período de teste. As métricas apresentadas continuam sendo as do passo 4.

**Referências de comparação:**

- *persistência:* o Dst atual, ou o último bloco de Kp encerrado;
- *climatologia:* a frequência de tempestades no treino, usada como referência do BSS.

**Métricas**, com acertos *a*, alarmes falsos *b*, perdas *c* e negativos corretos *d*:

| Métrica | Definição |
|---|---|
| RMSE, MAE | √média(ŷ − y)², média\|ŷ − y\| |
| POD | a / (a + c) |
| FAR | b / (a + b) |
| TSS | POD − b / (b + d) |
| HSS | 2(ad − bc) / [(a + c)(c + d) + (a + b)(b + d)] |
| Brier | média(p − o)² |
| BSS | 1 − Brier / Brier da climatologia |

**Re-treino:** o modelo é re-treinado mensalmente no GitHub Actions. Uma nova versão só é
publicada se, na mesma divisão de teste:

- o RMSE médio for no máximo 1,02 vez o da versão atual; e
- o BSS médio não cair mais que 0,02.

Os modelos não são atualizados automaticamente a partir dos próprios erros.

### Resultados

Os números abaixo são da versão `v3-20261106` no período de teste de 01/01/2019 a 30/10/2026:
68.600 horas, das quais 2.174 foram horas de tempestade pelo Dst e 1.584 pelo Kp.

| Previsão | RMSE | RMSE persistência | Redução | POD | FAR | TSS | HSS | HSS persistência | BSS | Limiar |
|---|---|---|---|---|---|---|---|---|---|---|
| Dst +1 h | 3,82 | 4,40 | 13% | 0,93 | 0,12 | 0,92 | 0,90 | 0,87 | 0,85 | 0,42 |
| Dst +3 h | 7,48 | 8,88 | 16% | 0,83 | 0,29 | 0,82 | 0,76 | 0,73 | 0,67 | 0,28 |
| Dst +6 h | 10,53 | 12,58 | 16% | 0,72 | 0,37 | 0,70 | 0,66 | 0,61 | 0,53 | 0,27 |
| Dst +12 h | 14,08 | 16,79 | 16% | 0,50 | 0,50 | 0,48 | 0,48 | 0,45 | 0,33 | 0,24 |
| Dst +24 h | 16,76 | 20,59 | 19% | 0,41 | 0,73 | 0,37 | 0,30 | 0,28 | 0,16 | 0,12 |
| Kp +3 h | 0,76 | 1,01 | 25% | 0,60 | 0,54 | 0,59 | 0,51 | 0,40 | 0,36 | 0,21 |
| Kp +6 h | 0,91 | 1,17 | 23% | 0,58 | 0,72 | 0,54 | 0,36 | 0,29 | 0,22 | 0,12 |
| Kp +12 h | 1,04 | 1,34 | 22% | 0,41 | 0,83 | 0,36 | 0,21 | 0,19 | 0,08 | 0,09 |
| Kp +24 h | 1,15 | 1,50 | 23% | 0,31 | 0,90 | 0,24 | 0,12 | 0,10 | 0,04 | 0,06 |

O RMSE está em nT para o Dst e em unidades de Kp para o Kp. POD, FAR, TSS e HSS se referem ao
alerta.

- **Valor:** o valor previsto supera a persistência em todos os horizontes.
- **Alertas:** o alerta tem habilidade até cerca de 6 h. Em 12 a 24 h, a maioria dos alertas
  de Kp é falsa.
- **Variáveis de CME:** o conjunto anterior, sem variáveis de CME (v2, 70 variáveis), tem BSS
  a no máximo ±0,02 do atual em todos os horizontes. O catálogo CACTus não indica se a CME
  está direcionada à Terra.

**Importância por grupo de variáveis.** As variáveis de cada grupo são embaralhadas em
conjunto no período de teste (20.000 horas amostradas, 3 repetições), e mede-se o aumento do
erro: RMSE para o regressor e Brier para o classificador. Cada valor é a fatia do grupo no
aumento total. Os valores foram medidos com o código da v3, com dados até 30/10/2026.

| Modelo | Vento solar | Estado geomagnético | Recorrência | CMEs | Calendário |
|---|---|---|---|---|---|
| Kp +3 h, valor | 78% | 17% | 0% | 0% | 5% |
| Kp +3 h, probabilidade | 77% | 16% | 0% | 1% | 6% |
| Kp +24 h, valor | 54% | 12% | 19% | 8% | 8% |
| Kp +24 h, probabilidade | 40% | 10% | 18% | 15% | 17% |
| Dst +6 h, valor | 23% | 75% | 0% | 0% | 1% |
| Dst +6 h, probabilidade | 22% | 76% | 0% | 0% | 2% |

O Kp de curto prazo depende principalmente do vento solar que está chegando: embaralhar esse
grupo eleva o RMSE do Kp +3 h de 0,76 para 1,34. O Dst de curto prazo depende principalmente
do valor atual e da sua tendência. Em 24 h, recorrência e CMEs respondem por 27% do modelo de
valor do Kp. A partir do próximo treino, essa decomposição fica registrada para os nove
modelos (ver "Histórico dos modelos").

**Comparação com a NOAA (2022–2026).** A comparação usa o arquivo NCEI do 3-Day Forecast da
NOAA. As duas previsões são comparadas com o Kp observado (GFZ), nos mesmos blocos e com a
mesma antecedência. Para cada edição da NOAA, o modelo:

- usa apenas dados disponíveis até a hora cheia anterior;
- é a versão treinada até 2018, que nunca viu esse período.

| Antecedência | Blocos | MAE modelo | MAE NOAA | HSS do alerta, modelo | HSS do alerta, NOAA |
|---|---|---|---|---|---|
| ~6 h | 2.534 | 0,72 | 1,16 | 0,40 | 0,24 |
| ~12 h | 2.534 | 0,80 | 1,18 | 0,21 | 0,16 |
| ~24 h | 2.534 | 0,90 | 1,16 | 0,08 | 0,16 |

O modelo tem erro menor nas três antecedências e alertas melhores em 6 h e 12 h. A NOAA tem
alertas melhores em 24 h. Cerca de 82% dos alertas de G1 da NOAA nesse período não se
confirmaram.

Esta simulação usa o OMNI e o Kp definitivo, que são mais limpos que os dados de tempo real.
A verificação ao vivo descrita abaixo testa se o resultado se mantém na operação.

**Verificação ao vivo.** O `futurespace verify` roda diariamente. Ele associa cada previsão
emitida ao valor observado depois e registra o erro, o erro da persistência e, para o Kp, o
erro da previsão da NOAA emitida até o mesmo instante. Os resumos de 7 e 30 dias estão em
`GET /v1/verification/summary`.

### Limitações

- **Início das tempestades.** O modelo vê principalmente o vento solar medido no L1, cerca de
  1 h antes de ele chegar à Terra. As variáveis de CME não incluem a direção.
- **Eventos extremos.** Conjuntos de árvores não extrapolam além da faixa de treino, e eventos
  como o de maio de 2024 (Dst −406 nT) são subestimados.
- **Treino × operação.**
  - O OMNI é deslocado no tempo até a frente de choque, enquanto os dados ao vivo são medidas
    brutas no L1, feitas cerca de 45 minutos antes.
  - O Dst provisório e as trocas de sonda no L1 podem introduzir descontinuidades que não
    existem no OMNI.
- **Interpretação da importância.** A importância por permutação descreve o que o modelo usa,
  não causalidade física. Grupos correlacionados dividem o crédito.

## Visibilidade da aurora (experimental)

Este componente roda no backend para acumular dados e previsões. Ele ainda não aparece no
site.

A cada hora, o `futurespace aurora` calcula, para 37 locais e para cada uma das próximas 24
horas, uma pontuação de 0 a 1 das condições para ver a aurora (mais alto = condições melhores). Os locais estão em `src/futurespace/aurora/sites.py` e incluem:

- cidades nas zonas de aurora e perto delas;
- cidades de latitude média;
- referências do Hemisfério Sul e do Brasil;
- locais de câmeras all-sky da rede THEMIS da NASA.

A pontuação não é uma probabilidade: os pesos ainda não foram comparados com auroras
observadas, então ela serve para ordenar locais e horários, mas não diz com que frequência a
aurora vai aparecer. Uma probabilidade calibrada entrará numa coluna própria quando houver
observações.

A pontuação é calculada assim:

```
visibility_score = aurora_score × (1 − cloud_cover/100) × darkness × moon_factor
```

| Componente | Método (v0) |
|---|---|
| Kp esperado | previsão de Kp do projeto para o bloco de 3 horas, interpolada entre os blocos de +3/+6/+12/+24 h; antes do primeiro bloco previsto, o último Kp medido |
| `aurora_score` | função logística de (Kp esperado − Kp necessário), com escala de 0,5 Kp; o Kp necessário vem da linha de visibilidade usada no mapa da aurora do site, 66° − 1,5° × Kp de latitude magnética do dipolo |
| `darkness` | 0 com o Sol acima de −6°, 1 abaixo de −15°, linear entre os dois |
| `moon_factor` | 1 − 0,3 × fração iluminada quando a Lua está acima do horizonte |
| nuvens | previsão do Open-Meteo |

**Referência do OVATION.** Para a próxima hora, a maior probabilidade do OVATION num raio de
500 km é guardada separadamente, junto com a visibilidade correspondente
(`visibility_score_ovation`). A cerca de 110 km de altura, a aurora aparece no horizonte a
essa distância.

As duas estimativas ficam separadas porque estão em escalas diferentes: o OVATION raramente
passa de cerca de 50% e depende da hora local magnética, que a regra do Kp ignora. O arquivo
do OVATION coletado aqui serve para calibrar a próxima versão da fórmula.

**Astronomia.** As posições do Sol e da Lua usam as fórmulas de baixa precisão do
Astronomical Almanac, implementadas em `aurora/astro.py`. Comparadas com o astropy em 300
instantes e locais aleatórios entre 1990 e 2035, os erros máximos são:

- 0,008° na altura do Sol;
- 0,3° na altura da Lua;
- 0,002 na fração iluminada da Lua.

**Limitações.** Os pesos da v0 não foram calibrados com observações. A regra do Kp não
depende da hora local magnética e não tem limite polar para o anel da aurora.

**Verificação prevista:** classificar imagens de câmeras all-sky (THEMIS, com dados rotulados
como o OATH) e comparar as duas estimativas com a aurora observada.

## Histórico dos modelos e reprodutibilidade

Cada treino grava um registro (`report.json`) ao lado dos seus modelos. O `models/index.json`
lista todos os treinos, inclusive os recusados pela trava de publicação, com o motivo da
recusa.

| Campo | Conteúdo |
|---|---|
| `reproducibility.code` | commit do GitHub usado no treino; se havia alterações não commitadas |
| `reproducibility.data` | commit do dataset no Hugging Face que foi lido; linhas e cobertura de cada fonte |
| `reproducibility.hyperparameters` | parâmetros das árvores, grades de busca, horizontes, definição de tempestade, configuração da importância |
| `reproducibility.environment` | versões do Python, scikit-learn, numpy e pandas |
| `splits`, `features`, `feature_groups` | divisão temporal, lista ordenada de variáveis, composição dos grupos |
| `models[]` | por modelo: iterações, limiar, número de linhas, métricas de teste, importância por grupo |

```bash
curl https://future-space-six.vercel.app/v1/models                # todos os treinos
curl https://future-space-six.vercel.app/v1/models/v3-20261106    # um treino
```

O treino é determinístico (`random_state=0`, sem parada antecipada). Um modelo pode, portanto,
ser reproduzido:

1. fazendo checkout do commit registrado;
2. instalando as versões fixadas em `requirements-pipeline.txt`;
3. lendo o dataset na revisão registrada (`HuggingFaceStorage(repo, revision=<sha>)`).

A versão `v3-20261106` é anterior a esses registros, por isso os campos de código e de revisão
dos dados estão vazios. Os dados de treino dela terminam em 30/10/2026 às 14:00 UTC, e ela foi
publicada no commit `20c2da27` do dataset.

## Acesso aos dados

### Arquivos

Os arquivos Parquet podem ser lidos diretamente, sem a API:

```python
import pandas as pd  # requer pandas, pyarrow e huggingface_hub

omni = pd.read_parquet("hf://datasets/Strutf/future-space/parquet/omni-hourly/")
kp = pd.read_parquet("hf://datasets/Strutf/future-space/parquet/kp-gfz/")
cmes = pd.read_parquet("hf://datasets/Strutf/future-space/parquet/cme-cactus/")
```

```
parquet/<dataset>/<partição>.parquet          análise e treino
json/<dataset>/<partição>.json                lido pela API
json/<dataset>/manifest.json                  partições e cobertura
json/<dataset>/latest.json                    últimos valores
json/catalog.json                             catálogo
models/current.json                           versão dos modelos em uso
models/index.json                             histórico de todos os treinos
models/<versão>/bundle.joblib                 modelos treinados
models/<versão>/report.json                   registro daquela versão
models/rejected/<versão>-<hora>/report.json   treinos recusados pela trava de publicação
```

| Dataset | Parquet | CSV | JSON |
|---|---|---|---|
| OMNI, 1963 até hoje (560 mil linhas) | 25 MB | 60 MB | 148 MB |
| Kp do GFZ, 1932 até hoje (277 mil linhas) | 2,6 MB | — | 15 MB |
| Campo magnético no L1, 1 dia, 3 sondas | ~50 KB | — | ~340 KB |

### API

URL base: `https://future-space-six.vercel.app`. A documentação interativa fica em `/docs`.

| Endpoint | Retorna |
|---|---|
| `GET /v1/datasets` | datasets e sua cobertura |
| `GET /v1/datasets/{id}` | metadados, campos, unidades, licença e cobertura |
| `GET /v1/datasets/{id}/latest` | valores mais recentes |
| `GET /v1/datasets/{id}/data` | série temporal num intervalo (`start`, `end`, `fields`, `format=json\|csv`) |
| `GET /v1/datasets/predictions/latest` | previsões mais recentes |
| `GET /v1/compare/kp?date=AAAA-MM-DD` | Kp do modelo, da NOAA e observado, por bloco de 3 horas |
| `GET /v1/compare/dst?date=AAAA-MM-DD` | previsões do modelo e Dst observado, por hora |
| `GET /v1/verification/summary` | erros das previsões nos últimos 7 e 30 dias |
| `GET /v1/models` | histórico dos modelos, métricas e importância por grupo |
| `GET /v1/models/{id}` | registro de um treino |
| `GET /v1/datasets/aurora-visibility/latest` | última previsão de visibilidade da aurora (todos os locais, 0 a 24 h) |

```bash
curl "https://future-space-six.vercel.app/v1/datasets/omni-hourly/data?start=2024-05-10&end=2024-05-12&fields=bz_gsm,speed,kp,dst"
```

Datasets grandes (`aurora-ovation`, `aurora-visibility`) ficam disponíveis apenas em
Parquet. Para eles, `/data` responde `404` com o link dos arquivos.

Limites de uso:

- cada dataset tem um intervalo máximo por consulta (`max_query_days`; por exemplo, 3 dias para
  dados de 1 minuto); volumes grandes devem ser lidos dos arquivos Parquet;
- 60 requisições por minuto por IP; acima disso, a API responde `429` com `Retry-After`;
- parâmetros de consulta desconhecidos são recusados com `400`.

Ver [SECURITY.md](SECURITY.md).

## Executando o projeto

Requer Python 3.11 ou mais recente.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,hf]"

futurespace datasets                                 # lista o catálogo
futurespace update                                   # coleta o tempo real em ./data
futurespace update --schedule daily                  # OMNI e Kp do GFZ recentes
futurespace backfill omni-hourly --start 1963-01-01  # histórico completo do OMNI (~4 min)
futurespace backfill kp-gfz --start 1932-01-01       # histórico completo do Kp (~30 s)
futurespace backfill cme-cactus --start 1997-01-01   # CMEs desde 1997 (~40 s)
futurespace train                                    # treina e avalia (~10 min) em ./models
futurespace train --no-cme --out models-no-cme       # mesmo treino sem variáveis de CME
futurespace predict                                  # previsão com os dados atuais
futurespace aurora                                   # visibilidade da aurora por cidade
futurespace serve --reload                           # API em http://localhost:8000/docs
pytest -q                                            # testes
```

| Variável | Finalidade | Padrão |
|---|---|---|
| `FS_STORAGE` | armazenamento do pipeline: `local`, `hf` ou `s3` | `local` |
| `FS_DATA_DIR` | pasta para `local` | `data` |
| `FS_HF_REPO` | dataset no Hugging Face, ex.: `Strutf/future-space` | — |
| `HF_TOKEN` | token de escrita do Hugging Face (só no pipeline) | — |
| `FS_DATA_URL` | URL base alternativa de onde a API lê os JSON | — |
| `FS_RATE_LIMIT_PER_MINUTE` | requisições por minuto por IP na API (`0` desativa o limite) | `60` |
| `FS_S3_BUCKET`, `FS_S3_ENDPOINT_URL` | configuração para `s3` (Cloudflare R2, Backblaze B2 etc.) | — |

A API lê de `FS_DATA_URL` se estiver definida; senão, de `FS_HF_REPO`; senão, da pasta local.

### Implantação

O projeto roda inteiramente em planos gratuitos:

1. **Hugging Face:** um repositório de dataset público, gravado com um token *fine-grained*
   restrito a esse repositório.
2. **GitHub:** um repositório público, onde os minutos do Actions são ilimitados.
   Configuração:
   - secret `HF_TOKEN`, o token do passo 1;
   - variável `FS_HF_REPO`.

   O token existe apenas nos secrets do GitHub.
3. **Carga inicial:** rodar os backfills com `FS_STORAGE=hf`, ou enviar uma pasta `data/` local
   com `hf upload <repo> data . --repo-type dataset`.
4. **Modelos:** rodar o workflow `train` pela aba Actions.
5. **Vercel:** importar o repositório e definir `FS_HF_REPO`. A API só lê dados públicos e não
   precisa de token.

As recomendações de segurança estão em [SECURITY.md](SECURITY.md).

### Agendamento

O disparo nativo `on.schedule` do GitHub nunca funcionou neste repositório, nem com intervalo
de 5 minutos, embora as execuções manuais funcionem. Por isso, os workflows são disparados pelo
[cron-job.org](https://cron-job.org) por meio da API `workflow_dispatch` do GitHub:

| Job | Horário (UTC−3) | Workflow | Corpo da requisição |
|---|---|---|---|
| coleta, tempo real | `23 * * * *` | `ingest.yml` | `{"ref":"main","inputs":{"modo":"realtime"}}` |
| coleta, diária | `47 0 * * *` | `ingest.yml` | `{"ref":"main","inputs":{"modo":"daily"}}` |
| treino | `41 1 1 * *` | `train.yml` | `{"ref":"main"}` |

Configuração de cada job:

- **método:** `POST`;
- **cabeçalhos:** `Authorization: Bearer <token>`, `Accept: application/vnd.github+json` e
  `Content-Type: application/json`;
- **token:** *fine-grained*, apenas com *Actions: read and write* neste repositório;
- **resposta esperada:** `204`.

O agendamento nativo continua nos workflows como reserva. Todos os jobs que gravam dados
compartilham um grupo de concorrência e nunca gravam ao mesmo tempo.

Cargas históricas em produção: rodar o workflow `ingest` manualmente com `backfill` (id do
dataset) e `desde` (data inicial).

Restrições dos planos gratuitos:

- O GitHub desativa workflows agendados após 60 dias sem atividade no repositório.
- O vento solar do L1 fica na fonte por apenas cerca de 24 h, então mais de um dia sem coleta
  perde dados.
- O plano Hobby da Vercel limita as respostas a cerca de 4,5 MB, por isso a API limita o
  intervalo por consulta.

## Site

https://future-space-six.vercel.app é gerado pela própria API, sem etapa de build, em
português, inglês, espanhol e francês.

| Página | Caminho (PT) | Conteúdo |
|---|---|---|
| Início | `/` | resumo do projeto e previsão atual |
| Previsões | `/dashboard` | três abas para ontem, hoje e amanhã: Kp (modelo, NOAA, observado; padrão), probabilidade de tempestade do modelo, Dst por hora |
| Sobre | `/sobre` | física das tempestades, Kp, Dst, CMEs e as fórmulas usadas |
| Interativo | `/interativo` | simulador de CME, reprodução de tempestades históricas, mapa da aurora |

As páginas interativas rodam no navegador a partir de arquivos estáticos gerados por
`futurespace export-web --bundle <modelos treinados até 2018>`:

- `sim-model.json`: um modelo reduzido com restrições monotônicas (médias de 6 horas de
  velocidade, densidade e VBs). Ele é exportado do scikit-learn para JSON e avaliado em
  JavaScript; um teste verifica que os dois dão resultados idênticos.
- `replays.json`: dados horários das tempestades de 1989, 2003, 2015 e 2024, com as previsões
  emitidas 3 h antes.
- `world-land.json`: contornos dos continentes (Natural Earth, domínio público).

O mapa da aurora usa uma aproximação de dipolo para a latitude magnética e uma linha de
visibilidade calibrada nos mapas de aurora da NOAA: cerca de 66° − 1,5° × Kp de latitude
magnética.

Os moldes das páginas ficam em `src/futurespace/api/web/pages`, e os textos em
`web/locales/{pt,en,es,fr}.json`. Testes garantem que todos os arquivos de idioma têm as mesmas
chaves e variáveis.

## Estrutura do repositório

```
src/futurespace/
├── catalog.py          metadados dos datasets (campos, unidades, licenças, agendamento)
├── partitions.py       layout dos arquivos e utilidades de tempo
├── storage.py          armazenamento: local, Hugging Face, S3, HTTP
├── model_index.py      histórico dos modelos (models/index.json), legível sem bibliotecas de ML
├── cli.py              comando `futurespace`
├── aurora/
│   ├── sites.py        locais acompanhados na previsão da aurora
│   ├── astro.py        altura do Sol e da Lua, fração iluminada da Lua
│   └── visibility.py   previsão de visibilidade da aurora (v0)
├── pipeline/           cliente HTTP, writer (mescla e deduplicação), um conector por fonte
├── ml/
│   ├── features.py     tabela horária, variáveis, alvos e grupos de variáveis
│   ├── train.py        treino, validação temporal, avaliação, importância, registros
│   ├── metrics.py      RMSE, MAE, POD, FAR, TSS, HSS, Brier
│   ├── serve.py        previsões ao vivo e publicação dos modelos
│   ├── verify.py       verificação das previsões
│   └── export_web.py   modelos para o navegador e reprodução de tempestades
└── api/
    ├── app.py          aplicação FastAPI (sem pandas, para uso em serverless)
    ├── compare.py      comparação entre modelo e NOAA usada pelo painel
    └── web/            moldes das páginas, arquivos estáticos, traduções
api/index.py, vercel.json    entrada da Vercel
.github/workflows/           coleta, treino, testes
requirements-pipeline.txt    versões fixadas nos workflows que têm o token
```

Para adicionar uma fonte:

1. descrever o dataset em `catalog.py`;
2. implementar `(client, start, end) -> DataFrame` em `pipeline/connectors/`;
3. registrá-la em `connectors/__init__.py`.

O writer e a API passam a atendê-la sem outras mudanças.

## Roadmap

- [x] Coleta e arquivamento de 15 datasets; API pública; dataset em Parquet/JSON
- [x] Previsão de Kp e Dst de 1 a 24 h (valor, probabilidade, alerta)
- [x] Variáveis de CME do CACTus
- [x] Previsões ao vivo de hora em hora, verificação diária, comparação com a NOAA
- [x] Re-treino mensal com trava de publicação
- [x] Registros reprodutíveis de treino e importância por grupo de variáveis
- [x] Site em quatro idiomas com páginas interativas
- [x] Previsão de visibilidade da aurora por cidade no backend (v0), com o arquivo do OVATION
- [ ] Aurora: calibração com o OVATION e câmeras all-sky; publicação no site
- [ ] Direção e tempo de chegada das CMEs (NASA DONKI, WSA–Enlil)
- [ ] Novas fontes: magnetômetros INTERMAGNET (Vassouras e Tatuoca, no Brasil), INPE/EMBRACE,
      TEC ionosférico
- [ ] Previsão de flares (parâmetros SHARP do SDO/HMI); ligação de eventos entre fontes;
      detecção de anomalias
- [ ] Análise de calibração das probabilidades e métricas específicas para o início das
      tempestades

## Citação e créditos dos dados

Os metadados de citação deste repositório estão em [`CITATION.cff`](CITATION.cff). Para citar
um modelo específico, inclua a versão e o endereço do registro, por exemplo
`https://future-space-six.vercel.app/v1/models/v3-20261106`.

Fontes originais dos dados:

- **OMNI:** King, J. H., & Papitashvili, N. E. (2005). *JGR*, 110, A02104.
  doi:10.1029/2004JA010649. NASA GSFC SPDF / OMNIWeb.
- **Kp:** Matzka, J., et al. (2021). *Space Weather*, 19, e2020SW002641.
  doi:10.1029/2020SW002641. Dados: doi:10.5880/Kp.0001 (CC BY 4.0).
- **Dst:** World Data Center for Geomagnetism, Kyoto, et al. (2015). doi:10.17593/14515-74000.
- **CACTus:** Robbrecht, E., & Berghmans, D. (2004). *A&A*, 425, 1097–1106.
  doi:10.1051/0004-6361:20041302. SIDC, Observatório Real da Bélgica.
- **NOAA SWPC**, produtos de tempo real e arquivo de previsões da NCEI: domínio público.
- **OVATION Prime:** Newell, P. T., Sotirelis, T., & Wing, S. (2009). *JGR*, 114, A09207.
  doi:10.1029/2009JA014326.
- **Open-Meteo:** dados meteorológicos de Open-Meteo.com (CC BY 4.0).

Métodos citados:

- Newell, P. T., et al. (2007). *JGR*, 112, A01206. doi:10.1029/2006JA012015.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

## Licença

O código é distribuído sob a licença MIT. Cada dataset mantém a licença da fonte original,
indicada no catálogo (`/v1/datasets/{id}`).
