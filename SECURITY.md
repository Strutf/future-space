# Política de segurança

## Como reportar uma vulnerabilidade

**Não abra uma issue pública.** Use o reporte privado do GitHub:
aba **Security → Report a vulnerability** neste repositório.

Inclua o que for possível: descrição, passos para reproduzir e impacto. A resposta
inicial costuma sair em até 7 dias.

## Escopo

- O código deste repositório (pipeline de coleta e API)
- A API publicada e os workflows do GitHub Actions
- O dataset publicado no Hugging Face

Problemas nos dados das fontes originais (NOAA, NASA, GFZ, Kyoto) devem ser reportados
a elas. Se afetarem este projeto, como valores absurdos sendo publicados, avise também.

## Como o projeto protege credenciais

- O único segredo é o `HF_TOKEN` (escrita no dataset do Hugging Face). Ele é um token
  *fine-grained*, limitado a um único repositório, e fica apenas nos secrets do GitHub Actions.
- O token só existe no ambiente do passo que grava os dados. A instalação de dependências
  roda sem ele, e as dependências do pipeline têm versões fixas (`requirements-pipeline.txt`).
- Workflows rodam com `GITHUB_TOKEN` somente leitura e actions fixadas por SHA.
- A API é somente leitura, não usa credenciais e lê apenas dados públicos.
- Nenhum segredo deve ser commitado. `.env` está no `.gitignore`.

## Proteção da API contra abuso e sobrecarga

Na aplicação:
- **Rate limiting por IP:** 60 requisições por minuto (`FS_RATE_LIMIT_PER_MINUTE`, `0`
  desliga). Acima disso, a resposta é `429` com `Retry-After`. Na Vercel o contador é por
  instância da função: barra scripts e abusos simples, não um ataque distribuído.
- **Parâmetros desconhecidos são recusados** (`400`). Isso impede furar o cache da CDN
  variando a URL (`?x=1`, `?x=2`...).
- **Cache na CDN com `stale-while-revalidate` e `stale-if-error`:** repetições não chegam à
  função, e durante picos ou falhas a CDN continua servindo a última resposta boa.
- **Limites por consulta** (intervalo máximo por dataset, datas entre 1900 e 2100) e
  **timeout curto** (10 s) nas leituras do armazenamento.

No painel da Vercel (gratuito):
- A proteção contra DDoS de rede é automática.
- Em **Firewall → Attack Challenge Mode**: em caso de ataque, exige verificação de navegador
  de todos os visitantes, com um clique para ligar e desligar.
- Se o plano permitir, uma **regra de rate limit no WAF** protege todas as instâncias de uma vez.

O plano Hobby não gera cobrança: no pior caso a API fica fora do ar por um tempo. A coleta
(GitHub Actions) e os dados (Hugging Face) são independentes da API e continuam funcionando.

## Modelos de IA

Os modelos são salvos com `joblib`, um formato que **executa código ao ser carregado**.
Por isso:
- `futurespace predict` só carrega modelos do armazenamento do próprio projeto, apontados
  por `models/current.json`. Quem tem permissão de escrita ali já é de confiança.
- Nunca carregue um `bundle.joblib` de origem desconhecida.
