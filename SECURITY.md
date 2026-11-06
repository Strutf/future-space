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
