# Triagem de chamados de Helpdesk com Rede Neural

Sistema que recebe o **texto de um problema relatado pelo usuário** e usa uma
**rede neural** para determinar automaticamente:

- **Categoria do chamado:** Hardware, Redes, Software, Acessos/Senhas
- **Nível de urgência:** Baixo, Médio, Alto

Inclui geração de dados, vetorização TF-IDF, rede neural multitarefa, avaliação,
CLI, API HTTP e interface web — tudo em **Python + NumPy** (sem scikit-learn,
sem frameworks web).

## Requisitos e instalação

- Python **3.10 ou superior** (alvo: 3.15.x). Única dependência: **NumPy**.
- O código usa apenas a biblioteca padrão + NumPy, sem recursos específicos de versão.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python -m helpdesk_nn gerar-dados            # (opcional) cria data/chamados.csv com 480 exemplos
python -m helpdesk_nn treinar                # treina, avalia e salva em artifacts/
python -m helpdesk_nn prever "a VPN caiu e a equipe está parada"
python -m helpdesk_nn servir                 # http://127.0.0.1:8000
python -m unittest discover -s tests -v      # testes
```

`treinar` gera o CSV sozinho se ele não existir. Artefatos em `artifacts/`:
`modelo.json`, `metricas.json`, `historico.json` e **`relatorio.md`** (métricas e
matrizes de confusão).

### API

```bash
curl -X POST localhost:8000/api/classificar \
  -H 'Content-Type: application/json' \
  -d '{"texto": "esqueci minha senha e preciso entregar um relatório hoje"}'
```

Retorna categoria, urgência, probabilidades por classe e `revisao_humana: true`
quando a confiança em alguma das tarefas fica abaixo de 50%.

## Arquitetura

```
CSV (texto, categoria, urgencia)
  └─ divisão treino/val/teste (estratificada, por grupo de frase)
      └─ TF-IDF (unigramas+bigramas, sem acentos, L2) — ajustado só no treino
          └─ MLP:  Dense(64)+ReLU+Dropout ─┬─ cabeça categoria (softmax, 4)
                                           └─ cabeça urgência  (softmax, 3)
```

| Arquivo | Papel |
|---|---|
| `helpdesk_nn/data.py` | gerador sintético, leitura de CSV, divisão dos dados |
| `helpdesk_nn/vectorizer.py` | TF-IDF em NumPy |
| `helpdesk_nn/model.py` | rede multitarefa: forward, backprop manual, Adam, early stopping |
| `helpdesk_nn/metrics.py` | acurácia, precisão/recall/F1, matriz de confusão |
| `helpdesk_nn/train.py` | pipeline de treino e relatório |
| `helpdesk_nn/predict.py` | inferência (`Classificador`) |
| `helpdesk_nn/server.py` + `web/index.html` | API e interface |

## Decisões de projeto

- **Multitarefa:** as duas tarefas compartilham a camada oculta; perda =
  CE(categoria) + CE(urgência) + regularização L2.
- **Backpropagation manual**, validado nos testes por **verificação numérica de
  gradiente** (diferenças finitas).
- **Sem vazamento de dados:** o vocabulário/IDF é ajustado só com o treino.
- **Avaliação por grupo:** no teste entram *frases-base que a rede nunca viu*;
  sortear variações da mesma frase entre treino e teste inflaria as métricas.
- **Regularização:** dropout 0.5, L2, early stopping pela perda de validação.
- **Segurança do servidor:** escuta só em `127.0.0.1` por padrão, limite de corpo
  e de texto, cabeçalhos de segurança, e a interface usa `textContent` (sem XSS).
- **Modelo sem `pickle`:** JSON puro (pesos + metadados), sem `pickle` e sem zip.

## Resultados (referência, seed 42, 480 exemplos, teste com frases inéditas)

| Tarefa | Acurácia | F1 macro | Baseline (maioria) |
|---|---|---|---|
| Categoria | 0,871 | 0,869 | 0,229 |
| Urgência | 0,886 | 0,886 | 0,329 |

Acerto conjunto (categoria **e** urgência): **0,757**. Detalhes por classe e
matrizes de confusão em `artifacts/relatorio.md`.

## Limitações (leia antes de apresentar)

- Os dados são **sintéticos**: a urgência depende muito da linguagem de impacto
  ("toda a empresa parada", "sem pressa"), mais clara do que em chamados reais.
  Os números acima **não** estimam desempenho em produção.
- O teste tem ~67 exemplos; diferenças de 1–2 pontos são ruído.
- O vocabulário é lexical: sinônimos nunca vistos no treino não são entendidos.

## Usando um dataset real

Basta um CSV UTF-8 com as colunas `texto,categoria,urgencia` (coluna `grupo`
opcional). Rótulos fora do padrão são aceitos (as classes são lidas dos dados):

```bash
python -m helpdesk_nn treinar --dados meus_chamados.csv
```

## Extensões possíveis

- Trocar o TF-IDF por *embeddings* de frase (`sentence-transformers`): basta
  gerar a matriz `X` e reutilizar `MultiTaskMLP` — a rede aceita qualquer vetor.
- Calibrar probabilidades (temperature scaling) e ajustar o limiar de revisão humana.
