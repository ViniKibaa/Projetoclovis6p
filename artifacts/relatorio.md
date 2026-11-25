# Relatório de treino — Triagem de chamados

## Dados
- Exemplos: **480** (treino 342 · validação 68 · teste 70)
- Treino em **124** épocas (melhor época: 99), 7.25 s

## Configuração
`hidden=[64]`, `lr=0.002`, `batch=32`, `l2=0.001`, `dropout=0.5`, `seed=42`

## Resultados no conjunto de teste
| Tarefa | Acurácia | F1 macro | Baseline (classe majoritária) |
|---|---|---|---|
| Categoria | 0.871 | 0.869 | 0.229 |
| Urgência | 0.886 | 0.886 | 0.329 |

Acerto conjunto (categoria **e** urgência): **0.757**

### Categoria
| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| Hardware | 1.000 | 0.737 | 0.849 | 19 |
| Redes | 0.824 | 0.875 | 0.849 | 16 |
| Software | 0.826 | 1.000 | 0.905 | 19 |
| Acessos/Senhas | 0.875 | 0.875 | 0.875 | 16 |

| real \ previsto | Hardware | Redes | Software | Acessos/Senhas |
|---|---|---|---|---|
| **Hardware** | 14 | 1 | 3 | 1 |
| **Redes** | 0 | 14 | 1 | 1 |
| **Software** | 0 | 0 | 19 | 0 |
| **Acessos/Senhas** | 0 | 2 | 0 | 14 |

### Urgência
| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| Baixo | 0.815 | 1.000 | 0.898 | 22 |
| Médio | 0.905 | 0.760 | 0.826 | 25 |
| Alto | 0.955 | 0.913 | 0.933 | 23 |

| real \ previsto | Baixo | Médio | Alto |
|---|---|---|---|
| **Baixo** | 22 | 0 | 0 |
| **Médio** | 5 | 19 | 1 |
| **Alto** | 0 | 2 | 21 |
