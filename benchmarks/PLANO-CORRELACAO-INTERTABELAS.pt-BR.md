# Plano de implementação: correlação entre tabelas no `IndependentSynthesizer`

> Fork `dmux/SDVReal` · branch `feature/independent-multi-table-synthesizer` · Issue [#1](https://github.com/dmux/SDVReal/issues/1)
> Relacionados: [`BENCHMARK.pt-BR.md`](BENCHMARK.pt-BR.md) · [`PLANO-OTIMIZACAO.pt-BR.md`](PLANO-OTIMIZACAO.pt-BR.md) · [`CONCEITOS-SDV.pt-BR.md`](CONCEITOS-SDV.pt-BR.md)

> **Resultado do experimento:** o plano foi implementado e medido na branch `experiment/inter-table-correlation`. Resultados, desvios, melhorias e metas revistas em [`EXPERIMENTO-CORRELACAO.pt-BR.md`](EXPERIMENTO-CORRELACAO.pt-BR.md).

**Sobre os números.** Custos e ganhos neste documento são **estimativas** até serem medidos com a escada do CNPJ (seção 8). Os valores marcados como "hoje" vêm do benchmark.

---

## 1. Objetivo

Recuperar as principais correlações entre tabelas que o `IndependentSynthesizer` perde hoje, **sem abrir mão** do que justifica a existência dele:

| Requisito | Restrição |
|---|---|
| Modelos | continuar com **1 modelo por tabela** (nunca 1 por linha do pai, como o HMA) |
| Tempo | continuar **linear** no número de linhas; aumento máximo de **+20%** |
| Memória | continuar linear (ou fixa, no modo streaming); aumento máximo de **+15%** |
| Cardinalidade | manter o KS = 0,000 e a cauda exata (7.903 estabelecimentos) |
| Compatibilidade | tudo **desligado por padrão**; o comportamento atual não muda sem opt-in |
| Plano de otimização | compatível com subamostra, streaming e sampling paralelo |

### O que se quer recuperar (CNPJ)

| Correlação | Hoje | Meta | Técnica |
|---|---|---|---|
| Município ↔ UF | ❌ (na variante `full`) | ✅ | T1 |
| Porte da empresa ↔ número de filiais | ❌ | ✅ | T2 |
| Porte/capital ↔ atributos das filiais | ❌ | ✅ (relações monotônicas) | T3 |
| Exatamente 1 matriz por empresa | 85% a 93% | ~100% (regra rígida: 100%) | T4 |
| Filiais da mesma empresa parecidas entre si | ❌ | ✅ parcial | T5 |

---

## 2. Ideia central: inverter a ordem do sampling

```
HOJE
  1. sample de cada tabela, independente  ──►  2. _connect_tables: sorteia as FKs
     (o filho é gerado sem saber quem é o pai)

PROPOSTO (ordem topológica, pais antes dos filhos)
  1. sample do pai (+ coluna "nº de filhos", aprendida junto com as outras)      [T2]
  2. contagens por pai ──► np.repeat(pai, contagem) ──► FK definida ANTES do filho
  3. contexto do pai repetido por filho (+ posição do filho no grupo)            [T3, T4]
  4. sample do filho CONDICIONADO ao contexto, vetorizado no espaço latente      [T3]
     (+ efeito compartilhado por pai)                                             [T5]
```

O que o HMA consegue com um modelo por pai, aqui se obtém com **colunas extras** no modelo de cada tabela e **álgebra linear vetorizada** no sampling.

---

## 3. API proposta

Todos os parâmetros são opcionais e ficam desligados por padrão.

```python
IndependentSynthesizer(
    metadata,
    locales=['en_US'],
    verbose=True,
    # T1: tabelas de consulta tratadas como colunas categóricas do filho
    lookup_tables=None,              # ex.: ['municipios', 'cnaes', 'paises', ...]
    # T2: número de filhos modelado como coluna do pai
    model_cardinality=False,
    # T3: colunas do pai usadas como contexto no filho
    context_columns=None,            # ex.: {'estabelecimentos': {'empresas': ['porte_empresa', 'capital_social']}}
    # T4: posição do filho dentro do pai e regras de grupo
    sibling_order=None,              # ex.: {'estabelecimentos': {'by': 'identificador_matriz_filial', 'ascending': True}}
    group_rules=None,                # ex.: {'estabelecimentos': {'column': 'identificador_matriz_filial', 'first': '1', 'others': '2'}}
    # T5: semelhança entre irmãos (efeito aleatório por pai)
    sibling_correlation=False,
)
```

Utilitário auxiliar: `sdv.multi_table.independent.detect_lookup_tables(data, metadata, max_rows=10_000)` sugere a lista de `lookup_tables`.

---

## 4. Arquitetura e arquivos

| Arquivo | Mudança |
|---|---|
| `sdv/multi_table/independent.py` | novos parâmetros; reescrita do metadata (T1, T3, T4); `_augment_tables` com colunas de cardinalidade (T2); novo `_sample` em ordem topológica quando T2, T3 ou T4 estiverem ligados |
| `sdv/multi_table/_conditional.py` (novo) | sampling condicionado **por linha** e vetorizado da GaussianCopula (T3) e efeito aleatório por pai (T5) |
| `sdv/multi_table/_group_features.py` (novo) | posição no grupo, `group_rules` e estimativa da correlação intraclasse (T4, T5) |
| `tests/unit/multi_table/test_independent*.py` | testes por técnica |
| `tests/utils.py` | `generate_multi_table_schema(..., correlated=True)`: dados com correlações pai → filho conhecidas, para testar a recuperação |
| `tests/integration/multi_table/test_independent_correlation.py` (novo) | recuperação das correlações nos dados sintéticos com correlação conhecida |
| `benchmarks/cnpj/run.py` · `dataset.py` | flag `--correlation` e métricas de correlação entre tabelas (seção 8) |

**Pontos do código reaproveitados:**
- `BaseIndependentSampler._sample_table` já usa `keep_extra_columns=True`, então as colunas extras do pai (T2) saem no sampling.
- O padrão do HMA para colunas estendidas (`self.extended_columns[...] = FloatFormatter(...)`, `hma.py:443-451`) serve para a coluna de cardinalidade.
- `copulas.multivariate.GaussianMultivariate` expõe `correlation`, `univariates`, `columns` e `_transform_to_normal`. O condicionamento analítico existe em `_get_conditional_distribution`, mas aceita **um único vetor de condições por chamada**. Por isso a T3 precisa da versão por linha.

---

## 5. Técnicas

### T1: tabelas de consulta como colunas categóricas

**Problema.** Na variante `full`, `municipio`, `cnae_fiscal_principal` etc. são FKs. A FK é sorteada sem olhar o resto da linha, e aparecem combinações como município do RJ com UF = SP.

**Design.**
1. Em `__init__`, para cada tabela em `lookup_tables`:
   - remover do metadata modificado (`_modified_multi_table_metadata`) as relações em que ela é pai;
   - mudar o sdtype da coluna de FK no filho de `id` para `categorical`.
   Isso acontece **antes** de `_initialize_models`, então os synthesizers de cada tabela já nascem com a coluna categórica.
2. A tabela de consulta não é modelada. No sampling, ela é **copiada** dos dados reais, guardados no fit. São códigos públicos de referência, e a cópia garante que todo código gerado no filho exista no pai.
3. `detect_lookup_tables`: sugere tabelas com até `max_rows` linhas cujas colunas, além da PK, sejam só descritivas (`unknown` ou texto único).

**Por que funciona.** A coluna categórica entra na cópula do filho, que aprende a correlação com `uf` e com as demais colunas. Os valores gerados são sempre categorias vistas no fit, então a integridade referencial é mantida.

**Custo.** Fica até mais barato, porque são menos relações e menos atribuições de FK. Uma coluna categórica a mais por filho no lugar de uma coluna `id`.

**Riscos.** Categorias raras somem com subamostra (mitigado pela cobertura de categorias do O2 do plano de otimização). Lookups grandes (ex.: 5.572 municípios) aumentam o custo do encoder categórico, o que é aceitável.

**Testes.**
- Unit: reescrita do metadata, cópia da tabela de consulta e integridade.
- Integração: no CNPJ, % de pares (UF, município) sintéticos que existem nos dados reais ≥ 99%. Hoje, com FK, é ≈ 1/27.

**Esforço.** 0,5 a 1 dia.

---

### T2: cardinalidade condicionada ao pai

**Problema.** O número de filhos é sorteado sem olhar os atributos do pai. Uma empresa "grande" sintética tem a mesma chance de ter 1 filial que uma "pequena".

**Design.**
1. **Fit:** em `_augment_tables`, para cada relação (exceto 1-para-1 e lookups), adicionar à tabela processada do pai a coluna `__{child}__{fk}__num_children`, com as contagens já calculadas hoje (`value_counts().reindex(...)`). Registrar um `FloatFormatter` em `self.extended_columns`, no padrão do HMA. A cópula do pai aprende a correlação dessa coluna com porte, capital etc.
2. **Sample do pai:** a coluna sai junto (`keep_extra_columns=True`) como um valor contínuo `v` por pai.
3. **Mapeamento por posto,** que preserva a cauda exata:
   ```python
   reais = reamostragem_sem_reposicao(contagens_reais, n_pais)   # algoritmo atual
   counts = np.empty(n_pais, int)
   counts[np.argsort(v)] = np.sort(reais)                        # O(n log n), vetorizado
   ```
   Assim a distribuição das contagens é **exatamente** a real (KS = 0) e a *ordem* vem da cópula, ou seja, da correlação com os atributos do pai.
4. O ajuste de excesso ou déficit para fechar o total continua igual.
5. **Vários pais não-lookup** (ex.: a fixture `data_metadata_multiple_foreign_keys`): a relação **primária** (a primeira declarada, ou o parâmetro `primary_parent`) define contagens e contexto. As FKs secundárias continuam sendo atribuídas como hoje.

**Custo.** +1 coluna por relação no pai. O mapeamento por posto custa O(n log n) por relação.

**Riscos.** Contagens com muitos empates (a maioria das empresas tem 1 filial): o posto entre empatados é arbitrário. É inofensivo, porque só redistribui empates.

**Testes.**
- Unit: KS = 0 depois do mapeamento por posto; a correlação de Spearman entre `v` e as contagens é preservada.
- Integração com `correlated=True`: Spearman(atributo do pai, número de filhos) sintético a no máximo ±0,05 do real.

**Esforço.** ~1 dia.

---

### T3: filho condicionado ao contexto do pai

**Problema.** Os atributos do filho ignoram os do pai (porte ↔ CNAE, capital ↔ situação cadastral das filiais etc.).

**Design.**

1. **Metadata (`__init__`):** para cada `context_columns[child][parent] = [cols]`, adicionar ao metadata do filho as colunas `__ctx__{parent}__{col}`, com o sdtype da coluna no pai. Elas recebem transformadores do RDT no data processor do filho.

2. **Fit (`preprocess`, nos dados brutos):** fazer o JOIN das colunas do pai no filho pela FK. Filhos com FK nula ficam com contexto nulo, e o RDT trata esses nulos. A cópula do filho aprende a correlação entre as colunas do filho e as de contexto.

3. **Sampling condicionado, por linha e vetorizado** (`_conditional.py`):

   Para o modelo do filho, com colunas próprias `1` e colunas de contexto `2`, as matrizes são calculadas **uma vez**:
   ```
   A  = Σ₁₂ Σ₂₂⁻¹
   Σ̄  = Σ₁₁ − A Σ₂₁
   L  = cholesky(Σ̄ + εI)
   ```
   Para cada lote:
   ```
   ctx_raw   = np.repeat(contexto_do_pai, counts)            # T2 dá as contagens
   ctx_proc  = data_processor.transform(ctx_raw)             # só as colunas de contexto
   Z₂        = Φ⁻¹(F₂(ctx_proc))                             # marginais → normal padrão
   Z₁        = Z₂ @ Aᵀ + E @ Lᵀ,   E ~ N(0, I)               # n × k: uma multiplicação de matrizes
   X₁        = F₁⁻¹(Φ(Z₁))                                   # normal → marginais do filho
   saída     = data_processor.reverse_transform([X₁, ctx_proc]).drop(contexto)
   ```
   Não se usa o `sample_from_conditions` do SDV: ele agrupa por valor de condição e ficaria lento com milhões de contextos distintos.

4. **Netos:** como o sampling segue a ordem topológica, o contexto se propaga em cadeia: o filho sintético já carrega o contexto e pode servir de pai.

**Restrições da primeira versão.**
- Só para filhos modelados com **GaussianCopula**. Para outros synthesizers, levantar erro claro.
- Constraints de **reject sampling** no filho: aplicar o filtro e reamostrar só as linhas rejeitadas, com o mesmo contexto.
- Contexto de **alta cardinalidade** (ids, `municipio`) não é recomendado: emitir aviso.

**Custo.**
- Fit: +k colunas no filho (k de 1 a 3), estimado em +5 a 10%.
- Sample: uma multiplicação de matrizes n × k por lote (desprezível perto do Faker).
- Memória: n_filhos × k × 8 bytes, ~90 MB para 3,7M filiais com k = 3.

**Riscos.** A cópula captura relações **monotônicas**; relações não lineares ficam aproximadas. `Σ₂₂` pode ser mal condicionada com contextos colineares (mitigado com regularização `εI`).

**Testes.**
- Unit:
  - equivalência com `copulas._get_conditional_distribution` para uma única condição;
  - média condicional correta em dados gaussianos sintéticos;
  - reprodutibilidade com seed.
- Integração com `correlated=True`: a correlação de Spearman entre a coluna do pai e a do filho sintético fica a no máximo ±0,05 da real. Hoje é ≈ 0.

**Esforço.** ~2 dias.

---

### T4: posição do filho no grupo e regras de grupo

**Problema.** Regras entre filhos do mesmo pai: exatamente uma matriz por empresa.

**Design.**
1. **Fit:** ordenar os filhos de cada pai por `sibling_order[child]['by']` e criar a coluna de contexto `__pos__ = groupby(fk).cumcount()`, limitada a um teto K (ex.: 3) para virar praticamente categórica: "0", "1", "2", "3+". Ela entra no modelo do filho como mais uma coluna de contexto (mecanismo da T3).
2. **Sample:** a posição sai de graça ao expandir as contagens:
   ```python
   inicio = np.repeat(np.cumsum(counts) - counts, counts)
   pos = np.arange(counts.sum()) - inicio
   ```
   e o filho é gerado condicionado a ela.
3. **Regras rígidas (`group_rules`):** pós-processamento vetorizado. Com `first`/`others`, a posição 0 recebe `first` e as demais recebem `others`. Garante 100% da regra.

**Custo.** +1 coluna de contexto e operações vetorizadas O(n).

**Testes.**
- Unit: cálculo de `pos` para grupos de tamanhos variados e aplicação das regras.
- Integração no CNPJ: % de empresas com exatamente uma matriz ≥ 99% só com a posição como contexto, e = 100% com `group_rules`.

**Esforço.** ~1 dia.

---

### T5: semelhança entre irmãos (efeito aleatório por pai)

**Problema.** Filiais da mesma empresa compartilham características (mesmo CNAE, mesma região) além do que as colunas do pai explicam.

**Design.**
1. **Fit:** para cada coluna própria `j` do filho, no espaço normal da cópula (`_transform_to_normal` dos dados processados), estimar a **correlação intraclasse** por ANOVA de um fator, de forma vetorizada:
   ```
   ρⱼ = (MS_entre − MS_dentro) / (MS_entre + (n̄ − 1)·MS_dentro),   limitada a [0, 1)
   ```
   Isso é calculado com `groupby(fk).agg(['mean','count'])` sobre as colunas normais e guardado em `self._sibling_rho[child]` (um valor por coluna).
2. **Sample:** sorteia `u ~ N(0, I)` **por pai** (n_pais × k₁), repete por filho e mistura com o ruído do condicionamento da T3:
   ```
   Z₁ = Z₂ @ Aᵀ + √ρ ⊙ (u_pai @ Lᵀ) + √(1−ρ) ⊙ (E @ Lᵀ)
   ```
   A variância de cada coluna é preservada. A covariância cruzada entre colunas fica aproximada, e isso é validado pelo Column Pair Trends.

**Custo.** n_pais × k₁ sorteios e uma soma vetorizada.

**Riscos.** Grupos de tamanho 1 (a maioria das empresas) não informam ρ: o estimador usa só grupos com 2 filhos ou mais. ρ alto demais em colunas quase constantes por grupo.

**Testes.**
- Unit: ρ estimado ≈ ρ verdadeiro em dados com efeito aleatório conhecido.
- Integração no CNPJ: % de pares de irmãos com o mesmo CNAE, sintético a no máximo ±5 p.p. do real.

**Esforço.** ~1 dia.

---

## 6. Compatibilidade com o plano de otimização

| Otimização ([`PLANO-OTIMIZACAO.pt-BR.md`](PLANO-OTIMIZACAO.pt-BR.md)) | Impacto da correlação entre tabelas |
|---|---|
| O2: fit numa subamostra | as contagens (T2) e a correlação intraclasse (T5) são calculadas nos **dados completos de FK**; o JOIN de contexto (T3) é feito só na subamostra |
| O6: fit tabela a tabela | o contexto do pai precisa estar disponível no fit do filho: ler do Parquet do pai só as colunas de contexto, pela FK |
| O7: sampling em streaming | em memória ficam a PK e as colunas de contexto do pai (n_pais × (1 + k)); o filho é gerado por lotes de pais consecutivos |
| O8: sampling paralelo | cada worker recebe um intervalo de pais com as contagens e o contexto; seeds por `SeedSequence.spawn` |

Ordem recomendada: implementar a correlação **depois da Fase 1** do plano de otimização, para medir os dois efeitos separadamente.

---

## 7. Fases e cronograma

| Fase | Entrega | Esforço | Dependências |
|---|---|---|---|
| C0 | Métricas de correlação entre tabelas no benchmark (seção 8) + gerador `correlated=True` + linha de base | 1 dia | — |
| C1 | **T1 + T2** | 1,5 a 2 dias | C0 |
| C2 | **T3 + T4**, incluindo `_conditional.py` e a nova ordem do `_sample` | 3 dias | C1 (T2 fornece as contagens antes do filho) |
| C3 | **T5** | 1 dia | C2 |
| C4 | Integração com streaming e paralelismo, se as Fases 2 e 3 do plano de otimização já existirem | 1 a 2 dias | C3 |
| **Total** | | **~7,5 a 9 dias** | |

Cada fase vira um commit separado (`Refs #1`), com a escada do CNPJ rerodada e o `BENCHMARK.pt-BR.md` atualizado.

---

## 8. Métricas e protocolo de medição

### 8.1 Métricas novas no `benchmarks/cnpj/run.py`

| Métrica | Como medir | Real | Hoje (sintético) | Meta |
|---|---|---|---|---|
| `sdmetrics` **Intertable Trends** | `QualityReport` multi-tabela, propriedade `inter_table_trends` | — | a medir em C0 | ≥ linha de base + 10 p.p. |
| Spearman(porte, nº de filiais) | por empresa | a medir | ≈ 0 | ±0,05 do real |
| Consistência UF ↔ município | % de pares sintéticos que existem no real | 100% | a medir (`full`) | ≥ 99% |
| Exatamente 1 matriz por empresa | por empresa | 100% | 93,2% (2,5%) | ≥ 99% (100% com `group_rules`) |
| Irmãos com o mesmo CNAE | % de pares de filiais da mesma empresa | a medir | a medir | ±5 p.p. do real |

### 8.2 Gates de não regressão

| Gate | Critério |
|---|---|
| Integridade referencial | = 100% |
| KS de filhos por pai | ≤ 0,005 (cauda exata com escala 1) |
| Column Shapes / Column Pair Trends | ≥ linha de base − 2 p.p. |
| Tempo total | ≤ +20% contra o Independent sem correlação, mesmo tamanho |
| Pico de memória | ≤ +15% |

### 8.3 Protocolo

O mesmo do plano de otimização: release CNPJ 2026-09, `sample_0.05`, 3 repetições por ponto, footprint físico, máquina na tomada. Cada técnica é medida **isolada** e **acumulada** (T1 → T1+T2 → … → T1 a T5), para atribuir o ganho de qualidade e o custo de cada uma.

---

## 9. Riscos

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| A cópula não captura relações não lineares fortes | média | médio | documentar; permitir CTGAN no filho em versão futura, com um caminho de condicionamento diferente |
| `Σ₂₂` mal condicionada (contextos colineares) | média | baixo | regularização `εI`; aviso ao usuário |
| Contexto de alta cardinalidade escolhido pelo usuário | média | médio | aviso; documentar boas escolhas (poucas colunas, com significado) |
| Interação com constraints de reject sampling | baixa | médio | reamostrar só as linhas rejeitadas, com o mesmo contexto |
| O mapeamento por posto distorce a cardinalidade em escalas ≠ 1 | baixa | baixo | a reamostragem sem reposição já trata escalas ≠ 1; teste específico |
| Custo maior que o estimado | baixa | médio | gate de +20% de tempo; cada técnica pode ser desligada isoladamente |

---

## 10. Fora de escopo

| Item | Motivo |
|---|---|
| Um modelo por linha do pai (abordagem do HMA) | é exatamente o custo que este plano evita |
| Escolha automática das colunas de contexto | a primeira versão é manual; a automática (informação mútua pai ↔ filho) fica para depois de medir o ganho da manual |
| Agregados dos filhos como colunas do pai (ex.: % de filiais ativas) | exige condicionar todos os filhos de um pai em conjunto; avaliar depois da T5 |
| Condicionamento para synthesizers não gaussianos (CTGAN) | requer outro mecanismo (condicionamento por vetor na rede); plano separado |
