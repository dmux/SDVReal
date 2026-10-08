# Plano de otimização de tempo e memória: `IndependentSynthesizer`

> Fork `dmux/SDVReal` · branch `feature/independent-multi-table-synthesizer` · Issue [#1](https://github.com/dmux/SDVReal/issues/1)
> Base de referência: [`BENCHMARK.pt-BR.md`](BENCHMARK.pt-BR.md) (CNPJ release 2026-09, MacBook Pro M4, 16 GB)

**Sobre os números.** Os valores de "Hoje" são **medidos**. As metas e os ganhos por otimização são **estimativas**, projetadas a partir do tempo por fase, da memória por fase e do profiling (seção 1). Cada fase do plano termina com uma nova medição que confirma ou corrige essas estimativas.

---

## Sumário

| Cenário | Hoje (medido) | Meta após a Fase 1 | Meta após as Fases 2 e 3 |
|---|---|---|---|
| 2,5% da base (5,6M linhas) | 8 min · 7,4 GB | **~2,5 min · ~4 a 5 GB** | **~2 min · ~2 a 3 GB** |
| 5% da base (11,1M linhas) | não cabe (cortado a 12,9 GB) | ~4 min · ~7 a 8 GB | ~3 min · ~2 a 3 GB |
| Base inteira (222M linhas) | não cabe (~280 GB estimados) | não cabe (~150 GB) | **~30 a 45 min · ~3 a 5 GB** |

O plano tem quatro ideias centrais:
1. **Ajustar o modelo numa amostra.** O custo do fit fica fixo, sem crescer com o volume.
2. **Gerar a saída em streaming**, em lotes gravados em disco. A memória fica fixa, sem crescer com o volume.
3. **Vetorizar a geração de PII**, o maior gargalo de tempo.
4. **Paralelizar o sampling dentro de cada tabela**, em vez de entre tabelas.

---

## 1. Diagnóstico (linha de base medida)

### 1.1 Tempo por fase: 2,5% da base, variante `full` (10 tabelas)

| load | init | preprocess | fit | sample | validate | **total** |
|---|---|---|---|---|---|---|
| 8,5 s | 1,3 s | 83,9 s (18%) | 204,5 s (43%) | 154,2 s (33%) | 16,4 s (3,5%) | **477,4 s** |

### 1.2 Memória acumulada por fase: 5% da base (pico de footprint físico)

| Imports | + load (dados brutos) | + preprocess | + fit | + sample |
|---|---|---|---|---|
| ~0,4 GB | **7,1 GB** | 9,8 GB | 11,5 GB | > 12,9 GB → cortado |

As quatro parcelas grandes (dados brutos, dados processados, modelos e saída) **coexistem**, e todas crescem com o número de linhas.

### 1.3 Profiling: 0,25% da base

| Tabela | Linhas | Colunas | preprocess | fit | sample | % do tempo |
|---|---|---|---|---|---|---|
| `estabelecimentos` | 191 mil | 29 | 10,2 s | 25,2 s | 63,3 s | **72%** |
| `empresas` | 175 mil | 7 | 1,5 s | 5,4 s | 5,2 s | 9% |
| `simples` | 126 mil | 7 | 1,4 s | 8,2 s | 7,0 s | 12% |
| `socios` | 71 mil | 11 | 0,8 s | 1,8 s | 7,7 s | 7% |

| Ponto quente | Tempo (cumulativo, com profiler) | Causa |
|---|---|---|
| `rdt ... AnonymizedFaker._function` + Faker + `re.sub` | **~73 s** | um valor de PII por chamada (2,5M chamadas), com regex por valor |
| `scipy ... _continuous_distns._logpdf` | 28 s | máxima verossimilhança da distribuição `beta` padrão em cada coluna numérica e de data |
| `rdt ... categorical.map_labels` | 7,9 s | 5,3M chamadas, valor a valor |
| `sdv._utils._parse_datetime` | 3,6 s | parse de data valor a valor |

### 1.4 Conclusões do diagnóstico

1. **O gargalo de tempo é a PII e o ajuste da cópula**, não a camada multi-tabela nova.
2. **Paralelizar por tabela rende pouco**: pela lei de Amdahl, no máximo ~1,4×, porque `estabelecimentos` concentra 72% do tempo.
3. **O gargalo de memória é a coexistência das quatro parcelas.** Nenhuma delas sozinha estoura os 16 GB.
4. **Polars ou outro formato de DataFrame não muda o quadro.** Strings Arrow deram −15% no pico (medido), e o SDV, o RDT e o copulas são acoplados a pandas e numpy.

---

## 2. Princípios de projeto

| Princípio | Aplicação |
|---|---|
| **Custo fixo no aprendizado** | marginais e correlações da cópula estabilizam com centenas de milhares de linhas; acima disso, o fit só gasta tempo e memória |
| **Aprender a cardinalidade com a base inteira** | só precisa das colunas de FK (inteiros ou strings curtas): barato e exato |
| **Memória constante na geração** | as linhas sintéticas são independentes, então podem ser geradas e gravadas em lotes |
| **Vetorizar o que é feito valor a valor** | PII, mapeamento de categorias, parse de datas |
| **Paralelismo onde está o custo** | dentro da tabela dominante, por lotes |
| **Nenhuma otimização sem gate de qualidade** | cada fase reroda a escada do CNPJ com as métricas da seção 7 |

---

## 3. Visão geral das otimizações

| ID | Otimização | Fase | Ganho de tempo (estim.) | Ganho de memória (estim.) | Esforço | Risco |
|---|---|---|---|---|---|---|
| O1 | Pool vetorizado de PII | 1 | sample **3 a 5×** | pequeno | baixo | médio (repetição de valores) |
| O2 | Fit numa subamostra por tabela | 1 | preprocess + fit **~3×** em 2,5%; ~10× em 5% | B e C fixos em ~1 a 2 GB | baixo | médio (categorias raras) |
| O3 | Distribuições de ajuste rápido | 1 | fit **~2×** | — | trivial | médio (forma das caudas) |
| O4 | Texto como `string[pyarrow]` | 1 | ~5% | **−15% (medido)** | baixo | baixo |
| O5 | Liberar os dados brutos após o preprocess (benchmark) | 1 | — | −40 a 50% no pico | trivial | nenhum |
| O6 | Fit tabela a tabela, com leitura preguiçosa | 2 | — | ~−40% em B + C | médio | baixo |
| O7 | Sampling em streaming para Parquet | 2 | +escrita (~3 µs/linha) | saída **constante** (~0,3 GB) | médio | médio (unicidade de PK) |
| O8 | Sampling paralelo por lotes dentro da tabela | 3 | sample **~3 a 4×** | +0,5 a 1 GB por worker | médio | médio (seeds e chaves) |
| O9 | Vetorizar `map_labels` e o parse de datas no RDT | 4 | ~8% | — | médio | baixo |

---

## 4. Fases de execução

### Fase 0: instrumentação (pré-requisito) · ~0,5 dia

**Objetivo:** medir com precisão suficiente para atribuir cada ganho à otimização certa.

| Item | Detalhe |
|---|---|
| Tempo por tabela e por fase | cronometrar `_preprocess`, `fit_processed_data` e `_sample_batch` de cada synthesizer de tabela, envolvendo esses métodos num wrapper de tempo no `run.py`, e gravar em `results.jsonl` |
| Custo da camada multi-tabela | cronometrar à parte `_augment_tables` (cardinalidade) e `_connect_tables` (atribuição de FK) |
| Custo do Faker | soma do tempo de `AnonymizedFaker._reverse_transform` por coluna |
| Repetições | 3 execuções por ponto, reportando a mediana e a variação (hoje há 1 por ponto) |
| Metadados | registrar o SHA do git, a release, o hardware e se a máquina estava na tomada |
| Qualidade ampliada | `sdmetrics.reports.multi_table.QualityReport` (Column Shapes, Column Pair Trends, Cardinality, Intertable Trends) além das métricas atuais |

**Entregável:** nova linha de base, com mediana de 3 execuções, para 0,1%, 0,5%, 1% e 2,5%.

---

### Fase 1: ganhos rápidos · ~2 dias

#### O5: liberar os dados brutos após o preprocess

- **Problema.** O `benchmarks/cnpj/run.py` mantém `data` vivo até a métrica de qualidade, então os dados brutos coexistem com o modelo e com a saída (1.2).
- **Solução.** Logo depois do `preprocess`, guardar só o que a qualidade usa (as colunas `cnpj_basico` e `identificador_matriz_filial`), fazer `del data` e chamar `gc.collect()`.
- **Estimativa.** Pico em 5% de >12,9 GB para ~7 a 8 GB. O pico do load (7,1 GB) não muda.
- **Aceite.** O degrau de 5% completa dentro do limite de 12,8 GB.

#### O4: carregar o texto como `string[pyarrow]`

- **Problema.** Strings em `object` ocupam 3,2× mais que em Arrow (1,28 GB contra 0,40 GB com 1% da base, medido).
- **Solução.** Em `benchmarks/cnpj/dataset.py`, usar `to_pandas(types_mapper=pd.ArrowDtype)` só nas colunas de texto, já na leitura, sem converter depois.
- **Evidência.** O SDV funciona de ponta a ponta com `string[pyarrow]` (testado). `category` **não** deve ser usado: o sampling passou de 5 min contra 25 s.
- **Estimativa.** −15% no pico (medido com a conversão tardia). Com a conversão na leitura, o pico do load também cai.
- **Aceite.** Integridade e KS inalterados, e o pico menor que a linha de base.

#### O1: pool vetorizado de PII

- **Problema.** `AnonymizedFaker` chama o Faker uma vez por valor. Em 0,25% da base, são 2,5M chamadas e ~73 s cumulativos, a maior parte do sampling.
- **Solução.** Criar `PooledAnonymizedFaker(AnonymizedFaker)` em `sdv/multi_table/_transformers.py`:
  1. no `_fit`, ou no primeiro `_reverse_transform`, gerar um pool de `pool_size` valores com o Faker (padrão 20 a 50 mil);
  2. no `_reverse_transform(n)`, retornar `pool[np.random.randint(0, len(pool), n)]` e aplicar a taxa de nulos aprendida (`_nan_frequency`);
  3. para colunas com `cardinality_rule='unique'` (ex.: e-mail), gerar de forma vetorizada combinando o pool com um sufixo sequencial (`{usuario}{i}@{dominio}`), o que garante unicidade sem chamar o Faker por linha.
- **Integração.** O `IndependentSynthesizer` ganha o parâmetro `pii_pool_size=None` (desligado por padrão, para manter o comportamento atual). Quando definido, troca os `AnonymizedFaker` de todas as tabelas via `update_transformers` depois de `auto_assign_transformers`.
- **Estimativa.**
  - Custo fixo de geração do pool: ~20 mil × ~30 µs ≈ 0,6 s por coluna, ~6 s para as 10 colunas de PII.
  - Sampling: **de 154 s para ~40 s** em 2,5%.
- **Riscos.** Valores repetidos: com pool de 20 mil e 1,8M estabelecimentos, cada nome aparece ~90 vezes. Isso é aceitável para teste de carga, mas não serve para colunas que precisam ser únicas, que usam o passo 3.
- **Testes.** Unitários (tamanho, taxa de nulos, unicidade no modo `unique`, reprodutibilidade com seed) e integração com o CNPJ.
- **Aceite.** Sampling ≥ 3× mais rápido, 100% de unicidade nas colunas `unique` e taxa de nulos com diferença ≤ 0,5 p.p.

#### O2: fit numa subamostra por tabela

- **Problema.** Preprocess e fit crescem com o volume (84 s + 205 s em 2,5%), embora a cópula não ganhe precisão acima de algumas centenas de milhares de linhas.
- **Solução.** No `IndependentSynthesizer`, criar o parâmetro `fit_sample_size=None` (ex.: 500_000) e sobrescrever `preprocess`:
  1. **sobre os dados completos e baratos:** guardar os tamanhos reais das tabelas (`_table_sizes`), aprender a cardinalidade de cada relação (só as colunas de FK) e calcular faixas numéricas e de datas (`min`/`max`);
  2. **subamostra com cobertura de categorias:** amostragem aleatória de `fit_sample_size` linhas, mais pelo menos uma linha de cada categoria das colunas categóricas (limitada, por exemplo, às 10 mil categorias mais frequentes);
  3. chamar o preprocess da base **só com a subamostra**;
  4. **validação:** a integridade referencial é verificada uma vez nos dados completos e depois desligada no preprocess da subamostra, porque filhos amostrados apontam para pais fora dela.
- **Ajustes no código atual.** `_augment_tables` passa a usar os tamanhos e a cardinalidade calculados no passo 1, e não `len(processed_data)`.
- **Estimativa (2,5%).** Preprocess de 84 s para ~25 s e fit de 205 s para ~60 s. Acima de ~500 mil linhas por tabela, o custo fica fixo.
- **Riscos.** Categorias raras ausentes (mitigado pela cobertura do passo 2), extremos fora da faixa (mitigado pelo `min`/`max` do passo 1) e correlações fracas mais ruidosas (a subamostra de 500 mil limita esse ruído).
- **Aceite.** Column Shapes e Column Pair Trends do `sdmetrics` no máximo 2 p.p. abaixo da linha de base, e KS de cardinalidade ≤ 0,005.

#### O3: distribuições de ajuste rápido

- **Problema.** O ajuste da `beta` (padrão da GaussianCopula) custa 28 s cumulativos em `_logpdf` no profiling de 0,25%.
- **Solução.** Medir o tempo de fit e o Column Shapes para `norm`, `truncnorm`, `beta` e `uniform` nas colunas numéricas e de data do CNPJ, e definir um padrão por sdtype no `IndependentSynthesizer` (via `numerical_distributions`). Combinado com o O2, o custo da `beta` já cai, porque ela passa a ser ajustada só na subamostra.
- **Estimativa.** Fit ~2× mais rápido, de ~60 s para ~30 s (2,5%, com o O2).
- **Aceite.** Column Shapes no máximo 1 p.p. abaixo da `beta`.

**Saída esperada da Fase 1 (2,5% da base):**

| load | preprocess | fit | sample | validate | **total** | pico |
|---|---|---|---|---|---|---|
| 8 s | ~25 s | ~30 a 60 s | ~40 s | 16 s | **~2 a 2,5 min (≈3×)** | **~4 a 5 GB** |

---

### Fase 2: streaming · ~3 a 4 dias

O objetivo é eliminar a dependência entre memória e volume.

#### O6: fit tabela a tabela com leitura preguiçosa

- **Problema.** Hoje todas as tabelas brutas e processadas coexistem.
- **Solução.** Criar o método `IndependentSynthesizer.fit_from_parquet(table_paths: dict[str, str])`, que para cada tabela:
  1. lê **só as colunas de FK e PK** (`pyarrow.parquet.read_table(columns=...)`) para aprender a cardinalidade e os tamanhos;
  2. lê a **subamostra** do O2 em lotes (`iter_batches`, amostragem por reservatório);
  3. faz o preprocess e o fit do synthesizer da tabela;
  4. descarta os dados da tabela antes de ir para a próxima.
- **Limitação.** Constraints multi-tabela que precisam dos dados completos não entram no modo streaming na primeira versão. Constraints de uma tabela continuam funcionando.
- **Estimativa.** O pico do fit passa a ser o da maior tabela, e não a soma: −40% em B + C, que, com o O2, ficam em ~1 a 2 GB.

#### O7: sampling em streaming para Parquet

- **Problema.** A saída inteira fica em memória (parcela D), e foi ela que estourou o degrau de 5%.
- **Solução.** Criar o método `IndependentSynthesizer.sample_to_parquet(output_folder, scale=1.0, batch_size=1_000_000)`:
  1. percorrer as tabelas em **ordem topológica** (pais antes dos filhos);
  2. para cada pai, manter em memória **só a coluna de PK** sintética, que é o que os filhos precisam;
  3. para cada filho, sortear de uma vez o **vetor de contagens** por pai (o mesmo algoritmo atual, de reamostragem sem reposição);
  4. gerar o filho em lotes e preencher a FK de cada lote com o trecho correspondente de `np.repeat(pks_do_pai, contagens)`, mais a fração de nulos sorteada por lote. Como as linhas sintéticas são independentes, não é preciso embaralhar a ordem global;
  5. gravar cada lote com `pyarrow.parquet.ParquetWriter`, num arquivo por tabela.
- **Unicidade de PK entre lotes.** Os geradores de chave do RDT mantêm estado dentro de um processo. No O8, as chaves serão atribuídas por faixa (ver abaixo).
- **Estimativa.** A parcela D cai de ~6 GB (5%) para ~0,3 GB (um lote mais as PKs dos pais). Na base inteira, as PKs de 70M empresas ocupam ~0,6 GB. A escrita custa ~3 µs por linha.
- **Validação.** Fazer `validate_data` por lote, mais uma verificação final de unicidade de PK e de integridade de FK feita só com as colunas de chave.

**Saída esperada da Fase 2:**

| Cenário | Tempo | Pico |
|---|---|---|
| 2,5% | ~2 a 2,5 min | **~2 a 3 GB** |
| 5% | ~3 a 4 min | ~2 a 3 GB |
| Base inteira (222M) | ~1,5 a 2 h (sampling ainda sequencial) | **~3 a 5 GB** |

---

### Fase 3: paralelismo dentro da tabela · ~2 dias

#### O8: sampling paralelo por lotes

- **Problema.** Tudo roda num núcleo, e 9 dos 10 ficam parados. Paralelizar por tabela não resolve (Amdahl, 1.4).
- **Solução.**
  - Usar um `ProcessPoolExecutor` com N workers (padrão: número de núcleos de performance, 4 no M4). O synthesizer ajustado é enviado uma vez a cada worker.
  - Cada worker gera os lotes que recebe e grava o próprio arquivo (`tabela/part-00007.parquet`).
  - **Seeds:** `numpy.random.SeedSequence(seed).spawn(N)` dá um fluxo independente e reprodutível por worker.
  - **Chaves:** o worker *k* recebe uma faixa de índices `[offset_k, offset_k + n_k)`, e a PK é gerada a partir do índice (ex.: `cnpj = f'{indice:014d}'`), o que garante unicidade sem estado compartilhado.
  - **FKs:** o processo principal envia a cada lote o trecho de FK já calculado no passo 4 do O7.
  - **Threads de BLAS:** fixar `OMP_NUM_THREADS=1` e `OPENBLAS_NUM_THREADS=1` nos workers, para não haver mais threads do que núcleos.
- **Estimativa.** Sampling ~3 a 4× mais rápido com 4 workers. Memória: +0,5 a 1 GB por worker, com pico total de ~4 a 6 GB.
- **Aceite.** Mesmo resultado com a mesma seed e o mesmo N, 100% de unicidade de PK e a mesma qualidade da Fase 2.

**Saída esperada da Fase 3:**

| Cenário | Tempo | Pico |
|---|---|---|
| 2,5% | **~2 min (≈4×)** | ~4 a 6 GB |
| Base inteira (222M) | **~30 a 45 min** | **~4 a 6 GB** |

Detalhamento da base inteira: leitura das FKs e das subamostras (~2 min) + preprocess e fit (~1 a 1,5 min) + sampling (~8 a 12 min; ~2,5 µs/linha) + escrita (~5 a 10 min) + validação por lote (~10 min, opcional).

---

### Fase 4: refinamentos (opcional) · ~1 a 2 dias

| ID | Item | Ganho (estim.) | Observação |
|---|---|---|---|
| O9 | Vetorizar `map_labels` (categóricas) e `_parse_datetime` | ~8% do tempo | patch no RDT e no SDV, a propor upstream |
| — | Dados processados em `float32` | ~−50% na parcela B | arriscado: o copulas trabalha em `float64`; precisa de validação numérica |
| — | Compressão `zstd` nos Parquet de saída | disco ~−60% | custo de CPU pequeno |

---

## 5. Projeções consolidadas

### 5.1 Tempo

| Cenário | Hoje | Fase 1 | Fase 2 | Fase 3 |
|---|---|---|---|---|
| 2,5% (5,6M) | 8 min | ~2,5 min | ~2,5 min | **~2 min** |
| 5% (11,1M) | não cabe | ~4 min | ~3,5 min | **~3 min** |
| Base inteira (222M) | não cabe | não cabe | ~1,5 a 2 h | **~30 a 45 min** |

### 5.2 Memória (pico de footprint)

| Cenário | Hoje | Fase 1 | Fase 2 | Fase 3 |
|---|---|---|---|---|
| 2,5% | 7,4 GB | ~4 a 5 GB | **~2 a 3 GB** | ~4 a 6 GB (workers) |
| 5% | > 12,9 GB | ~7 a 8 GB | **~2 a 3 GB** | ~4 a 6 GB |
| Base inteira | ~280 GB | ~150 GB | **~3 a 5 GB** | ~4 a 6 GB |

### 5.3 Contribuição de cada otimização para a memória

| Parcela (1.2) | Hoje (5%) | Otimizações | Depois |
|---|---|---|---|
| A. Dados brutos | 7,1 GB | O5 (liberar), O4 (Arrow), O6 (leitura preguiçosa) | só as colunas de FK e a subamostra: ~1 GB |
| B. Dados processados | +2,6 GB | O2 (subamostra), O6 (uma tabela por vez) | ~0,5 a 1 GB, fixo |
| C. Modelos em ajuste | +1,7 GB | O2, O6 | ~0,3 a 0,5 GB, fixo |
| D. Saída sintética | estourou | O7 (streaming) | ~0,3 GB por lote + PKs dos pais |

---

## 6. Cronograma

| Fase | Esforço | Dependências | Entregável |
|---|---|---|---|
| 0: instrumentação | 0,5 dia | — | linha de base com 3 repetições |
| 1: ganhos rápidos | 2 dias | Fase 0 | O1 a O5, a escada medida e o relatório |
| 2: streaming | 3 a 4 dias | Fase 1 (O2) | `fit_from_parquet`, `sample_to_parquet`, 5% e a base inteira medidos |
| 3: paralelismo | 2 dias | Fase 2 (O7) | sampling paralelo e a base inteira em 30 a 45 min |
| 4: refinamentos | 1 a 2 dias | opcional | patches de RDT e SDV |
| **Total** | **~8 a 10 dias** | | |

Cada fase vira um commit (ou PR) separado, referenciando a issue #1, com a escada do CNPJ rerodada e o `BENCHMARK.pt-BR.md` atualizado.

---

## 7. Gates de qualidade (obrigatórios em cada fase)

| Métrica | Linha de base | Critério |
|---|---|---|
| Integridade referencial (`validate_data`) | 100% | **= 100%** |
| Unicidade de PK | 100% | **= 100%** |
| KS de estabelecimentos por empresa | 0,000 | ≤ 0,005 |
| KS de sócios por empresa | 0,000 | ≤ 0,005 |
| Máx. de filhos por pai (cauda) | 7.903 = real | igual ao real com escala 1 |
| `sdmetrics` Column Shapes | a medir na Fase 0 | ≥ linha de base − 2 p.p. |
| `sdmetrics` Column Pair Trends | a medir na Fase 0 | ≥ linha de base − 2 p.p. |
| Taxa de nulos por coluna | real | diferença ≤ 0,5 p.p. |
| Unicidade de PII `unique` (ex.: e-mail) | — | = 100% |
| Empresas com exatamente uma matriz | 93,2% (2,5%) | relatar; não pode piorar |

Se algum gate falhar, a otimização fica **desligada por padrão** (parâmetro opcional) até ser corrigida.

---

## 8. Protocolo de medição

- **Dados:** CNPJ release 2026-09, amostra `sample_0.05` (a mesma do benchmark atual). A base inteira é medida no fim das Fases 2 e 3.
- **Ambiente:** MacBook Pro M4 com 16 GB, **ligado na tomada**, com navegador e IDE fechados, para reduzir o ruído de memória (hoje outros apps ocupavam ~55% da RAM).
- **Repetições:** 3 por ponto, reportando a mediana e a variação.
- **Memória:** pico de footprint físico (`ri_lifetime_max_phys_footprint`), com o mesmo corte de segurança de 80% da RAM.
- **Registro:** `results.jsonl` com o SHA do git, a fase, os parâmetros (`fit_sample_size`, `pii_pool_size`, número de workers, `batch_size`) e todas as métricas da seção 7.
- **Comparação:** cada otimização é medida isolada (ligando só ela) e em conjunto, para atribuir o ganho certo a cada uma.

---

## 9. Riscos gerais

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| Estimativas otimistas no sampling (o peso do Faker inflado pelo profiler) | média | médio | a Fase 0 mede o Faker sem profiler antes de qualquer otimização |
| A subamostra degrada correlações fracas | média | médio | gate do `sdmetrics`; `fit_sample_size` ajustável |
| Colisão de PK no sampling paralelo | baixa | alto | PK por faixa de índice; verificação final de unicidade |
| Mudanças internas do SDV ou do RDT quebram as extensões | baixa | médio | tudo dentro de `IndependentSynthesizer` e de transformers próprios, sem patch no RDT até a Fase 4 |
| Disco na base inteira | baixa | baixo | ~141 GB livres; saída com `zstd` estimada em ~10 a 20 GB |

---

## 10. Fora de escopo, com justificativa

| Ideia | Por que não |
|---|---|
| Reescrever com Polars 2.0 | o SDV, o RDT e o copulas são acoplados a pandas e numpy; o ganho real medido do formato Arrow foi de −15%, que o O4 já captura |
| `category` no lugar de `object` | quebra o desempenho do sampling (> 5 min contra 25 s, medido) |
| Paralelizar **entre** tabelas | máximo de ~1,4× (Amdahl), porque `estabelecimentos` concentra 72% do tempo |
| Otimizar o `HMASynthesizer` | o custo é estrutural (um modelo por linha do pai, colunas em cascata); 52× mais lento já em 30 mil linhas |
| Modelar correlações entre tabelas | é um ganho de qualidade, não de desempenho; plano separado (condicionar filhos a colunas do pai) |
