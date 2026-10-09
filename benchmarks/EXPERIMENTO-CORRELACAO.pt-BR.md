# Experimento: correlação entre tabelas no `IndependentSynthesizer`

> Branch `experiment/inter-table-correlation` (a partir de `feature/independent-multi-table-synthesizer`) · Issue [#1](https://github.com/dmux/SDVReal/issues/1)
> Plano avaliado: [`PLANO-CORRELACAO-INTERTABELAS.pt-BR.md`](PLANO-CORRELACAO-INTERTABELAS.pt-BR.md) · Relacionados: [`BENCHMARK.pt-BR.md`](BENCHMARK.pt-BR.md) · [`PLANO-OTIMIZACAO.pt-BR.md`](PLANO-OTIMIZACAO.pt-BR.md)
> Execução: 2026-10-09, MacBook Pro M4 (10 núcleos, 16 GB), na tomada. Dados: CNPJ release 2026-09, `sample_0.05`.

## 1. Resumo

| Pergunta | Resposta |
|---|---|
| O plano funciona? | **Em parte.** T2, T4 com regras e T5 numérica funcionam como previsto. Mas a T3 com contexto categórico e a T5 com colunas categóricas, **do jeito que o plano descreve**, quase não recuperam nada no CNPJ. A meta de UF↔município (≥ 99%) **não é alcançável com a T1**. |
| Houve melhorias reais? | **Sim, quatro**, todas medidas: contexto categórico em one-hot com regressão exata (I1), cópia categórica entre irmãos calibrada por pai (I2), cardinalidade estratificada (I3, `cardinality_by`) e `FixedCombinations` integrada à cópia entre irmãos (I4). |
| Ficou melhor que o HMA? | **Em custo e cardinalidade, claramente; em qualidade, na maioria das métricas.** No CNPJ a 0,01%, a configuração recomendada é **34× mais rápida** (12 s contra 419 s), preserva a cauda da cardinalidade e vence o HMA em 4 de 7 associações entre tabelas (3 delas ≈ real), perdendo em 2 e empatando em 1 (seção 7). |
| Custo | Configuração recomendada: **+2% de tempo em 2,5% (5,6 M de linhas)** e +4% em 0,1%. A memória fica dentro do limite até 1% (+14% a +19%) e passa dele em 2,5% (+23%). Esse é o único gate reprovado e fica como item aberto (seção 8). |
| Compatibilidade | Tudo desligado por padrão. Com as opções desligadas, a saída é **idêntica bit a bit** à da branch atual (hash SHA-256 das tabelas sintéticas, `PYTHONHASHSEED=0`). |

### Comparativo principal (CNPJ, variante `core`, 0,1% = 230 mil linhas)

| Métrica | Real | Hoje (baseline) | Plano (T2..T5 como escrito) | **Recomendado**¹ |
|---|---|---|---|---|
| KS filiais/empresa · sócios/empresa | 0 | 0,000 · 0,000 | 0,000 · 0,000 | 0,000 · 0,001 |
| Exatamente 1 matriz por empresa | 100% | 84,9% | 100% | **100%** |
| Consistência UF↔município | 100% | 12,9% | 13,4% | **100%** |
| Spearman(porte, nº de sócios) | 0,131 | −0,004 | 0,067 | **0,134** |
| Cramér's V porte ~ `simples.opcao_mei` | 0,157 | 0,007 | 0,068 | 0,050 |
| Cramér's V porte ~ `estab.situacao_cadastral` | 0,133 | 0,007 | 0,051 | **0,081** |
| Cramér's V natureza ~ `estab.situacao_cadastral` | 0,235 | 0,021 | 0,050 | **0,073** |
| Cramér's V natureza ~ `socios.qualificacao_socio` | 0,500 | 0,037 | 0,135 | 0,132 |
| Irmãos com o mesmo CNAE (média por empresa) | 0,680 | 0,021 | 0,038 | **0,637** |
| Irmãos com a mesma UF (média por empresa) | 0,863 | 0,135 | 0,157 | **0,854** |
| Irmãos com a mesma situação cadastral | 0,755 | 0,393 | 0,556 | **0,712** |
| sdmetrics Column Pair Trends | — | 0,615 | 0,603 | 0,608 |
| Tempo total (wall) | — | 31,8 s | 36,2 s (+14%) | 33,0 s (**+4%**) |
| Pico de memória | — | 1,43 GB | 1,45 GB | 1,44 GB |

¹ `FC+T2S+T3C+T4+T4R+T5`: `FixedCombinations(uf, municipio)`, cardinalidade estratificada por porte, contexto `porte_empresa` e `natureza_juridica` (sem `capital_social`), posição e regra de matriz, irmãos com cópia categórica.

---

## 2. O que foi implementado

### 2.1 Técnicas do plano (T1 a T5)

| Técnica | Parâmetro | Arquivo |
|---|---|---|
| T1: tabelas de consulta copiadas, FK modelada como categórica | `lookup_tables`, `detect_lookup_tables()` | `sdv/multi_table/independent.py` |
| T2: nº de filhos como coluna do pai + mapeamento por posto | `model_cardinality` | idem |
| T3: filho condicionado ao contexto do pai, linha a linha e vetorizado | `context_columns` | `sdv/multi_table/_conditional.py` |
| T4: posição entre irmãos como contexto + regras de grupo | `sibling_order`, `group_rules` | `sdv/multi_table/_group_features.py` |
| T5: efeito aleatório por pai (ICC por ANOVA de um fator) | `sibling_correlation` | idem |

Desvios do plano na implementação:
- **T2 também em relações 1-para-1** (`simples` ⊂ `empresas`): define *quais* empresas têm o registro filho.
- **T3 funciona sem a T2.** As FKs são sempre atribuídas antes de gerar o filho.
- **T5 estima o ICC sobre o resíduo** depois do condicionamento da T3, e não sobre o valor bruto. Assim não conta duas vezes o que o contexto já explica.
- O gerador de dados com correlação conhecida virou uma função separada, `tests/utils.py::generate_correlated_parent_child`, e não um parâmetro `correlated=True`.

### 2.2 Melhorias encontradas durante o experimento

| # | Melhoria | Parâmetro | Problema do plano que resolve |
|---|---|---|---|
| I1 | Contexto categórico como **indicadores padronizados fora da cópula**, com uma matriz de correlação estendida (regressão exata nas categorias); a posição da T4 também | `categorical_context='one_hot'` (padrão) | No plano, a categoria do pai entra pelo `UniformEncoder`, cuja **ordem das categorias é arbitrária**. Um efeito não monotônico some, e o resultado varia com essa ordem (η² = 0,14 ± 0,17 entre seeds). Mesmo com one-hot, o jitter dentro da categoria atenua o efeito pela metade. |
| I2 | **Cópia da categoria do 1º irmão**, com a probabilidade que leva à taxa real de "mesmo valor que o 1º irmão", calibrada **por pai** e com **um sorteio por linha** compartilhado entre as colunas | `sibling_categorical='copy'` (padrão) | O efeito aleatório latente da T5 aproxima valores no espaço normal, mas raramente produz a *mesma categoria* (CNAE com ~1.300 códigos: 0,03 contra 0,68 real). Calibrar por linha deixava a empresa com 7.903 filiais dominar. |
| I3 | **Cardinalidade estratificada**: as contagens reais são reamostradas sem reposição *dentro* de cada estrato do pai | `cardinality_by` | A T2 depende de a cópula do pai capturar a relação. No CNPJ o marginal Beta de `capital_social` (cauda até 1,2×10¹¹) e a massa de zeros achatam a correlação latente (0,07 contra Spearman real de 0,29). |
| I4 | Colunas de uma `FixedCombinations` são **copiadas juntas** entre irmãos; as colunas copiadas vêm do metadata original | automático | Copiar `uf` sem `municipio` criava pares inválidos (99,5%). E a constraint funde as colunas no metadata modificado, o que tirava `uf` da cópia. |
| — | ICC estimado em blocos, por somas por grupo; o sampling guarda só as colunas processadas que os filhos usam | interno | Memória: em 1%, de +20% para +14%. |

---

## 3. Protocolo

- **Dados com correlação conhecida** (`benchmarks/correlation/synthetic.py`): 20 mil pais, seeds 0, 1 e 2. `size` determina o nº de filhos e `amount`; `tier` (categórica, independente de `size`) desloca `discount` de forma não monotônica; o 1º filho é `HQ`; irmãos compartilham um efeito em `score` e, em 80% dos casos, a `region`.
- **CNPJ** (`benchmarks/correlation/run_cnpj_campaign.sh` e `..._e.sh`): `benchmarks/cnpj/run.py` com `--correlation`, `--encoding plan|improved`, `--sdmetrics` e `--repeat`. Cada configuração roda em um subprocesso, e o pico é o footprint físico. Na fração de 0,1%, 3 repetições por configuração. O sdmetrics `QualityReport` roda em uma subamostra de 2.000 empresas e seus descendentes.
- **Métricas entre tabelas novas:**
  - Spearman entre atributos do pai e nº de filhos;
  - Cramér's V de pares (coluna do pai, coluna do filho) no JOIN;
  - semelhança entre irmãos de duas formas: por pares (dominada pela empresa gigante, onde só 9,5% dos pares têm a mesma UF) e **média por empresa**, que é a usada nas conclusões;
  - consistência UF↔município.
- **Determinismo:** para a mesma configuração, as métricas de qualidade são idênticas entre repetições (seed fixa). O desvio padrão mostrado nas tabelas vem do tempo e da subamostra do sdmetrics.

> **Sobre o sdmetrics:** na subamostra de 2.000 empresas, Column Shapes varia ±0,02 entre repetições da *mesma* configuração. O Intertable Trends faz a média de muitos pares sem relação real (datas, capital), então quase não se move (0,627 a 0,660). As métricas direcionadas da seção 5 são as que discriminam as técnicas.

---

## 4. Resultados: dados com correlação conhecida

20 mil pais, média ± desvio entre 3 seeds (`benchmarks/correlation/results_synthetic.jsonl`).

| Configuração | Spearman(size, nº filhos) | Spearman(size, amount) | η² tier→discount | 1 HQ por pai | ICC score | Mesma região (pares) | Intertable Trends |
|---|---|---|---|---|---|---|---|
| **Real** | 0,600 | 0,688 | 0,525 | 97,2% | 0,638 | 0,717 | — |
| baseline | −0,001 | 0,003 | 0,000 | 42,5% | 0,000 | 0,201 | 0,684 |
| T2 | 0,522 | −0,005 | 0,000 | 42,4% | 0,000 | 0,200 | 0,690 |
| T3 (I1) | −0,001 | 0,582 | **0,432 ± 0,006** | 44,3% | 0,001 | 0,201 | **0,844** |
| T3 (plano) | −0,001 | 0,589 | 0,140 ± 0,168 | 44,1% | 0,001 | 0,201 | 0,804 |
| T4 (posição, I1) | — | — | — | **75,2%** | — | — | 0,684 |
| T4 + regras | — | — | — | **100%** | — | — | 0,686 |
| T5 (I2) | — | — | — | 42,4% | 0,635 | **0,611** | 0,683 |
| T5 (plano) | — | — | — | 42,4% | 0,635 | 0,346 | 0,683 |
| **T2..T5 (melhorado)** | 0,522 | 0,531 | **0,437** | 100% | 0,638 | **0,605** | **0,851** |
| T2..T5 (plano) | 0,522 | 0,541 | 0,141 ± 0,170 | 100% | 0,635 | 0,345 | 0,807 |

O KS de filhos por pai foi 0,000 em todas as configurações, com integridade de 100%.

Leituras:
- A T3 do plano tem desempenho **instável**: η² vai de ~0 a ~0,3 conforme a ordem em que o `UniformEncoder` organiza as categorias. A I1 é estável e chega a 82% do real.
- A T4 só com a posição (codificação do plano, com jitter) dá 54%. Com a posição em one-hot (I1), 75%. A meta do plano (≥ 99% só com a posição) não é alcançável por uma cópula gaussiana, que não representa relações binárias determinísticas. Para 100%, é preciso `group_rules`.
- A T3 sem a T2 desloca o marginal do filho (Column Shapes 0,979 → 0,958). O contexto que o filho vê no sampling passa a ser ponderado por pai, e não por filho. **Usar T3 junto com T2 ou T2S.**
- `segment↔amount` não melhora com nenhuma opção: a cópula do **pai** não preserva `size↔segment` (0,14 contra 0,94 real). É uma limitação single-table que a correlação entre tabelas herda.

---

## 5. Resultados: CNPJ, variante `core`, 0,1% (230 mil linhas)

### 5.1 Cada técnica isolada e acumulada

| Configuração | Wall (s) | Fit (s) | Sample (s) | Spearman porte~nº sócios | V porte~MEI | V porte~situação | V natureza~qualif. sócio | 1 matriz | Column Pair Trends |
|---|---|---|---|---|---|---|---|---|---|
| **Real** | | | | 0,131 | 0,157 | 0,133 | 0,500 | 100% | |
| baseline | 31,8 | 10,9 | 7,8 | −0,004 | 0,007 | 0,007 | 0,037 | 84,9% | 0,615 |
| T2 | 33,8 | 12,3 | 8,0 | 0,067 | 0,011 | 0,007 | 0,032 | 84,9% | 0,617 |
| **T2S** (I3) | 32,1 | 10,9 | 7,8 | **0,133** | 0,002 | 0,005 | 0,036 | 85,0% | 0,617 |
| T3 (I1, com capital) | 32,1 | 12,3 | 6,1 | −0,005 | **0,095** | 0,060 | 0,126 | 89,2% | **0,581** ⚠ |
| T3 (plano) | 33,5 | 13,8 | 6,1 | −0,005 | 0,077 | 0,048 | 0,132 | 87,0% | 0,592 |
| **T3C** (I1, sem capital) | 32,4 | 11,2 | 7,8 | −0,005 | 0,085 | 0,056 | 0,126 | 90,5% | 0,607 |
| T4 | 32,1 | 11,2 | 7,3 | — | — | 0,015 | — | 95,2% | 0,617 |
| T4 + T4R | 31,9 | 11,2 | 7,2 | — | — | 0,015 | — | **100%** | 0,617 |
| T5 | 33,2 | 11,4 | 8,0 | — | — | 0,014 | — | 85,0% | 0,613 |
| FC | 32,1 | 10,8 | 7,8 | — | — | — | — | 84,9% | 0,620 |
| T2+T3 | 33,5 | 13,8 | 6,2 | 0,067 | 0,082 | 0,055 | 0,137 | 89,2% | 0,588 |
| T2+T3+T4+T4R+T5 | 34,3 | 14,1 | 6,3 | 0,067 | 0,075 | 0,056 | 0,140 | 100% | 0,582 ⚠ |
| T2..T5 (plano) | 36,2 | 16,2 | 6,3 | 0,067 | 0,068 | 0,051 | 0,135 | 100% | 0,603 |
| T2S+T3C+T4+T4R+T5 | 33,1 | 11,4 | 7,8 | 0,133 | 0,048 | 0,047 | 0,131 | 100% | 0,603 |
| **FC+T2S+T3C+T4+T4R+T5** | **33,0** | 11,3 | 8,0 | **0,134** | 0,050 | **0,081** | 0,132 | **100%** | 0,608 |

⚠ = reprovado no gate de Column Pair Trends (≥ baseline − 2 p.p.).

Desvio padrão de tempo em 3 repetições: ≤ 0,45 s.

**T3 com `capital_social` reprova o gate.** O `capital_social` sintético do pai sai mal modelado: o marginal Beta erra até os quartis (p75 de 215 mil contra 10 mil no real). Condicionar os filhos nele **propaga o erro do pai** para as correlações internas do filho. Sem capital (T3C), o Column Pair Trends fica em −0,8 p.p. e quase todo o ganho de associação se mantém.

**Por que a T2S e a T3 juntas perdem um pouco de V porte~MEI** (0,095 → 0,050): com a T2S, a presença do registro em `simples` passa a seguir o porte (a taxa por porte bate exatamente com o real: 0,91/0,76/0,04). Assim a associação aparece no **JOIN** (quem tem Simples), e a T3 explica menos *dentro* dos registros existentes. Na métrica direta de presença, a T2S acerta 100%.

### 5.2 Semelhança entre irmãos (média por empresa)

| Configuração | Mesmo CNAE | Mesma UF | Mesmo município | Mesma situação |
|---|---|---|---|---|
| **Real** | 0,680 | 0,863 | 0,531 | 0,755 |
| baseline | 0,021 | 0,135 | 0,011 | 0,393 |
| T5 (plano: efeito latente) | 0,035 | 0,137 | 0,018 | 0,477 |
| T2..T5 (plano) | 0,038 | 0,157 | 0,017 | 0,556 |
| **T5 (I2: cópia)** | **0,640** | **0,841** | **0,491** | **0,728** |
| T2S+T3C+T4+T4R+T5 | 0,649 | 0,840 | 0,514 | 0,732 |
| FC+T2S+T3C+T4+T4R+T5 | 0,637 | 0,854 | 0,835 ⚠ | 0,712 |

⚠ Com `FixedCombinations`, `municipio` é copiado sempre que `uf` é copiada, para manter o par válido. Isso deixa os irmãos parecidos demais no município (0,835 contra 0,531). É o custo de ter 100% de consistência UF↔município.

---

## 6. Resultados: variante `full` (10 tabelas) e T1

0,1%, 3 repetições:

| Configuração | Wall (s) | Consistência UF↔município | Mesmo CNAE (pares) | V porte~MEI | Column Pair Trends |
|---|---|---|---|---|---|
| full: baseline | 28,8 | **0,0%** | 0,017 | 0,008 | 0,638 |
| full: T1 | 32,4 | 12,9% | 0,016 | 0,009 | 0,636 |
| full: T1..T5 | 35,3 | 14,1% | 0,564 | 0,124 | 0,605 |

- **Sem a T1, as tabelas de domínio são sintetizadas** com códigos fictícios: nenhum par (UF, município) existe no real. Com a T1, a variante `full` se comporta como a `core`.
- **A meta do plano para a T1 (≥ 99%) não é atingível com a T1.** A cópula do filho não aprende a dependência funcional município → UF dentro da tabela, e o resultado fica em 12,9%. **A solução é `FixedCombinations(['uf', 'municipio'])`, que leva a 100%** (seção 5.1, linha FC) sem custo de tempo mensurável.
- A T1 custa +12% de tempo (3,6 s), por causa do encoder categórico de 5.572 municípios no lugar de uma FK. A estimativa do plano era de redução de custo; ela não se confirmou.

---

## 7. Comparação com o HMA (0,01%, 30 mil linhas)

O HMA, o baseline e T2..T5 têm 1 execução cada. A configuração recomendada (FC+T2S+T3C+T4+T4R+T5, código final) tem 3, com métricas de qualidade idênticas porque a seed é fixa.

| Métrica | Real | Independent baseline | Independent T2..T5 | **Independent recomendado** | HMA |
|---|---|---|---|---|---|
| Tempo total | — | 13,2 s | 11,4 s | **12,4 s** | 419,4 s (34× o recomendado) |
| Pico de memória | — | 1,01 GB | 1,03 GB | 1,03 GB | 1,27 GB |
| KS filiais/empresa · sócios/empresa | 0 | 0 · 0 | 0 · 0 | 0 · 0,002 | 0,028 · 0,141 |
| Máximo de filiais por empresa | 7.903 | 7.903 | 7.903 | 7.903 | 211 |
| 1 matriz por empresa | 100% | 45,9% | 100% | 100% | 81,1% |
| Par (UF, município) válido | 100% | 11,0% | 11,7% | **100%** | 12,1% |
| Spearman porte~nº sócios | 0,143 | −0,001 | 0,074 | **0,146** | 0,086 |
| Spearman capital~nº sócios | 0,284 | 0,005 | **0,056** | −0,041 | 0,045 |
| V porte~MEI | 0,152 | 0,013 | **0,101** | 0,037 | 0,038 |
| V porte~situação (estab.) | 0,183 | 0,011 | 0,061 | **0,196** | 0,066 |
| V natureza~situação (estab.) | 0,261 | 0,034 | 0,100 | **0,264** | 0,101 |
| V natureza~identificador sócio | 0,109 | 0,067 | 0,049 | 0,047 | **0,102** |
| V natureza~qualif. sócio | 0,616 | 0,070 | **0,279** | 0,277 | 0,081 |
| Mesmo CNAE entre filiais (pares) | 0,694 | 0,184 | **0,618** | 0,392 | 0,497 |
| Mesmo CNAE entre filiais (média por empresa) | 0,676 | — | — | 0,563 | — |
| sdmetrics Column Shapes | — | 0,743 | 0,746 | **0,754** | 0,664 |
| sdmetrics Column Pair Trends | — | 0,573 | **0,609** | 0,606 | 0,550 |
| sdmetrics Intertable Trends | — | 0,622 | **0,676** | 0,662 | 0,605 |

Na escala em que o HMA ainda roda, a configuração recomendada:
- **supera o HMA em 4 de 7 associações**, três delas praticamente iguais ao real: natureza~situação 0,264 contra 0,261; porte~situação 0,196 contra 0,183; porte~nº de sócios 0,146 contra 0,143;
- **perde em 2**: natureza~identificador do sócio e capital~nº de sócios;
- **empata em 1**: porte~MEI;
- vence nos 3 agregados do sdmetrics e mantém a cauda exata da cardinalidade (7.903 contra 211);
- leva o par (UF, município) a 100%.

As configurações T2..T5 e recomendada se complementam:
- **T2..T5** acerta melhor porte~MEI (associação dentro dos registros do Simples) e o CNAE entre filiais medido por pares. A métrica por pares é dominada pela empresa com 7.903 filiais (52% dos estabelecimentos nesta escala).
- **Recomendada** acerta as associações com a situação do estabelecimento. A cópia entre filiais é calibrada por empresa, o que favorece a métrica por empresa, e não a métrica por pares.

O HMA tem 1 execução nos parâmetros padrão, e a comparação vale só para esta escala (§7 de [`REVISAO.md`](REVISAO.md)).

---

## 8. Escala e gates de não regressão

| Fração (linhas) | Baseline | T2..T5 (antes da otimização de memória) | T2..T5 (depois) | Recomendado (FC+T2S+T3C+T4+T4R+T5) |
|---|---|---|---|---|
| 0,5% (1,1 M) | 131 s · 2,66 GB | 145 s (+10%) · 2,69 GB (+1%) | — | 132 s (**+0,2%**) · 2,82 GB (+6%) |
| 1% (2,2 M) | 256 s · 3,86 GB | 286 s (+12%) · 4,63 GB (+20%) | 286 s (+12%) · 4,40 GB (**+14%**) | 259 s (**+1,1%**) · 4,60 GB (+19%) |
| 2,5% (5,6 M) | 619 s · 7,47 GB | 699 s (+13%) · 9,11 GB (+22%) | 698 s (+13%) · 9,29 GB (+24%) | 632 s (**+2,0%**) · 9,18 GB (+23%) |

| Gate (seção 8.2 do plano) | Resultado |
|---|---|
| Integridade referencial = 100% | ✅ em todas as 84 execuções do CNPJ e 39 do dado sintético |
| KS filhos por pai ≤ 0,005 | ✅ 0,000 a 0,001 (a T2S reamostra por estrato, então não é exatamente 0) |
| Column Shapes / Column Pair Trends ≥ baseline − 2 p.p. | ✅ recomendado (−0,7 p.p.) · ❌ T3 com `capital_social` (−3,4 p.p.) |
| Tempo ≤ +20% | ✅ recomendado +0,2% a +4%; T2..T5 +10% a +14% |
| Memória ≤ +15% | ✅ até 0,5% · ❌ +19% a +24% em 1% e 2,5% |

**Item aberto (memória).** Em 2,5%, o excedente se divide entre:
- **preprocess**, +0,6 a 0,9 GB: taxas de cópia calculadas coluna a coluna com `DataFrame`s de objetos, e cópias dos valores brutos de contexto;
- **fit**, +0,7 GB: colunas de contexto e dummies na tabela processada do filho;
- **sample**, +1 GB.

Próximos passos sugeridos:
- calcular as taxas de cópia com códigos inteiros (`factorize`) em vez de objetos;
- montar as dummies em `float32`;
- liberar as tabelas processadas de cada tabela assim que os filhos forem gerados.

O modo streaming do plano de otimização (O7) resolve isso de forma estrutural.

---

## 9. Achados colaterais (fora do escopo, mas relevantes)

1. **A branch atual não é reproduzível entre processos.** Com a mesma seed, o resultado muda de uma execução para outra: `_connect_tables` percorre os filhos de um `set` (`Metadata._get_child_map`), cuja ordem depende da randomização de hash de strings. Com `PYTHONHASHSEED=0` a saída fica estável. O novo caminho condicional usa uma ordem determinística (`_get_sampling_order`). Correção sugerida para o caminho padrão: ordenar os filhos.
2. **O marginal Beta é inadequado para `capital_social`.** A cauda vai até 1,2×10¹¹ e há massa em zero. O `LogScaler` também não resolveu, nem o marginal nem a correlação. É o principal limitador da T2 e da T3 com capital no CNPJ, e merece um estudo próprio (`gaussian_kde` ou transformação por quantis).
3. **A semelhança entre irmãos medida por pares é dominada pela empresa com 7.903 filiais.** O real dá UF igual em só 9,5% dos pares, contra 86% na média por empresa. As métricas por empresa estão em `sibling_same_per_empresa_*`.

---

## 10. Recomendações

1. **Adotar no `IndependentSynthesizer`**: T2S (`cardinality_by` com poucas colunas categóricas bem modeladas no pai), T3 com contexto **categórico** (`categorical_context='one_hot'`), T4 com `group_rules` e T5 com `sibling_categorical='copy'`. Documentar o uso de `FixedCombinations` para pares hierárquicos (UF/município).
2. **Não adotar** como estão no plano: o contexto categórico via `UniformEncoder` (instável) e o efeito latente para colunas categóricas (sem efeito). Revisar as metas da T1 (≥ 99% de UF↔município) e da T4 só com posição (≥ 99%).
3. **Evitar como contexto** colunas que o pai modela mal (no CNPJ, `capital_social`) e colunas redundantes entre si (no dado sintético, `segment` junto com `size`).
4. **Antes de levar para `main`:** resolver o gate de memória (seção 8) e decidir se a T2 por cópula fica. Ela é dominada pela T2S no CNPJ, mas a cópula ainda ordena as contagens *dentro* dos estratos quando as duas estão ligadas.

---

## 11. Reprodução

```bash
# testes
.venv/bin/python -m pytest tests/unit/multi_table/test_independent_correlation.py \
    tests/integration/multi_table/test_independent_correlation.py

# dados com correlação conhecida (~75 s)
.venv/bin/python benchmarks/correlation/synthetic.py --num-parents 20000 --seeds 0 1 2

# CNPJ (~3 h no total)
benchmarks/correlation/run_cnpj_campaign.sh ~/.cache/sdv-cnpj/2026-09/sample_0.05
benchmarks/correlation/run_cnpj_campaign_e.sh ~/.cache/sdv-cnpj/2026-09/sample_0.05

# uma configuração específica
.venv/bin/python benchmarks/cnpj/run.py --sample-dir ~/.cache/sdv-cnpj/2026-09/sample_0.05 \
    --variant core --fractions 0.001 --correlation FC T2S T3C T4 T4R T5 --sdmetrics

# tabelas deste documento
.venv/bin/python benchmarks/correlation/summarize.py benchmarks/correlation/results_cnpj.jsonl \
    --group label --query "fraction==0.001" --columns wall_seconds pct_empresas_one_matriz_synthetic
```

Os resultados brutos estão em `benchmarks/correlation/results_synthetic.jsonl` e `results_cnpj.jsonl` (uma linha JSON por execução), e os logs em `campaign_cnpj*.log` e `synthetic.log`.

**Atenção à cronologia:** as execuções da campanha principal com T5 usaram a calibração de cópia por linha, que veio antes da calibração por pai (I2/I4) e da otimização de memória. As linhas com prefixo `E:`, `best:`, `scale:best` e `(mem-opt)` usam o código final. As métricas por empresa (seção 5.2) existem só nessas execuções.
