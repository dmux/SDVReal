# Revisão técnica: síntese multi-tabela com HMA, `IndependentSynthesizer` e `IndependentSynthesizer` com correlação entre tabelas

> **Para quem:** especialista em ML / dados sintéticos, convidado a emitir um juízo de valor independente.
> **Pergunta central:** a comparação entre o `HMASynthesizer` padrão do SDV, o `IndependentSynthesizer` (v1) e o `IndependentSynthesizer` com correlação entre tabelas (v2) **faz sentido**? E os métodos **cumprem o propósito** de gerar dados sintéticos **representativos**, com **distribuições semelhantes às reais**?
> **Repositório:** fork `dmux/SDVReal`, branch `experiment/inter-table-correlation` (a partir de `feature/independent-multi-table-synthesizer`). Issue #1.
> **Documentos de origem:** [`BENCHMARK.pt-BR.md`](BENCHMARK.pt-BR.md) (v1 × HMA) · [`PLANO-CORRELACAO-INTERTABELAS.pt-BR.md`](PLANO-CORRELACAO-INTERTABELAS.pt-BR.md) (plano da v2) · [`EXPERIMENTO-CORRELACAO.pt-BR.md`](EXPERIMENTO-CORRELACAO.pt-BR.md) (resultados da v2) · [`CONCEITOS-SDV.pt-BR.md`](CONCEITOS-SDV.pt-BR.md).
> **Data:** 2026-10-09.

Este documento foi escrito por quem implementou os métodos. Por isso a seção 7 lista, de forma deliberadamente crítica, os pontos em que a avaliação pode estar enviesada. Pedimos ao revisor que a leia antes de aceitar qualquer conclusão das seções 5 e 6.

---

## Sumário

1. [O que se pede ao revisor](#1-o-que-se-pede-ao-revisor)
2. [O problema: síntese de dados relacionais](#2-o-problema-síntese-de-dados-relacionais)
3. [Evolução dos métodos](#3-evolução-dos-métodos)
4. [Desenho experimental](#4-desenho-experimental)
5. [Resultados](#5-resultados)
6. [Leitura dos resultados pelos autores](#6-leitura-dos-resultados-pelos-autores)
7. [Ameaças à validade (autocrítica)](#7-ameaças-à-validade-autocrítica)
8. [Avaliações que faltam](#8-avaliações-que-faltam)
9. [Questionário de revisão](#9-questionário-de-revisão)
10. [Artefatos e reprodução](#10-artefatos-e-reprodução)

---

## 1. O que se pede ao revisor

Responder, com justificativa, a quatro perguntas:

| # | Pergunta | Onde olhar |
|---|---|---|
| Q1 | **A comparação é justa?** As três abordagens são comparadas nas mesmas condições, com métricas que não favorecem nenhuma delas indevidamente? | §4, §5.1, §7.1, §7.2 |
| Q2 | **As métricas medem "representatividade"?** Cobrem os aspectos que importam (marginais, dependências intra e entre tabelas, cardinalidade, regras de grupo) e são sensíveis o bastante? | §2.2, §4.3, §7.3 |
| Q3 | **Os resultados sustentam as conclusões?** Em especial: (a) a v2 é melhor que a v1; (b) a v2 é melhor que o HMA; (c) a v2 é escalável. | §5, §6, §7 |
| Q4 | **O propósito é cumprido?** Os dados da v2 servem como substitutos estatísticos dos reais para análise, testes e treino de modelos? Em que condições, e com quais riscos? | §6, §7.5, §8 |

Um questionário estruturado está na §9.

---

## 2. O problema: síntese de dados relacionais

### 2.1 Definição

Um banco relacional é um conjunto de tabelas $T_1, \dots, T_k$ ligadas por chaves estrangeiras (FK), que formam um grafo (geralmente acíclico) de relações pai → filho. Sintetizar o banco significa gerar $\tilde T_1, \dots, \tilde T_k$ tal que a distribuição conjunta **do banco inteiro** se pareça com a real, e não apenas a de cada tabela.

No caso de uma relação pai $P$ → filho $C$, a distribuição a reproduzir é

$$
p(\text{banco}) = p(P)\;\prod_{i \in P} \Big[\, p(n_i \mid P_i)\; p(C_{i,1}, \dots, C_{i,n_i} \mid P_i, n_i) \Big]
$$

em que $n_i$ é o número de filhos do pai $i$ e $C_{i,j}$ é o $j$-ésimo filho. Para uma cadeia mais profunda, o termo do filho se repete recursivamente com os netos.

### 2.2 O que "representativo" significa aqui

A fidelidade de um banco sintético é avaliada em camadas. A tabela abaixo é a taxonomia usada neste trabalho; o revisor pode questioná-la (Q2).

| Camada | Exemplo no CNPJ | Termo da fórmula acima |
|---|---|---|
| L1. Marginais | distribuição de `capital_social`, de `uf` | $p(P)$, $p(C)$ por coluna |
| L2. Dependências intra-tabela | `porte_empresa` × `natureza_juridica` | $p(P)$ conjunta |
| L3. Cardinalidade | nº de estabelecimentos por empresa (cauda: uma empresa com 7.903) | $p(n_i)$ |
| L4. Cardinalidade condicionada ao pai | empresas maiores têm mais sócios | $p(n_i \mid P_i)$ |
| L5. Dependência pai → filho | porte da empresa × situação cadastral da filial | $p(C_{ij} \mid P_i)$ |
| L6. Dependência entre irmãos | filiais da mesma empresa tendem a ter o mesmo CNAE e a mesma UF | $p(C_{i,1}, \dots, C_{i,n_i})$ não fatorável |
| L7. Regras de grupo | exatamente 1 matriz por empresa | restrição sobre $(C_{i,1}, \dots, C_{i,n_i})$ |
| L8. Integridade referencial | toda FK aponta para uma PK existente | restrição dura |
| L9. Restrições de domínio | (UF, município) válido | restrição intra-linha |

Ficam **fora** desta avaliação, de propósito (ver §8):
- **utilidade** para ML (treinar no sintético e testar no real, TSTR);
- **privacidade**: distância ao registro real mais próximo e risco de reidentificação.

### 2.3 Por que é difícil

- **Explosão combinatória:** modelar $p(C_{i,1..n_i} \mid P_i)$ para $n_i$ variável, com $n_i$ podendo chegar a milhares.
- **Profundidade:** dependências que atravessam vários níveis (avô → neto).
- **Escala:** bancos reais têm milhões de linhas. Qualquer custo superlinear em linhas ou em pais inviabiliza o método.
- **Caudas pesadas:** cardinalidade e valores monetários com distribuição de lei de potência.

---

## 3. Evolução dos métodos

### 3.1 Linha do tempo

| Etapa | Método | Onde | Ideia |
|---|---|---|---|
| 2016 | **HMA**, Hierarchical Modeling Algorithm (Patki, Wedge, Veeramachaneni, *The Synthetic Data Vault*, IEEE DSAA 2016) | SDV Community, `sdv/multi_table/hma.py` | Agrega recursivamente, nas linhas do pai, os parâmetros de modelos ajustados nos filhos |
| 2026-07/08 | Trava no HMA: erro com mais de 5 tabelas ou profundidade maior que 2 (commits `43310d3a`, `adb52dd8` do upstream) | `hma.py:261` | Direciona schemas maiores para o SDV Enterprise |
| 2026-10 | **`IndependentSynthesizer` v1** | este fork, `sdv/multi_table/independent.py` | Um modelo por tabela, sem dependência entre tabelas; só a cardinalidade é preservada |
| 2026-10 | v1 corrigida com dados reais | idem, commit `dd681257` | Reamostragem sem reposição das contagens reais (preserva a cauda exata) e correção de dtype de FK nula |
| 2026-10 | **`IndependentSynthesizer` v2** (este experimento) | idem + `_conditional.py`, `_group_features.py` | Recupera dependências entre tabelas (L4 a L7, L9) com colunas extras e álgebra linear vetorizada, mantendo um modelo por tabela |

### 3.2 HMA

**Mecanismo.**
- **Fit:** para cada relação pai → filho e para **cada linha do pai**, ajusta uma cópula gaussiana nas linhas filhas daquele pai. Os parâmetros desse modelo viram **colunas extras** da linha do pai: número de filhos, parâmetros das marginais e o triângulo da matriz de correlação. Em seguida, o pai, já estendido, é modelado por outra cópula gaussiana. Com mais níveis, isso se repete recursivamente: os parâmetros dos netos sobem para o filho, e os do filho para o avô.
- **Sample:** gera as linhas do pai, incluindo as colunas de parâmetros. Para **cada** linha do pai, reconstrói uma cópula de filho com aqueles parâmetros e gera os filhos dela.

**Fatoração implícita:**

$$
p(P, \theta) \text{ via cópula do pai estendido}, \qquad C_{i,\cdot} \sim \text{Cópula}(\theta_i),\ \ n_i \text{ parte de } \theta_i
$$

**O que captura bem:**
- L4 e L5, via $\theta_i$ condicionado a $P_i$;
- parte de L6: filhos do mesmo pai vêm do mesmo modelo, então compartilham médias e correlações.

**Limitações:**

| Limitação | Causa | Evidência |
|---|---|---|
| Custo do fit proporcional ao nº de pais | Um modelo por linha do pai (`_get_extension`, `hma.py:304`) | Fit 23× mais lento que a v1 com 30 mil linhas |
| Custo do sampling proporcional ao nº de pais | Recria um synthesizer por pai, com laço em Python (`_recreate_child_synthesizer`, `_sample_children`) | Sample de 389 a 776 s com 30 mil linhas; timeout de 30 min com 119 mil |
| Explosão de colunas | O triângulo de correlação do filho, $m(m-1)/2$ colunas, sobe para o pai e se multiplica com a profundidade | Limite interno `MAX_NUMBER_OF_COLUMNS = 1000` |
| Modelos por pai estimados com poucas linhas | A maioria das empresas tem 1 filial: estimar uma correlação a partir de 1 linha é degenerado | Comportamento conhecido; não foi medido isoladamente aqui |
| Perda da cauda de cardinalidade | O nº de filhos é uma coluna contínua da cópula do pai, com marginal paramétrica | Máximo de filiais por empresa: 7.903 no real, 127 a 211 no HMA; KS de sócios por empresa = 0,14 |
| Trava de schema | Mais de 5 tabelas ou profundidade maior que 2 gera erro | A variante `full` do CNPJ (10 tabelas) é recusada |

### 3.3 `IndependentSynthesizer` v1

**Mecanismo:**
- uma cópula gaussiana (synthesizer padrão do SDV) **por tabela**, sem as colunas de FK;
- para cada relação, guarda a distribuição empírica do nº de filhos por pai e a taxa de FK nula;
- no sampling, gera cada tabela de forma independente e depois atribui as FKs: as contagens reais são reamostradas **sem reposição** e distribuídas **aleatoriamente** entre os pais sintéticos.

**Fatoração implícita:**

$$
p(P)\;\cdot\;p(n)\;\cdot\;\prod_{ij} p(C_{ij}), \quad \text{com } n \perp P \text{ e } C \perp (P, n, \text{irmãos})
$$

| Camada | v1 |
|---|---|
| L1, L2 | ✅ dentro dos limites da cópula gaussiana |
| L3 | ✅ **exata** (KS = 0, cauda preservada) |
| L4 a L7 | ❌ por construção |
| L8 | ✅ |
| Escala | linear em linhas e tabelas; sem limite de profundidade |

### 3.4 `IndependentSynthesizer` v2 (correlação entre tabelas)

O princípio é **inverter a ordem do sampling**: os pais são gerados primeiro, as FKs são definidas **antes** de gerar os filhos, e cada filho é gerado **condicionado** ao seu próprio pai sintético. Tudo é feito com colunas extras no modelo de cada tabela e com álgebra linear vetorizada. Continua havendo **um modelo por tabela**.

Componentes (todos opt-in; o comportamento padrão é idêntico à v1):

| Técnica | Camada | Formalização |
|---|---|---|
| **T1** `lookup_tables` | L9 (parcial) | Tabelas de domínio (municípios, CNAEs…) são copiadas do real; a FK do filho vira uma coluna categórica do modelo do filho |
| **T2** `model_cardinality` | L4 | Adiciona ao pai $v_i = \log(1 + n_i + U)$, com $U \sim \mathcal U(0,1)$. A cópula do pai aprende $\text{corr}(v, P)$. No sampling, as contagens reais reamostradas são atribuídas **por posto** de $\tilde v$: `counts[argsort(v)] = sort(reais)`. A distribuição das contagens é exatamente a real, e a ordem vem da cópula |
| **I3/T2S** `cardinality_by` | L4 | Contagens reais guardadas **por estrato** do pai (categorias; faixas de quantil para numéricas) e reamostradas dentro do estrato do pai sintético |
| **T3** `context_columns` | L5 | Colunas do pai são unidas ao filho no fit. No sampling, com a correlação latente particionada em próprias (1) e contexto (2): $Z_1 = Z_2 A^\top + E L^\top$, com $A = \Sigma_{12}\Sigma_{22}^{-1}$, $LL^\top = \Sigma_{11} - A\Sigma_{21}$, $E \sim \mathcal N(0, I)$. Uma multiplicação de matrizes $n \times k$, condicionando **cada linha ao seu pai** |
| **I1** `categorical_context='one_hot'` | L5 | Colunas de contexto categóricas entram como indicadores padronizados **fora** da cópula, com uma correlação estendida calculada à parte (regressão linear exata nas dummies) |
| **T4** `sibling_order` + `group_rules` | L7 | A posição do filho entre os irmãos entra como contexto (dummies). A regra rígida "1º filho = matriz" é aplicada como pós-processamento |
| **T5** `sibling_correlation` | L6 (numéricas) | Efeito aleatório por pai: $Z_1 = \mu + \sqrt{\rho}\odot(u_{\text{pai}}L^\top) + \sqrt{1-\rho}\odot(e_{\text{linha}}L^\top)$. $\rho$ é a correlação intraclasse dos resíduos, estimada por ANOVA de um fator |
| **I2** `sibling_categorical='copy'` | L6 (categóricas) | Cada filho que não é o 1º copia o valor do 1º irmão com probabilidade $q = (\pi_{\text{real}} - \pi_{\text{sint}}) / (1 - \pi_{\text{sint}})$, em que $\pi$ é a taxa de "mesmo valor que o 1º irmão" **média por pai**. Um sorteio por linha é compartilhado entre as colunas (cópia comonotônica) |
| **I4** `FixedCombinations` + cópia em grupo | L9 | Constraint do SDV que restringe (UF, município) a combinações reais; colunas da mesma constraint são copiadas juntas entre irmãos |

T1 a T5 vêm do plano original; I1 a I4 foram introduzidas porque o plano, medido, falhou nesses pontos (§5.3).

**Fatoração implícita da v2:**

$$
p(P, v) \;\cdot\; p(n \mid \text{estrato}(P), \text{posto}(v)) \;\cdot\; \prod_{ij} p\big(C_{ij} \mid \text{ctx}(P_i), \text{pos}_{ij}, u_i\big) \;\circ\; \text{cópia}(C_{i1}) \;\circ\; \text{regras}
$$

**O que a v2 ainda não modela:**
- dependências não monotônicas entre colunas numéricas (limite da cópula gaussiana);
- contexto de mais de um pai (só o pai "primário" de cada filho);
- contexto do avô para o neto, a não ser pelo que o pai já carrega;
- agregados dos filhos que afetam o pai (por exemplo, "% de filiais ativas" como atributo da empresa);
- restrições com rejeição (reject sampling) no caminho condicional, que lança erro.

### 3.5 Quadro comparativo conceitual

| | HMA | v1 | v2 |
|---|---|---|---|
| Modelos | 1 por **linha** do pai, por relação + 1 por tabela | 1 por tabela | 1 por tabela |
| Custo | ≈ O(nº de pais × custo de um modelo pequeno), em laço Python | O(linhas), vetorizado | O(linhas), vetorizado |
| Profundidade / nº de tabelas | ≤ 2 / ≤ 5 (trava) | ilimitado | ilimitado |
| L3 cardinalidade | aproximada; perde a cauda | **exata** | exata (T2), quase exata (T2S) |
| L4 cardinalidade × pai | ✅ via $\theta_i$ | ❌ | ✅ |
| L5 pai → filho | ✅ via $\theta_i$ | ❌ | ✅ (linear no latente) |
| L6 irmãos | parcial (mesmo $\theta_i$) | ❌ | ✅ numéricas (ICC); categóricas por cópia |
| L7 regras de grupo | aproximada | ❌ | ✅ exata, mas como regra **declarada** |
| L9 domínio | só com constraints | só com constraints | só com constraints (integrado à cópia) |
| Conhecimento de domínio exigido | nenhum | nenhum | **escolha de contexto, estratos e regras** |

---

## 4. Desenho experimental

### 4.1 Dados

| Conjunto | Descrição | Por que |
|---|---|---|
| **Sintético com correlação conhecida** (`tests/utils.py::generate_correlated_parent_child`) | 20 mil pais, cerca de 27 mil filhos, 3 seeds. `size` (log-normal) → nº de filhos e `amount`; `tier` (categórica, independente de `size`) → `discount`, com efeito **não monotônico**; o 1º filho é `HQ`; irmãos compartilham um efeito em `score` e, em 80% dos casos, a `region` | Verdade conhecida para cada camada L4 a L7 |
| **CNPJ** (Receita Federal, release 2026-09) | Base completa: 70 M de empresas, 73 M de estabelecimentos, 28 M de sócios e 50 M de registros do Simples. Amostra **por empresa** (`int(cnpj_basico) % 10000 < fração × 10000`) com todos os descendentes | Dados reais, schema real, cauda extrema (uma empresa com 7.903 estabelecimentos), PII gerada por Faker |

Variantes do schema CNPJ:
- `core`: 4 tabelas, profundidade 2, códigos de domínio como colunas categóricas. É a única aceita pelo HMA.
- `full`: 10 tabelas, profundidade 3, domínios como tabelas-pai. É recusada pelo HMA.

### 4.2 Configurações comparadas

| Rótulo | Método |
|---|---|
| HMA | `HMASynthesizer` com parâmetros padrão (marginais `beta`) |
| v1 / baseline | `IndependentSynthesizer` sem opções |
| v2 (plano) | T2 + T3 + T4 + T4R + T5 com as codificações do plano original |
| v2 (melhorada) | T2 + T3 + T4 + T4R + T5 com I1 e I2 |
| **v2 (recomendada)** | FC + T2S + T3C + T4 + T4R + T5 (T3C = contexto sem `capital_social`) |
| técnicas isoladas e acumuladas | para atribuir o ganho e o custo de cada uma |

### 4.3 Métricas

| Métrica | Camada | Definição | Limitações conhecidas |
|---|---|---|---|
| KS de filhos por pai | L3 | Estatística KS entre as distribuições de contagem real e sintética | Insensível a qual pai recebe quantos filhos |
| Máximo de filhos por pai | L3 | Valor máximo | Um único número; mede a cauda |
| Spearman(atributo do pai, nº de filhos) | L4 | Por pai, incluindo zeros | Muitos empates (maioria com 0 ou 1 filho) |
| Cramér's V (coluna do pai, coluna do filho) | L5 | Sobre o JOIN pai-filho; comparado em **magnitude** com o real | Não verifica se o **padrão** da associação é o mesmo, só a intensidade |
| Similaridade de contingência (1 − TVD) | L5 | Distribuição conjunta do par, real × sintético | Dominada pelas categorias frequentes e pelo pai gigante em frações pequenas |
| Semelhança entre irmãos, média por empresa | L6 | Para empresas com 2 ou mais filiais: fração de pares de filiais com o mesmo valor, média entre empresas | **A mesma grandeza que a I2 calibra** (ver §7.2) |
| 1 matriz por empresa | L7 | % de empresas com exatamente uma filial `identificador_matriz_filial == 1` | Com `group_rules` é 100% **por construção** |
| UF↔município consistente | L9 | % de linhas sintéticas cujo par (UF, município) existe no real | Com `FixedCombinations` é 100% **por construção** |
| sdmetrics `QualityReport` | L1, L2, L3, L5 | Column Shapes, Column Pair Trends, Cardinality, Intertable Trends, em uma subamostra de 2.000 empresas e seus descendentes | Ruído de ±0,02 entre repetições; o Intertable Trends faz a média de muitos pares sem relação real e por isso discrimina pouco |
| Integridade | L8 | `metadata.validate_data(sintético)` | Binária |
| Tempo por fase, pico de memória | custo | Subprocesso isolado; footprint físico no macOS (inclui páginas comprimidas) | Uma máquina, sem isolamento total de carga |

### 4.4 Protocolo

- Cada configuração roda em um subprocesso. No CNPJ a 0,1%, há **3 repetições**.
- O sampling usa seed fixa. Por isso, **as métricas de qualidade são idênticas entre repetições**: as repetições medem apenas a variância de tempo e a da subamostra do sdmetrics, **não a variância amostral do gerador** (ver §7.4).
- O dado sintético com correlação conhecida usa 3 seeds de **dados**, que geram variância real.
- **HMA:** uma execução a 0,01% (30 mil linhas) nesta campanha, mais uma execução anterior na mesma fração, registrada em `BENCHMARK.pt-BR.md`. Em frações maiores o HMA não termina dentro de 30 minutos.

---

## 5. Resultados

### 5.1 Três vias: HMA × v1 × v2, CNPJ `core`, 0,01% (30 mil linhas)

Esta é a única escala em que o HMA roda (1 execução cada). A coluna v2 usa T2 + T3 + T4 + T4R + T5, melhorada, com a calibração anterior à I2/I4.

| Métrica | Real | HMA | v1 | v2 |
|---|---|---|---|---|
| Tempo total | — | 419 s¹ | 13,2 s | **11,4 s** |
| Pico de memória | — | 1,27 GB | 1,01 GB | 1,03 GB |
| Integridade | — | ✅ | ✅ | ✅ |
| KS estabelecimentos / empresa | 0 | 0,028 | **0,000** | **0,000** |
| KS sócios / empresa | 0 | 0,141 | **0,000** | **0,000** |
| Máximo de estabelecimentos por empresa | 7.903 | 211 | **7.903** | **7.903** |
| 1 matriz por empresa | 100% | 81,1% | 45,9% | **100%**² |
| Spearman(porte, nº de sócios) | 0,143 | **0,086** | −0,001 | 0,074 |
| V porte ~ `simples.opcao_mei` | 0,152 | 0,038 | 0,013 | **0,101** |
| V porte ~ `estab.situacao_cadastral` | 0,183 | **0,066** | 0,011 | 0,061 |
| V natureza ~ `estab.situacao_cadastral` | 0,261 | **0,101** | 0,034 | 0,100 |
| V natureza ~ `socios.identificador_socio` | 0,109 | **0,102** | 0,067 | 0,049 |
| V natureza ~ `socios.qualificacao_socio` | 0,616 | 0,081 | 0,070 | **0,279** |
| Mesmo CNAE entre filiais (pares) | 0,694 | 0,497 | 0,184 | **0,618** |
| sdmetrics Column Shapes | — | 0,664 | 0,743 | **0,746** |
| sdmetrics Column Pair Trends | — | 0,550 | 0,573 | **0,609** |
| sdmetrics Intertable Trends | — | 0,605 | 0,622 | **0,676** |

¹ Em uma execução anterior com os mesmos dados, o HMA levou 833 s (fit 42,7 s, sample 776 s) e chegou a 79,5% em "1 matriz" e 127 de máximo. A variação de tempo e de resultado entre execuções do HMA não foi investigada.
² Por regra declarada (`group_rules`), e não por modelagem.

### 5.2 v1 × v2, CNPJ `core`, 0,1% (230 mil linhas, 3 repetições)

| Métrica | Real | v1 | v2 (plano) | v2 (recomendada) |
|---|---|---|---|---|
| Tempo total | — | 31,8 ± 0,2 s | 36,2 s (+14%) | 33,0 s (+4%) |
| KS estab. / sócios | 0 | 0,000 / 0,000 | 0,000 / 0,000 | 0,000 / 0,001 |
| 1 matriz | 100% | 84,9% | 100%² | 100%² |
| UF↔município consistente | 100% | 12,9% | 13,4% | 100%³ |
| Spearman(porte, nº de sócios) | 0,131 | −0,004 | 0,067 | **0,134** |
| Spearman(capital, nº de sócios) | 0,292 | 0,004 | 0,050 | **−0,058** ⚠ |
| V porte ~ MEI | 0,157 | 0,007 | 0,068 | 0,050 |
| V porte ~ situação (estab.) | 0,133 | 0,007 | 0,051 | 0,081 |
| V natureza ~ situação (estab.) | 0,235 | 0,021 | 0,050 | 0,073 |
| V natureza ~ qualif. sócio | 0,500 | 0,037 | 0,135 | 0,132 |
| Irmãos: mesmo CNAE | 0,680 | 0,021 | 0,038 | 0,637 |
| Irmãos: mesma UF | 0,863 | 0,135 | 0,157 | 0,854 |
| Irmãos: mesmo município | 0,531 | 0,011 | 0,017 | 0,835 ⚠ |
| Irmãos: mesma situação | 0,755 | 0,393 | 0,556 | 0,712 |
| Column Pair Trends | — | 0,615 | 0,603 | 0,608 |

³ Por constraint (`FixedCombinations`).

⚠ Pioras ou excessos introduzidos pela v2:
- Spearman(capital, nº de sócios) fica negativo, porque a T2S estratifica só por porte, e o capital sintético é mal modelado no pai;
- os irmãos ficam parecidos demais no município, porque o município é copiado junto com a UF para manter o par válido.

### 5.3 Dados com correlação conhecida (20 mil pais, 3 seeds)

| Métrica | Real | v1 | v2 (plano) | v2 (melhorada) |
|---|---|---|---|---|
| Spearman(size, nº de filhos) | 0,600 | −0,001 | 0,522 | 0,522 |
| Spearman(size, amount) | 0,688 | 0,003 | 0,541 | 0,531 |
| η² tier → discount (efeito não monotônico) | 0,525 | 0,000 | **0,141 ± 0,170** | **0,437 ± 0,001** |
| 1 HQ por pai | 97,2% | 42,5% | 100%² | 100%² |
| ICC de score entre irmãos | 0,638 | 0,000 | 0,635 | 0,638 |
| Mesma região entre irmãos (pares) | 0,717 | 0,201 | 0,345 | 0,605 |
| sdmetrics Intertable Trends | — | 0,684 | 0,807 | **0,851** |
| KS de filhos por pai | 0 | 0 | 0 | 0 |

Com o plano original, o efeito de uma categoria do pai **depende da ordem arbitrária** em que o `UniformEncoder` organiza as categorias (desvio de 0,17 entre seeds). A I1 elimina essa instabilidade.

### 5.4 Escala (CNPJ `core`)

| Fração (linhas) | v1 | v2 recomendada | Δ tempo | Δ memória |
|---|---|---|---|---|
| 0,5% (1,1 M) | 131 s · 2,66 GB | 132 s · 2,82 GB | +0,2% | +6% |
| 1% (2,2 M) | 256 s · 3,86 GB | 259 s · 4,60 GB | +1,1% | +19% |
| 2,5% (5,6 M) | 619 s · 7,47 GB | 632 s · 9,18 GB | +2,0% | **+23%** |

O tempo é linear nas duas versões. A memória da v2 cresce mais rápido que a da v1 e passa da meta de +15% a partir de 1%. Com o HMA, o tempo projetado para 0,05% já passa de 30 minutos.

---

## 6. Leitura dos resultados pelos autores

Afirmações que os autores **sustentam**, e as que **não sustentam**:

| Afirmação | Força da evidência | Comentário |
|---|---|---|
| A v1 preserva a cardinalidade (L3) melhor que o HMA | **Forte** | KS 0 contra 0,03 a 0,14; a cauda de 7.903 é preservada, contra 127 a 211 |
| A v1 escala e o HMA não | **Forte** | 52× (execução anterior) ou 37× (esta) mais rápida a 30 mil linhas; o HMA estoura o timeout a 119 mil; a v1 é linear até 5,6 M |
| A v1 não captura L4 a L7 | **Forte** | Por construção; medido ≈ 0 |
| A v2 recupera parte de L4 a L7 sem perder L3 nem a escala | **Moderada a forte** | Ganhos consistentes nas métricas direcionadas; custo de +0,2% a +4% de tempo |
| A v2 é **melhor que o HMA** em dependências entre tabelas | **Fraca a moderada** | Uma execução, na única escala em que o HMA roda (30 mil linhas, dominada por uma empresa gigante), com o HMA nos parâmetros padrão. Nas 7 associações, a v2 vence em 3 (MEI, qualificação do sócio, CNAE entre irmãos), perde em 3 (Spearman porte~sócios, porte~situação, natureza~identificador) e empata em 1. Nos agregados do sdmetrics, a vantagem é modesta |
| A v2 reproduz a **magnitude** das associações pai → filho | **Fraca** | Recupera de 1/3 a 2/3 do Cramér's V real (por exemplo, 0,08 contra 0,13 e 0,13 contra 0,50). O **padrão** da associação não foi verificado |
| A v2 reproduz a semelhança entre irmãos | **Moderada, com ressalva de circularidade** | A métrica é próxima da grandeza calibrada (§7.2) |
| As regras de grupo e de domínio ficam 100% | **Trivial** | É uma propriedade da regra ou constraint declarada, não um mérito do modelo |
| As melhorias I1 a I4 são melhores que o plano original | **Forte** no dado sintético (verdade conhecida, 3 seeds) · **moderada** no CNPJ | |

**Resposta provisória dos autores a Q4.** A v2 é um substituto razoável para:
- testes de sistemas e de carga;
- análises de cardinalidade e de estrutura;
- análises que dependem das dependências pai → filho **em direção e ordem de grandeza**.

**Não** foi demonstrada para:
- inferência estatística fina sobre associações entre tabelas;
- treino de modelos de ML que dependam dessas associações;
- qualquer uso com requisitos formais de privacidade.

---

## 7. Ameaças à validade (autocrítica)

### 7.1 Comparação com o HMA

1. **Uma única escala e uma execução.** A 0,01%, uma empresa com 7.903 estabelecimentos concentra cerca de 50% dos estabelecimentos e domina várias métricas sobre o JOIN.
2. **HMA sem ajuste.** Foram usados os parâmetros padrão (marginais `beta`). Não testamos `norm`, `truncnorm`, `gaussian_kde` nem a remoção de colunas problemáticas. Um HMA ajustado poderia se sair melhor.
3. **Variabilidade do HMA não caracterizada.** Duas execuções nos mesmos dados deram 833 s / 79,5% e 419 s / 81,1%. A origem não foi investigada; pode ser carga da máquina ou a ordem de iteração de `set`s.
4. **O HMA nunca foi avaliado na variante `full`**, que ele recusa. A comparação de três vias vale só para o schema simples.

### 7.2 Circularidade entre calibração e métrica

1. **Cópia entre irmãos (I2):** é calibrada na taxa de "mesmo valor que o 1º irmão", média por pai. A métrica de avaliação é a fração de **pares** iguais, média por empresa. As grandezas são diferentes, mas fortemente relacionadas. O acerto da v2 nessa métrica é em parte esperado por construção.
2. **T2S** estratifica por `porte_empresa` e é avaliada com Spearman(porte, nº de sócios) e com a taxa de Simples por porte. O acerto nessas duas é quase garantido. O teste honesto é uma relação **não usada** na estratificação, e nela a T2S **piora**: Spearman(capital, nº de sócios) = −0,058.
3. **`group_rules` e `FixedCombinations`** produzem 100% por construção.
4. **Avaliação no mesmo dado do fit.** Não há conjunto de teste (*held-out*). Medimos a fidelidade ao treino, não a generalização.
5. **Métricas escolhidas pelos autores** depois de ver os problemas. Exemplo: a métrica de irmãos por empresa foi criada depois de constatar que a métrica por pares era dominada pela empresa gigante. A escolha é defensável, mas não foi pré-registrada.

### 7.3 Sensibilidade das métricas

1. **O sdmetrics Intertable Trends quase não discrimina** (0,627 a 0,676 em todas as configurações), por fazer a média de muitos pares sem associação real.
2. **Column Shapes varia ±0,02 entre repetições da mesma configuração**, por causa da subamostra de 2.000 empresas. Diferenças menores que isso são ruído.
3. O **Cramér's V** compara só a intensidade. Uma associação de mesma magnitude com padrão diferente (porte "01" associado a MEI no real e porte "05" no sintético) passaria despercebida. A similaridade de contingência foi medida, mas é dominada pelas categorias frequentes.
4. **Não há métricas de dependência de ordem superior**, como as distribuições de agregados por pai (média, desvio e moda das filiais de cada empresa comparadas entre real e sintético), nem um classificador real × sintético sobre features agregadas por pai.

### 7.4 Variância e reprodutibilidade

1. **CNPJ com seed fixa:** as 3 repetições têm métricas idênticas. **Não há intervalo de confiança para a qualidade no CNPJ.** É preciso repetir com seeds diferentes de sampling e, idealmente, com subamostras diferentes (bootstrap de empresas).
2. **A v1 (e a branch base) não é reproduzível entre processos:** a ordem dos filhos vem de um `set` com hash randomizado. A v2 é determinística.

### 7.5 Qualidade single-table como teto

1. **O Column Shapes da v1 já é baixo** (≈ 0,77 no CNPJ). O marginal Beta é inadequado para `capital_social`: o p75 sintético é 215 mil, contra 10 mil no real. Toda dependência entre tabelas que passa por colunas mal modeladas herda o erro. **A T3 com `capital_social` reprovou o gate** de Column Pair Trends (−3,4 p.p.).
2. A cópula gaussiana **não preserva relações intra-tabela** com categóricas de ordem arbitrária. No dado sintético, `size↔segment` cai de 0,94 para 0,14. Esse limite vale para os três métodos, inclusive o HMA.
3. **Uma conclusão sobre representatividade "do banco" está limitada pela representatividade de cada tabela.** Esta avaliação não ataca esse problema.

### 7.6 Privacidade (não avaliada)

1. **A reamostragem exata das contagens** reproduz o multiconjunto real de cardinalidades. A existência de uma empresa com exatamente 7.903 estabelecimentos e 41 sócios é um fato raro, potencialmente identificável. Com a T2S e o contexto, essa contagem fica associada a atributos (porte, natureza) parecidos com os reais.
2. As tabelas de domínio são **copiadas** (T1), mas são códigos públicos.
3. Os campos de PII (nomes, telefones, e-mails, logradouros) são gerados por Faker, e não aprendidos.
4. **Não foram calculadas** distância ao registro mais próximo (DCR), NNDR nem ataques de inferência de pertencimento.

### 7.7 Generalidade

1. Um único banco real (CNPJ) e um gerador sintético simples (1 relação).
2. O contexto, os estratos e as regras do CNPJ foram **escolhidos com conhecimento do domínio**. A v2 não é *plug-and-play*: sem essas escolhas, o ganho é menor. A escolha automática de contexto (por exemplo, por informação mútua) não foi implementada.
3. Uma máquina só (M4, 16 GB). A memória medida inclui páginas comprimidas (footprint do macOS).

---

## 8. Avaliações que faltam

Em ordem de prioridade sugerida:

| # | Avaliação | O que resolveria |
|---|---|---|
| 1 | **Repetições com seeds de sampling e subamostras diferentes** (≥ 5) e intervalos de confiança | §7.4 |
| 2 | **Métricas held-out e não calibradas:** dividir as empresas em treino e teste, avaliar em relações não usadas para calibrar (por exemplo, natureza × nº de filiais, situação × CNAE entre irmãos) | §7.2 |
| 3 | **HMA ajustado e repetido:** distribuições alternativas, 3 ou mais execuções, e a 0,05% com timeout maior | §7.1 |
| 4 | **Utilidade (TSTR):** treinar um modelo no sintético e testar no real, em tarefas que cruzam tabelas (prever nº de filiais ou situação cadastral a partir de atributos da empresa e das filiais agregadas) | Q4 |
| 5 | **Detecção:** classificador real × sintético (C2ST) sobre features agregadas por empresa (nº de filiais, % ativas, nº de UFs distintas, CNAE modal) | §7.3 |
| 6 | **Privacidade:** DCR/NNDR por tabela e por "empresa com filhos"; checar a exposição da cauda exata | §7.6 |
| 7 | **Comparação com outros métodos multi-tabela:** RCTGAN (Gueye et al., 2023), REaLTabFormer (Solatorio & Dupriez, 2023), ClavaDDPM (Pang et al., NeurIPS 2024) e o synthesizer escalável do SDV Enterprise, se houver acesso | posição relativa ao estado da arte |
| 8 | **Marginais melhores no pai** (por exemplo, transformação por quantis em `capital_social`) e nova medição da T3 com capital | §7.5 |

---

## 9. Questionário de revisão

Para cada item, marque **Concordo / Concordo parcialmente / Discordo** e justifique.

**Comparabilidade (Q1)**
1. É justo comparar o HMA nos parâmetros padrão com a v2, cujos parâmetros foram escolhidos com conhecimento do domínio CNPJ?
2. A comparação de três vias em uma única fração (0,01%), dominada por uma empresa gigante, permite alguma conclusão sobre qualidade? Ou só sobre custo?
3. Os gates de não regressão (§8.2 do plano: Column Pair Trends ≥ base − 2 p.p., tempo ≤ +20%, memória ≤ +15%) são critérios razoáveis?

**Métricas (Q2)**
4. A taxonomia L1 a L9 cobre o que você entende por "representatividade" de um banco relacional? O que falta?
5. Cramér's V em magnitude é adequado para dependências pai → filho? Qual métrica você recomendaria?
6. Medir a semelhança entre irmãos com a mesma família de grandeza usada na calibração (I2) invalida o resultado, ou é aceitável com a ressalva?
7. O sdmetrics Intertable Trends deveria ser a métrica principal? Por quê não discrimina neste caso?

**Conclusões (Q3)**
8. A evidência sustenta que a v2 é **melhor que a v1**? Em quais camadas?
9. A evidência sustenta que a v2 é **melhor que o HMA**? Ou apenas "comparável, com custo muito menor"?
10. A recuperação de 1/3 a 2/3 da magnitude das associações pai → filho é suficiente para algum uso? Para qual?
11. As pioras introduzidas (capital × nº de sócios negativo; município parecido demais entre irmãos) são aceitáveis em troca dos ganhos?

**Propósito (Q4)**
12. Para quais usos você aprovaria os dados da v2 (testes, análise exploratória, treino de modelos, publicação)?
13. A reprodução exata do multiconjunto de cardinalidades é um risco de privacidade relevante neste contexto (dados públicos da Receita) e em contextos privados?
14. A dependência de conhecimento de domínio (contexto, estratos, regras) é uma limitação séria para o uso geral?

**Metodologia**
15. A abordagem "um modelo por tabela + condicionamento linear no latente + regras" é uma direção promissora em relação a modelos generativos relacionais profundos (difusão, transformers)? Ou é um beco sem saída, limitado pela cópula gaussiana?
16. Quais das avaliações da §8 você considera **obrigatórias** antes de qualquer conclusão pública?

**Parecer final:** ☐ Aceitar as conclusões · ☐ Aceitar com revisões · ☐ Revisões maiores · ☐ Rejeitar as conclusões

---

## 10. Artefatos e reprodução

| Artefato | Caminho |
|---|---|
| Implementação v1 e v2 | `sdv/multi_table/independent.py`, `_conditional.py`, `_group_features.py` |
| HMA (upstream) | `sdv/multi_table/hma.py`, `sdv/sampling/hierarchical_sampler.py` |
| Testes (25 novos) | `tests/unit/multi_table/test_independent_correlation.py`, `tests/integration/multi_table/test_independent_correlation.py` |
| Gerador com correlação conhecida | `tests/utils.py::generate_correlated_parent_child` |
| Benchmark CNPJ | `benchmarks/cnpj/` (README com as opções) |
| Experimento e campanhas | `benchmarks/correlation/` (`synthetic.py`, `metrics.py`, `summarize.py`, `run_cnpj_campaign*.sh`) |
| Resultados brutos (1 linha JSON por execução) | `benchmarks/correlation/results_cnpj.jsonl` (81 execuções), `results_synthetic.jsonl` (39) |
| Logs | `benchmarks/correlation/*.log` |
| Relatórios detalhados | `BENCHMARK.pt-BR.md`, `EXPERIMENTO-CORRELACAO.pt-BR.md` |

```bash
# ambiente: Python 3.12, .venv do repositório (pip install -e .)
.venv/bin/python -m pytest tests/unit/multi_table/test_independent_correlation.py \
    tests/integration/multi_table/test_independent_correlation.py

# dados com correlação conhecida (~75 s)
.venv/bin/python benchmarks/correlation/synthetic.py --num-parents 20000 --seeds 0 1 2

# CNPJ: download e preparo em benchmarks/cnpj/README.md; campanhas (~3 h)
benchmarks/correlation/run_cnpj_campaign.sh ~/.cache/sdv-cnpj/2026-09/sample_0.05
benchmarks/correlation/run_cnpj_campaign_e.sh ~/.cache/sdv-cnpj/2026-09/sample_0.05

# HMA × v1 × v2 em 0,01%
for c in "" "T2 T3 T4 T4R T5"; do
  .venv/bin/python benchmarks/cnpj/run.py --sample-dir ~/.cache/sdv-cnpj/2026-09/sample_0.05 \
      --variant core --fractions 0.0001 --sdmetrics --correlation $c
done
.venv/bin/python benchmarks/cnpj/run.py --sample-dir ~/.cache/sdv-cnpj/2026-09/sample_0.05 \
    --synthesizer hma --variant core --fractions 0.0001 --timeout 1800 --sdmetrics
```
