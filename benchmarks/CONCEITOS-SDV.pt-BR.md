# Conceitos do SDV: guia para quem é de tecnologia, mas não da área de dados sintéticos

> Complementa [`BENCHMARK.pt-BR.md`](BENCHMARK.pt-BR.md) e [`PLANO-OTIMIZACAO.pt-BR.md`](PLANO-OTIMIZACAO.pt-BR.md).
> Os exemplos usam a base pública do CNPJ da Receita Federal.

O SDV (Synthetic Data Vault) é uma biblioteca que aprende como os seus dados "se comportam" e depois **fabrica dados novos, falsos, mas com o mesmo comportamento**.

---

## 1. Dado sintético: o que é e para que serve

Imagine que você precisa testar um sistema de cadastro de empresas, mas não pode usar os dados reais, seja por LGPD, por sigilo ou porque o ambiente de teste não é seguro.

- **Copiar os dados reais** é proibido.
- **Gerar dados aleatórios** não serve: CEP com 3 dígitos, empresa aberta em 2090, capital social negativo. O sistema não é testado de verdade.
- **Dado sintético** é o meio-termo: nenhuma linha é real, mas o conjunto *parece* real. As distribuições, as proporções e as relações batem com a base original.

Usos típicos: ambientes de QA e desenvolvimento, testes de carga, compartilhar dados com terceiros e treinar modelos sem expor pessoas.

---

## 2. O fluxo básico: aprender e depois gerar

Funciona como qualquer modelo de machine learning, com duas etapas:

```
dados reais ──► fit() ──► modelo ──► sample() ──► dados sintéticos
               (aprende)            (gera quantas linhas quiser)
```

- **`fit`**: o modelo estuda os dados. Aprende, por exemplo, que a maioria das empresas tem capital social baixo e poucas têm capital alto.
- **`sample`**: o modelo "inventa" linhas novas que seguem o que ele aprendeu. Dá para pedir 10 linhas ou 10 milhões.

Depois do fit, o modelo **não guarda os dados reais**, só os padrões. Dá para salvar o modelo e gerar dados em outro lugar sem levar a base junto.

---

## 3. Metadata: o "manual" dos dados

Antes de aprender, o SDV precisa saber **o que cada coluna significa**. Isso fica no metadata, uma espécie de schema enriquecido:

| Conceito | O que é | Exemplo no CNPJ |
|---|---|---|
| **Tabela** | uma planilha | `empresas`, `estabelecimentos` |
| **sdtype** (tipo semântico) | o *significado* da coluna, não só o tipo técnico | `numerical`, `categorical`, `datetime`, `id`, `email`, `phone_number` |
| **Chave primária (PK)** | o identificador único de cada linha | `cnpj_basico` em `empresas` |
| **Chave estrangeira (FK)** | a coluna que aponta para outra tabela | `cnpj_basico` em `estabelecimentos` aponta para a empresa |
| **Relação** | a ligação pai → filho entre tabelas | uma empresa tem N estabelecimentos |

O sdtype importa porque muda a forma de gerar:
- uma coluna `uf` do tipo categórico deve gerar valores que existem ("SP", "RJ");
- uma coluna `email` deve gerar e-mails com formato válido;
- uma coluna `id` deve gerar valores únicos.

---

## 4. Transformadores (RDT): traduzir tudo para números

Os modelos matemáticos só entendem números, e os dados têm texto, datas e nulos. Quem faz a tradução, nos dois sentidos, é a biblioteca **RDT** (Reversible Data Transforms):

```
"SP"        ──► 0,37        (categoria vira número)
2019-05-12  ──► 1557619200  (data vira número)
nulo        ──► coluna extra "era nulo? 0/1"
```

A palavra-chave é **reversível**. Depois de gerar números, o RDT traduz de volta para "SP", datas e nulos. No benchmark, essa etapa é o **preprocess**.

---

## 5. Dados pessoais (PII): não aprender, inventar

Nome de sócio, e-mail e telefone **não devem ser aprendidos**, senão o modelo poderia reproduzir pessoas reais. Para essas colunas o SDV não modela nada: usa a biblioteca **Faker**, que inventa valores plausíveis ("Ana Souza", "(11) 9xxxx-xxxx").

É mais seguro, mas lento, porque o Faker gera um valor por vez. No benchmark ele aparece como **o maior gargalo de tempo**.

---

## 6. Como o modelo "aprende": a cópula gaussiana

O modelo padrão é a **GaussianCopula**. A intuição tem duas partes:

1. **Cada coluna isolada** (a "marginal"): o modelo aprende a forma da distribuição de cada coluna, por exemplo "capital social: muitos valores baixos e uma cauda longa de valores altos". Escolher essa forma é o que o parâmetro `distribution` controla (`norm`, `beta` etc.).
2. **Como as colunas andam juntas** (a "correlação"): por exemplo, empresas mais antigas tendem a ter capital maior.

A cópula é o truque matemático que **junta as duas coisas**: gera valores que respeitam a forma de cada coluna *e* as correlações entre elas.

Existem modelos mais sofisticados, como o **CTGAN**, uma rede neural que captura padrões mais complexos. Ele é mais lento e mais pesado.

---

## 7. Multi-tabela: onde fica difícil

Gerar uma tabela isolada é relativamente simples. O desafio de verdade são **várias tabelas ligadas**, porque é preciso preservar três coisas:

| O que preservar | Pergunta | Exemplo |
|---|---|---|
| **Integridade referencial** | toda FK aponta para um pai que existe? | nenhum estabelecimento pode ter um `cnpj_basico` de empresa inexistente |
| **Cardinalidade** | quantos filhos cada pai costuma ter? | a maioria das empresas tem 1 estabelecimento, e algumas redes têm milhares |
| **Correlação entre tabelas** | o pai influencia o conteúdo do filho? | empresa grande tende a ter filiais em mais estados |

### As duas estratégias que comparamos

**HMA (o padrão do SDV Community).** Ele tenta capturar as três coisas. Para isso, treina um mini-modelo **para cada linha do pai**, ou seja, um para cada empresa, e "achata" o que aprendeu em colunas extras do pai. Captura correlação entre tabelas, mas o custo explode: no benchmark foi 52× mais lento e não terminou com 119 mil linhas. Além disso, uma trava recusa schemas com mais de 5 tabelas.

**Independent (a abordagem que implementamos).** Ele divide o problema:
1. treina **um modelo por tabela**, de forma independente;
2. aprende só **quantos filhos cada pai tem** (a cardinalidade);
3. na geração, cria cada tabela sozinha e depois "costura" as FKs respeitando essas contagens.

Garante a integridade e a cardinalidade, e é rápido e linear. O preço é **não capturar a correlação entre tabelas**: uma empresa sintética "pequena" pode receber 500 filiais, porque quem gera as filiais não sabe nada sobre a empresa.

**Analogia.** O HMA é um alfaiate que faz cada roupa sob medida para cada cliente: perfeito, mas lento. O Independent é uma fábrica que produz camisas e calças em série e depois monta os conjuntos respeitando quantas peças cada cliente leva: rápido, mas a calça não "combina" de propósito com a camisa.

---

## 8. Restrições (constraints)

São regras de negócio que o modelo sozinho pode não respeitar, como "a data de fim é maior que a de início" ou "o salário fica entre X e Y". O SDV permite declará-las, e a geração passa a respeitá-las, filtrando ou transformando os dados.

O benchmark mostrou um exemplo de regra que **nenhum dos dois synthesizers garante**: toda empresa tem exatamente uma matriz. Esse tipo de regra entre linhas do mesmo pai precisaria de uma constraint ou de um pós-processamento.

---

## 9. Avaliação: como saber se o dado sintético ficou bom

A biblioteca irmã **SDMetrics** compara o real com o sintético em duas perguntas:

- **Diagnóstico (válido?)**: chaves únicas, FKs válidas, valores dentro das faixas permitidas. Deve dar 100%.
- **Qualidade (parecido?)**:
  - **Column Shapes**: a distribuição de cada coluna é parecida?
  - **Column Pair Trends**: as correlações entre colunas se mantêm?
  - **Cardinality**: os filhos por pai se mantêm? No benchmark usamos o teste **KS**, em que 0 significa distribuições idênticas.

---

## 10. Juntando tudo

```
   Metadata ─────────── define tipos, chaves e relações
       │
 dados reais
       │
   RDT (preprocess) ─── texto/datas → números
       │
   Synthesizer (fit) ── GaussianCopula por tabela + cardinalidade (Independent)
       │                ou um modelo por linha do pai (HMA)
   Synthesizer (sample)  gera números novos + Faker para PII
       │
   RDT (reverse) ────── números → texto/datas
       │
   costura das FKs ──── integridade referencial
       │
 dados sintéticos ──►  SDMetrics: válido? parecido?
```

Em uma frase: **o SDV aprende a "receita" dos seus dados (tipos, distribuições, correlações e relações) e cozinha quantos pratos novos você quiser com essa receita, sem servir nenhum ingrediente original.**

---

## Glossário rápido

| Termo | Significado |
|---|---|
| **Synthesizer** | o modelo que aprende (`fit`) e gera (`sample`) |
| **Marginal** | a distribuição de uma coluna isolada |
| **Cópula** | a técnica que combina as marginais com as correlações entre colunas |
| **sdtype** | o tipo semântico de uma coluna (categórica, data, e-mail, id…) |
| **PK / FK** | chave primária / chave estrangeira |
| **Cardinalidade** | quantos filhos cada pai tem numa relação |
| **Integridade referencial** | toda FK aponta para uma PK que existe |
| **PII** | dado pessoal identificável (nome, e-mail, telefone) |
| **RDT** | biblioteca que transforma dados em números e de volta |
| **SDMetrics** | biblioteca que mede a validade e a qualidade do dado sintético |
| **KS (Kolmogorov-Smirnov)** | teste que compara duas distribuições; 0 = idênticas |
| **HMA** | synthesizer multi-tabela padrão do SDV Community; captura relações, mas não escala |
| **IndependentSynthesizer** | synthesizer multi-tabela implementado neste fork; escala de forma linear, sem correlação entre tabelas |
