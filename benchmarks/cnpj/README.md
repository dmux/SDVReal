# CNPJ scale benchmark

Load test of SDV multi-table synthesizers on the Brazilian company registry (CNPJ open data
published monthly by the Receita Federal): companies, establishments (headquarters and
branches), partners and Simples Nacional records.

This benchmark is **not** part of the test suite: it needs a ~7.5 GB download and takes about an
hour. Data stays outside the repository (default: `~/.cache/sdv-cnpj`).

## 1. Download

The files are published on the Receita Federal Nextcloud share `YggdBLfdninEJX9`, one folder
per release (`YYYY-MM`). The public WebDAV endpoint can be used to list and download them:

```bash
RELEASE=2026-09
DEST=~/.cache/sdv-cnpj/$RELEASE && mkdir -p $DEST && cd $DEST
SHARE=https://arquivos.receitafederal.gov.br/public.php/webdav
curl -s -u "YggdBLfdninEJX9:" -X PROPFIND -H "Depth: 1" $SHARE/$RELEASE/ \
  | grep -oE '[A-Za-z]+[0-9]?\.zip' | sort -u \
  | xargs -P 4 -I{} curl -sS --retry 5 -C - -u "YggdBLfdninEJX9:" -o {} $SHARE/$RELEASE/{}
```

## 2. Prepare a company-level sample

```bash
python benchmarks/cnpj/prepare.py --release-dir ~/.cache/sdv-cnpj/2026-09 --max-fraction 0.05
```

Streams every zip once and keeps the companies whose `int(cnpj_basico) % 10000` is below
`max_fraction * 10000`, together with **all** their establishments, partners and Simples
records, so referential integrity and the children-per-company distribution are preserved.
Smaller fractions are subsets of the prepared sample.

## 3. Run

```bash
python benchmarks/cnpj/run.py --sample-dir ~/.cache/sdv-cnpj/2026-09/sample_0.05 \
    --synthesizer independent --variant full --fractions 0.001 0.0025 0.005 0.01 0.025 0.05
python benchmarks/cnpj/run.py --sample-dir ~/.cache/sdv-cnpj/2026-09/sample_0.05 \
    --synthesizer hma --variant core --fractions 0.0001 0.0005 0.001 0.0025 --timeout 1800
```

Every configuration runs in its own subprocess. The parent tracks the peak physical footprint
(macOS `ri_lifetime_max_phys_footprint`, which includes compressed pages; RSS elsewhere) and
kills the run above `--memory-limit-gb` (default 80% of RAM) or `--timeout`. Results are
appended to `results.jsonl` in the sample folder.

### Inter-table correlation options

`--correlation` enables the correlation options of the `IndependentSynthesizer` (see
[`../EXPERIMENTO-CORRELACAO.pt-BR.md`](../EXPERIMENTO-CORRELACAO.pt-BR.md)):

| Token | Option |
|---|---|
| `T1` | `lookup_tables`: the domain tables (`full` variant only) |
| `T2` | `model_cardinality=True` |
| `T2S` | `cardinality_by`: children counts stratified by `porte_empresa` |
| `T3` | `context_columns`: `porte_empresa`, `natureza_juridica`, `capital_social` |
| `T3C` | `context_columns`: `porte_empresa`, `natureza_juridica` |
| `T4` | `sibling_order` of the establishments by `identificador_matriz_filial` |
| `T4R` | `group_rules`: one headquarters (`1`) per company, the others `2` |
| `T5` | `sibling_correlation=True` |
| `FC` | `FixedCombinations(['uf', 'municipio'])` on the establishments |

```bash
python benchmarks/cnpj/run.py --sample-dir ~/.cache/sdv-cnpj/2026-09/sample_0.05 \
    --variant core --fractions 0.001 --correlation FC T2S T3C T4 T4R T5 --sdmetrics --repeat 3
```

* `--encoding plan` uses the encodings of the original plan (`categorical_context='processed'`,
  `sibling_categorical='latent'`) for comparison.
* `--sdmetrics` adds the sdmetrics `QualityReport` on a subset of 2,000 companies.
* `--label` names the configuration in `results.jsonl`; `--repeat` repeats every fraction.
* Correlation metrics are always reported: Spearman of company attributes vs children counts,
  Cramér's V of company/child column pairs, sibling similarity (pairs and mean per company) and
  the share of synthetic (`uf`, `municipio`) pairs that exist in the real data.

The full comparative campaign is in `benchmarks/correlation/run_cnpj_campaign.sh` and
`run_cnpj_campaign_e.sh`.

### Schema variants

| Variant | Tables | Depth | Notes |
|---|---|---|---|
| `core` | `empresas` → `estabelecimentos`, `socios`, `simples` | 2 | Domain codes are categorical columns. Accepted by `HMASynthesizer`. |
| `full` | `core` + `cnaes`, `motivos`, `municipios`, `naturezas`, `paises`, `qualificacoes` | 3 | 12 relationships, 3 foreign keys from `socios` to `qualificacoes`/`paises`. Rejected by `HMASynthesizer`. |

SDV Community does not support composite keys, so the establishment primary key is the 14 digit
`cnpj` (`cnpj_basico` + `cnpj_ordem` + `cnpj_dv`). Names, trade names, streets, phones, e-mails
and postcodes are PII sdtypes generated with Faker (`pt_BR`), so no real personal data is
reproduced in the synthetic output.

### Metrics

* Time per phase: `load`, `init`, `preprocess`, `fit`, `sample`, `validate`, `quality`.
* Peak memory (physical footprint).
* `integrity`: `metadata.validate_data(synthetic_data)` passes.
* `ks_*_per_empresa`: KS statistic between the real and synthetic children-per-company
  distributions (0 = identical).
* `pct_empresas_one_matriz_*`: share of companies with exactly one headquarters
  (`identificador_matriz_filial == 1`). It is 100% in the real data; the independent
  synthesizer does not model this cross-row rule.

## Results — release 2026-09, MacBook Pro M4 (10 cores), 16 GB RAM

Full registry: 70.1M companies, 73.4M establishments, 28.3M partners, 50.4M Simples records
(222M rows). Streaming the whole release into the 5% sample took 84s.

### `IndependentSynthesizer`, `full` variant (10 tables, 12 relationships)

| Sample | Rows | Fit + sample (total) | Peak memory | Integrity | KS est./company | KS partners/company | One HQ per company (real 100%) |
|---|---|---|---|---|---|---|---|
| 0.1% | 237k | 27s | 1.4 GB | ✅ | 0.000 | 0.000 | 84.8% |
| 0.25% | 570k | 54s | 1.4 GB | ✅ | 0.000 | 0.000 | 90.3% |
| 0.5% | 1.13M | 102s | 2.1 GB | ✅ | 0.000 | 0.000 | 91.9% |
| 1% | 2.24M | 203s | 3.4 GB | ✅ | 0.000 | 0.000 | 92.5% |
| 2.5% | 5.57M | 477s | 7.4 GB | ✅ | 0.000 | 0.000 | 93.2% |
| 5% | 11.1M | killed while sampling at 12.9 GB (80% of RAM) | — | — | — | — | — |

Time grows linearly (≈ 85 µs per row; 23.5× rows ⇒ 17.7× time). On this machine the memory
ceiling is between 5.6M and 11.1M rows: loading 11.1M rows into pandas alone takes 7.1 GB.

### `HMASynthesizer` vs `IndependentSynthesizer`, `core` variant (4 tables, depth 2)

| Sample | Rows | Independent | HMA |
|---|---|---|---|
| 0.01% | 30k | 16s, 0.9 GB, KS 0.000 / 0.000 | 833s (52×), 1.3 GB, KS 0.031 / 0.140, max establishments 127 (real 7,903) |
| 0.05% | 119k | 38s, 0.9 GB | killed by the 30 min timeout |
| 0.1% | 230k | 59s, 1.2 GB | — |
| 0.25% | 563k | 121s, 1.6 GB | — |
