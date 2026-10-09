#!/usr/bin/env bash
# Comparative campaign of the inter-table correlation options on the CNPJ sample.
# Usage: benchmarks/correlation/run_cnpj_campaign.sh [sample_dir]
set -u
cd "$(dirname "$0")/../.."
SAMPLE=${1:-$HOME/.cache/sdv-cnpj/2026-09/sample_0.05}
OUT=benchmarks/correlation/results_cnpj.jsonl
PY=.venv/bin/python
run() { $PY benchmarks/cnpj/run.py --sample-dir "$SAMPLE" --output "$OUT" --sdmetrics "$@" 2>/dev/null | grep -v '^RAM'; }

echo "== A. core 0.1%: each technique isolated and accumulated (3 repetitions)"
run --variant core --fractions 0.001 --repeat 3 --label baseline
for cfg in "T2" "T3" "T4" "T4 T4R" "T5" "T2 T3" "T2 T3 T4 T4R" "T2 T3 T4 T4R T5"; do
  run --variant core --fractions 0.001 --repeat 3 --correlation $cfg
done
run --variant core --fractions 0.001 --repeat 3 --correlation T3 --encoding plan --label "T3(plan)"
run --variant core --fractions 0.001 --repeat 3 --correlation T5 --encoding plan --label "T5(plan)"
run --variant core --fractions 0.001 --repeat 3 --correlation T2 T3 T4 T4R T5 --encoding plan --label "T2+T3+T4+T4R+T5(plan)"

echo "== B. full 0.1%: lookup tables"
run --variant full --fractions 0.001 --repeat 3 --label "full:baseline"
run --variant full --fractions 0.001 --repeat 3 --correlation T1 --label "full:T1"
run --variant full --fractions 0.001 --repeat 3 --correlation T1 T2 T3 T4 T4R T5 --label "full:T1..T5"

echo "== C. core scale ladder: baseline vs all techniques"
for fraction in 0.005 0.01 0.025; do
  run --variant core --fractions $fraction --label "scale:baseline"
  run --variant core --fractions $fraction --correlation T2 T3 T4 T4R T5 --label "scale:T2..T5"
done

echo "== D. HMA reference at 0.01%"
run --variant core --fractions 0.0001 --label "tiny:baseline"
run --variant core --fractions 0.0001 --correlation T2 T3 T4 T4R T5 --label "tiny:T2..T5"
run --synthesizer hma --variant core --fractions 0.0001 --timeout 1800 --label "tiny:HMA"
echo "== done"
