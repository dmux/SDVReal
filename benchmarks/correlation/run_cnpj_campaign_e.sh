#!/usr/bin/env bash
# Follow-up campaign: stratified cardinality (T2S) and per-company sibling metrics.
set -u
cd "$(dirname "$0")/../.."
SAMPLE=${1:-$HOME/.cache/sdv-cnpj/2026-09/sample_0.05}
OUT=benchmarks/correlation/results_cnpj.jsonl
PY=.venv/bin/python
run() { $PY benchmarks/cnpj/run.py --sample-dir "$SAMPLE" --output "$OUT" --sdmetrics "$@" 2>/dev/null | grep -v '^RAM'; }

echo "== E. core 0.1%: stratified cardinality and per-company sibling metrics"
run --variant core --fractions 0.001 --repeat 3 --correlation T2S
run --variant core --fractions 0.001 --repeat 3 --correlation T2S T3 T4 T4R T5 --label "best:T2S+T3+T4+T4R+T5"
run --variant core --fractions 0.001 --repeat 3 --correlation T3C
run --variant core --fractions 0.001 --repeat 3 --correlation T2S T3C T4 T4R T5 --label "best:T2S+T3C+T4+T4R+T5"
run --variant core --fractions 0.001 --repeat 3 --correlation FC
run --variant core --fractions 0.001 --repeat 3 --correlation FC T2S T3C T4 T4R T5 --label "best:FC+T2S+T3C+T4+T4R+T5"
run --variant core --fractions 0.001 --label "E:baseline"
run --variant core --fractions 0.001 --correlation T5 --label "E:T5"
run --variant core --fractions 0.001 --correlation T5 --encoding plan --label "E:T5(plan)"
run --variant core --fractions 0.001 --correlation T2 T3 T4 T4R T5 --encoding plan --label "E:T2..T5(plan)"
for fraction in 0.005 0.01 0.025; do
  run --variant core --fractions $fraction --correlation FC T2S T3C T4 T4R T5 --label "scale:best"
done
run --variant core --fractions 0.01 0.025 --correlation T2 T3 T4 T4R T5 --label "scale:T2..T5(mem-opt)"
echo "== done E"
