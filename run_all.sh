#!/bin/bash
# Runs the whole experiment (15 runs). Usage:  JOBS=4 ./run_all.sh     (JOBS = parallel runs, default 2)
# For a quick smoke test:                       TMAX=2 OUT=test_runs ./run_all.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=${OUT:-runs}; TMAX=${TMAX:-80}; mkdir -p "$OUT"
cat > jobs.txt << JOBS
--name ref     --seed 1 --out $OUT --tmax $TMAX
--name p12     --seed 1 --perturb 1e-12 --out $OUT --tmax $TMAX
--name p10     --seed 1 --perturb 1e-10 --out $OUT --tmax $TMAX
--name p08     --seed 1 --perturb 1e-8  --out $OUT --tmax $TMAX
--name permute --seed 1 --variant permute --out $OUT --tmax $TMAX
--name matmul  --seed 1 --variant matmul  --out $OUT --tmax $TMAX
--name seed2  --seed 2  --out $OUT --tmax $TMAX
--name seed3  --seed 3  --out $OUT --tmax $TMAX
--name seed4  --seed 4  --out $OUT --tmax $TMAX
--name seed5  --seed 5  --out $OUT --tmax $TMAX
--name seed6  --seed 6  --out $OUT --tmax $TMAX
--name seed7  --seed 7  --out $OUT --tmax $TMAX
--name seed8  --seed 8  --out $OUT --tmax $TMAX
--name seed9  --seed 9  --out $OUT --tmax $TMAX
--name seed10 --seed 10 --out $OUT --tmax $TMAX
JOBS
xargs -P "${JOBS:-2}" -L 1 python3 nbody.py < jobs.txt
echo "all runs finished"
