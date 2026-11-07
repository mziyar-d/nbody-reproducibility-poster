# Same Code, Same Input, Different Answer

Code and analysis for the poster **"Same Code, Same Input, Different Answer: Why open code is not enough for reproducible chaotic simulations"**
(HKUST Library symposium *Empowering Researchers for Future Open Scholarship & Publishing*, 2 December 2026).

A small, fully open N-body experiment that shows two things at once:

1. **Individual trajectories are not reproducible.** Changing one star's position by 1e-12, or changing only the floating-point arithmetic (star order, matrix-form force sum), makes trajectories diverge exponentially.
2. **The cluster's overall statistics are.** Late-time mean mass radii of every variant stay close to the reference, compared with the natural spread between independent clusters.

So for chaotic simulations "reproducible" should mean *reproducing the distribution of outcomes*, and open code alone does not give you that. See the poster's *Reproducibility card* below for what to publish alongside the code.

## Contents

| File | What it does |
|---|---|
| `nbody.py` | The simulation: Plummer sphere, softened direct summation, fixed-step leapfrog (NumPy). One call = one run. |
| `run_all.sh` | Runs all 15 simulations (parallel with `JOBS=n`). |
| `analysis_v2.py` | Divergence curves, statistical comparison, poster figures and `summary_v2.json`. **Used for the poster.** |
| `analysis.py` | First analysis version (instantaneous deviation metric). Kept for transparency; superseded by `analysis_v2.py`. |
| `summary_v2.json` | Numbers reported on the poster. |
| `figures/` | Figures used on the poster. |
| `CITATION.cff` | How to cite this repository. |


## Quick start

```bash
pip install -r requirements.txt

# smoke test (about a minute; each run is short)
TMAX=2 OUT=test_runs JOBS=1 ./run_all.sh

# full experiment (15 runs, t = 80)
OUT=final JOBS=2 ./run_all.sh

# analysis and figures
RUNS=final python3 analysis_v2.py
```

`run_all.sh` needs a bash shell (Linux, macOS, WSL, Colab). On Windows without WSL, run the lines written to `jobs.txt` one by one with `python nbody.py <args>`.

Setting `LATE_START` (default 70) changes the late-time window; `BG` changes the figure background colour.

## The experiment

* N = 500 equal-mass stars, Plummer sphere in N-body units (G = M = 1, virial radius 1; crossing time about 2.83).
* Softening 0.05, fixed time step 2^-7, total time 80 (about 28 crossing times), snapshot every 0.125.
* Reference run `ref` uses seed 1. Seeds 1 to 10 form the ten independent clusters used for the natural run-to-run spread (`ref` is seed 1).

| Run | What changes relative to `ref` |
|---|---|
| `p12`, `p10`, `p08` | x-coordinate of star 0 shifted by 1e-12, 1e-10, 1e-8 |
| `permute` | identical input, star order permuted (different summation order) |
| `matmul` | identical input, forces summed in matrix form (different rounding) |
| `seed2` ... `seed10` | different random initial conditions |

**Metrics.** Divergence: median star-position difference to `ref`, divided by the initial half-mass radius; "diverged" means this reaches 10%. Statistics: radii containing 10%, 50% and 90% of the mass about the centre of mass. Comparison: for t >= 70, mean radius of each variant minus that of `ref`, divided by the standard deviation across the ten clusters.

## Results of the published run

| Change | Time to reach 10% of half-mass radius |
|---|---|
| shift 1e-8 | 35.75 |
| shift 1e-10 | 43.1 |
| shift 1e-12 | 51.0 |
| permuted star order | 57.5 |
| matrix-form forces | 57.5 |

* Each 100x smaller perturbation delays divergence by only about 7.6 time units (about 2.7 crossing times). Estimated e-folding time about 1.7.
* Late-time mean radii differ from `ref` by less than 0.7 sigma in 14 of 15 comparisons. The largest is 2.1 sigma (90% mass radius, `p10`). This is not systematic with perturbation size (`p12`: 0.03, `p08`: 0.53), but ten clusters and a ten-unit window cannot settle it.
* Maximum relative energy error over all runs: 2.3e-5.

Exact numbers on another machine or library version will differ in detail. That is the point of the project; compare statistics, not trajectories.

## Environment of the published run

* Google Colab, 2 CPU cores, Python **3.13.18**, NumPy **2.4.6**, Matplotlib **3.10.0**
* About 7 minutes per run (about 3 minutes for `matmul`), about 45 minutes for all 15 with `JOBS=2`.

## Reproducibility card (this repository)

| Item | Where |
|---|---|
| Environment | Colab runtime above; `requirements.txt` |
| Input | generated inside `nbody.py` from the seed; all parameters are at the top of the file and in the table above |
| Build | pure Python and NumPy, nothing compiled |
| Metrics | defined above, with the 10% and 0.7 / 2.1 sigma reporting rules fixed before reading the results |
| Credit | `CITATION.cff`, archive DOI (to be added), licence below |

## Limitations

Small model (N = 500, softened gravity, no binaries or stellar evolution), one code, 15 runs, one computing environment. It shows the principle, not the size of the effect in production codes such as NBODY6. Hardware and compiler differences (GPU, other operating systems) were not tested; only two arithmetic variants were.

## Use of AI tools

AI tools assisted with drafting code, designing the experiment and editing text. The authors ran all simulations and checked the results.

## Cite

See `CITATION.cff` 

## Licence

MIT (see `LICENSE`).

## Contact

**mziyarda@gmail.com**
**maziyar.dalili@iasbs.ac.ir**

