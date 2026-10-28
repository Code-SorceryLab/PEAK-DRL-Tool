# Narval jobs

Everything runs from `$SCRATCH/peak` (a sparse copy of `code/`, **never run git there**) through one
array runner: each task runs one line of a task file. `jobs/make_tasks.py` writes the task files;
`jobs/run_tasks.sh` runs them (`rc=$?` logged on every END line). Submit ≤18 cores and ≤3 h per task so
backfill starts them. The scripts suit any Alliance cluster (account `def-polako_cpu`); on Narval
`export EXCLUDE=nc10901` (its /cvmfs fails: rc=126 at once).

    mkdir -p runs/logs && .venv/bin/python -m jobs.make_tasks          # all task files, with their sizes
    sub() { sbatch --parsable ${DEP:+--dependency=afterany:$DEP} ${EXCLUDE:+--exclude=$EXCLUDE} -J "peak-$1" -c "$2" -t "${4:-3:00:00}" \
            --array="${3:-0-$(( $(wc -l < jobs/tasks/$1.txt) - 1 ))}" -o "runs/logs/%x_%A_%a.out" \
            jobs/run_tasks.sh "jobs/tasks/$1.txt"; }       # sub <stage> <cores> [array] [time]; DEP=<jobid> waits
    check() { sacct -j "$1" -X -n -P -o State | sed 's/ by.*//' | sort | uniq -c; grep -L "END rc=0" runs/logs/*_"$1"_*.out; }

Before uploading: strip CR from every `.sh` (`tr -d '\r'`), then on Narval `bash -n` it and check
`tr -cd '\r' < f | wc -c` is 0. Upload only the files that differ (md5 both sides).

## Step 6: does it train on Narval?

    sub test 3 0-3 0:30:00            # one 5-gen probe per game into runs/_test; then: ls runs/_test/balance

## Step 7: quick derisking sweep (its own copy, so ga_best.yaml and runs/ stay untouched)

    mkdir -p $SCRATCH/peak_quick/runs/logs && cp -r code run_all.sh jobs $SCRATCH/peak_quick/ \
        && ln -sfn $SCRATCH/peak/.venv $SCRATCH/peak_quick/.venv
    cd $SCRATCH/peak_quick && export PEAK_DIR=$SCRATCH/peak_quick      # sbatch passes the environment on
    .venv/bin/python -m jobs.make_tasks quick_sweep quick_final quick_best
    A=$(sub quick_sweep 6); F=$(DEP=$A sub quick_final 1 0-0 0:30:00); B=$(DEP=$F sub quick_best 3)
    # then check $A, $F, $B (every END rc=0) and that code/neuro/ga_best.yaml here names all four games

Five generations, one seed: the 2D GA-sweep pipeline (probes, ablation, sweep, confirm), Mega Man's
section sweep and confirm, then every persona on its winners (`--best`). Delete the copy when it passes.

## Step 8: the full pipeline (stage by stage: check each, resubmit failed ids, then go on)

A resubmitted task keeps the probes that finished (balance and gasweep skip them), so a timeout only
costs the unfinished cells. Resubmit failed ids with `--array=<ids>`, after grouping by `NodeList`.

| stage | task file | cores | what it makes |
|---|---|---|---|
| 1 | `mutants` (1) | 1 | every level's mutants, from the levels as they are now |
| 2 | `make_tasks` again | — | the AST task lists read the mutants on disk |
| 3 | `sweep2d` (18) + `sweep_mm` (48) | 16 / 12 | 40-gen probes, ablation, GA sweeps |
| 4 | `gabest` (1) | 1 | every sweep report rebuilt, `code/neuro/ga_best.yaml` |
| 5 | `confirm_mm` (6) | 12 | Mega Man's composite per persona |
| 6 | `gabest` (1) | 1 | the final `ga_best.yaml` (print it, keep a copy) |
| 7 | `full` (372) | 3 | 200-gen agents, default GA and `--best`, every 2D level |
| 8 | `reports` (1) | 1 | `runs/balance/report_*` from the probe dirs |
| 9 | `ast_retrain` (~1040, in chunks: `sub ast_retrain 3 0-499`, then `OFFSET=500 sub ast_retrain 3 0-537`) | 3 | every retrained variant; Mega Man's base-retrained is its full training |
| 10 | `ast_play` (~510) | 1 | playtests, replays, random baselines |
| 11 | `final` (1) | 1 | `runs/ast/metrics.json` |

Then copy `runs/` home without `episodes.csv`, rebuild the pages at home (`python -m code.stats.ast_metrics
--page`, `python -m code.neuro.command_center --publish`) and commit.
