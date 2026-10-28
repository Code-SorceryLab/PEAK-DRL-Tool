#!/bin/bash
#SBATCH --account=def-polako_cpu
#SBATCH --mem-per-cpu=2G
# One Slurm array task = one line of the task file $1 (jobs/make_tasks.py writes them), run from $PEAK_DIR
# (default $SCRATCH/peak), on any Alliance cluster. Name, array size, cores, time, log and broken nodes to
# exclude go on the sbatch line; see jobs/README.md:
#   sbatch -J peak-full --array=0-371 -c 3 -t 3:00:00 -o runs/logs/%x_%A_%a.out jobs/run_tasks.sh jobs/tasks/full.txt
cd "${PEAK_DIR:-$SCRATCH/peak}" || exit 1
module load python/3.11
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
PY=.venv/bin/python
W=${SLURM_CPUS_PER_TASK:-1}
n=$((SLURM_ARRAY_TASK_ID + ${OFFSET:-0} + 1))      # OFFSET=500: a long file in chunks (arrays count against the job limit)
line=$(sed -n "${n}p" "$1")
[ -n "$line" ] || { echo "no line $n in $1"; exit 2; }
echo "[$(date '+%F %T')] START n=$n host=$(hostname) $line"
# a broken node (its /cvmfs python fails: rc=126 at once) would grab task after task: hold it 10 min instead
$PY -c "" || { echo "$(hostname): $PY does not start"; sleep 600; exit 126; }
eval "$line"
rc=$?
echo "[$(date '+%F %T')] END rc=$rc $line"
exit $rc
