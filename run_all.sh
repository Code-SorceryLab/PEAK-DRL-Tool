#!/usr/bin/env bash
# PEAK runs: ./run_all.sh sweeps | megaman | ast-megaman | test   (inside a Slurm job, or locally: WORKERS=4 ./run_all.sh ...)
cd "$(dirname "$0")"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
PY=.venv/bin/python; [[ -x $PY ]] || PY=.venv/Scripts/python.exe  # Windows venv layout
W=${WORKERS:-${SLURM_CPUS_PER_TASK:-4}}
GAMES=(${GAMES:-mario meatboy bomberman})  # GAMES=meatboy ./run_all.sh sweeps: one game per job
PERSONAS=(${PERSONAS:-novice clumsy experienced speedrunner killer collector})  # PERSONAS=clumsy: one persona per job
SEEDS=(1234 2025 31337); GENS=40
SECTIONS=(CutMan-1-Street CutMan-2-Climb CutMan-3-Ledges CutMan-4-Shaft CutMan-5-Rooftop CutMan-6-Descent CutMan-7-Corridor CutMan-8-Boss)
stamp() { echo "[$(date '+%F %T')] $*"; }
throttle() { while (( $(jobs -rp | wc -l) >= W )); do wait -n; done; }
probe() { $PY -m code.neuro.balance --game "$1" --persona "$2" --gens $GENS --seeds "${SEEDS[@]}" --workers $W "${@:3}"; }
train() {  # section seed gens [warm-start run]
  local extra=(); [[ -n $4 ]] && extra=(--seed-from "$4")
  $PY -m code.neuro.trainer --game megaman --level "$1" --seed "$2" --gens "$3" --no-serve \
      --run-dir "runs/megaman/$1/seed$2" "${extra[@]}" > "runs/megaman/logs/$1-seed$2.log" 2>&1
  stamp "END train $1 seed$2 rc=$?"
}

sweeps() {
  stamp "PHASE 1 full sweep (rays)"
  for g in "${GAMES[@]}"; do for p in "${PERSONAS[@]}"; do
    stamp "START full $g $p"; probe $g $p --sensors rays; stamp "END full $g $p rc=$?"
  done; done
  stamp "PHASE 2 sensor ablation"
  for g in "${GAMES[@]}"; do for p in "${PERSONAS[@]}"; do for s in rays grid; do
    stamp "START ablation $g $p $s"; probe $g $p --sensors $s --ablation; stamp "END ablation $g $p $s rc=$?"
  done; done; done
  for g in "${GAMES[@]}"; do for p in "${PERSONAS[@]}"; do probe $g $p --compare --ablation; done; done
  stamp "PHASE 3 GA sweep"
  for g in "${GAMES[@]}"; do for p in "${PERSONAS[@]}"; do
    stamp "START gasweep $g $p"
    $PY -m code.neuro.gasweep --game $g --persona $p --gens $GENS --seeds "${SEEDS[@]}" --workers $W --sensors rays --confirm
    stamp "END gasweep $g $p rc=$?"
  done; done
  $PY -m code.neuro.command_center
  stamp "ALL DONE"
}

megaman() {
  mkdir -p runs/megaman/logs
  stamp "MM PHASE 1 train 80 gens"
  for s in "${SECTIONS[@]}"; do for seed in "${SEEDS[@]}"; do throttle; train "$s" "$seed" 80 & done; done; wait
  stamp "MM PICK 1"; $PY -m code.neuro.relay --pick; cp runs/megaman/relay.json runs/megaman/relay_round1.json
  stamp "MM PHASE 2 fine-tune 150 gens"
  for s in "${SECTIONS[@]}"; do
    src=$($PY -c "import json;print(json.load(open('runs/megaman/relay.json'))['legs']['$s']['run'])")
    for seed in 4001 4002 4003; do throttle; train "$s" "$seed" 150 "$src" & done
  done; wait
  stamp "MM PICK 2"; $PY -m code.neuro.relay --pick
  stamp "MM RELAY"; $PY -m code.neuro.relay --episodes 20
  for s in "${SECTIONS[@]}"; do $PY -m code.neuro.relay --section "$s" --episodes 20; done
  stamp "MM ALL DONE"
}

ast_megaman() {  # one Slurm array task (--array=0-$((19 * personas - 1))): a persona's section of base-retrained or a Mega Man mutant
  local tasks=() p v s
  for p in "${PERSONAS[@]}"; do
    for v in base-retrained m-jump_velocity_down; do for s in "${SECTIONS[@]}"; do tasks+=("$p $v $s"); done; done
    for v in s3-seal_route s3-widen_gap s3-off_route; do tasks+=("$p $v CutMan-3-Ledges"); done
  done
  local id=${SLURM_ARRAY_TASK_ID:?run it as an array task}
  (( id < ${#tasks[@]} )) || { echo "task $id: there are only ${#tasks[@]} (0-$(( ${#tasks[@]} - 1 )))"; exit 2; }
  set -- ${tasks[$id]}
  stamp "START ast megaman $*"
  $PY -m code.neuro.ast_eval retrain --game megaman --persona "$1" --mutant "$2" --level "$3" --workers $W
  stamp "END ast megaman $* rc=$?"
}

smoke() {  # a few minutes: proves the install, the personas and Mega Man work on this machine
  $PY -m pytest code/tests -q -p no:debugging || exit 1
  GENS=5; SEEDS=(1234)
  probe mario clumsy --sensors rays --out runs/_test
  mkdir -p runs/_test/logs
  $PY -m code.neuro.trainer --game megaman --level CutMan-7-Corridor --seed 1 --gens 5 --no-serve --run-dir runs/_test/megaman
  stamp "TEST DONE"
}

case "$1" in
  sweeps) sweeps ;; megaman) megaman ;; ast-megaman) ast_megaman ;; test) smoke ;;
  *) echo "usage: $0 sweeps|megaman|ast-megaman|test"; exit 2 ;;
esac
