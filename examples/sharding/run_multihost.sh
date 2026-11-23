#!/usr/bin/env bash
# Runs examples/sharding as one process per host across several CUDA machines.
#
#   examples/sharding/run_multihost.sh [options] HOST... [-- EXTRA_SHARDING_ARGS...]
#
#   examples/sharding/run_multihost.sh 4090 5090
#   examples/sharding/run_multihost.sh --gpus=1 --coordinator=10.0.0.12:8476 4090 5090 -- --partitioner=gspmd
#
# HOSTs are ssh aliases. The first one is process 0 and hosts the coordinator.
# Steps: sync the local tree to every host with `ssync --once`, build in
# parallel on every host, then start every process at the same time.
#
# Options:
#   --gpus=N            GPUs per host (default: the smallest GPU count across
#                       hosts, rounded down to a power of two; every process
#                       must own the same number of devices)
#   --coordinator=IP:PORT
#                       address the other hosts use to reach process 0
#                       (default: process 0's address as resolved by process 1,
#                       port 8476)
#   --port=PORT         coordinator port when it's resolved automatically
#   --bazel-flag=FLAG   extra flag for bazel build/run (repeatable)
#   --no-sync           skip ssync
#   --no-build          skip the separate build step
#   --dry-run           print what would run on each host, then exit
#
# Logs go to $TMPDIR/zml-multihost-<namespace>/<host>.log and to the terminal,
# prefixed with the host.
set -euo pipefail

gpus=
coordinator=
port=8476
sync=1
build=1
dry_run=0
bazel_flags=(--@zml//platforms:cuda=true)
hosts=()
extra_args=()

usage() { sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'; }

while (($#)); do
  case $1 in
    --gpus=*)        gpus=${1#--gpus=} ;;
    --coordinator=*) coordinator=${1#--coordinator=} ;;
    --port=*)        port=${1#--port=} ;;
    --bazel-flag=*)  bazel_flags+=("${1#--bazel-flag=}") ;;
    --no-sync)       sync=0 ;;
    --no-build)      build=0 ;;
    --dry-run)       dry_run=1 ;;
    -h|--help)       usage; exit 0 ;;
    --)              shift; extra_args=("$@"); break ;;
    -*)              echo "unknown option: $1" >&2; exit 2 ;;
    *)               hosts+=("$1") ;;
  esac
  shift
done

if ((${#hosts[@]} == 0)); then
  usage >&2
  exit 2
fi

is_power_of_two() { (($1 > 0 && ($1 & ($1 - 1)) == 0)); }

log() { printf '\033[1m==> %s\033[0m\n' "$*" >&2; }

# Streams a command's output to the terminal with a [host] prefix and to a log file.
prefixed() {
  local host=$1 logfile=$2
  shift 2
  "$@" 2>&1 | tee -a "$logfile" | while IFS= read -r line; do printf '[%s] %s\n' "$host" "$line"; done
  return "${PIPESTATUS[0]}"
}

process_count=${#hosts[@]}
if ! is_power_of_two "$process_count"; then
  echo "error: ZML needs a power-of-two device count; use 1, 2, 4 or 8 hosts (got $process_count)" >&2
  exit 2
fi

namespace="zml-sharding-$(date +%Y%m%d-%H%M%S)-$$"
log_dir="${TMPDIR:-/tmp}/zml-multihost-$namespace"
mkdir -p "$log_dir"

# Rank 0 is often a different host than this Mac, so stop remote processes on Ctrl-C.
pids=()
cleanup() {
  trap - INT TERM
  log "Stopping remote processes"
  for host in "${hosts[@]}"; do
    ssh -o BatchMode=yes "$host" "pkill -f -- '--namespace=$namespace'" >/dev/null 2>&1 || true
  done
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  exit 130
}
trap cleanup INT TERM

if ((sync)) && ((!dry_run)); then
  for host in "${hosts[@]}"; do
    log "Syncing to $host"
    ssync --once "$host"
  done
fi

if [[ -z $gpus ]]; then
  gpus=
  for host in "${hosts[@]}"; do
    count=$(ssh -o BatchMode=yes "$host" 'nvidia-smi -L | wc -l' | tr -d '[:space:]')
    log "$host has $count GPUs"
    if [[ -z $gpus ]] || ((count < gpus)); then gpus=$count; fi
  done
  # Round down to a power of two.
  p=1
  while ((p * 2 <= gpus)); do p=$((p * 2)); done
  gpus=$p
fi
if ! is_power_of_two "$gpus"; then
  echo "error: --gpus must be a power of two (got $gpus)" >&2
  exit 2
fi
local_devices=0
for ((i = 1; i < gpus; i++)); do local_devices+=",$i"; done

if [[ -z $coordinator ]]; then
  rank0_name=$(ssh -G "${hosts[0]}" | awk '/^hostname /{print $2}')
  if ((process_count > 1)); then
    # The address process 1 resolves for process 0 is the one every other process will dial.
    ip=$(ssh -o BatchMode=yes "${hosts[1]}" "getent ahostsv4 '$rank0_name' | awk 'NR==1{print \$1}'" || true)
  fi
  if [[ -z ${ip:-} || $ip == 127.* ]]; then
    ip=$(ssh -o BatchMode=yes "${hosts[0]}" "hostname -I | awk '{print \$1}'")
  fi
  coordinator="$ip:$port"
fi

log "Namespace $namespace"
log "Coordinator $coordinator on ${hosts[0]}, $process_count processes x $gpus GPUs (local devices $local_devices)"
log "Logs in $log_dir"

remote_quote() { printf '%q ' "$@"; }

if ((build)); then
  build_cmd="cd ~/zml && bazel build $(remote_quote "${bazel_flags[@]}") //examples/sharding"
  if ((dry_run)); then
    for host in "${hosts[@]}"; do echo "[$host] $build_cmd"; done
  else
    log "Building on ${hosts[*]}"
    pids=()
    for host in "${hosts[@]}"; do
      prefixed "$host" "$log_dir/$host.log" ssh -o BatchMode=yes "$host" "$build_cmd" &
      pids+=($!)
    done
    failed=0
    for i in "${!pids[@]}"; do
      if ! wait "${pids[$i]}"; then
        echo "error: build failed on ${hosts[$i]} (see $log_dir/${hosts[$i]}.log)" >&2
        failed=1
      fi
    done
    ((failed)) && exit 1
  fi
fi

log "Starting processes"
pids=()
for rank in "${!hosts[@]}"; do
  host=${hosts[$rank]}
  args=(
    --target=cuda
    "--coordinator=$coordinator"
    "--process-index=$rank"
    "--process-count=$process_count"
    "--local-devices=$local_devices"
    "--namespace=$namespace"
    ${extra_args[@]+"${extra_args[@]}"}
  )
  run_cmd="cd ~/zml && bazel run $(remote_quote "${bazel_flags[@]}") //examples/sharding -- $(remote_quote "${args[@]}")"
  if ((dry_run)); then
    echo "[$host] $run_cmd"
    continue
  fi
  prefixed "$host" "$log_dir/$host.log" ssh -o BatchMode=yes "$host" "$run_cmd" &
  pids+=($!)
done
((dry_run)) && exit 0

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "error: process $i failed on ${hosts[$i]} (see $log_dir/${hosts[$i]}.log)" >&2
    failed=1
  fi
done
if ((failed)); then
  exit 1
fi
log "All $process_count processes finished"
