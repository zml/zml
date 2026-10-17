#!/usr/bin/env bash
# Sweep backend choices over an unchanged benchmark workload selection.
set -euo pipefail
if (( $# < 2 )); then
    echo "usage: $0 BENCH NEW_OUTPUT_DIRECTORY [benchmark workload arguments...]" >&2
    echo "TILES='8 16 32 64 128', QUANTIZE_LANES='1 4', PERSISTENT_CTAS='', NSYS=''" >&2
    exit 2
fi
bench=$(realpath "$1")
out=$(realpath -m "$2")
shift 2
# The driver owns output/validation options. In particular, --list can exit
# successfully without producing or checking a single baseline.
for arg in "$@"; do
    case "$arg" in
        --help|-h|--list|--list=*|--json|--json=*|--dump|--dump=*|--dump-dir|--dump-dir=*|--compare|--compare=*|--compare-dir|--compare-dir=*|--compare-exact|--compare-exact=*)
            printf 'The tuning driver owns this option: %s\n' "$arg" >&2
            exit 2 ;;
        --backend=cute) ;;
        --backend|--backend=*)
            printf 'The CuTe tuning driver requires --backend=cute\n' >&2
            exit 2 ;;
    esac
done
[[ -x $bench ]] || { printf 'Benchmark is not executable: %s\n' "$bench" >&2; exit 2; }
mkdir "$out"
mkdir "$out/baseline"
sha256sum "$bench" > "$out/benchmark.sha256"
# Default backend selection would also run unrelated implementations.
set -- --backend=cute "$@"
unset ZML_MOE_CUTE_TILE_N ZML_MOE_CUTE_TWO_CTA ZML_MOE_CUTE_QUANTIZE_LANES ZML_MOE_CUTE_PERSISTENT_CTAS
failures=0

run() {
    local name=$1
    shift
    local -a profiler=()
    if [[ -n ${NSYS:-} ]]; then
        export SKIP_PJRT_PROFILER=true
        profiler=("$NSYS" profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none "--output=$out/$name")
    fi
    printf 'Running %s\n' "$name"
    # Clear inherited overrides when replaying the saved command, then restore
    # the exact settings used here. JSON records the remaining provenance.
    local variable
    local -a command_env=(env)
    for variable in ZML_MOE_CUTE_TILE_N ZML_MOE_CUTE_TWO_CTA ZML_MOE_CUTE_QUANTIZE_LANES ZML_MOE_CUTE_PERSISTENT_CTAS CUDA_VISIBLE_DEVICES XLA_FLAGS SKIP_PJRT_PROFILER ZML_AUTOTUNE_CACHE_DIR; do
        command_env+=(-u "$variable")
    done
    for variable in ZML_MOE_CUTE_TILE_N ZML_MOE_CUTE_TWO_CTA ZML_MOE_CUTE_QUANTIZE_LANES ZML_MOE_CUTE_PERSISTENT_CTAS CUDA_VISIBLE_DEVICES XLA_FLAGS SKIP_PJRT_PROFILER ZML_AUTOTUNE_CACHE_DIR; do
        if [[ -v $variable ]]; then command_env+=("$variable=${!variable}"); fi
    done
    printf '%q ' "${command_env[@]}" "${profiler[@]}" "$bench" "$@" "--json=$out/$name.jsonl" > "$out/$name.command"
    printf '\n' >> "$out/$name.command"
    if ! "${profiler[@]}" "$bench" "$@" "--json=$out/$name.jsonl" > "$out/$name.stdout" 2> "$out/$name.stderr"; then
        printf 'FAILED: %s (see %s/%s.stderr)\n' "$name" "$out" "$name" >&2
        failures=$((failures + 1))
    fi
}

run default "$@" "--dump-dir=$out/baseline"
if (( failures )); then exit 1; fi
common=("$@" "--compare-dir=$out/baseline" --compare-exact)

# Pairing is automatic for wide prefill tiles and off for direct decode.
# Forcing single CTAs provides the matching control for each wide tile.
for tile in ${TILES-8 16 32 64 128}; do
    export ZML_MOE_CUTE_TILE_N=$tile
    run "tile-$tile-auto" "${common[@]}"
    if [[ $tile == 64 || $tile == 128 ]]; then
        export ZML_MOE_CUTE_TWO_CTA=false
        run "tile-$tile-single" "${common[@]}"
        unset ZML_MOE_CUTE_TWO_CTA
    fi
done
unset ZML_MOE_CUTE_TILE_N
for lanes in ${QUANTIZE_LANES-1 4}; do
    export ZML_MOE_CUTE_QUANTIZE_LANES=$lanes
    run "quantize-$lanes" "${common[@]}"
done
unset ZML_MOE_CUTE_QUANTIZE_LANES
for ctas in ${PERSISTENT_CTAS-}; do
    export ZML_MOE_CUTE_PERSISTENT_CTAS=$ctas
    run "ctas-$ctas" "${common[@]}"
done
printf '%d failed configurations; results: %s\n' "$failures" "$out"
(( failures == 0 ))
