#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

neuron_explorer_bin="$0.runfiles/+neuron_packages+libpjrt_neuron/sandbox/bin/neuron-explorer"
data_path="$1"
default_profile_root="$2"
shift 2

mkdir -p "${data_path}"

ingest_run() {
  local run_dir="$1"
  local execution_dir="${run_dir}/execution"
  local run_id
  run_id="$(basename "${run_dir}")"

  if [[ ! -d "${execution_dir}" ]]; then
    echo "skipping ${run_dir}: missing execution directory" >&2
    return
  fi

  for profile_session in "${execution_dir}"/*/*; do
    [[ -f "${profile_session}/ntrace.pb" ]] || continue

    local system_dir
    system_dir="$(mktemp -d)"
    for file in ntrace.pb trace_info.pb cpu_util.pb host_mem.pb; do
      [[ -f "${profile_session}/${file}" ]] && cp "${profile_session}/${file}" "${system_dir}/${file}"
    done

    "${neuron_explorer_bin}" view \
      --ingest-only \
      --force \
      --data-path "${data_path}" \
      --display-name "${run_id}:system" \
      -d "${system_dir}"

    rm -rf "${system_dir}"
  done

  for neff in "${execution_dir}"/*/*/*.neff; do
    local neff_name
    local program_name
    local ntff_prefix
    neff_name="$(basename "${neff}" .neff)"
    program_name="$(grep -aEom1 'MODULE_[A-Za-z0-9_]+' "${neff}" | sed -E 's/^MODULE_//; s/_[0-9]+$//')"
    if [[ -z "${program_name}" ]]; then
      echo "failed to find program name in ${neff}" >&2
      exit 1
    fi
    ntff_prefix="$(dirname "${neff}")/${neff_name}"

    if ! compgen -G "${ntff_prefix}_rank_*.ntff" >/dev/null && [[ ! -f "${ntff_prefix}.ntff" ]]; then
      "${neuron_explorer_bin}" capture \
        -n "${neff}" \
        -s "${ntff_prefix}.ntff" \
        --num-exec=2 \
        --profile-nth-exec=2 \
        --io-from=neff
    fi

    for ntff in "${ntff_prefix}"_rank_*.ntff "${ntff_prefix}".ntff; do
      [[ -f "${ntff}" ]] || continue

      local core_suffix
      local display_name
      core_suffix="$(basename "${ntff}" .ntff | sed -E 's/^.*_rank_([0-9]+).*$/nc\1/; s/^.*_(vnc_[0-9]+)$/\1/')"
      display_name="${run_id}:${program_name}_${core_suffix}"

      "${neuron_explorer_bin}" view \
        --ingest-only \
        --force \
        --data-path "${data_path}" \
        --display-name "${display_name}" \
        -n "${neff}" \
        -s "${ntff}"
    done
  done
}

[[ $# -eq 0 ]] && set -- "${default_profile_root}"

for path in "$@"; do
  if [[ -d "${path}/execution" ]]; then
    ingest_run "${path}"
  else
    for run_dir in "${path}"/*; do
      [[ -d "${run_dir}" ]] && ingest_run "${run_dir}"
    done
  fi
done
