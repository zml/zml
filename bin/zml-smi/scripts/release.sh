#!/usr/bin/env bash
set -euo pipefail

# --- UI ---

if [ -t 1 ]; then
  DIM='\033[2m'
  RESET='\033[0m'
  BLUE='\033[1;38;2;140;180;255m'
  GREEN='\033[1;32m'
  RED='\033[1;31m'
  YELLOW='\033[1;33m'
  CYAN='\033[1;36m'
  WHITE='\033[1;37m'
else
  DIM='' RESET='' BLUE='' GREEN='' RED='' YELLOW='' CYAN='' WHITE=''
fi

info()    { printf "${BLUE}  │${RESET} %b\n" "$*"; }
success() { printf "${GREEN}  ✓${RESET} %b\n" "$*"; }
warn()    { printf "${YELLOW}  !${RESET} %b\n" "$*"; }
fail()    { printf "${RED}  ✗ %b${RESET}\n" "$*" >&2; exit 1; }
step()    { printf "\n${WHITE}  ▸ %b${RESET}\n" "$*"; }

confirm() {
  if [ "$YES" = true ]; then
    return 0
  fi
  printf "${WHITE}  ? %b ${DIM}[y/N]${RESET} " "$*"
  local answer
  read -r answer
  case "$answer" in
    y|Y|yes|Yes) return 0 ;;
    *) return 1 ;;
  esac
}

usage() {
  printf >&2 "
  ${WHITE}Usage${RESET}
    bazel run //bin/zml-smi:release -- [flags] <version>

  ${WHITE}Arguments${RESET}
    ${CYAN}version${RESET}         Semantic version, e.g. ${DIM}1.0.0${RESET} (tagged as ${DIM}zml-smi-v1.0.0${RESET})

  ${WHITE}Flags${RESET}
    ${CYAN}--dry-run${RESET}       Build only — skip tagging, uploading, and publishing
    ${CYAN}--verify${RESET}        Verify uploaded files by comparing sha256 against the CDN
    ${CYAN}--publish${RESET}       Push git tag and create a GitHub release
    ${CYAN}-y, --yes${RESET}       Skip all confirmation prompts
    ${CYAN}-h, --help${RESET}      Show this help

  ${WHITE}Environment${RESET}
    ${CYAN}BUCKET${RESET}          S3-compatible bucket name
    ${CYAN}BASE_URL${RESET}        Public base URL serving the bucket
    ${CYAN}RCLONE_REMOTE${RESET}   Rclone remote name ${DIM}(default: r2)${RESET}

    ${DIM}If rclone is not configured, you can set the remote via env vars:${RESET}
    ${DIM}  RCLONE_CONFIG_R2_TYPE=s3${RESET}
    ${DIM}  RCLONE_CONFIG_R2_PROVIDER=Cloudflare${RESET}
    ${DIM}  RCLONE_CONFIG_R2_ACCESS_KEY_ID=your_access_key${RESET}
    ${DIM}  RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=your_secret_key${RESET}
    ${DIM}  RCLONE_CONFIG_R2_ENDPOINT=https://<account_id>.r2.cloudflarestorage.com${RESET}

  ${WHITE}Examples${RESET}
    ${DIM}# Interactive, local only${RESET}
    bazel run //bin/zml-smi:release -- 0.4.0

    ${DIM}# Full release, non-interactive${RESET}
    bazel run //bin/zml-smi:release -- --publish --verify --yes 0.4.0

"
  exit 0
}

# --- Checks ---

check_env() {
  [ -z "${BUILD_WORKSPACE_DIRECTORY:-}" ] && fail "Must be run via 'bazel run //bin/zml-smi:release -- <version>'"
  local missing=()
  [ -z "${BUCKET:-}" ] && missing+=("BUCKET")
  [ -z "${BASE_URL:-}" ] && missing+=("BASE_URL")
  if [ ${#missing[@]} -gt 0 ]; then
    fail "Missing required environment variables: ${CYAN}${missing[*]}"
  fi
}

# --- Steps ---

changelog() {
  local prev_tag
  prev_tag="$(git describe --tags --abbrev=0 --match 'zml-smi-v*' 2>/dev/null || true)"
  local range="${prev_tag:+${prev_tag}..HEAD}"

  git log ${range:-HEAD} --pretty=format:"- %s" -- bin/zml-smi/ | grep -vi WIP || true
}

tag() {
  step "Tagging ${CYAN}${TAG}${WHITE}"

  # Cache changelog before tagging, otherwise the new tag makes the range empty
  CACHED_CHANGELOG="$(changelog)"

  if [ -z "$CACHED_CHANGELOG" ]; then
    warn "No changes since last release"
  fi

  if git rev-parse "$TAG" &>/dev/null; then
    warn "Tag ${TAG} already exists, skipping"
    return
  fi

  if [ "$DRY_RUN" = true ]; then
    warn "Dry run — skipping tag creation"
  elif confirm "Create tag ${CYAN}${TAG}${RESET}${WHITE}?"; then
    git tag -a "$TAG" -m "$(printf '%s\n\n%s' "$TAG" "$CACHED_CHANGELOG")"
    success "Created tag ${CYAN}${TAG}"
  else
    warn "Skipped tagging"
    return
  fi
  printf "\n"

  printf "${DIM}"
  echo "$CACHED_CHANGELOG"
  printf "${RESET}"
}

build() {
  step "Building release archives"

  if ! confirm "Build all platforms?"; then
    warn "Skipped build"
    return 1
  fi

  printf "${DIM}%s${RESET}\n" "────────────────────────────────────────"

  bazel build --config=release //bin/zml-smi:archives

  printf "${DIM}%s${RESET}\n" "────────────────────────────────────────"
  mapfile -t FILES < <(bazel cquery --config=release --output=files //bin/zml-smi:archives 2>/dev/null)
  success "Built ${CYAN}${#FILES[@]}${RESET}${GREEN} artifacts"
}

upload() {
  local rclone_remote="${RCLONE_REMOTE:-r2}"
  local install_url="${BASE_URL}/zml-smi/${VERSION}"

  step "Patching install script"

  INSTALL_SCRIPT="$(mktemp)"
  sed -e "s|^BASE_URL=.*|BASE_URL=\"${install_url}\"|" \
      -e "s|^VERSION=.*|VERSION=\"${VERSION}\"|" \
    "${BUILD_WORKSPACE_DIRECTORY}/bin/zml-smi/scripts/remote-install.sh" > "$INSTALL_SCRIPT"
  success "BASE_URL=${CYAN}${install_url}"

  if [ "$DRY_RUN" = true ]; then
    step "Uploading to ${CYAN}${rclone_remote}:${BUCKET}"
    warn "Dry run — skipping upload"
    return
  fi

  step "Uploading to ${CYAN}${rclone_remote}:${BUCKET}"

  if ! confirm "Upload artifacts?"; then
    warn "Skipped upload"
    return
  fi

  for f in "${FILES[@]}"; do
    if [ ! -f "$f" ]; then
      warn "SKIP $(basename "$f") ${DIM}(not built for this host)${RESET}"
      continue
    fi
    local name remote_name
    name="$(basename "$f")"
    remote_name="${name/zml-smi-/zml-smi-v${VERSION}-}"
    rclone copyto "$f" "${rclone_remote}:${BUCKET}/zml-smi/${VERSION}/${remote_name}"
    success "${remote_name}"
  done

  rclone copyto "$INSTALL_SCRIPT" "${rclone_remote}:${BUCKET}/zml-smi/${VERSION}/install.sh"
  success "zml-smi/${VERSION}/install.sh"

  if confirm "Set ${CYAN}${VERSION}${RESET}${WHITE} as the default install version?"; then
    rclone copyto "$INSTALL_SCRIPT" "${rclone_remote}:${BUCKET}/zml-smi/install.sh"
    success "zml-smi/install.sh"
    DEFAULT_INSTALL=true
  else
    warn "Skipped default install script"
    DEFAULT_INSTALL=false
  fi
}

verify() {
  local install_url="${BASE_URL}/zml-smi/${VERSION}"

  step "Verifying uploads"

  for f in "${FILES[@]}"; do
    if [ ! -f "$f" ]; then
      continue
    fi
    local name remote_name
    name="$(basename "$f")"
    remote_name="${name/zml-smi-/zml-smi-v${VERSION}-}"
    local local_sha remote_sha
    local_sha="$(shasum -a 256 "$f" | awk '{print $1}')"
    remote_sha="$(curl -fsSL "${install_url}/${remote_name}" | shasum -a 256 | awk '{print $1}')"
    if [ "$local_sha" = "$remote_sha" ]; then
      success "${remote_name}"
    else
      fail "${remote_name} — sha256 mismatch (local: ${local_sha}, remote: ${remote_sha})"
    fi
  done

  local local_sha remote_sha
  local_sha="$(shasum -a 256 "$INSTALL_SCRIPT" | awk '{print $1}')"
  remote_sha="$(curl -fsSL "${install_url}/install.sh" | shasum -a 256 | awk '{print $1}')"
  if [ "$local_sha" = "$remote_sha" ]; then
    success "install.sh"
  else
    fail "install.sh — sha256 mismatch (local: ${local_sha}, remote: ${remote_sha})"
  fi
}

release_notes() {
  local install_url="${BASE_URL}/zml-smi/${VERSION}"
  local log="$CACHED_CHANGELOG"

  local downloads=""
  for f in "${FILES[@]}"; do
    [ -f "$f" ] || continue
    local name remote_name
    name="$(basename "$f")"
    [[ "$name" == *.sha256 ]] && continue
    local sha_file=""
    for sf in "${FILES[@]}"; do
      if [ "$(basename "$sf")" = "${name}.sha256" ]; then
        sha_file="$sf"
        break
      fi
    done
    if [ -z "$sha_file" ] || [ ! -f "$sha_file" ]; then
      fail "Missing sha256 sidecar for ${name}"
    fi
    local sha
    sha="$(awk '{print $1}' "$sha_file")"
    local actual
    actual="$(shasum -a 256 "$f" | awk '{print $1}')"
    if [ "$sha" != "$actual" ]; then
      fail "${name} sha256 mismatch — sidecar: ${sha}, actual: ${actual}"
    fi
    remote_name="${name/zml-smi-/zml-smi-v${VERSION}-}"
    downloads="${downloads}| [${remote_name}](${install_url}/${remote_name}) | \`${sha}\` |"$'\n'
  done

  cat <<EOF
## Changelog

${log}

## Install

\`\`\`
curl -fsSL ${BASE_URL}/zml-smi/install.sh | sudo bash
\`\`\`

## Downloads

| Archive | SHA-256 |
|---------|---------|
${downloads}
EOF
}

github_release() {
  step "Creating GitHub release"

  local notes
  notes="$(release_notes)"

  if [ "$DRY_RUN" = true ]; then
    warn "Dry run — skipping GitHub release"
    info "Title: ${CYAN}${TAG}"
    info "Notes:"
    printf "${DIM}%s${RESET}\n" "$notes"
    return
  fi

  if ! confirm "Create GitHub release ${CYAN}${TAG}${RESET}${WHITE}?"; then
    warn "Skipped GitHub release"
    return
  fi

  printf "${DIM}%s${RESET}\n" "────────────────────────────────────────"
  gh release create "$TAG" \
    --title "$TAG" \
    --notes "$notes"
  printf "${DIM}%s${RESET}\n" "────────────────────────────────────────"
  success "Created release ${CYAN}${TAG}"
}

push_tag() {
  step "Pushing tag"

  if [ "$DRY_RUN" = true ]; then
    warn "Dry run — skipping push"
    return
  fi

  if ! confirm "Push ${CYAN}${TAG}${RESET}${WHITE} to origin?"; then
    warn "Skipped push"
    return
  fi

  printf "${DIM}%s${RESET}\n" "────────────────────────────────────────"
  git push origin "$TAG"
  printf "${DIM}%s${RESET}\n" "────────────────────────────────────────"
  success "Pushed ${CYAN}${TAG}${RESET}${GREEN} to origin"
}

# --- Main ---

main() {
  DRY_RUN=false
  VERIFY=false
  PUBLISH=false
  YES=false

  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run)  DRY_RUN=true; shift ;;
      --verify)   VERIFY=true; shift ;;
      --publish)  PUBLISH=true; shift ;;
      -y|--yes)   YES=true; shift ;;
      -h|--help)  usage ;;
      -*)         fail "Unknown flag: $1" ;;
      *)          break ;;
    esac
  done

  VERSION="${1:-}"
  [ -z "$VERSION" ] && usage

  check_env

  TAG="zml-smi-v${VERSION}"

  local title="zml-smi release ${TAG}"
  if [ "$DRY_RUN" = true ]; then
    title="${title} (dry run)"
  fi
  local pad
  pad="$(printf '═%.0s' $(seq 1 $((${#title} + 4))))"

  printf "\n"
  printf "${BLUE}  ╔${pad}╗${RESET}\n"
  printf "${BLUE}  ║${RESET}  ${WHITE}${title}${RESET}  ${BLUE}║${RESET}\n"
  printf "${BLUE}  ╚${pad}╝${RESET}\n"

  cd "$BUILD_WORKSPACE_DIRECTORY"
  tag
  build
  upload
  if [ "$VERIFY" = true ] && [ "$DRY_RUN" = false ]; then
    verify
  fi
  if [ "$PUBLISH" = true ] && [ "${DEFAULT_INSTALL:-true}" = true ]; then
    push_tag
    github_release
  fi
  rm -f "${INSTALL_SCRIPT:-}"

  printf "\n"
  if [ "$DRY_RUN" = true ]; then
    printf "  ${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
    printf "  ${YELLOW}  Dry run complete - nothing was published${RESET}\n"
    printf "  ${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
  else
    printf "  ${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
    printf "  ${GREEN}  Release complete!${RESET}\n"
    if [ "${DEFAULT_INSTALL:-true}" = true ]; then
      printf "  ${DIM}  curl -fsSL ${BASE_URL}/zml-smi/install.sh | sudo bash${RESET}\n"
    else
      printf "  ${DIM}  curl -fsSL ${BASE_URL}/zml-smi/${VERSION}/install.sh | sudo bash${RESET}\n"
    fi
    printf "  ${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
  fi
  printf "\n"
}

main "$@"
