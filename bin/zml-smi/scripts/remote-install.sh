#!/usr/bin/env bash
set -euo pipefail

BASE_URL=""
BINARY_NAME="zml-smi"

resolve_dirs() {
  if [ -n "${ZML_SMI_INSTALL_DIR:-}" ] || [ -n "${ZML_SMI_BIN_DIR:-}" ]; then
    INSTALL_DIR="${ZML_SMI_INSTALL_DIR:-/usr/lib/zml-smi}"
    BIN_DIR="${ZML_SMI_BIN_DIR:-/usr/bin}"
    return
  fi

  INSTALL_DIR="/usr/lib/zml-smi"
  BIN_DIR="/usr/bin"

  if mkdir -p "$INSTALL_DIR" 2>/dev/null && [ -w "$INSTALL_DIR" ] &&
     mkdir -p "$BIN_DIR" 2>/dev/null && [ -w "$BIN_DIR" ]; then
    return
  fi

  warn "Cannot write to /usr/lib or /usr/bin — falling back to ~/.local"
  INSTALL_DIR="${HOME}/.local/lib/zml-smi"
  BIN_DIR="${HOME}/.local/bin"
}

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
warn()    { printf "${YELLOW}  !${RESET} %b\n" "$*" >&2; }
fail()    { printf "${RED}  ✗ %b${RESET}\n" "$*" >&2; exit 1; }
step()    { printf "\n${WHITE}  ▸ %b${RESET}\n" "$*"; }

print_logo() {
  printf "\n"
  printf "${BLUE}  █████╗███╗   ███╗██╗  ${RESET}\n"
  printf "${BLUE}  ╚═██╔╝████╗ ████║██║  ${RESET}\n"
  printf "${BLUE}   ██╔╝ ██╔████╔██║██║  ${RESET}\n"
  printf "${BLUE}  █████╗██║ ╚═╝ ██║████╗${RESET}\n"
  printf "${BLUE}  ╚════╝╚═╝     ╚═╝╚═══╝${RESET}\n"
}

detect_platform() {
  step "Detecting platform"

  OS="$(uname -s)"
  ARCH="$(uname -m)"

  case "$OS" in
    Linux)   OS_LABEL="linux" ;;
    Darwin)  OS_LABEL="macos" ;;
    MINGW*|MSYS*|CYGWIN*)
      fail "Windows is not supported. Use WSL2 instead." ;;
    *)
      fail "Unsupported operating system: ${OS}" ;;
  esac

  case "$ARCH" in
    x86_64|amd64)   ARCH_LABEL="x86_64" ;;
    aarch64|arm64)   ARCH_LABEL="aarch64" ;;
    *)
      fail "Unsupported architecture: ${ARCH}" ;;
  esac

  success "${OS_LABEL} ${ARCH_LABEL}"

  DOWNLOAD_URL="${BASE_URL}/${BINARY_NAME}-${OS_LABEL}-${ARCH_LABEL}.tar.zst"
}

check_deps() {
  step "Checking dependencies"
  local missing=()

  if command -v curl &>/dev/null; then
    DOWNLOADER="curl"
    success "curl"
  elif command -v wget &>/dev/null; then
    DOWNLOADER="wget"
    success "wget"
  else
    missing+=("curl or wget")
  fi

  if command -v zstd &>/dev/null; then
    success "zstd"
  else
    missing+=("zstd")
  fi

  if command -v tar &>/dev/null; then
    success "tar"
  else
    missing+=("tar")
  fi

  if [ ${#missing[@]} -gt 0 ]; then
    fail "Missing required tools: ${missing[*]}"
  fi
}

download_and_install() {
  step "Downloading & extracting"

  rm -rf "$INSTALL_DIR"
  mkdir -p "$INSTALL_DIR"

  if [ "$DOWNLOADER" = "curl" ]; then
    if ! curl -fsSL "$DOWNLOAD_URL" | tar --zstd -xf - --strip-components=1 -C "$INSTALL_DIR"; then
      fail "Download failed — is this OS/arch supported?"
    fi
  else
    if ! wget -q -O- "$DOWNLOAD_URL" | tar --zstd -xf - --strip-components=1 -C "$INSTALL_DIR"; then
      fail "Download failed — is this OS/arch supported?"
    fi
  fi

  chmod 755 "${INSTALL_DIR}/${BINARY_NAME}"
  success "Installed to ${INSTALL_DIR}/"

  mkdir -p "$BIN_DIR"
  ln -sf "${INSTALL_DIR}/${BINARY_NAME}" "${BIN_DIR}/${BINARY_NAME}"
  success "Symlink at ${BIN_DIR}/${BINARY_NAME}"
}

check_path() {
  if ! echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
    printf "\n"
    warn "${BIN_DIR} is not in your PATH"
    step "Add to your shell profile:"
    info "${CYAN}export PATH=\"${BIN_DIR}:\$PATH\""
    printf "\n"
  fi
}

main() {
  print_logo
  resolve_dirs
  detect_platform
  check_deps

  info "Install to:  ${CYAN}${INSTALL_DIR}${RESET}"
  info "Bin dir:     ${CYAN}${BIN_DIR}${RESET}"

  download_and_install
  check_path

  printf "\n"
  printf "  ${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
  printf "  ${GREEN}  Installation complete!${RESET}\n"
  printf "  ${DIM}  Run '${BINARY_NAME}' to get started.${RESET}\n"
  printf "  ${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
  printf "\n"
}

main
