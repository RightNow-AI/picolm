#!/bin/bash
#
# PicoLM + PicoClaw installer for Raspberry Pi & Linux
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/RightNow-AI/picolm/main/install.sh | bash
#
# Or locally:
#   chmod +x install.sh && ./install.sh
#
set -e

# ---- Config ----
INSTALL_DIR="${PICOLM_DIR:-$HOME/.picolm}"
MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_NAME="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
THREADS=4
PICOCLAW_CONFIG_DIR="${HOME}/.picoclaw"

# Source fetch (used when running via curl | bash)
PICOLM_REPO="RightNow-AI/picolm"
PICOLM_REF="${PICOLM_REF:-main}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*"; exit 1; }

# ---- Helpers ----
fetch_picolm_src() {
  # Fetches picolm/ sources into "$INSTALL_DIR/src" when the script is not run from a repo checkout.
  # Prefers git if available; falls back to downloading a tarball via curl + tar.
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  info "Fetching PicoLM source (${PICOLM_REPO}@${PICOLM_REF})..."

  if command -v git >/dev/null 2>&1; then
    if git clone --depth=1 --branch "$PICOLM_REF" "https://github.com/${PICOLM_REPO}.git" "$tmp/repo" >/dev/null 2>&1; then
      cp "$tmp/repo"/picolm/*.c "$INSTALL_DIR/src/" 2>/dev/null || true
      cp "$tmp/repo"/picolm/*.h "$INSTALL_DIR/src/" 2>/dev/null || true
      cp "$tmp/repo"/picolm/Makefile "$INSTALL_DIR/src/" 2>/dev/null || true
      return 0
    fi
    warn "git clone failed, falling back to tarball download..."
  fi

  # Tarball fallback (works without git)
  local tar_url="https://codeload.github.com/${PICOLM_REPO}/tar.gz/${PICOLM_REF}"
  if ! curl -fsSL "$tar_url" -o "$tmp/src.tgz"; then
    return 1
  fi
  tar -xzf "$tmp/src.tgz" -C "$tmp" >/dev/null 2>&1 || return 1

  # Find extracted folder (name can vary: picolm-main, picolm-${ref}, etc.)
  local extracted
  extracted="$(find "$tmp" -maxdepth 1 -type d -name 'picolm-*' -o -name 'picolm' | head -n 1)"
  if [ -z "${extracted:-}" ] || [ ! -d "$extracted/picolm" ]; then
    return 1
  fi

  cp "$extracted"/picolm/*.c "$INSTALL_DIR/src/" 2>/dev/null || true
  cp "$extracted"/picolm/*.h "$INSTALL_DIR/src/" 2>/dev/null || true
  cp "$extracted"/picolm/Makefile "$INSTALL_DIR/src/" 2>/dev/null || true
}

# ---- Detect platform ----
ARCH=$(uname -m)
OS=$(uname -s)

info "Platform: ${OS} ${ARCH}"

if [ "$OS" != "Linux" ]; then
    warn "This script is designed for Linux (Raspberry Pi)."
    warn "On other platforms, build manually: cd picolm && make native"
fi

# Detect CPU count for thread count
if [ -f /proc/cpuinfo ]; then
    NCPU=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo)
    THREADS=$NCPU
    info "Detected ${NCPU} CPU cores, will use ${THREADS} threads"
fi

# Check free RAM
if [ -f /proc/meminfo ]; then
    MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    MEM_MB=$((MEM_KB / 1024))
    info "Available RAM: ${MEM_MB} MB"
    if [ "$MEM_MB" -lt 256 ]; then
        error "PicoLM needs at least 256 MB RAM. Detected: ${MEM_MB} MB"
    elif [ "$MEM_MB" -lt 512 ]; then
        warn "Low RAM (${MEM_MB} MB). Consider reducing context: picolm ... -c 512"
    fi
fi

# ---- Step 1: Install build dependencies ----
info "Checking build dependencies..."

DEPS_NEEDED=""
command -v gcc   >/dev/null 2>&1 || DEPS_NEEDED="$DEPS_NEEDED gcc"
command -v make  >/dev/null 2>&1 || DEPS_NEEDED="$DEPS_NEEDED make"
command -v curl  >/dev/null 2>&1 || DEPS_NEEDED="$DEPS_NEEDED curl"
command -v tar   >/dev/null 2>&1 || DEPS_NEEDED="$DEPS_NEEDED tar"

if [ -n "$DEPS_NEEDED" ]; then
    info "Installing: $DEPS_NEEDED"
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq $DEPS_NEEDED
    elif command -v apk >/dev/null 2>&1; then
        sudo apk add $DEPS_NEEDED
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y $DEPS_NEEDED
    else
        error "Cannot install deps. Please install manually: $DEPS_NEEDED"
    fi
fi

# ---- Step 2: Create directory structure ----
info "Creating directories..."
mkdir -p "$INSTALL_DIR/bin" "$INSTALL_DIR/models" "$INSTALL_DIR/src"

# ---- Step 3: Get picolm source ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

if [ -f "$SCRIPT_DIR/picolm/picolm.c" ]; then
    info "Using local source from $SCRIPT_DIR/picolm/"
    cp "$SCRIPT_DIR"/picolm/*.c "$INSTALL_DIR/src/" 2>/dev/null || true
    cp "$SCRIPT_DIR"/picolm/*.h "$INSTALL_DIR/src/" 2>/dev/null || true
    cp "$SCRIPT_DIR"/picolm/Makefile "$INSTALL_DIR/src/" 2>/dev/null || true
else
    warn "PicoLM source not found next to install.sh (likely running via curl | bash)."
    if ! fetch_picolm_src; then
        error "Failed to fetch PicoLM source. Try: git clone https://github.com/${PICOLM_REPO}.git && ./install.sh"
    fi
fi

# ---- Step 4: Build picolm ----
if [ -f "$INSTALL_DIR/src/picolm.c" ]; then
    info "Building picolm for ${ARCH}..."
    cd "$INSTALL_DIR/src"

    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        make pi
    elif [ "$ARCH" = "armv7l" ] || [ "$ARCH" = "armv6l" ]; then
        make pi-arm32
    else
        make native
    fi

    if [ -f picolm ]; then
        cp picolm "$INSTALL_DIR/bin/"
        info "Built: $INSTALL_DIR/bin/picolm"
    else
        error "Build failed — picolm binary not produced"
    fi
else
    error "Build skipped — source files missing in $INSTALL_DIR/src (unexpected)."
fi

# ---- Step 5: Download model ----
MODEL_PATH="$INSTALL_DIR/models/$MODEL_NAME"

if [ -f "$MODEL_PATH" ]; then
    info "Model already exists: $MODEL_PATH"
else
    info "Downloading TinyLlama 1.1B Q4_K_M (638 MB)..."
    info "This may take a while on slow connections."
    if ! curl -L --progress-bar -o "$MODEL_PATH" "$MODEL_URL"; then
        rm -f "$MODEL_PATH"
        error "Model download failed. Check your internet connection and try again."
    fi
    # Verify download isn't empty or truncated
    FILE_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -lt 100000000 ]; then
        rm -f "$MODEL_PATH"
        error "Downloaded file too small (${FILE_SIZE} bytes). Expected ~638MB. Try again."
    fi
    info "Downloaded: $MODEL_PATH"
fi

# ---- Step 6: Build PicoClaw (if Go is available) ----
PICOCLAW_SRC=""
if [ -d "$SCRIPT_DIR/picoclaw" ]; then
    PICOCLAW_SRC="$SCRIPT_DIR/picoclaw"
fi

if [ -n "$PICOCLAW_SRC" ] && command -v go >/dev/null 2>&1; then
    GO_VERSION=$(go version | awk '{print $3}')
    info "Found Go ($GO_VERSION) — building PicoClaw..."
    cd "$PICOCLAW_SRC"
    if make deps 2>/dev/null || go mod tidy; then
        if make build 2>/dev/null || go build -o picoclaw ./cmd/picoclaw/; then
            if [ -f picoclaw ]; then
                cp picoclaw "$INSTALL_DIR/bin/"
                info "Built: $INSTALL_DIR/bin/picoclaw"
            fi
        else
            warn "PicoClaw build failed. You can build manually later:"
            warn "  cd $PICOCLAW_SRC && make deps && make build"
        fi
    else
        warn "Failed to resolve Go dependencies."
    fi
    cd "$SCRIPT_DIR"
elif [ -n "$PICOCLAW_SRC" ]; then
    warn "Go not found — skipping PicoClaw build."
    warn "Install Go (https://go.dev/dl/) and re-run, or build manually:"
    warn "  cd picoclaw && make deps && make build"
fi

# ---- Step 7: Quick test ----
PICOLM="$INSTALL_DIR/bin/picolm"

if [ -x "$PICOLM" ] && [ -f "$MODEL_PATH" ]; then
    info "Running quick test..."
    RESULT=$("$PICOLM" "$MODEL_PATH" \
        -p "The capital of France is" \
        -n 10 -t 0 -j "$THREADS" 2>/dev/null) || true
    if [ -n "$RESULT" ]; then
        info "Test output: $RESULT"
    else
        warn "Quick test produced no output (model may need more context)"
    fi
    echo ""
fi

# ---- Step 8: Generate PicoClaw config ----
info "Generating PicoClaw config..."
mkdir -p "$PICOCLAW_CONFIG_DIR"

PICOCLAW_CFG="$PICOCLAW_CONFIG_DIR/config.json"

if [ -f "$PICOCLAW_CFG" ]; then
    warn "PicoClaw config already exists: $PICOCLAW_CFG"
    warn "To use PicoLM, ensure your providers section includes:"
    echo ""
    cat <<EOF
  "picolm": {
    "binary": "$INSTALL_DIR/bin/picolm",
    "model": "$MODEL_PATH",
    "max_tokens": 256,
    "threads": $THREADS,
    "template": "chatml"
  }
EOF
else
    cat > "$PICOCLAW_CFG" <<EOF
{
  "agents": {
    "defaults": {
      "provider": "picolm",
      "model": "picolm-local",
      "workspace": "$PICOCLAW_CONFIG_DIR/workspace",
      "restrict_to_workspace": true,
      "max_tokens": 256,
      "temperature": 0.7,
      "max_tool_iterations": 5
    }
  },
  "providers": {
    "picolm": {
      "binary": "$INSTALL_DIR/bin/picolm",
      "model": "$MODEL_PATH",
      "max_tokens": 256,
      "threads": $THREADS,
      "template": "chatml"
    }
  },
  "channels": {
    "telegram": {
      "enabled": false,
      "token": ""
    }
  },
  "gateway": {
    "host": "0.0.0.0",
    "port": 18790
  }
}
EOF
    info "Created: $PICOCLAW_CFG"
fi

# ---- Step 9: Add to PATH ----
PROFILE_FILE="$HOME/.bashrc"
[ -f "$HOME/.zshrc" ] && PROFILE_FILE="$HOME/.zshrc"

if ! grep -q "picolm/bin" "$PROFILE_FILE" 2>/dev/null; then
    echo "" >> "$PROFILE_FILE"
    echo "# PicoLM" >> "$PROFILE_FILE"
    echo "export PATH=\"\$PATH:$INSTALL_DIR/bin\"" >> "$PROFILE_FILE"
    info "Added $INSTALL_DIR/bin to PATH in $PROFILE_FILE"
fi

# ---- Done ----
echo ""
echo "============================================"
info "PicoLM installation complete!"
echo "============================================"
echo ""
echo "  Binary:  $INSTALL_DIR/bin/picolm"
echo "  Model:   $MODEL_PATH"
echo "  Config:  $PICOCLAW_CFG"
echo "  Threads: $THREADS"
echo ""
echo "  To activate PATH now (or open a new terminal):"
echo "    source $PROFILE_FILE"
echo ""
echo "  Quick test (PicoLM direct):"
echo "    picolm $MODEL_PATH -p \"Hello\" -n 50 -j $THREADS"
echo ""
echo "  Chat with PicoClaw agent:"
echo "    picoclaw agent -m \"What is the meaning of life?\""
echo ""
echo "  Interactive chat:"
echo "    picoclaw agent"
echo ""
echo "  Start Telegram/Discord gateway:"
echo "    picoclaw gateway"
echo ""
echo "  Memory usage: ~45 MB (PicoLM) + ~10 MB (PicoClaw) + model on disk via mmap"
echo ""