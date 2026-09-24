#!/usr/bin/env bash
set -euo pipefail

tool_dir="${RUNNER_TEMP}/quality-tools"
mkdir -p "$tool_dir"

install_archive() {
  local url="$1"
  local archive="$2"
  local digest="$3"
  local binary="$4"

  curl --fail --location --silent --show-error "$url" --output "$tool_dir/$archive"
  echo "$digest  $tool_dir/$archive" | sha256sum --check --status
  tar -xzf "$tool_dir/$archive" -C "$tool_dir" "$binary"
  chmod +x "$tool_dir/$binary"
}

install_archive \
  "https://github.com/rhysd/actionlint/releases/download/v1.7.12/actionlint_1.7.12_linux_amd64.tar.gz" \
  "actionlint_1.7.12_linux_amd64.tar.gz" \
  "8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8" \
  "actionlint"

install_archive \
  "https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz" \
  "gitleaks_8.30.1_linux_x64.tar.gz" \
  "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb" \
  "gitleaks"

echo "$tool_dir" >> "$GITHUB_PATH"
