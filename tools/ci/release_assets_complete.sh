#!/usr/bin/env bash
# Prints the GPU release assets that are still missing from a release, one per line, and exits 0 only when none is.
#
# The release workflow keeps the release a DRAFT until every plugin and its sha256 sidecar is attached, so a
# user (or decide_accelerate()) never meets a plugin without its sidecar, or a release with only some plugins.
# The six names are what decide_download_runtime asks for (src/include/decide_plugin_artifacts.hpp:
# DecidePluginFileNameFor and DecidePluginSha256AssetName). Reads asset names from stdin (one per line) when a
# second argument is "-" (used by the self-test), otherwise from the repository's releases (drafts included).
#
#   tools/ci/release_assets_complete.sh v2026.10.14            # needs gh and a token
#   printf '%s\n' a b | tools/ci/release_assets_complete.sh v2026.10.14 -
set -euo pipefail

tag="${1:?usage: $0 <tag> [-]}"
required=(
  libanofox_decide_cuda_plugin.so
  cuda_plugin.sha256
  libanofox_decide_migraphx_plugin.so
  migraphx_plugin.sha256
  libanofox_decide_mlx_plugin.dylib
  mlx_plugin.sha256
)

if [ "${2:-}" = "-" ]; then
  present="$(cat)"
else
  # The releases list, not `gh release view`: a tag lookup does not find a draft.
  present="$(gh api "repos/${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is not set}/releases?per_page=100" --paginate \
    --jq ".[] | select(.tag_name == \"${tag}\") | .assets[].name")"
fi

missing=0
for name in "${required[@]}"; do
  if ! grep -qxF -- "${name}" <<<"${present}"; then
    echo "${name}"
    missing=1
  fi
done
exit "${missing}"
