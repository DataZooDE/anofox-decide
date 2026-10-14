#!/usr/bin/env bash
# Fail unless DECIDE_PLUGIN_RELEASE_TAG (src/include/decide_plugin_artifacts.hpp) equals the tag being released.
#
# decide_download_runtime fetches the GPU plugins from the release that constant names. A release whose source
# still points at the previous tag would hand every user the previous release's plugins, and the loader accepts
# them whenever the ABI version has not changed, so nothing would tell them. Used by the release workflows;
# runnable by hand: tools/ci/check_plugin_pin.sh v2026.10.14
set -euo pipefail

tag="${1:-${GITHUB_REF_NAME:-}}"
header="$(dirname "$0")/../../src/include/decide_plugin_artifacts.hpp"
if [ -z "${tag}" ]; then
  echo "usage: $0 <tag>" >&2
  exit 2
fi
pinned="$(sed -n 's/.*DECIDE_PLUGIN_RELEASE_TAG = "\(.*\)".*/\1/p' "${header}")"
echo "pinned in source: ${pinned:-<not found>}"
echo "tag being built:  ${tag}"
if [ -z "${pinned}" ] || [ "${pinned}" != "${tag}" ]; then
  echo "::error::DECIDE_PLUGIN_RELEASE_TAG is '${pinned}' but this is '${tag}'." \
       "Bump it in src/include/decide_plugin_artifacts.hpp before tagging, so the release contains" \
       "code that points at itself."
  exit 1
fi
