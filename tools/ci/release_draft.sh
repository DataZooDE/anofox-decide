#!/usr/bin/env bash
# One GitHub release per tag for the three GPU plugin workflows (CUDA, MIGraphX, MLX), which all run on the same tag.
#
# Why this exists: a draft release is not bound to its tag until it is published, so `gh release create --draft` in
# three racing workflows never fails with "already exists"; each creates its own draft. Before this script every
# release left two empty drafts behind, and `gh release upload <tag>` picked whichever draft the API listed first,
# so the assets could in principle split across drafts. Here every step works on ONE release, the CANONICAL one:
# the release with the tag and the lowest id (the first created). Ids only grow, so every racer agrees on it.
#
#   tools/ci/release_draft.sh ensure  <tag>            # create the draft unless one exists, drop empty duplicates, print the canonical id
#   tools/ci/release_draft.sh id      <tag>            # print the canonical id (fails when there is no release)
#   tools/ci/release_draft.sh assets  <tag>            # print the canonical release's asset names, one per line
#   tools/ci/release_draft.sh upload  <tag> <file>...  # attach files to the canonical release, replacing same-named assets
#   tools/ci/release_draft.sh publish <tag>            # publish the canonical release
#   printf '%s\n' "<id> <draft> <n_assets>" ... | tools/ci/release_draft.sh extras -   # offline: ids that `ensure` deletes
#
# Needs gh, a token with contents: write, and GITHUB_REPOSITORY, except for `extras -`.
set -euo pipefail

cmd="${1:?usage: $0 ensure|id|assets|upload|publish <tag> | extras -}"
shift

# "<id> <draft> <n_assets>" for every release carrying the tag (drafts included: a tag lookup does not find a draft).
releases() {
  gh api "repos/${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is not set}/releases?per_page=100" --paginate \
    --jq ".[] | select(.tag_name == \"$1\") | \"\(.id) \(.draft) \(.assets | length)\""
}

canonical() { sort -n | awk 'NR == 1 { print $1 }'; }

# Empty drafts other than the canonical release: the losers of the race. A published release or one with assets
# is never touched, whatever its id.
extras() { sort -n | awk 'NR > 1 && $2 == "true" && $3 == 0 { print $1 }'; }

canonical_id() {
  local id
  id="$(releases "$1" | canonical)"
  [ -n "${id}" ] || { echo "::error::no release for $1" >&2; return 1; }
  echo "${id}"
}

case "${cmd}" in
  extras)
    [ "${1:-}" = "-" ] || { echo "usage: $0 extras -" >&2; exit 2; }
    extras
    ;;
  id)
    canonical_id "${1:?tag}"
    ;;
  assets)
    id="$(canonical_id "${1:?tag}")"
    gh api "repos/${GITHUB_REPOSITORY}/releases/${id}/assets?per_page=100" --paginate --jq '.[].name'
    ;;
  ensure)
    tag="${1:?tag}"
    if [ -z "$(releases "${tag}")" ]; then
      gh release create "${tag}" --draft --verify-tag --generate-notes --repo "${GITHUB_REPOSITORY}" >/dev/null
      echo "created a draft for ${tag}" >&2
    fi
    # Let a racer's draft become visible, then keep only the canonical one. Every racer runs this, so whoever
    # looks last cleans up; deleting an already deleted draft is not an error.
    sleep "${RELEASE_DRAFT_SETTLE_SECONDS:-20}"
    for extra in $(releases "${tag}" | extras); do
      echo "deleting duplicate empty draft ${extra} of ${tag}" >&2
      gh api --method DELETE "repos/${GITHUB_REPOSITORY}/releases/${extra}" >/dev/null 2>&1 || true
    done
    canonical_id "${tag}"
    ;;
  upload)
    tag="${1:?tag}"; shift
    [ "$#" -gt 0 ] || { echo "usage: $0 upload <tag> <file>..." >&2; exit 2; }
    id="$(canonical_id "${tag}")"
    for file in "$@"; do
      name="$(basename "${file}")"
      old="$(gh api "repos/${GITHUB_REPOSITORY}/releases/${id}/assets?per_page=100" --paginate \
               --jq ".[] | select(.name == \"${name}\") | .id")"
      for asset in ${old}; do
        gh api --method DELETE "repos/${GITHUB_REPOSITORY}/releases/assets/${asset}" >/dev/null
      done
      gh api --method POST "https://uploads.github.com/repos/${GITHUB_REPOSITORY}/releases/${id}/assets?name=${name}" \
        -H 'Content-Type: application/octet-stream' --input "${file}" --jq '"uploaded \(.name) (\(.size) bytes)"'
    done
    ;;
  publish)
    id="$(canonical_id "${1:?tag}")"
    gh api --method PATCH "repos/${GITHUB_REPOSITORY}/releases/${id}" -F draft=false --jq '"published \(.tag_name) (\(.html_url))"'
    ;;
  *)
    echo "unknown command ${cmd}" >&2
    exit 2
    ;;
esac
