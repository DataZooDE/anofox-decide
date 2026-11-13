# Building from source

Clone with submodules (`git clone --recurse-submodules`; they include the shared `posthog-telemetry` and
`datazoo-banner` libraries), then:

```bash
# vcpkg is required: vcpkg.json depends on openssl, and ONNX Runtime is built from the vcpkg overlay port.
git clone https://github.com/microsoft/vcpkg && ./vcpkg/bootstrap-vcpkg.sh -disableMetrics
export VCPKG_TOOLCHAIN_PATH=$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake

GEN=ninja make release                # release build; ONNX Runtime built via vcpkg (the first build is slow)
GEN=ninja make release DECIDE_ORT_VCPKG=0 CMAKE_PREFIX_PATH=<ort-install>   # faster local build against an existing ONNX Runtime
make test_release                     # offline suite: SQL tests + Catch2 (no network, no weights)
make test-live                        # live TypeSafe + Liquid D1 tests; SKIP without TYPESAFE_API_KEY / LIQUID_API_KEY
./build/release/test/unittest test/sql/decide_contract.test       # a single file
```

Without `VCPKG_TOOLCHAIN_PATH` the configure step stops with a cascade of CMake errors in which only the
first line is the real cause (`Could not find toolchain file: .../vcpkg_installed//share/vcpkg/...`); the
"no build program" and "compiler not set" lines after it are consequences, reported even when Ninja and the
compilers are installed. You need CMake (3.19 or newer to build the bundled C++ unit tests), a C++17
toolchain, and vcpkg.

A locally built extension is unsigned: start DuckDB with `duckdb -unsigned` and
`LOAD '/path/to/anofox_decide.duckdb_extension';`, or use the `duckdb` binary from `build/release`, which
has the extension linked in.

Some tests need model weights or a network and warn-and-pass without them: Catch2 `[tokenizer]` goldens
(`JULIA_WEIGHTS_DIR`, `LAYA_TYPED_DIR`), `[local]` real graphs (`JULIA_ONNX`, `LAYA_MULTILINGUAL_DIR` +
`LAYA_ONNX`, `LAYA_TYPED_DIR` + `LAYA_TYPED_ONNX`), and the export pytest in `tools/export_julia`
(`JULIA_WEIGHTS_DIR`, `JULIA_SRC_MODEL`, `JULIA_ONNX`). Run the suite with `TYPESAFE_API_KEY` and
`LIQUID_API_KEY` unset, as CI does, so tests do not depend on your own key. Tests always run with
`DATAZOO_DISABLE_TELEMETRY=1` (the Makefile does this).

## The Claude Code skills

The Claude Code plugin is in [`plugins/anofox-decide/`](../plugins/anofox-decide/README.md): one `SKILL.md` per
skill, with the marketplace entry in `.claude-plugin/marketplace.json` at the repository root. To try your
edits, start Claude Code with `claude --plugin-dir ./plugins/anofox-decide`. The `[docs]` Catch2 tests
(`test/cpp/test_decide_docs.cpp`) treat the skills like the documentation, with these additions:

- Every SQL block of a skill **runs** against the static extension, blocks in order with one database per skill,
  so write them for the `stub` model. A block that needs a key, the network, a GPU or files on disk starts with
  `-- needs: <what>`. Such a block is only parsed.
- Every `decide_*` name a skill writes in backticks, or names in its frontmatter `description`, must exist, and
  every public `decide_*` function must appear in at least one skill. A new function therefore needs a line in a skill.
- Each skill's `name` matches its folder, and the `version` of `plugin.json` and of every skill equals the release.
- Link to the documentation with `https://github.com/DataZooDE/anofox-decide/blob/main/...` URLs: an installed
  plugin is copied out of the repository, so relative links would not resolve.

## Releases

Releases are calendar-versioned tags (`vYYYY.MM.DD`); see [CHANGELOG.md](../CHANGELOG.md). The version strings
to bump: `vcpkg.json`, `src/anofox_decide_extension.cpp`, `DECIDE_PLUGIN_RELEASE_TAG` in
`src/include/decide_plugin_artifacts.hpp`, the status line in `README.md`, and the Claude Code plugin
(`plugins/anofox-decide/.claude-plugin/plugin.json` and the `version:` line of every
`plugins/anofox-decide/skills/*/SKILL.md`). The docs tests fail when the plugin's versions lag behind.

The GPU plugins are built separately; see [GPU_BACKENDS.md](GPU_BACKENDS.md) and [`plugins/cuda`](../plugins/cuda/README.md).
