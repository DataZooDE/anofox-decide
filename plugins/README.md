# plugins/

> **Note:** the GPU runtime plugins in `cuda/`, `migraphx/` and `mlx/` are superseded. Local models now run on the
> GPU through [anofox-ggml-adapter](https://github.com/DataZooDE/anofox-ggml-adapter)'s plugins (plugin ABI v3,
> shared with anofox-tabfm), which `decide_accelerate()` downloads from the adapter's release. These directories
> are no longer used by the extension and are kept only until they are removed.

Two kinds of plugin live here:

| Folder | What it is |
|---|---|
| [`anofox-decide/`](anofox-decide/README.md) | The **Claude Code plugin**: four skills that teach an AI assistant the SQL API. Install it with `/plugin marketplace add DataZooDE/anofox-decide`, then `/plugin install anofox-decide@anofox-decide` ([README](../README.md#claude-code-skills)) |
| [`cuda/`](cuda/README.md), `migraphx/`, `mlx/` | The former **GPU runtime plugins** (superseded, see the note above; [GPU acceleration](../docs/api/04-gpu.md) describes the current ones) |
