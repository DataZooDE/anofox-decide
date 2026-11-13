# plugins/

Two kinds of plugin live here:

| Folder | What it is |
|---|---|
| [`anofox-decide/`](anofox-decide/README.md) | The **Claude Code plugin**: four skills that teach an AI assistant the SQL API. Install it with `/plugin marketplace add DataZooDE/anofox-decide`, then `/plugin install anofox-decide@anofox-decide` ([README](../README.md#claude-code-skills)) |
| [`cuda/`](cuda/README.md), `migraphx/`, `mlx/` | The **GPU runtime plugins** that the extension loads to score local models on a GPU ([GPU acceleration](../docs/api/04-gpu.md)) |
