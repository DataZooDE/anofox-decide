# plugins/

The GPU plugins are not here: they are [anofox-ggml-adapter](https://github.com/DataZooDE/anofox-ggml-adapter)'s (plugin ABI v3,
shared with anofox-tabfm), and `decide_accelerate()` downloads them from the adapter's release
([GPU acceleration](../docs/api/04-gpu.md)).

| Folder | What it is |
|---|---|
| [`anofox-decide/`](anofox-decide/README.md) | The **Claude Code plugin**: four skills that teach an AI assistant the SQL API. Install it with `/plugin marketplace add DataZooDE/anofox-decide`, then `/plugin install anofox-decide@anofox-decide` ([README](../README.md#claude-code-skills)) |
