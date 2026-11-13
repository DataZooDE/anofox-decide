# anofox-decide — Claude Code plugin

Four skills that teach [Claude Code](https://claude.com/claude-code) the SQL API of the
[anofox_decide](https://github.com/DataZooDE/anofox-decide) DuckDB extension.

```
/plugin marketplace add DataZooDE/anofox-decide
/plugin install anofox-decide@anofox-decide
```

| Skill | Covers |
|---|---|
| [`anofox-decide-questions`](skills/anofox-decide-questions/SKILL.md) | binary, choice and score questions; one or several per text; NULLs, refusals, scoring many rows |
| [`anofox-decide-models`](skills/anofox-decide-models/SKILL.md) | providers, registering models, the remote opt-in, API keys and secrets, settings, errors |
| [`anofox-decide-local`](skills/anofox-decide-local/SKILL.md) | Julia-1 and Laya in-process, downloads, your own ONNX export, long text, batching, GPU |
| [`anofox-decide-evaluate`](skills/anofox-decide-evaluate/SKILL.md) | metrics, curves, comparing models, calibration, thresholds, abstaining |

Claude loads a skill on its own when a request matches its description. The skills are not slash commands.
To develop the skills in this repository, run `claude --plugin-dir ./plugins/anofox-decide`. How they are
tested is described in [docs/BUILDING.md](https://github.com/DataZooDE/anofox-decide/blob/main/docs/BUILDING.md#the-claude-code-skills).
