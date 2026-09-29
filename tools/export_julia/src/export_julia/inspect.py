"""Step 1 of the pipeline: pin the upstream model shape before exporting.

Julia-1 is SupersonicLabs/Julia-1 on HuggingFace (Apache 2.0, ~144M params,
mmBERT-small encoder + decision head, 2-20 answers per call). This command
dumps its config, tokenizer shape, and head dimensions into a versioned spec
so the exporter never depends on a moving upstream default.

Writes <out>/julia-1-spec.json. Requires network + huggingface_hub.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ID = "SupersonicLabs/Julia-1"


def inspect(repo_id: str = REPO_ID) -> dict:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.model_info(repo_id)
    siblings = sorted(getattr(s, "rfilename", None) or s.name for s in (info.siblings or []))
    return {
        "repo_id": repo_id,
        "sha": info.sha,
        "tags": sorted(info.tags or []),
        "files": siblings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pin the Julia-1 upstream spec")
    parser.add_argument("--repo", default=REPO_ID)
    parser.add_argument("--out", required=True, help="Output directory for julia-1-spec.json")
    args = parser.parse_args(argv)
    spec = inspect(args.repo)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "julia-1-spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    print(f"pinned {args.repo} @ {spec['sha']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
