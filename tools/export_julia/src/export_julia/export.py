"""Step 2 of the pipeline: Julia-1 -> ONNX graph + tensor map.

Wiring derived from SupersonicLabs/Julia-1 @ a85b1273 (julia/model.py,
julia/data.py, julia_config.json — see docs/julia-1-spec.json). Scores path
only (return_actions=False): encoder + type addition + full-sequence head +
marker gather + scorer, masked slots filled with -1e4 and dropped before
softmax on the SQL side.

Inputs: a julia-1-spec.json from inspect.py plus a local snapshot of the
weights (huggingface_hub snapshot_download, then offline).

Outputs in <out>/:
  julia1.onnx            full graph with weights
  julia1_tensor_map.json input/output names for the C++ side
  julia1_tiny.onnx       weight-free tiny graph for test/fixtures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file


class ManualSelfAttention(nn.Module):
    """torch-free dynamic port of nn.MultiheadAttention (batch_first).

    Parameter names mirror nn.MultiheadAttention exactly (in_proj_weight,
    in_proj_bias, out_proj) so upstream JuliaDecisionModel weights load
    unchanged. Math replicates the forward path: QKV projection, per-head
    scaled dot-product attention with key padding mask, output projection.
    No transposes to [T, B, C] and no baked reshape constants, so the
    TorchScript trace stays fully dynamic in batch/sequence/markers.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        b, t, _ = x.shape
        h, d = self.num_heads, self.head_dim
        qkv = torch.nn.functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        q = q.view(b, t, h, d).transpose(1, 2)
        k = k.view(b, t, h, d).transpose(1, 2)
        v = v.view(b, t, h, d).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d**0.5)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, self.embed_dim)
        return self.out_proj(out)


class ManualEncoderLayer(nn.Module):
    """norm_first TransformerEncoderLayer port (activation=relu, matching the
    upstream default in JuliaDecisionModel). Submodule names mirror the
    upstream layout so state dicts load 1:1."""

    def __init__(self, width: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = ManualSelfAttention(width, nhead)
        self.linear1 = nn.Linear(width, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x), src_key_padding_mask))
        x = x + self.dropout(self.linear2(self.dropout(torch.relu(self.linear1(self.norm2(x))))))
        return x


class JuliaScoresOnly(nn.Module):
    """Scores path of JuliaDecisionModel (model.py, return_actions=False).

    The decision head is a manual port (see ManualSelfAttention): upstream
    uses nn.TransformerEncoderLayer whose TorchScript trace bakes sequence
    constants into reshape nodes, breaking every off-trace shape. Parameter
    names are identical, so upstream weights load unchanged; equivalence to
    the upstream module is gated by test_upstream_equivalence.
    """

    def __init__(self, encoder: nn.Module, head_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        width = encoder.config.hidden_size
        # Plain container (not nn.TransformerEncoder) so forward stays manual,
        # but the `head.layers.N.*` key layout matches upstream exactly.
        self.head = nn.Module()
        self.head.layers = nn.ModuleList(
            [ManualEncoderLayer(width, max(1, width // 64), 4 * width, dropout) for _ in range(head_layers)]
        )
        self.type_emb = nn.Embedding(3, width)
        self.scorer = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden + self.type_emb(qtype)[:, None, :]
        padding = ~attention_mask.bool()
        for layer in self.head.layers:
            hidden = layer(hidden, src_key_padding_mask=padding)
        positions = marker_pos[:, :, None].expand(-1, -1, hidden.shape[-1])
        markers = hidden.gather(1, positions)
        scores = self.scorer(markers).squeeze(-1).float()
        return scores.masked_fill(~marker_mask, -1e4)


def load_scores_model(weights_dir: Path, head_layers: int, dropout: float,
                      attn: str = "eager") -> JuliaScoresOnly:
    """Assemble the scores model; encoder shape from snapshot config, all
    weights from the top-level model.safetensors (the encoder/ subdir
    carries config only)."""
    from transformers import AutoConfig, AutoModel

    enc_config = AutoConfig.from_pretrained(weights_dir / "encoder", trust_remote_code=False)
    encoder = AutoModel.from_config(enc_config, trust_remote_code=False, attn_implementation=attn)
    model = JuliaScoresOnly(encoder, head_layers=head_layers, dropout=dropout)
    state = load_file(str(weights_dir / "model.safetensors"))
    # Safetensors keys already carry their module prefixes ("encoder.*",
    # "head.*", "type_emb.*", "scorer.*"): pass through as-is.
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Only the scores path must resolve fully. act_head.* (actions path) and
    # the temperature buffer are intentionally excluded from the SQL graph.
    unresolved = [k for k in missing if not k.startswith("act_head.")]
    if unresolved:
        raise ValueError(f"weights missing for: {unresolved[:8]}")
    surplus = [k for k in unexpected if not (k.startswith("act_head.") or k == "temperature")]
    if surplus:
        raise ValueError(f"unexpected weight keys: {surplus[:8]}")
    model.eval()
    return model


def export_onnx(model: nn.Module, out_path: Path, opset: int = 17) -> None:
    model.eval()
    b, t, m = 1, 64, 3
    args = (
        torch.ones(b, t, dtype=torch.long),
        torch.ones(b, t, dtype=torch.long),
        torch.tensor([[4, 8, 12]], dtype=torch.long),
        torch.ones(b, m, dtype=torch.bool),
        torch.zeros(b, dtype=torch.long),
    )
    torch.onnx.export(
        model,
        args,
        str(out_path),
        input_names=["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
        output_names=["scores"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "marker_pos": {0: "batch", 1: "markers"},
            "marker_mask": {0: "batch", 1: "markers"},
            "qtype": {0: "batch"},
            "scores": {0: "batch", 1: "markers"},
        },
        opset_version=opset,
        # TorchScript path: the dynamo exporter needs onnxscript, and the
        # eager-attention graph traces cleanly. Parity test decides.
        dynamo=False,
    )


def make_tiny_model(seed: int = 0) -> JuliaScoresOnly:
    """Random-init miniature of the scores architecture for repo fixtures.

    No trained weights anywhere near this path (license wall): shapes mirror
    the real graph (5 inputs, scores output, dynamic batch/seq/markers) so
    the C++ loader and parity harness run in CI without the 577MB artifact.
    """
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(seed)
    enc_config = ModernBertConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        vocab_size=1000,
        max_position_embeddings=128,
        pad_token_id=0,
        cls_token_id=1,
        sep_token_id=1,
        mask_token_id=4,
        layer_types=["full_attention"],
    )
    encoder = ModernBertModel(enc_config)
    return JuliaScoresOnly(encoder, head_layers=1, dropout=0.0)


TENSOR_MAP = {
    "inputs": ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"],
    "outputs": ["scores"],
    "masked_fill": -10000.0,
}


def export_model(spec_path: str, weights_dir: str, out_dir: str, opset: int = 17) -> dict:
    spec = json.loads(Path(spec_path).read_text())
    weights = Path(weights_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    head_layers = int(spec.get("head", {}).get("head_layers", 2))
    dropout = float(spec.get("head", {}).get("dropout", 0.1))
    model = load_scores_model(weights, head_layers, dropout)
    graph = out / "julia1.onnx"
    export_onnx(model, graph, opset)
    (out / "julia1_tensor_map.json").write_text(json.dumps(TENSOR_MAP, indent=2) + "\n")
    return {"graph": str(graph), "tensor_map": str(out / "julia1_tensor_map.json")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Julia-1 to ONNX")
    parser.add_argument("--spec", required=True, help="julia-1-spec.json from inspect.py")
    parser.add_argument("--weights", required=True, help="Local snapshot_download directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    print(f"exporting {spec['repo_id']} @ {spec['sha']}")
    artifacts = export_model(args.spec, args.weights, args.out, args.opset)
    print(json.dumps(artifacts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
