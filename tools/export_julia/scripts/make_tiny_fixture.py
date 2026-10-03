#!/usr/bin/env python3
"""Random-init tiny model as a weight-free graph + safetensors for the C++ tests.

Writes graph_tiny.onnx, tensor_map_tiny.json and tiny.safetensors (fp16, like the
real Laya checkpoints) into the output directory. No trained weights anywhere:
the values are torch's seeded initialisation.
  make_tiny_fixture.py ../../test/fixtures/local_tiny
"""
import sys
from pathlib import Path

from safetensors.torch import save_file

from export_julia.export import make_tiny_model
from export_julia.weight_free import export_weight_free


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model = make_tiny_model(0).eval()
    sd = {k: v.detach().half().contiguous() for k, v in model.state_dict().items()}
    ck = out / "tiny.safetensors"
    save_file(sd, str(ck))
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    print(export_weight_free(model, ck, out, "tiny", {"hidden_size": 32}, 17))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
