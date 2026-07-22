"""Convert the official Hugging Face LeWM weights into the evaluation checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source_dir = root / "third_party" / "le-wm"
    sys.path.insert(0, str(source_dir))

    import stable_pretraining as spt
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    source = args.cache_dir / "hf_pusht"
    config_path = source / "config.json"
    weights_path = source / "weights.pt"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            "Download config.json and weights.pt first: "
            "bash scripts/download_assets.sh checkpoint"
        )

    config = json.loads(config_path.read_text())
    encoder = spt.backbone.utils.vit_hf(
        config["encoder"]["size"],
        patch_size=config["encoder"]["patch_size"],
        image_size=config["encoder"]["image_size"],
        pretrained=False,
        use_mask_token=False,
    )
    mlp = lambda key: MLP(
        input_dim=config[key]["input_dim"],
        output_dim=config[key]["output_dim"],
        hidden_dim=config[key]["hidden_dim"],
        norm_fn=torch.nn.BatchNorm1d,
    )
    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**config["predictor"]),
        action_encoder=Embedder(**config["action_encoder"]),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)

    output = args.cache_dir / "pusht" / "lewm_object.ckpt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, output)
    print(output)


if __name__ == "__main__":
    main()
