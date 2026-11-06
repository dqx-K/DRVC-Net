"""Print trainable counts for controlled and external architectures."""

from __future__ import annotations

import argparse
from dataclasses import replace

from drvcnet.config import load_config
from drvcnet.constants import ModelVariant
from drvcnet.models import DRVCNet, build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--include-external", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for variant in ModelVariant:
        shift_lambda = 0.1 if variant is ModelVariant.B7 else 0.0
        candidate = replace(
            config, model=replace(config.model, variant=variant, shift_lambda=shift_lambda)
        )
        model = DRVCNet(candidate)
        print(f"{variant.value}\t{model.trainable_parameter_count():,}")
    if args.include_external:
        for architecture in ("mmgcn", "sdt", "css"):
            candidate = replace(
                config,
                model=replace(
                    config.model,
                    architecture=architecture,
                    variant=ModelVariant.B2,
                    shift_lambda=0.0,
                ),
            )
            model = build_model(candidate)
            print(f"{architecture.upper()}\t{model.trainable_parameter_count():,}")


if __name__ == "__main__":
    main()
