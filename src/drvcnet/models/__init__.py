"""DRVC-Net model components."""

from torch import nn

from drvcnet.config import ExperimentConfig
from drvcnet.models.drvc_net import DRVCNet, ModelOutput
from drvcnet.models.external import CSSBaseline, MMGCNBaseline, SDTBaseline


def build_model(config: ExperimentConfig) -> nn.Module:
    architectures = {
        "drvc": DRVCNet,
        "mmgcn": MMGCNBaseline,
        "sdt": SDTBaseline,
        "css": CSSBaseline,
    }
    return architectures[config.model.architecture](config)


__all__ = [
    "CSSBaseline",
    "DRVCNet",
    "MMGCNBaseline",
    "ModelOutput",
    "SDTBaseline",
    "build_model",
]
