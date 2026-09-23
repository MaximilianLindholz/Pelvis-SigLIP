"""
models.py
=========
Backbone + linear-head wrappers for the two Pelvis-SigLIP release checkpoints.

Trimmed, verbatim-logic copy of the relevant classes from the original
`finetune.py` used to produce the paper's Table 2 numbers — only the two
architectures that are actually released (MedSigLIP for sequence-type,
SigLIP-2 for view-orientation) are kept here, to keep this repo's
dependency footprint minimal.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _Head(nn.Module):
    """Linear classification head on top of a pooled backbone feature."""

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return self.fc(x)


class MedSigLIPModel(nn.Module):
    """SigLIP vision tower (google/medsiglip-448) + linear head.

    Released checkpoint: sequence-type classification (8 classes).
    """

    IMG_SIZE = 448  # native resolution — position embeddings hardcoded to 32x32 patches
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    GRAYSCALE = False

    def __init__(self, n_classes: int):
        from transformers import SiglipVisionModel

        super().__init__()
        self.backbone = SiglipVisionModel.from_pretrained("google/medsiglip-448")
        self.backbone.gradient_checkpointing_enable()
        self.head = _Head(1152, n_classes)

    def forward(self, x):
        feat = self.backbone(pixel_values=x).pooler_output
        return self.head(feat)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        logits = self(x)
        return logits.softmax(dim=-1)


class SigLIP2Model(nn.Module):
    """open_clip SigLIP-2 (ViT-L-16-SigLIP2-384) vision tower + linear head.

    Released checkpoint: view-orientation classification (4 classes).
    """

    IMG_SIZE = 384
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    GRAYSCALE = False

    def __init__(self, n_classes: int):
        import open_clip

        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-16-SigLIP2-384", pretrained="webli"
        )
        self.backbone = model.visual
        self.head = _Head(1024, n_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        logits = self(x)
        return logits.softmax(dim=-1)


# Only these two are trained/released in this repo. finetune.py on the
# cluster has the full 7-model registry (DINOv3, PubMedCLIP, MedSAM,
# MRI-CORE, BiomedCLIP too) used for the paper's full benchmark comparison.
MODEL_REGISTRY = {
    "MedSigLIP": MedSigLIPModel,
    "SigLIP-2": SigLIP2Model,
}

# Which (model, task) pair each architecture is released for.
RELEASE_TARGETS = [
    ("MedSigLIP", "Sequence"),
    ("SigLIP-2", "Orientation"),
]
