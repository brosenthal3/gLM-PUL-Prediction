"""
From finetuned model, save the pretrained model part
Used to generate embeddings from finetuned models, so that they can be visualized in the umap
"""

import sys
from pathlib import Path
from genecat.model.model_registry import load_model

print("Loading", sys.argv[1])

finetuned_model, hyper_params = load_model(
    path=Path(sys.argv[1]),
    with_weights=True,  # load the pretrained model
    return_hyperparams=True,
)

print("Saving pretrained model to", sys.argv[2])

with open(Path(sys.argv[2]).with_suffix(".pt"), "wb") as f:
    finetuned_model.pretrained_model.save_model(  # ty:ignore[possibly-missing-attribute]
        file=f,
        info=hyper_params["model_info"],
    )  # ty:ignore[call-non-callable]