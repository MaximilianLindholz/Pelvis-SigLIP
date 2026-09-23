# Pelvis-SigLIP

Two fine-tuned vision-language models that identify what a female pelvic
MRI series actually *is* — sequence type (T1, T2, DWI, ...) and view
orientation (axial, sagittal, ...) — directly from the image, without
relying on the DICOM series description. Those descriptions are often
missing, inconsistent between scanner vendors, or corrupted, which makes
it hard to automatically compare a patient's scans across visits or
assemble a research cohort from a multi-site database.

From the paper *"Pelvis-SigLIP: Benchmarking Vision-Language Models for
Female Pelvic MRI Series Retrieval across Zero-Shot, Linear-Probe, and
Fine-Tuning"* (MICCAI 2026 Workshop CAPI-WOMEN, Oral). The paper's main
finding: general-purpose and medical vision-language models fail at this
task out of the box (14.5–29.4% accuracy, below a 36.2% majority-class
baseline), but a small amount of fine-tuning fixes it — the two
checkpoints below reach 95.1% and 90.5% accuracy on a held-out patient
test set.

## Released models

| Model | Task | Test accuracy | Hugging Face |
|---|---|---|---|
| Fine-tuned MedSigLIP | Sequence type (8 classes: T1, T1FS, T2, T2FS, DCE, DWI, ADC, other) | 95.1% | [MaximilianLindholz/pelvis-siglip-medsiglip-sequence](https://huggingface.co/MaximilianLindholz/pelvis-siglip-medsiglip-sequence) |
| Fine-tuned SigLIP-2 | View orientation (4 classes: axial, sagittal, coronal, oblique) | 90.5% | [MaximilianLindholz/pelvis-siglip-siglip2-orientation](https://huggingface.co/MaximilianLindholz/pelvis-siglip-siglip2-orientation) |

Both models take a single 2D image (one slice of the series) as input —
no metadata required. Weights are hosted on Hugging Face, not in this
repository; the code here downloads them automatically the first time
you run it.

## Try it yourself

The fastest way to see this working on your own data:

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open `notebooks/explore_checkpoints.ipynb` and set `NII_PATH` to a
   local `.nii` or `.nii.gz` MRI file.
3. Run all cells. It will:
   - extract a central 2D slice from your volume,
   - show what a **zero-shot** model predicts (the naive baseline that
     the paper shows fails), for comparison,
   - then run both fine-tuned Pelvis-SigLIP models and show their
     predictions.

`google/medsiglip-448` (the MedSigLIP backbone) is a gated model on
Hugging Face — request access on
[its model page](https://huggingface.co/google/medsiglip-448), then run
`huggingface-cli login` once before using the sequence-type checkpoint.

### Minimal usage (no notebook)

```python
import torch
from huggingface_hub import snapshot_download
from src.models import MedSigLIPModel
from src.data import preprocess_image_array
import json, torch.nn as nn

ckpt_dir = snapshot_download("MaximilianLindholz/pelvis-siglip-medsiglip-sequence")
config = json.load(open(f"{ckpt_dir}/config.json"))
classes = json.load(open(f"{ckpt_dir}/label_classes.json"))

from transformers import SiglipVisionModel
model = MedSigLIPModel(config["n_classes"])
model.backbone = SiglipVisionModel.from_pretrained(f"{ckpt_dir}/backbone")
model.head = nn.Linear(config["embed_dim"], config["n_classes"])
model.head.load_state_dict(torch.load(f"{ckpt_dir}/head.pt"))
model.eval()

# `slice_arr` is a 2D numpy array or PIL Image — e.g. one slice of your scan
x = preprocess_image_array(slice_arr, config["img_size"], config["mean"], config["std"])
probs = model.predict(x).squeeze(0).tolist()   # one probability per class, same order as `classes`

# `classes` and `probs` are index-aligned — classes[i] is the class for probs[i].
# This mapping is fixed at training time (classes = LabelEncoder.classes_) and
# saved verbatim into label_classes.json, so it's always consistent with the
# checkpoint it ships alongside.
for cls, p in sorted(zip(classes, probs), key=lambda cp: -cp[1]):
    print(f"{cls:>8s}  {p:.3f}")
# top prediction: max(zip(classes, probs), key=lambda cp: cp[1])[0]
```

## Repository contents

```
Pelvis-SigLIP/
├── src/
│   ├── models.py     # model architectures (MedSigLIP / SigLIP-2 backbone + linear head)
│   └── data.py        # image preprocessing shared by both models
├── notebooks/
│   └── explore_checkpoints.ipynb   # run both models on your own .nii/.nii.gz file
├── environment.yml / requirements.txt
└── README.md
```

This repository intentionally contains just enough code to *use* the two
released checkpoints. The training and evaluation pipeline that produced
them (5-fold cross-validation, hyperparameter setup, etc.) is described
in the paper; get in touch if you specifically need that code.

## Citation

```bibtex
@inproceedings{lindholz2026pelvissiglip,
  title     = {Pelvis-SigLIP: Benchmarking Vision-Language Models for Female Pelvic MRI Series Retrieval across Zero-Shot, Linear-Probe, and Fine-Tuning},
  author    = {Lindholz, Maximilian and Ruppel, Richard and Hamm, Charlie Alexander and Kn{\"u}pfer, Anika and Eminovic, Semil and Schmidt, Robin and Haack, Anna-Maria and El-Nahry, Yasmin and Aleixo, Carolina and Hutter, Jana and Mechsner, Sylvia and Penzkofer, Tobias},
  booktitle = {Proceedings of the MICCAI 2026 Workshop CAPI-WOMEN},
  year      = {2026},
  note      = {Oral}
}
```

## License

Code: MIT. Model weights are released for clinical research use — see
each model's page on Hugging Face for details. This is a research tool,
not a diagnostic device; it has not been validated for clinical decision
support.
