# LookHere :eyes: (NeurIPS 2024 :tada:)
Official code for "LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate"

arXiv link: https://arxiv.org/abs/2405.13985

NeurIPS link: https://neurips.cc/virtual/2024/poster/93643

ImageNet-HR dataset: https://huggingface.co/datasets/antofuller/ImageNet-HR


LookHere is a patch position encoding technique for ViTs. LookHere's main advantage is its extrapolation ability — outperforming 2D-RoPE by 21.7% on ImageNet-1k when trained at 224x224 px and tested at 1024x1024 px.

## Basic usage

First, let's download pretrained weights. We evaluate these models in our paper (trained for 150 epochs on ImageNet-1k).
```cli
wget https://huggingface.co/antofuller/LookHere/resolve/main/LH_180_weights_and_config.pth
wget https://huggingface.co/antofuller/LookHere/resolve/main/LH_90_weights_and_config.pth
wget https://huggingface.co/antofuller/LookHere/resolve/main/LH_45_weights_and_config.pth
```

```python
import torch
from lookhere import LookHere
from data_prep import ImageNetDataset
from sklearn.metrics import accuracy_score

# prepare model
checkpoint = torch.load("LH_180_weights_and_config.pth")
model = LookHere(device="cuda", lh_config=checkpoint["config"])
model.load_state_dict(checkpoint["weights"])
model = model.eval()

# prepare minival
minival_dataset = ImageNetDataset(
  dataset=load_dataset("imagenet-1k", split="train[99%:]"),
  do_augment=False,
  img_size=224,
)
minival_loader = DataLoader(
  minival_dataset,
  batch_size=batch_size,
  shuffle=False,
  num_workers=8,
)

# make some predictions
with torch.no_grad():
    for batch in minival_loader:
        images, labels = batch
        images = images.cuda()  # (batch_size, 3, 224, 224)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(images)  # (batch_size, 1_000)
        preds = logits.argmax(dim=-1)  # (batch_size)
```

Extrapolation is easy!

```python
img_size = 1024
model.set_pos_embed(int(img_size / 16))  # 16 is the patch size, so the grid size is (img_size/patch_size, img_size/patch_size)
```

# Please Cite
```bib
@misc{fuller2024lookhere,
      title={LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate}, 
      author={Anthony Fuller and Daniel G. Kyrollos and Yousef Yassin and James R. Green},
      year={2024},
      eprint={2405.13985},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
