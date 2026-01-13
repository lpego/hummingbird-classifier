# hummingbird-classifier-models
This folder is meant to contain the pre-trained models for [hummingbird-classifier](https://github.com/lpego/hummingbird-classifier). 

## Quickstart
If you want to get started quickly, just download the [`hummingbird-classifier minimum data package`](https://drive.google.com/file/d/1KWaAEvdTckdlPAlN-YcgULySAQoi0unG/view?usp=sharing) with demo data and the smallest model, unzip it in your working directory, for example: `C:\\my-folder\hummingbird-classifier`. 

The [Zenodo archive](https://zenodo.org/records/18232453) contains all the model checkpoints and training reports for various architectures, all trained on the same data and split. 

In particular, `.ckpt` files are the model weights that can be directly imported in PyTorch. 
Other file formats like `.pt` and `.npy` are used in model evaluation (e.g. [Weights & Biases](https://wandb.ai/site)). 

## Contributions
<!-- AUTHORS -->
- Luca Pegoraro (WSL) - luca.pegoraro@wsl.ch
- Michele Volpi (SDSC) - michele.volpi@sdsc.ethz.ch
