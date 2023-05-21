This repo was used for my MSc thesis on Weakly Supervised Representation Learning for Abstract Visual Reasoning (Grade 21/22) and the NeurIPS 2020 workshop paper, "Odd one out representation learning" (https://arxiv.org/abs/2012.07966).
The abstract-reasoning datatset based on DSprites is available under `data/rpm_data.py`.
The various weakly supervised datasets used for this work are available under `data/dsprites_data.py`. These include iterable (infinite i.i.d. random samples) and iterable (finite samples) of the DSprites Pairs and Triplets datasets, which sample pairs of samples which share generative factors, and triplets of samples which respect the odd-one-out constraint in their generative factors, respectively.


Code for training weakly supervised representation learning models, and evaluating them on a downstream abstract visual reasoning task

`pip install -r requirements.txt`

To train the weakly supervised model, use

`python train_infinitedataset.py -h`

To see a list of parameters. The available models include VAE, B-VAE, TC-VAE, AdaGVAE, and the contribution from this paper, TVAE (Triplet-VAE). A variety of disentanglement metrics are also available, which are discussed in the paper: the FactorVAE score, the DCI score, the BVAE score, and the UDR score.

To train the abstract reasoning model (a WReN) model, use

`python train_abstractreasoning.py -h`

To see a list of parameters.
