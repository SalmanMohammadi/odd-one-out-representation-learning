This repo was used for my MSc thesis on Weakly Supervised Representation Learning for Abstract Visual Reasoning (Grade 21/22) and the NeurIPS 2020 workshop paper, "Odd one out representation learning" (https://arxiv.org/abs/2012.07966). It's currently going through a major refactor to make it production-ready, and to clean up some of the mess I made while hacking code near deadlines for results.

Code for training weakly supervised representation learning models, and evaluating them on a downstream abstract visual reasoning task

`pip install -r requirements.txt`

To train the weakly supervised model, use

python train_infinitedataset.py -h 

To see a list of parameters.

To train the abstract reasoning model, use

python train_abstractreasoning.py -h

To see a list of parameters.
