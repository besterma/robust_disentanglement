# Robust Disentanglement of a Few Factors at a Time
This repository contains the released code for:

### Robust Disentanglement of a Few Factors at a Time
Benjamin Estermann*, Markus Marks*, Mehmet Fatih Yanik

[arXiv](https://arxiv.org/abs/2010.13527)

### Abstract
Disentanglement  is  at  the  forefront  of  unsupervised  learning,  as  disentangledrepresentations of data improve generalization, interpretability, and performancein downstream tasks.  Current unsupervised approaches remain inapplicable forreal-world datasets since they are highly variable in their performance and fail toreach levels of disentanglement of (semi-)supervised approaches. We introducepopulation-based training (PBT) for improving consistency in training variationalautoencoders (VAEs) and demonstrate the validity of this approach in a supervisedsetting (PBT-VAE). We then use Unsupervised Disentanglement Ranking (UDR)as an unsupervised heuristic to score models in our PBT-VAE training and showhow models trained this way tend to consistently disentangle only a subset of thegenerative factors. Building on top of this observation we introduce the recursiverPU-VAE approach.  We train the model until convergence, remove the learnedfactors from the dataset and reiterate.  In doing so, we can label subsets of thedataset with the learned factors and consecutively use these labels to train one modelthat fully disentangles the whole dataset.  With this approach, we show strikingimprovement in state-of-the-art unsupervised disentanglement performance androbustness across multiple datasets and metrics.

### Cite
If you make use of this code in your own work, please cite our paper:
```
@article{estermann2020robust,
    title={Robust Disentanglement of a Few Factors at a Time},
    author={Estermann, Benjamin and Marks, Markus and Yanik, Mehmet Fatih},
    journal={NeurIPS},
    year={2020}
}
```

### Acknowledgements
Parts of the code loosely based on the implementation by [voiler](https://github.com/voiler/PopulationBasedTraining).
Including modified code from [Disentanglement-lib](https://github.com/google-research/disentanglement_lib) and [beta-tcvae](https://github.com/rtqichen/beta-tcvae)

### Usage

Configuration is done using [gin config](https://github.com/google/gin-config). See folder 'example_gins' for example configurations of the different pbt modes.
Due to computational restraints, we had to split the rPU-VAE runs into different subruns. The config for leaf_run 0 can be found in `config_x_run.gin`.
The config for the consecutive leaf-runs can then be found in `config_leafrun.gin`. The final supervised run running on the labels generated during the leaf-runs is configurated in `config_supervised_leafrun.gin`.
To generate the labels needed for this run, use `prepare_leaf_run.py`. To compute MIG and DCI disentanglement of a finished run, you can use `compute_metrics_vae.py`

`config_supervised_reference_run.gin` includes the config used for the fully- and semisupervised runs.

All runs can be started the same way:

`$ python pbt4vae/main.py --gin_config ./config.gin`

If you want more control over which GPUs should be used, and write the output into a logfile, use following command:

`$ unbuffer CUDA_VISIBLE_DEVICES=2 python pbt4vae/main.py --gin_config ./config.gin | tee logfile.log`

if you dont have the expect package for conda installed, you can install it with:
`conda install -c eumetsat expect`

### Contact
[Benjamin Estermann](mailto:benjamin.estermann@bluewin.ch), [Markus Marks](mailto:marksm@ethz.ch)
