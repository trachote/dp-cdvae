# Diffusion Probabilistic CDVAE (DP-CDVAE)
DP-CDVAE is a generative model for crystal structure generation developed on crystal diffusional variational autoencoder ([CDVAE](https://github.com/txie-93/cdvae)).
This is an implemented code of a paper: Diffusion probabilistic models enhance variational autoencoder for crystal structure
generative modeling.

Link to [[Paper](https://arxiv.org/abs/2308.02165)]

training command:
```
python train.py --config_path conf/ddpm_carbon_dime.yaml --output_path out_dir
```
reconstruction command:
```
python evaluate.py --model_path out_dir --task recon
```
generation command:
```
python evaluate.py --model_path out_dir --task gen
```
compute reconstruction & generation metrics
```
python compute_metrics.py --root_path out_dir --task recon gen
```

# Graph Neural Networks
DimeNet++ and GemNetT have been modified to compatible with crystal structures from [CDVAE](https://github.com/txie-93/cdvae) code.

# References
DP-CDVAE
```
@misc{pakornchote2023diffusion,
      title={Diffusion probabilistic models enhance variational autoencoder for crystal structure generative modeling}, 
      author={Teerachote Pakornchote and Natthaphon Choomphon-anomakhun and Sorrjit Arrerut and Chayanon Atthapak and Sakarn Khamkaeo and Thiparat Chotibut and Thiti Bovornratanaraks},
      year={2023},
      eprint={2308.02165},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

CDVAE
```
@inproceedings{
xie2022crystal,
title={Crystal Diffusion Variational Autoencoder for Periodic Material Generation},
author={Tian Xie and Xiang Fu and Octavian-Eugen Ganea and Regina Barzilay and Tommi S. Jaakkola},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=03RLpj-tc_}
}
```

DimeNet++
```
@misc{gasteiger2022fast,
      title={Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules}, 
      author={Johannes Gasteiger and Shankari Giri and Johannes T. Margraf and Stephan Günnemann},
      year={2022},
      eprint={2011.14115},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

GemNetT
```
@inproceedings{
klicpera2021gemnet,
title={GemNet: Universal Directional Graph Neural Networks for Molecules},
author={Johannes Klicpera and Florian Becker and Stephan G{\"u}nnemann},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021},
url={https://openreview.net/forum?id=HS_sOaxS9K-}
}
```
