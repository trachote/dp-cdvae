# Diffusion Probabilistic CDVAE (DP-CDVAE)
DP-CDVAE is a generative model for crystal structure generation developed on crystal diffusional variational autoencoder ([CDVAE](https://github.com/txie-93/cdvae)).
This is an implemented code of a paper: Diffusion probabilistic models enhance variational autoencoder for crystal structure
generative modeling.

[[Paper](https://arxiv.org/abs/2308.02165)]

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
