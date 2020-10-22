import numpy as np
import torch
import gin

from torch.utils.data import DataLoader

from beta_tcvae.vae_quant import VAE


@gin.configurable(blacklist=["random_state"])
def get_init_batch_size(random_state):
    return int(
        random_state.choice(np.logspace(3, 10, base=2, dtype=int, num=8))
    )  # 8 - 1024


@gin.configurable(blacklist=["random_state"])
def get_init_lr(random_state):
    return random_state.choice(np.logspace(-5, 0, num=30, base=10))


@gin.configurable(blacklist=["random_state"])
def get_init_beta(random_state):
    return int(random_state.choice(np.logspace(1, 15, base=1.5, num=24, dtype=int)[1:]))
    # [1:] because else we would have 1 double


class TorchIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, ground_truth_data, random_seed):
        self.random_state = np.random.RandomState(random_seed)
        self.ground_truth_data = ground_truth_data

    def __iter__(self):
        while True:
            x = self.ground_truth_data.sample_observations(1, self.random_state)[0]
            yield np.moveaxis(x, 2, 0)

    def __len__(self):
        return np.prod(self.ground_truth_data.factors_num_values)


@gin.configurable(whitelist=["hyper_params", "perturb_factors"])
def exploit_and_explore(
    top_checkpoint_path,
    bot_checkpoint_path,
    hyper_params,
    random_state,
    perturb_factors=(2, 1.2, 0.8, 0.5),
):
    """Copy parameters from the better model and the hyperparameters
    and running averages from the corresponding optimizer."""
    # Copy model parameters
    # TODO only change parts of loaded checkpoint that really changed, no need to create new dict
    print("Running function exploit_and_explore")
    checkpoint = torch.load(top_checkpoint_path, map_location=torch.device("cpu"))
    state_dict = checkpoint["model_state_dict"]
    hyperparam_state_dict = checkpoint["hyperparam_state_dict"]
    optimizer_state_dict = checkpoint["optim_state_dict"]
    batch_size = checkpoint["batch_size"]
    scores = checkpoint["scores"]
    model_random_state = checkpoint["random_state"]
    training_params = checkpoint["training_params"]
    if "lr" in hyper_params:
        perturb = random_state.choice(perturb_factors)
        for param_group in optimizer_state_dict["param_groups"]:
            param_group["lr"] *= perturb
    if "batch_size" in hyper_params:
        perturb = random_state.choice(perturb_factors)
        batch_size = int(
            np.minimum(np.ceil(perturb * batch_size), 1024)
        )  # limit due to memory constraints
    if "beta" in hyper_params:
        perturb = random_state.choice(perturb_factors)
        beta = int(np.ceil(perturb * hyperparam_state_dict["beta"]))
        hyperparam_state_dict["beta"] = beta
    checkpoint = dict(
        model_state_dict=state_dict,
        hyperparam_state_dict=hyperparam_state_dict,
        optim_state_dict=optimizer_state_dict,
        batch_size=batch_size,
        training_params=training_params,
        scores=scores,
        random_state=model_random_state,
    )
    torch.save(checkpoint, bot_checkpoint_path)


def _compute_gaussian_kl(z_mean, z_logvar):
    return np.mean(0.5 * (np.square(z_mean) + np.exp(z_logvar) - z_logvar - 1), axis=0)


def _representation_function(x, model):
    """Computes representation vector for input images."""
    # x = np.moveaxis(x, 3, 1)
    if type(x) is np.ndarray:
        x = torch.from_numpy(x)
    x = x.to(model.device)
    zs, zs_params = model.encode(x)

    means = zs_params[:, :, 0].cpu().detach().numpy()

    logvars = np.abs(zs_params[:, :, 1].cpu().detach().numpy().mean(axis=0))

    return means, _compute_gaussian_kl(means, logvars)
    # return zs.cpu().numpy()                # if we want a sample from the distribution


def _encoder(x, model):
    # x = np.moveaxis(x, 3, 1)
    if type(x) is np.ndarray:
        x = torch.from_numpy(x)
    x = x.to(model.device)
    zs, zs_params = model.encode(x)

    return zs_params[:, :, 0].cpu().detach().numpy()


def _encoder_pytorch(x, model):
    """Encode image to latent representation, x and zs are both pyTorch tensors"""
    x = x.to(model.device)
    zs, zs_params = model.encode(x)
    # zs_params shape (n_samples, z_dim, 2), where [:,:,0] is means, [:,:,1] is logvars
    return zs, zs_params


def _decoder_pytorch(z, model):
    """Decode latent representation to image, z and xs_params are both pyTorch tensors"""
    z = z.to(model.device)
    _, xs_params = model.decode(z)
    return xs_params.sigmoid()


def _compute_kl_divs_vae(path, dataset, device, map_gpu):
    num_samples_means = 100
    model = VAE(
        z_dim=10, use_cuda=True, tcvae=True, conv=True, device=device, num_channels=1
    )
    checkpoint = torch.load(
        path,
        map_location={
            "cuda:3": map_gpu,
            "cuda:2": map_gpu,
            "cuda:1": map_gpu,
            "cuda:0": map_gpu,
        },
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    z_dim = model.z_dim
    nparams = model.q_dist.nparams
    N = num_samples_means * 128
    qz_params = torch.Tensor(N, z_dim, nparams).to(device)
    data_loader = DataLoader(
        dataset=dataset, batch_size=128, shuffle=True, num_workers=0
    )
    for i in range(num_samples_means):
        images = iter(data_loader).next()
        zs, zs_params = model.encode(images.to(device))
        qz_params[i * 128 : (i + 1) * 128] = zs_params

    qz_params = qz_params.detach().cpu()
    means = qz_params[..., 0].numpy()
    logvars = np.abs(qz_params[..., 1].numpy().mean(axis=0))

    return _compute_gaussian_kl(means, logvars)
