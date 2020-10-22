import sys

import os
import time
import numpy as np
import math
from numbers import Number
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
import gin
from torch.autograd import Variable
from torch.utils.data import DataLoader

import beta_tcvae.lib.dist as dist
import beta_tcvae.lib.utils as utils
import beta_tcvae.lib.datasets as dset
from beta_tcvae.lib.flows import FactorialNormalizingFlow

from beta_tcvae.elbo_decomposition import elbo_decomposition

# from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces  # noqa: F401


class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096),
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img


class ConvEncoder(nn.Module):
    def __init__(self, output_dim, num_channels):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim
        self.num_channels = num_channels

        self.conv1 = nn.Conv2d(num_channels, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, self.num_channels, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim, num_channels):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, num_channels, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img


class Model(object):
    """Abstract class on how to define models"""


@gin.configurable()
class VAE(nn.Module):
    def __init__(
        self,
        z_dim,
        use_cuda=False,
        prior_dist=None,
        q_dist=None,
        include_mutinfo=True,
        tcvae=False,
        conv=False,
        mss=False,
        device=0,
        beta=1,
        num_channels=1,
    ):
        super(VAE, self).__init__()
        print("initializing VAE on device", device)

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = beta
        self.mss = mss
        self.x_dist = dist.Bernoulli(device=device)
        self.device = device
        self.num_channels = num_channels

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = (
            dist.Normal(device=device) if prior_dist is None else prior_dist
        )
        self.q_dist = dist.Normal(device=device) if q_dist is None else q_dist
        # hyperparameters for prior p(z)
        self.register_buffer("prior_params", torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        if conv:
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams, num_channels)
            self.decoder = ConvDecoder(z_dim, num_channels)
        else:
            assert num_channels == 1, "More than 1 channel only for conv supported"
            self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
            self.decoder = MLPDecoder(z_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.to_device(self.device)

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    def to_device(self, device):
        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)
        self.x_dist.set_device(device)
        self.prior_dist.set_device(device)
        self.q_dist.set_device(device)
        self.to(device)

    # return dictionary containing all hyperparameters
    def get_hyperparam_state_dict(self):
        return dict(
            use_cuda=self.use_cuda,
            z_dim=self.z_dim,
            include_mutinfo=self.include_mutinfo,
            tcvae=self.tcvae,
            lamb=self.lamb,
            beta=self.beta,
            mss=self.mss,
            x_dist=copy.deepcopy(self.x_dist),
            device=self.device,
            prior_dist=copy.deepcopy(self.prior_dist),
            q_dist=copy.deepcopy(self.q_dist),
            num_channels=self.num_channels,
        )

    # load hyperparameters from state dict
    def load_hyperparam_state_dict(self, state_dict):
        self.use_cuda = state_dict["use_cuda"]
        self.z_dim = state_dict["z_dim"]
        self.include_mutinfo = state_dict["include_mutinfo"]
        self.tcvae = state_dict["tcvae"]
        self.lamb = state_dict["lamb"]
        self.beta = state_dict["beta"]
        self.mss = state_dict["mss"]
        self.x_dist = state_dict["x_dist"]
        self.device = state_dict["device"]
        self.prior_dist = state_dict["prior_dist"]
        self.q_dist = state_dict["q_dist"]
        self.num_channels = state_dict.get("num_channels", 1)

    def backward(self, obj_list):
        objective, _ = obj_list
        objective.mean().mul(-1).backward()

    def nan_in_objective(self, obj_list):
        objective, _ = obj_list
        return utils.isnan(objective).any()

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        assert x.shape == torch.Size([x.size(0), self.num_channels, 64, 64]), (
            "Wrong input data dimensions"
            + str(x.shape)
            + ", expected"
            + str(torch.Size([x.size(0), self.num_channels, 64, 64]))
        )
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(
            x.size(0), self.z_dim, self.q_dist.nparams
        )
        # zs_params shape (n_samples, z_dim, 2), where [:,:,0] is means, [:,:,1] is logvars
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        x_params = self.decoder.forward(z).view(z.size(0), self.num_channels, 64, 64)
        xs = self.x_dist.sample(params=x_params)  ## Why sample for x_dist as well?
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def forward(self, x):
        xs, _, _, _ = self.reconstruct_img(x)
        return xs

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[:: M + 1] = 1 / N
        W.view(-1)[1 :: M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size):
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        assert x.shape == torch.Size([batch_size, self.num_channels, 64, 64]), (
            "Wrong input data dimensions"
            + str(x.shape)
            + ", expected"
            + str(torch.Size([batch_size, self.num_channels, 64, 64]))
        )
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)
        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = (
            self.prior_dist.log_density(zs, params=prior_params)
            .view(batch_size, -1)
            .sum(1)
        )
        logqz_condx = (
            self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)
        )

        elbo = logpx + logpz - logqz_condx

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            return elbo, elbo.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams),
        )

        if not self.mss:
            # minibatch weighted sampling
            logqz_prodmarginals = (
                logsumexp(_logqz, dim=1, keepdim=False)
                - math.log(batch_size * dataset_size)
            ).sum(1)
            logqz = logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(
                batch_size * dataset_size
            )
        else:
            # minibatch stratified sampling
            logiw_matrix = Variable(
                self._log_importance_weight_matrix(batch_size, dataset_size).type_as(
                    _logqz.data
                )
            )
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz,
                dim=1,
                keepdim=False,
            ).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) - self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals)
                    + (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = (
                    logpx
                    - (logqz_condx - logqz)
                    - self.beta * (logqz - logqz_prodmarginals)
                    - (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = (
                    logpx
                    - self.beta * (logqz - logqz_prodmarginals)
                    - (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )

        return modified_elbo, elbo.detach()


@gin.configurable()
class UDRVAE(nn.Module):
    def __init__(
        self,
        z_dim,
        use_cuda=False,
        prior_dist=None,
        q_dist=None,
        include_mutinfo=True,
        tcvae=False,
        conv=False,
        mss=False,
        device=0,
        beta=1,
        num_channels=1,
        num_models=5,
    ):
        super(UDRVAE, self).__init__()
        print("initializing UDRVAE on device", device)
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = beta
        self.mss = mss
        self.num_models = num_models
        self.num_channels = num_channels

        prior_dist = dist.Normal(device=device) if prior_dist is None else prior_dist
        q_dist = dist.Normal(device=device) if q_dist is None else q_dist

        for i in range(num_models):
            model = VAE(
                z_dim=z_dim,
                use_cuda=use_cuda,
                prior_dist=copy.deepcopy(prior_dist),
                q_dist=copy.deepcopy(q_dist),
                include_mutinfo=include_mutinfo,
                tcvae=tcvae,
                conv=conv,
                mss=mss,
                device=device,
                beta=beta,
                num_channels=num_channels,
            )
            self.add_module(str(i), model)
        self.device = device

        self.to_device(self.device)

    def to_device(self, device):

        for name, module in self.named_children():
            module.to_device(device)
        self.to(device)

    # return dictionary containing all hyperparameters
    def get_hyperparam_state_dict(self):
        return next(self.children()).get_hyperparam_state_dict()

    # load hyperparameters from state dict
    def load_hyperparam_state_dict(self, state_dict):
        for name, module in self.named_children():
            module.load_hyperparam_state_dict(state_dict)

        self.device = state_dict["device"]

        if self.use_cuda:
            self.to(self.device)

    def backward(self, obj_list):
        [obj.mean().mul(-1).backward() for obj, elbo in obj_list]

    def nan_in_objective(self, obj_list):
        return np.any([utils.isnan(obj).any() for obj, elbo in obj_list])

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        return [
            module.model_sample(batch_size) for name, module in self.named_children()
        ]

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        return [module.encode(x) for name, module in self.named_children()]

    def decode(self, z):
        return [module.decode(z) for name, module in self.named_children()]

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        return [module.reconstruct_img(x) for name, module in self.named_children()]

    def forward(self, x):
        return [module.forward(x) for name, module in self.named_children()]

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[:: M + 1] = 1 / N
        W.view(-1)[1 :: M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()

    def elbo(self, x, dataset_size):
        return [module.elbo(x, dataset_size) for name, module in self.named_children()]


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False):
    if args.dataset == "shapes":
        train_set = dset.Shapes()
    elif args.dataset == "faces":
        train_set = dset.Faces()
    else:
        raise ValueError("Unknown dataset " + str(args.dataset))

    kwargs = {"num_workers": 4, "pin_memory": use_cuda}
    train_loader = DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    return train_loader


win_samples = None
win_test_reco = None
win_latent_walk = None
win_train_elbo = None


def display_samples(model, x, vis):
    global win_samples, win_test_reco, win_latent_walk

    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    sample_mu = sample_mu
    images = list(sample_mu.view(-1, 1, 64, 64).data.cpu())
    win_samples = vis.images(
        images, 10, 2, opts={"caption": "samples"}, win=win_samples
    )

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()
    test_reco_imgs = torch.cat(
        [test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0
    ).transpose(0, 1)
    win_test_reco = vis.images(
        list(test_reco_imgs.contiguous().view(-1, 1, 64, 64).data.cpu()),
        10,
        2,
        opts={"caption": "test reconstruction image"},
        win=win_test_reco,
    )

    # plot latent walks (change one variable while all others stay the same)
    zs = zs[0:3]
    batch_size, z_dim = zs.size()
    xs = []
    delta = torch.autograd.Variable(torch.linspace(-2, 2, 7), volatile=True).type_as(zs)
    for i in range(z_dim):
        vec = (
            Variable(torch.zeros(z_dim))
            .view(1, z_dim)
            .expand(7, z_dim)
            .contiguous()
            .type_as(zs)
        )
        vec[:, i] = 1
        vec = vec * delta[:, None]
        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = list(torch.cat(xs, 0).data.cpu())
    win_latent_walk = vis.images(
        xs, 7, 2, opts={"caption": "latent walk"}, win=win_latent_walk
    )


def plot_elbo(train_elbo, vis):
    global win_train_elbo
    win_train_elbo = vis.line(
        torch.Tensor(train_elbo), opts={"markers": True}, win=win_train_elbo
    )


def anneal_kl(args, vae, iteration):
    if args.dataset == "shapes":
        warmup_iter = 7000
    elif args.dataset == "faces":
        warmup_iter = 2500

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
    else:
        vae.beta = args.beta


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-d",
        "--dataset",
        default="shapes",
        type=str,
        help="dataset name",
        choices=["shapes", "faces"],
    )
    parser.add_argument(
        "-dist", default="normal", type=str, choices=["normal", "laplace", "flow"]
    )
    parser.add_argument(
        "-n", "--num-epochs", default=50, type=int, help="number of training epochs"
    )
    parser.add_argument("-b", "--batch-size", default=2, type=int, help="batch size")
    parser.add_argument(
        "-l", "--learning-rate", default=1e-3, type=float, help="learning rate"
    )
    parser.add_argument(
        "-z", "--latent-dim", default=10, type=int, help="size of latent dimension"
    )
    parser.add_argument("--beta", default=1, type=float, help="ELBO penalty term")
    parser.add_argument("--tcvae", action="store_true")
    parser.add_argument("--exclude-mutinfo", action="store_true")
    parser.add_argument("--beta-anneal", action="store_true")
    parser.add_argument("--lambda-anneal", action="store_true")
    parser.add_argument(
        "--mss", action="store_true", help="use the improved minibatch estimator"
    )
    parser.add_argument("--conv", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--visdom", action="store_true", help="whether plotting in visdom is desired"
    )
    parser.add_argument("--save", default="test1")
    parser.add_argument(
        "--log_freq", default=200, type=int, help="num iterations per log"
    )
    args = parser.parse_args()

    args.batch_size = 10

    torch.cuda.set_device(args.gpu)
    #    torch.cuda.set_device(args.gpu)

    # data loader
    #    train_loader = setup_data_loaders(args, use_cuda=True)
    train_loader = setup_data_loaders(args, use_cuda=False)

    # setup the VAE
    if args.dist == "normal":
        prior_dist = dist.Normal()
        q_dist = dist.Normal()
    elif args.dist == "laplace":
        prior_dist = dist.Laplace()
        q_dist = dist.Laplace()
    elif args.dist == "flow":
        prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
        q_dist = dist.Normal()

    vae = VAE(
        z_dim=args.latent_dim,
        use_cuda=True,
        prior_dist=prior_dist,
        q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo,
        tcvae=args.tcvae,
        conv=args.conv,
        mss=args.mss,
    )

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)

    # setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom(env=args.save, port=4500)

    train_elbo = []

    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    # initialize loss accumulator
    elbo_running_mean = utils.RunningAverageMeter()
    while iteration < num_iterations:
        for i, x in enumerate(train_loader):
            iteration += 1
            batch_time = time.time()
            vae.train()
            anneal_kl(args, vae, iteration)
            optimizer.zero_grad()
            # transfer to GPU
            x = x.to()
            # wrap the mini-batch in a PyTorch Variable
            x = Variable(x)
            # do ELBO gradient and accumulate loss
            obj, elbo = vae.elbo(x, dataset_size)
            if utils.isnan(obj).any():
                raise ValueError("NaN spotted in objective.")
            obj.mean().mul(-1).backward()
            elbo_running_mean.update(elbo.mean().item())
            optimizer.step()

            # report training diagnostics
            if iteration % args.log_freq == 0:
                train_elbo.append(elbo_running_mean.avg)
                print(
                    "[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f)"
                    % (
                        iteration,
                        time.time() - batch_time,
                        vae.beta,
                        vae.lamb,
                        elbo_running_mean.val,
                        elbo_running_mean.avg,
                    )
                )

                vae.eval()

                # plot training and test ELBOs
                if args.visdom:
                    display_samples(vae, x, vis)
                    plot_elbo(train_elbo, vis)

                utils.save_checkpoint(
                    {"state_dict": vae.state_dict(), "args": args}, args.save, 0
                )
                eval("plot_vs_gt_" + args.dataset)(
                    vae,
                    train_loader.dataset,
                    os.path.join(
                        args.save, "gt_vs_latent_{:05d}.png".format(iteration)
                    ),
                )

    # Report statistics after training
    vae.eval()
    utils.save_checkpoint({"state_dict": vae.state_dict(), "args": args}, args.save, 0)
    dataset_loader = DataLoader(
        train_loader.dataset, batch_size=1000, num_workers=1, shuffle=False
    )
    (
        logpx,
        dependence,
        information,
        dimwise_kl,
        analytical_cond_kl,
        marginal_entropies,
        joint_entropy,
    ) = elbo_decomposition(vae, dataset_loader)
    torch.save(
        {
            "logpx": logpx,
            "dependence": dependence,
            "information": information,
            "dimwise_kl": dimwise_kl,
            "analytical_cond_kl": analytical_cond_kl,
            "marginal_entropies": marginal_entropies,
            "joint_entropy": joint_entropy,
        },
        os.path.join(args.save, "elbo_decomposition.pth"),
    )
    eval("plot_vs_gt_" + args.dataset)(
        vae, dataset_loader.dataset, os.path.join(args.save, "gt_vs_latent.png")
    )
    return vae


if __name__ == "__main__":
    model = main()
