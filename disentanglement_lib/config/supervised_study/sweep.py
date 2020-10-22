# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hyperparameter sweeps and configs for the study "unsupervised_study_v1".

Challenging Common Assumptions in the Unsupervised Learning of Disentangled
Representations. Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch,
Sylvain Gelly, Bernhard Schoelkopf, Olivier Bachem. arXiv preprint, 2018.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from disentanglement_lib.config import study
from disentanglement_lib.utils import resources
import disentanglement_lib.utils.hyperparams as h
from six.moves import range


def get_datasets():
    """Returns all the data sets."""
    datasets = h.sweep(
        "dataset.name",
        h.categorical(
            [
                # "dsprites_full", "color_dsprites", "noisy_dsprites",
                # "scream_dsprites", "smallnorb", "cars3d", "shapes3d"
                "shapes3d"
            ]
        ),
    )
    nr_channels_vae = h.sweep(
        "vae_quant.VAE.num_channels",
        h.categorical(
            [
                # 1, 3, 3, 3, 1, 3, 3
                3
            ]
        ),
    )

    return h.zipit([datasets, nr_channels_vae])


# def get_num_models():
#     return h.sweep(
#         "UDRVAE.num_models",
#         h.categorical([
#             "5, 15"
#         ]))


def get_num_latent(sweep):
    return h.sweep("encoder.num_latent", h.discrete(sweep))


def get_seeds(num):
    """Returns random seeds."""
    return h.sweep("model.random_seed", h.categorical(list(range(num))))


def get_default_models():
    # udr_vae config.
    model_name = h.fixed("model.name", "pbt_beta_tc_vae")
    udr_vae = h.zipit([model_name])
    all_models = h.chainit(
        [
            udr_vae,
        ]
    )
    return all_models


def get_config():
    """Returns the hyperparameter configs for different experiments."""
    return h.product(
        [
            get_datasets(),
            get_default_models(),
            get_seeds(1),
        ]
    )


class PbtVaeStudy(study.Study):
    """Defines the study for the paper."""

    def get_model_config(self, model_num=0):
        """Returns model bindings and config file."""
        config = get_config()[model_num]
        model_bindings = h.to_bindings(config)
        model_config_file = resources.get_file(
            "config/supervised_study/model_configs/shared.gin"
        )
        return model_bindings, model_config_file

    def get_eval_config_files(self):
        """Returns evaluation config files."""
        return list(
            resources.get_files_in_folder("config/supervised_study/metric_configs/")
        )

    def get_postprocess_config_files(self):
        """Returns postprocessing config files."""
        return []
