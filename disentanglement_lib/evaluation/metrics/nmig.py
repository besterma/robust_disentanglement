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

"""Mutual Information Gap from the beta-TC-VAE paper.

Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import gin.tf


@gin.configurable(
    "nmig",
    blacklist=[
        "ground_truth_data",
        "labels",
        "representation_function",
        "random_state",
    ],
)
def compute_nmig(
    ground_truth_data,
    representation_function,
    random_state,
    labels=None,
    num_train=gin.REQUIRED,
    batch_size=16,
    active=None,
):
    """Computes the normal mutual information gap.

    Args:
      ground_truth_data: GroundTruthData to be sampled from. If labels is not None, numpy dataset
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      random_state: Numpy random state used for randomness.
      num_train: Number of points used for training.
      batch_size: Batch size for sampling.

    Returns:
      Dict with average mutual information gap.
    """
    # allow partial mig so far only for dsprites
    logging.info("Generating training set.")
    if labels is not None:
        mus_train, ys_train = utils.generate_batch_factor_code_pytorch(
            ground_truth_data,
            labels,
            representation_function,
            num_train,
            random_state,
            batch_size,
        )
    else:
        mus_train, ys_train = utils.generate_batch_factor_code(
            ground_truth_data,
            representation_function,
            num_train,
            random_state,
            batch_size,
        )
    assert mus_train.shape[1] == num_train
    return _compute_nmig(mus_train, ys_train, active)


@gin.configurable("nmig_leaf", whitelist=["num_train", "batch_size"])
def compute_nmig_leaf(
    datasets_list,
    labels_list,
    representation_function,
    random_state,
    num_train=gin.REQUIRED,
    batch_size=16,
):
    assert len(datasets_list) == len(labels_list)

    nr_factors = len(datasets_list)
    effective_nr_factors = 0
    nr_samples_list = []
    for i in range(nr_factors):
        nr_samples_list.append(np.min((num_train, len(datasets_list[i]))))
        effective_nr_factors += (
            1 if len(labels_list[i].shape) == 1 else labels_list[i].shape[1]
        )

    mus_train = []
    ys_train = []
    for i in range(nr_factors):
        print(i)
        mu, y = utils.generate_batch_factor_code_pytorch(
            datasets_list[i],
            labels_list[i],
            representation_function=representation_function,
            num_points=nr_samples_list[i],
            random_state=random_state,
            batch_size=16,
        )
        # mu shape: (10, num_points)
        # y shape: (1, num_points
        mus_train.append(mu)
        ys_train.append(y)

    m = np.zeros((mus_train[0].shape[0], effective_nr_factors))
    entropy = np.zeros((effective_nr_factors))

    s_i = 0
    for i in range(nr_factors):
        discretized_mus = utils.make_discretizer(mus_train[i])
        discretized_ys = utils.make_discretizer(ys_train[i])
        disc_mi = utils.discrete_mutual_info(discretized_mus, discretized_ys)
        size = disc_mi.shape[1]
        m[:, s_i : s_i + size] = disc_mi
        entropy[s_i : s_i + size] = utils.discrete_entropy(discretized_ys)
        s_i += size

    sorted_m = np.sort(m, axis=0)[::-1]
    individual_mig = np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:])
    print("ind mig", individual_mig)
    mig = np.mean(individual_mig)

    nr_gt = effective_nr_factors

    if nr_gt == 1:
        nmig = np.max(np.divide(m, entropy[:]))
    else:
        m = np.divide(m, entropy[:])
        partials = np.zeros((nr_gt))
        best_ids = np.argmax(m, axis=0)
        for i in range(nr_gt):
            mask = np.ones((nr_gt), dtype=np.bool)
            mask[i] = 0
            best_id = best_ids[i]
            partials[i] = m[best_id, i] - np.max(m[best_id, mask])
        nmig = np.mean(partials)
        print("ind nmig", partials)

    score_dict = dict()
    score_dict["discrete_mig"] = mig
    score_dict["discrete_nmig"] = nmig

    return score_dict


def _compute_nmig(mus_train, ys_train, active):
    """Computes score based on both training and testing codes and factors."""
    print("start nmig")
    score_dict = {}
    discretized_mus = utils.make_discretizer(mus_train)
    m = utils.discrete_mutual_info(discretized_mus, ys_train)
    # m shape: (10, nr_ground_truth)
    print("finished discretizing")
    assert m.shape[0] == mus_train.shape[0]
    assert m.shape[1] == ys_train.shape[0]
    entropy = utils.discrete_entropy(ys_train)
    if active is not None:
        assert len(active) <= ys_train.shape[0]
        m = m[:, active]
        entropy = entropy[active]
    nr_lt = m.shape[0]
    nr_gt = m.shape[1]
    # m is [num_latents, num_factors]

    sorted_m = np.sort(m, axis=0)[::-1]
    individual_mig = np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:])
    print("ind mig", individual_mig)
    mig = np.mean(individual_mig)

    if nr_gt == 1:
        nmig = np.max(np.divide(m, entropy[:]))
    else:
        m = np.divide(m, entropy[:])
        partials = np.zeros((nr_gt))
        best_ids = np.argmax(m, axis=0)
        for i in range(nr_gt):
            mask = np.ones((nr_gt), dtype=np.bool)
            mask[i] = 0
            best_id = best_ids[i]
            partials[i] = m[best_id, i] - np.max(m[best_id, mask])
        nmig = np.mean(partials)
        print("ind nmig", partials)
    score_dict["discrete_mig"] = mig
    score_dict["discrete_nmig"] = nmig

    return score_dict
