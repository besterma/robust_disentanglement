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

"""Implementations of the UDR score.

Methods for computing the UDR and UDR-A2A scores specified in "Unsupervised
Model Selection for Variational Disentangled Representation Learning"
(https://arxiv.org/abs/1905.12614)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import numpy as np
import scipy
from sklearn import linear_model
from sklearn import preprocessing
import gin.tf
from torch.utils.data import DataLoader


def relative_strength_disentanglement(corr_matrix):
    """Computes disentanglement using relative strength score."""
    score_x = np.nanmean(
        np.nan_to_num(
            np.power(np.ndarray.max(corr_matrix, axis=0), 2)
            / np.sum(corr_matrix, axis=0),
            0,
        )
    )
    score_y = np.nanmean(
        np.nan_to_num(
            np.power(np.ndarray.max(corr_matrix, axis=1), 2)
            / np.sum(corr_matrix, axis=1),
            0,
        )
    )
    return (score_x + score_y) / 2


def spearman_correlation_conv(vec1, vec2):
    """Computes Spearman correlation matrix of two representations.

    Args:
      vec1: 2d array of representations with axis 0 the batch dimension and axis 1
        the representation dimension.
      vec2: 2d array of representations with axis 0 the batch dimension and axis 1
        the representation dimension.

    Returns:
      A 2d array with the correlations between all pairwise combinations of
      elements of both representations are computed. Elements of vec1 correspond
      to axis 0 and elements of vec2 correspond to axis 1.
    """
    assert vec1.shape == vec2.shape
    corr_y = []
    for i in range(vec1.shape[1]):
        corr_x = []
        for j in range(vec2.shape[1]):
            corr, _ = scipy.stats.spearmanr(vec1[:, i], vec2[:, j], nan_policy="omit")
            corr_x.append(corr)
        corr_y.append(np.stack(corr_x))
    return np.transpose(np.absolute(np.stack(corr_y, axis=1)))


def lasso_correlation_matrix(vec1, vec2, random_state=None):
    """Computes correlation matrix of two representations using Lasso Regression.

    Args:
      vec1: 2d array of representations with axis 0 the batch dimension and axis 1
        the representation dimension.
      vec2: 2d array of representations with axis 0 the batch dimension and axis 1
        the representation dimension.
      random_state: int used to seed an RNG used for model training.

    Returns:
      A 2d array with the correlations between all pairwise combinations of
      elements of both representations are computed. Elements of vec1 correspond
      to axis 0 and elements of vec2 correspond to axis 1.
    """
    # assert vec1.shape == vec2.shape
    model = linear_model.Lasso(random_state=random_state, alpha=0.1)
    model.fit(vec1, vec2)
    return np.transpose(np.absolute(model.coef_))


def _generate_representation_batch(
    ground_truth_data, representation_functions, batch_size, random_state
):
    """Sample a single mini-batch of representations from the ground-truth data.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_functions: functions that takes observations as input and
        outputs a dim_representation sized representation for each observation and
        a vector of the average kl divergence per latent.
      batch_size: size of batches of representations to be collected at one time.
      random_state: numpy random state used for randomness.

    Returns:
      representations: List[batch_size, dim_representation] List of representation
        batches for each of the representation_functions.
    """
    # Sample a mini batch of latent variables
    observations = ground_truth_data.sample_observations(batch_size, random_state)
    # Compute representations based on the observations.
    return [fn(observations) for fn in representation_functions]


def _generate_representation_dataset(
    ground_truth_data,
    representation_functions,
    batch_size,
    num_data_points,
    random_state,
):
    """Sample dataset of represetations for all of the different models.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_functions: functions that takes observations as input and
        outputs a dim_representation sized representation for each observation and
        a vector of the average kl divergence per latent.
      batch_size: size of batches of representations to be collected at one time.
      num_data_points: total number of points to be sampled for training set.
      random_state: numpy random state used for randomness.

    Returns:
      representation_points: (num_data_points, dim_representation)-sized numpy
        array with training set features.
      kl: (dim_representation) - The average KL divergence per latent in the
        representation.
    """
    if num_data_points % batch_size != 0:
        raise ValueError("num_data_points must be a multiple of batch_size")

    representation_points = []
    kl_divergence = []

    for i in range(int(num_data_points / batch_size)):
        representation_batch = _generate_representation_batch(
            ground_truth_data, representation_functions, batch_size, random_state
        )

        for j in range(len(representation_functions)):
            # Initialize the outputs if it hasn't been created yet.
            if len(representation_points) <= j:
                kl_divergence.append(
                    np.zeros(
                        (
                            int(num_data_points / batch_size),
                            representation_batch[j][1].shape[0],
                        )
                    )
                )
                representation_points.append(
                    np.zeros((num_data_points, representation_batch[j][0].shape[1]))
                )
            kl_divergence[j][i, :] = representation_batch[j][1]
            representation_points[j][
                i * batch_size : (i + 1) * batch_size, :
            ] = representation_batch[j][0]
    return representation_points, [np.mean(kl, axis=0) for kl in kl_divergence]


def _generate_representation_dataset_pytorch(
    dataset, representation_functions, batch_size, num_data_points, random_state
):
    """Sample dataset of represetations for all of the different pytorch models.

    Args:
      dataset: numpy dataset with shape (
      representation_functions: functions that takes observations as input and
        outputs a dim_representation sized representation for each observation and
        a vector of the average kl divergence per latent.
      batch_size: size of batches of representations to be collected at one time.
      num_data_points: total number of points to be sampled for training set.
      random_state: numpy random state used for randomness.

    Returns:
      representation_points: (num_data_points, dim_representation)-sized numpy
        array with training set features.
      kl: (dim_representation) - The average KL divergence per latent in the
        representation.
    """
    if num_data_points > len(dataset):
        if num_data_points < batch_size:
            batch_size = num_data_points
            print("UDR: Danger, num_data_points < batch_size, reduced batch_size")
        temp_num_dp = num_data_points
        num_data_points = len(dataset) - (len(dataset) % batch_size)
        print(
            "UDR: reduced num_data_points from",
            temp_num_dp,
            "to",
            num_data_points,
            "due to smaller dataset",
        )

    if num_data_points % batch_size != 0:
        raise ValueError("num_data_points must be a multiple of batch_size")

    dataset = dataset.copy()
    random_state.shuffle(dataset)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    representation_points = []
    kl_divergence = []

    for i, sample in enumerate(train_loader):
        if i >= num_data_points // batch_size:
            break

        representation_batch = [
            rep_func(sample) for rep_func in representation_functions
        ]

        for j in range(len(representation_functions)):
            # Initialize the outputs if it hasn't been created yet.
            if len(representation_points) <= j:
                kl_divergence.append(
                    np.zeros(
                        (
                            int(num_data_points / batch_size),
                            representation_batch[j][1].shape[0],
                        )
                    )
                )
                representation_points.append(
                    np.zeros((num_data_points, representation_batch[j][0].shape[1]))
                )
            kl_divergence[j][i, :] = representation_batch[j][1]
            representation_points[j][
                i * batch_size : (i + 1) * batch_size, :
            ] = representation_batch[j][0]
    return representation_points, [np.mean(kl, axis=0) for kl in kl_divergence]


def _generate_representation_and_label_dataset_pytorch(
    dataset, labels, representation_functions, batch_size, num_data_points, random_state
):
    """Sample dataset of represetations for all of the different pytorch models.

    Args:
      dataset: numpy dataset with shape (
      representation_functions: functions that takes observations as input and
        outputs a dim_representation sized representation for each observation and
        a vector of the average kl divergence per latent.
      batch_size: size of batches of representations to be collected at one time.
      num_data_points: total number of points to be sampled for training set.
      random_state: numpy random state used for randomness.

    Returns:
      representation_points: (num_data_points, dim_representation)-sized numpy
        array with training set features.
      kl: (dim_representation) - The average KL divergence per latent in the
        representation."""
    if num_data_points % batch_size != 0:
        raise ValueError("num_data_points must be a multiple of batch_size")

    assert len(labels) == len(dataset), "Dataset and labels dont match"

    dataset = dataset.copy()
    indices = np.array(range(len(dataset)))
    random_state.shuffle(indices)
    label_points = labels[indices][:num_data_points]

    train_loader = DataLoader(
        dataset=dataset[indices],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    representation_points = []
    kl_divergence = []

    for i, sample in enumerate(train_loader):
        if i >= num_data_points // batch_size:
            break

        representation_batch = [
            rep_func(sample) for rep_func in representation_functions
        ]

        for j in range(len(representation_functions)):
            # Initialize the outputs if it hasn't been created yet.
            if len(representation_points) <= j:
                kl_divergence.append(
                    np.zeros(
                        (
                            int(num_data_points / batch_size),
                            representation_batch[j][1].shape[0],
                        )
                    )
                )
                representation_points.append(
                    np.zeros((num_data_points, representation_batch[j][0].shape[1]))
                )
            kl_divergence[j][i, :] = representation_batch[j][1]
            representation_points[j][
                i * batch_size : (i + 1) * batch_size, :
            ] = representation_batch[j][0]
    return (
        representation_points,
        label_points,
        [np.mean(kl, axis=0) for kl in kl_divergence],
    )


@gin.configurable(
    "udr_sklearn",
    blacklist=["ground_truth_data", "representation_functions", "random_state"],
)
def compute_udr_sklearn(
    ground_truth_data,
    representation_functions,
    random_state,
    batch_size,
    num_data_points,
    correlation_matrix="lasso",
    filter_low_kl=True,
    include_raw_correlations=True,
    kl_filter_threshold=0.01,
    pytorch=False,
):
    """Computes the UDR score using scikit-learn.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_functions: functions that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      random_state: numpy random state used for randomness.
      batch_size: Number of datapoints to compute in a single batch. Useful for
        reducing memory overhead for larger models.
      num_data_points: total number of representation datapoints to generate for
        computing the correlation matrix.
      correlation_matrix: Type of correlation matrix to generate. Can be either
        "lasso" or "spearman".
      filter_low_kl: If True, filter out elements of the representation vector
        which have low computed KL divergence.
      include_raw_correlations: Whether or not to include the raw correlation
        matrices in the results.
      kl_filter_threshold: Threshold which latents with average KL divergence
        lower than the threshold will be ignored when computing disentanglement.

    Returns:
      scores_dict: a dictionary of the scores computed for UDR with the following
      keys:
        raw_correlations: (num_models, num_models, latent_dim, latent_dim) -  The
          raw computed correlation matrices for all models. The pair of models is
          indexed by axis 0 and 1 and the matrix represents the computed
          correlation matrix between latents in axis 2 and 3.
        pairwise_disentanglement_scores: (num_models, num_models, 1) - The
          computed disentanglement scores representing the similarity of
          representation between pairs of models.
        model_scores: (num_models) - List of aggregated model scores corresponding
          to the median of the pairwise disentanglement scores for each model.
    """
    logging.info("Generating training set.")
    if pytorch:
        inferred_model_reps, kl = _generate_representation_dataset_pytorch(
            ground_truth_data,
            representation_functions,
            batch_size,
            num_data_points,
            random_state,
        )
    else:
        inferred_model_reps, kl = _generate_representation_dataset(
            ground_truth_data,
            representation_functions,
            batch_size,
            num_data_points,
            random_state,
        )

    num_models = len(inferred_model_reps)
    logging.info("Number of Models: %s", num_models)

    logging.info("Training sklearn models.")
    latent_dim = inferred_model_reps[0].shape[1]
    corr_matrix_all = np.zeros((num_models, num_models, latent_dim, latent_dim))

    # Normalize and calculate mask based off of kl divergence to remove
    # uninformative latents.
    kl_mask = []
    for i in range(len(inferred_model_reps)):
        scaler = preprocessing.StandardScaler()
        scaler.fit(inferred_model_reps[i])
        inferred_model_reps[i] = scaler.transform(inferred_model_reps[i])
        inferred_model_reps[i] = inferred_model_reps[i] * np.greater(kl[i], 0.01)
        kl_mask.append(kl[i] > kl_filter_threshold)

    # kl_mask = list of bool arrays size z_dim

    disentanglement = np.zeros((num_models, num_models, 1))
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue

            if correlation_matrix == "lasso":
                corr_matrix = lasso_correlation_matrix(
                    inferred_model_reps[i], inferred_model_reps[j], random_state
                )
            else:
                corr_matrix = spearman_correlation_conv(
                    inferred_model_reps[i], inferred_model_reps[j]
                )
            # corr matrix shape [z_dim, z_dim]
            corr_matrix_all[i, j, :, :] = corr_matrix
            if filter_low_kl:
                corr_matrix = corr_matrix[kl_mask[i], ...][..., kl_mask[j]]
            if not (kl_mask[i].any() and kl_mask[j].any()):
                disentanglement[i, j] = 0
            else:
                disentanglement[i, j] = relative_strength_disentanglement(corr_matrix)

    scores_dict = {}
    if include_raw_correlations:
        scores_dict["raw_correlations"] = corr_matrix_all.tolist()
    scores_dict["pairwise_disentanglement_scores"] = disentanglement.tolist()

    model_scores = []
    for i in range(num_models):
        model_scores.append(np.median(np.delete(disentanglement[:, i], i)))

    scores_dict["model_scores"] = model_scores

    return scores_dict


@gin.configurable(
    "udr_sklearn_2",
    blacklist=[
        "ground_truth_data",
        "representation_functions",
        "random_state",
        "labels",
    ],
)
def compute_udr_sklearn_2(
    ground_truth_data,
    representation_functions,
    random_state,
    batch_size,
    num_data_points,
    correlation_matrix="lasso",
    filter_low_kl=True,
    include_raw_correlations=True,
    kl_filter_threshold=0.01,
    pytorch=False,
    known_label_factor=0.5,
    labels=None,
    inverse_kl_weighting=False,
):
    """Computes the UDR score using scikit-learn.

    Args:
      ground_truth_data: GroundTruthData to be sampled from.
      representation_functions: functions that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      random_state: numpy random state used for randomness.
      batch_size: Number of datapoints to compute in a single batch. Useful for
        reducing memory overhead for larger models.
      num_data_points: total number of representation datapoints to generate for
        computing the correlation matrix.
      correlation_matrix: Type of correlation matrix to generate. Can be either
        "lasso" or "spearman".
      filter_low_kl: If True, filter out elements of the representation vector
        which have low computed KL divergence.
      include_raw_correlations: Whether or not to include the raw correlation
        matrices in the results.
      kl_filter_threshold: Threshold which latents with average KL divergence
        lower than the threshold will be ignored when computing disentanglement.
      pytorch: Flag specifying if the dataset and representation functions are
        of pytorch type
      known_label_factor: if labels are provided, it specifies how strongly to
        reduce their influence on the final score
      labels: labels fitting the dataset, pytorch needs to be True

    Returns:
      scores_dict: a dictionary of the scores computed for UDR with the following
      keys:
        raw_correlations: (num_models, num_models, latent_dim, latent_dim) -  The
          raw computed correlation matrices for all models. The pair of models is
          indexed by axis 0 and 1 and the matrix represents the computed
          correlation matrix between latents in axis 2 and 3.
        pairwise_disentanglement_scores: (num_models, num_models, 1) - The
          computed disentanglement scores representing the similarity of
          representation between pairs of models.
        model_scores: (num_models) - List of aggregated model scores corresponding
          to the median of the pairwise disentanglement scores for each model.
    """
    logging.info("Generating training set.")
    if pytorch:
        if labels is not None:
            (
                inferred_model_reps,
                sampled_labels,
                kl,
            ) = _generate_representation_and_label_dataset_pytorch(
                ground_truth_data,
                labels,
                representation_functions,
                batch_size,
                num_data_points,
                random_state,
            )
            scaler = preprocessing.StandardScaler()
            sampled_labels = scaler.fit_transform(sampled_labels)

        else:
            inferred_model_reps, kl = _generate_representation_dataset_pytorch(
                ground_truth_data,
                representation_functions,
                batch_size,
                num_data_points,
                random_state,
            )
    else:
        inferred_model_reps, kl = _generate_representation_dataset(
            ground_truth_data,
            representation_functions,
            batch_size,
            num_data_points,
            random_state,
        )

    num_models = len(inferred_model_reps)
    logging.info("Number of Models: %s", num_models)

    logging.info("Training sklearn models.")
    latent_dim = inferred_model_reps[0].shape[1]
    corr_matrix_all = np.zeros((num_models, num_models, latent_dim, latent_dim))

    # Normalize and calculate mask based off of kl divergence to remove
    # uninformative latents.
    kl_mask = []
    for i in range(num_models):
        scaler = preprocessing.StandardScaler()
        scaler.fit(inferred_model_reps[i])
        inferred_model_reps[i] = scaler.transform(inferred_model_reps[i])
        inferred_model_reps[i] = inferred_model_reps[i] * np.greater(kl[i], 0.01)
        kl_mask.append(kl[i] > kl_filter_threshold)

    if labels is not None:
        known_label_mask = []
        for i in range(len(inferred_model_reps)):
            if correlation_matrix == "lasso":
                corr_matrix = lasso_correlation_matrix(
                    inferred_model_reps[i], sampled_labels, random_state
                )
            else:
                corr_matrix = spearman_correlation_conv(
                    inferred_model_reps[i], sampled_labels
                )
            if len(corr_matrix.shape) == 1:
                corr_matrix = np.expand_dims(corr_matrix, axis=-1)
            known_label_mask.append(np.any(np.greater(corr_matrix, 0.8), axis=1))

    disentanglement = np.zeros((num_models, num_models, 1))
    for i in range(num_models):
        for j in range(num_models):
            if i == j:
                continue

            if correlation_matrix == "lasso":
                corr_matrix = lasso_correlation_matrix(
                    inferred_model_reps[i], inferred_model_reps[j], random_state
                )
            else:
                corr_matrix = spearman_correlation_conv(
                    inferred_model_reps[i], inferred_model_reps[j]
                )
            corr_matrix_all[i, j, :, :] = corr_matrix
            if labels is not None and (
                known_label_mask[i].any() and known_label_mask[j].any()
            ):
                corr_matrix[
                    np.ix_(known_label_mask[i], known_label_mask[j])
                ] *= known_label_factor
            if inverse_kl_weighting:
                assert (
                    filter_low_kl
                ), "Inverse kl weighting only possible if filter_low_kl is set"
                corr_matrix *= 1 / kl[j]
            if filter_low_kl:
                corr_matrix = corr_matrix[kl_mask[i], ...][..., kl_mask[j]]
            if not (kl_mask[i].any() and kl_mask[j].any()):
                disentanglement[i, j] = 0
            else:
                disentanglement[i, j] = relative_strength_disentanglement(corr_matrix)

    scores_dict = {}
    if include_raw_correlations:
        scores_dict["raw_correlations"] = corr_matrix_all.tolist()
    scores_dict["pairwise_disentanglement_scores"] = disentanglement.tolist()

    model_scores = []
    for i in range(num_models):
        model_scores.append(np.median(np.delete(disentanglement[:, i], i)))

    scores_dict["model_scores"] = model_scores

    return scores_dict
