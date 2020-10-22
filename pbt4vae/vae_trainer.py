import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from functools import partial
from contextlib import nullcontext

import gin
from pbt4vae.utils import (
    _representation_function,
    _encoder,
)


from disentanglement_lib.evaluation.udr.metrics.udr import (
    compute_udr_sklearn as compute_udr,
)
from disentanglement_lib.evaluation.metrics.nmig import compute_nmig, compute_nmig_leaf
from disentanglement_lib.evaluation.metrics.dci import compute_dci


class Trainer(object):
    """Abstract class for trainers."""

    def train(self, dataset, random_seed):
        raise NotImplementedError()

    def save_checkpoint(self, checkpoint_path, random_state):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint_path):
        raise NotImplementedError()

    def export_best_model(self, checkpoint_path, dataset):
        raise NotImplementedError()

    def eval(self, dataset, labels):
        raise NotImplementedError()

    def release_memory(self):
        raise NotImplementedError()

    def set_id(self, new_id):
        raise NotImplementedError()


@gin.configurable(blacklist=["random_state", "device"])
class GeneralVaeTrainer(Trainer):
    def __init__(
        self,
        model_class,
        optimizer_class,
        device=None,
        hyper_params=None,
        is_test_run=False,
        score_random_seed=None,
        random_state=None,
        epoch_train_steps=737280,
        batch_size=None,
        batch_size_init_function=None,
        beta=None,
        beta_init_function=None,
        lr=None,
        lr_init_function=None,
    ):

        # Technically only needed for first time model creation, after that they will be overridden during chkpt loading
        beta = beta_init_function(random_state) if "beta" in hyper_params else beta
        self.batch_size = (
            batch_size_init_function(random_state)
            if "batch_size" in hyper_params
            else batch_size
        )
        lr = lr_init_function(random_state) if "lr" in hyper_params else lr
        use_cuda = device != torch.device("cpu") and device != "cpu"

        self.to_device = 0 if use_cuda else "cpu"

        # Other parameters are supplied by gin config
        self.model = model_class(
            use_cuda=use_cuda,
            device=0 if use_cuda else "cpu",
            beta=beta,
        )
        self.optimizer = optimizer_class(self.model.parameters(), lr=lr)
        self.task_id = None
        self.device = device
        self.training_params = dict()
        self.scores = []
        self.hyper_params = hyper_params
        self.score_random_seed = score_random_seed
        self.epoch_train_steps = epoch_train_steps
        self.is_test_run = is_test_run

    def set_id(self, new_id):
        self.task_id = new_id

    def release_memory(self):
        self.model.to_device(torch.device("cpu"))

    def save_checkpoint(self, checkpoint_path, random_state):
        print(self.task_id, "trying to save checkpoint")
        checkpoint = dict(
            model_state_dict=self.model.state_dict(),
            hyperparam_state_dict=self.model.get_hyperparam_state_dict(),
            optim_state_dict=self.optimizer.state_dict(),
            batch_size=self.batch_size,
            training_params=self.training_params,
            scores=self.scores,
            random_state=random_state,
        )
        torch.save(checkpoint, checkpoint_path)
        print(self.task_id, "finished saving checkpoint")

    def load_checkpoint(self, checkpoint_path):
        print(self.task_id, "trying to load checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.load_hyperparam_state_dict(checkpoint["hyperparam_state_dict"])
        self.model.to_device(self.to_device)
        self.optimizer.load_state_dict(checkpoint["optim_state_dict"])
        self.batch_size = checkpoint["batch_size"]
        self.training_params = checkpoint["training_params"]
        self.scores = checkpoint["scores"]
        print(self.task_id, "finished loading checkpoint")
        return checkpoint["random_state"]

    def export_best_model(self, checkpoint_path, dataset):
        print(self.task_id, "trying to export best model")
        checkpoint = dict(
            model_state_dict=self.model.state_dict(),
            hyperparam_state_dict=self.model.get_hyperparam_state_dict(),
            batch_size=self.batch_size,
            training_params=self.training_params,
            scores=self.scores,
        )
        torch.save(checkpoint, checkpoint_path)
        print(self.task_id, "finished exporting best model")

    def save_training_params(self):
        epoch = 0
        while epoch in self.training_params:
            epoch += 1
        param_dict = dict(epoch=epoch)
        optim_state_dict = self.optimizer.state_dict()
        if "lr" in self.hyper_params:
            param_dict["lr"] = optim_state_dict["param_groups"][0].get("lr", "empty")
        if "batch_size" in self.hyper_params:
            param_dict["batch_size"] = self.batch_size
        if "beta" in self.hyper_params:
            param_dict["beta"] = self.model.beta

        self.training_params[epoch] = param_dict

    def train(self, dataset, random_seed=None):
        start = time.time()
        with nullcontext() if self.device == torch.device("cpu") else self.device:
            pin_memory = not (self.device == torch.device("cpu"))
            drop_last = len(dataset) % self.batch_size == 1
            train_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=pin_memory,
                drop_last=drop_last,
            )
        dataset_size = len(train_loader.dataset)
        print(
            self.task_id,
            "start training with parameters B",
            self.model.beta,
            "lr",
            self.optimizer.param_groups[0]["lr"],
            "batch size",
            self.batch_size,
        )
        current_train_steps = 0
        if self.epoch_train_steps is None:
            max_train_steps = np.max((10000, dataset_size))
            print(
                "task",
                self.task_id,
                "using adaptive epoch train steps",
                max_train_steps,
            )
        else:
            max_train_steps = self.epoch_train_steps
        if max_train_steps % self.batch_size == 1:
            max_train_steps -= 1
        with nullcontext() if self.device == torch.device("cpu") else self.device:
            while current_train_steps < max_train_steps:
                for i, x in enumerate(train_loader):
                    if current_train_steps >= max_train_steps:
                        break
                    if current_train_steps + x.size(0) > max_train_steps:
                        x = x[: max_train_steps - current_train_steps]
                        current_train_steps = max_train_steps
                    else:
                        current_train_steps += x.size(0)
                    if current_train_steps % 200000 == 0:
                        print(
                            "task",
                            self.task_id,
                            "iteration",
                            current_train_steps,
                            "of",
                            max_train_steps,
                        )

                    if self.is_test_run and current_train_steps % 10000 != 0:
                        continue

                    self.model.train()
                    self.optimizer.zero_grad()
                    x = x.to(device=self.to_device)
                    x = Variable(x)
                    obj = self.model.elbo(x, dataset_size)
                    if self.model.nan_in_objective(obj):
                        raise ValueError("NaN spotted in objective.")
                    self.model.backward(obj)
                    self.optimizer.step()
        self.save_training_params()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()
        print(
            "Trainer",
            self.task_id,
            "finished training in",
            time.time() - start,
            "seconds",
        )
        del train_loader

    def generate_udr_repr_function(self):
        self.model.eval()
        return partial(_representation_function, model=self.model)


@gin.configurable(blacklist=["random_state", "device"])
class UdrVaeTrainer(GeneralVaeTrainer):
    def __init__(self, inverse_kl_weighting=False, **kw):
        super().__init__(**kw)
        self.inverse_kl_weighting = inverse_kl_weighting

    def export_best_model(self, checkpoint_path, dataset):
        """
        export best model out of th UDRVAE according to UDR metric
        :type checkpoint_path: string
        :type dataset: ground_truth_data.GroundTruthData or numpy array
        """
        print(self.task_id, "exporting best model")
        with nullcontext() if self.device == torch.device("cpu") else self.device:
            self.model.eval()
            random_state = np.random.RandomState(self.score_random_seed)
            representation_functions = []
            models = list(self.model.children())
            for model in models:
                model.eval()
                representation_functions.append(
                    partial(_representation_function, model=model)
                )

            udr_score_dict = compute_udr(
                dataset, representation_functions, random_state, pytorch=True
            )
        best_model_id = np.argmax(udr_score_dict["model_scores"])

        for name, module in self.model.named_children():
            if name == str(best_model_id):
                export_model = module
                print(
                    "Best model:",
                    name,
                    "type",
                    type(module),
                    "score",
                    np.max(udr_score_dict["model_scores"]),
                )
                break

        checkpoint = dict(
            model_state_dict=export_model.state_dict(),
            hyperparam_state_dict=export_model.get_hyperparam_state_dict(),
            batch_size=self.batch_size,
            training_params=self.training_params,
            scores=self.scores,
        )
        torch.save(checkpoint, checkpoint_path)
        print(self.task_id, "saved best model", best_model_id)

    @torch.no_grad()
    def eval(self, dataset, labels=None):
        with nullcontext() if self.device == torch.device("cpu") else self.device:
            self.model.eval()
            print(self.task_id, "Evaluate Model with B", self.model.beta)
            start = time.time()
            representation_functions = []
            models = list(self.model.children())
            for model in models:
                model.eval()
                representation_functions.append(
                    partial(_representation_function, model=model)
                )

            random_state = np.random.RandomState(self.score_random_seed)

            udr_score_dict = compute_udr(
                dataset, representation_functions, random_state, pytorch=True
            )

        final_score = np.mean(udr_score_dict["model_scores"])
        score_dict = dict(final_score=final_score, udr_score=udr_score_dict)
        self.scores.append(score_dict)
        print(
            self.task_id,
            "Model with B",
            self.model.beta,
            "got final score",
            final_score,
        )
        print(self.task_id, "Eval took", time.time() - start, "seconds")
        if self.is_test_run:
            final_score = np.random.choice([0.05, 0.9], size=(1,))
            print("For testing purposes, it got score", final_score)
        return final_score


@gin.configurable(blacklist=["random_state", "device"])
class MigVaeTrainer(GeneralVaeTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def eval(self, dataset, labels):
        with nullcontext() if self.device == torch.device("cpu") else self.device:
            self.model.eval()
            print(self.task_id, "Evaluate Model with B", self.model.beta)
            start = time.time()
            self.model.eval()
            representation_function = partial(_encoder, model=self.model)

            random_state = np.random.RandomState(self.score_random_seed)
            nmig_score_dict = compute_nmig(
                ground_truth_data=dataset,
                representation_function=representation_function,
                random_state=random_state,
                labels=labels,
            )

        final_score = nmig_score_dict["discrete_mig"]
        score_dict = dict(final_score=final_score, nmig_score=nmig_score_dict)
        self.scores.append(score_dict)
        print(
            self.task_id,
            "Model with B",
            self.model.beta,
            "got final score",
            final_score,
        )
        print(self.task_id, "Eval took", time.time() - start, "seconds")
        if self.is_test_run:
            final_score = np.random.choice([0.05, 0.9], size=(1,))
            print("For testing purposes, it got score", final_score)
        return final_score


@gin.configurable(blacklist=["random_state", "device"])
class DciVaeTrainer(GeneralVaeTrainer):
    def __init__(self, score_func=gin.REQUIRED, **kw):
        super().__init__(**kw)
        self.score_func = score_func

    def eval(self, dataset, labels):
        with nullcontext() if self.device == torch.device("cpu") else self.device:
            self.model.eval()
            print(self.task_id, "Evaluate Model with B", self.model.beta)
            start = time.time()
            self.model.eval()
            representation_function = partial(_encoder, model=self.model)

            random_state = np.random.RandomState(self.score_random_seed)
            dci_score_dict = compute_dci(
                ground_truth_data=dataset,
                representation_function=representation_function,
                random_state=random_state,
                labels=labels,
            )

        final_score = self.score_func(dci_score_dict)
        score_dict = dict(final_score=final_score, score_dict=dci_score_dict)
        self.scores.append(score_dict)
        print(
            self.task_id,
            "Model with B",
            self.model.beta,
            "got final score",
            final_score,
        )
        print(self.task_id, "Eval took", time.time() - start, "seconds")
        if self.is_test_run:
            final_score = np.random.choice([0.05, 0.9], size=(1,))
            print("For testing purposes, it got score", final_score)
        return final_score


@gin.configurable(blacklist=["random_state", "device"])
class LeafMigVaeTrainer(GeneralVaeTrainer):
    def __init__(self, indice_label_paths=gin.REQUIRED, **kw):
        super().__init__(**kw)
        self.indice_label_paths = indice_label_paths

    def eval(self, dataset, labels):
        with nullcontext() if self.device == torch.device("cpu") else self.device:
            self.model.eval()
            print(self.task_id, "Evaluate Model with B", self.model.beta)
            start = time.time()
            representation_function = partial(_encoder, model=self.model)
            random_state = np.random.RandomState(self.score_random_seed)
            nr_label_sets = len(self.indice_label_paths)
            indice_string = "indices_{}"
            labels_string = "labels_{}"

            individual_scores = []
            for i in range(nr_label_sets):
                indice_labels = np.load(self.indice_label_paths[i])
                current_dataset = dataset

                datasets_list = []
                labels_list = []

                nr_factors = 0
                while indice_string.format(nr_factors) in indice_labels:
                    nr_factors += 1

                for j in range(nr_factors):
                    current_labels = indice_labels[labels_string.format(j)]
                    assert len(current_dataset) == len(current_labels)

                    datasets_list.append(current_dataset)
                    labels_list.append(current_labels)

                    current_indices = indice_labels[indice_string.format(j)]
                    current_dataset = current_dataset[current_indices]

                score_dict = compute_nmig_leaf(
                    datasets_list=datasets_list,
                    labels_list=labels_list,
                    representation_function=representation_function,
                    random_state=random_state,
                )
                score = score_dict["discrete_mig"]
                individual_scores.append(score)

        final_score = np.median(individual_scores)

        score_dict = dict(final_score=final_score, individual_scores=individual_scores)
        self.scores.append(score_dict)
        print(
            self.task_id,
            "Model with B",
            self.model.beta,
            "got final score",
            final_score,
        )
        print(self.task_id, "Eval took", time.time() - start, "seconds")
        if self.is_test_run:
            final_score = np.random.choice([0.05, 0.9], size=(1,))
            print("For testing purposes, it got score", final_score)
        return final_score
