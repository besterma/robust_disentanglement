import pickle
from shutil import copyfile
import numpy as np
import torch
import torch.multiprocessing as _mp
import os
import time
import gin
from torch.optim import Adam
from torch.utils.data import DataLoader
from pbt4vae.utils import TorchIterableDataset, _representation_function
from scipy import signal
from contextlib import nullcontext

from functools import reduce, partial
from tqdm import tqdm
from itertools import product
import random


from beta_tcvae.vae_quant import VAE

mp = _mp.get_context("spawn")


@gin.configurable(
    whitelist=[
        "trainer_class",
        "exploit_and_explore_func",
        "cutoff",
        "dataset_path",
        "label_path",
        "big_init_population_factor",
    ]
)
class GeneralExplorer(mp.Process):
    def __init__(
        self,
        is_stop_requested,
        train_queue,
        score_queue,
        finished_queue,
        gpu_id,
        result_dict,
        gin_string,
        model_dir,
        random_states,
        trainer_class,
        init_population_size,
        dataset_path=None,
        label_path=None,
        big_init_population_factor=1,
        exploit_and_explore_func=gin.REQUIRED,
        cutoff=gin.REQUIRED,
    ):

        print("Init Explorer")
        super().__init__()
        self.is_stop_requested = is_stop_requested
        self.train_queue = train_queue
        self.score_queue = score_queue
        self.finished_queue = finished_queue
        self.gpu_id = gpu_id
        self.result_dict = result_dict
        self.random_state = np.random.RandomState()
        self.gin_config = gin_string
        self.model_dir = model_dir
        self.dataset_path = dataset_path
        self.label_path = label_path
        self.epoch = 0
        self.iteration = 0
        self.exploit_and_explore_func = exploit_and_explore_func
        self.epoch_start_time = 0
        self.trainer_class = trainer_class
        self.cutoff = cutoff
        self.big_init_population_factor = big_init_population_factor
        self.population_size = init_population_size

        self.device = (
            torch.cuda.device(0) if (not gpu_id == "cpu") else torch.device("cpu")
        )

        self.device_id = 0 if (not gpu_id == "cpu") else torch.device("cpu")

        self.current_best_model_path = None
        self.current_best_model_score = -1

        if "scores" not in self.result_dict:
            self.result_dict["scores"] = dict()
        if "parameters" not in self.result_dict:
            self.result_dict["parameters"] = dict()

        self.set_rng_states(random_states)
        self.dataset = None

    def run(self):
        print(
            "Running in loop of explorer in epoch ", self.epoch, "on gpu", self.gpu_id
        )
        gin.external_configurable(Adam, module="torch")
        gin.parse_config(self.gin_config)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with nullcontext() if self.device == torch.device("cpu") else self.device:
            print("Explorer init dataset")

            self.dataset = (
                np.load(self.dataset_path)["images"].astype(np.float32) / 255.0
            )

            if len(self.dataset.shape) == 3:
                self.dataset = np.expand_dims(self.dataset, axis=1)

            self.dataset_path = os.path.join(
                self.model_dir, "datasets", "dataset_iteration-000.npz"
            )

            np.savez_compressed(
                self.dataset_path, images=(self.dataset * 255.0).astype(np.uint8)
            )

            print("Explorer dataset initialized")
            while not self.finished_queue.empty():
                task = self.finished_queue.get()
                if self.label_path is not None:
                    task["label_path"] = self.label_path
                self.train_queue.put(task)
            self.epoch_start_time = time.time()
            while True:
                status = self.main_loop()
                if status != 0:
                    break

        print("Explorer finishing")
        return

    @torch.no_grad()
    def main_loop(self):
        raise NotImplementedError()

    def exploit_and_explore(self, tasks):
        raise NotImplementedError()

    def export_scores(self, tasks):
        print("Explorer export scores")
        with open(os.path.join(self.model_dir, "parameters/scores.txt"), "a+") as f:
            f.write(
                str(self.iteration)
                + ". Iteration "
                + str(self.epoch)
                + ". Epoch Scores:"
            )
            for task in tasks:
                f.write(
                    "\n\tId: " + str(task["id"]) + " - Score: " + str(task["score"])
                )
            f.write("\n")

            self.result_dict["scores"][self.epoch] = tasks

    def export_best_model(self, top_checkpoint_path, score):
        model_path = os.path.join(
            self.model_dir,
            "bestmodels/model_iteration-{:03d}_epoch_{:03d}.pth".format(
                self.iteration, self.epoch
            ),
        )
        if score > self.current_best_model_score:
            self.current_best_model_path = model_path
            self.current_best_model_score = score
        copyfile(top_checkpoint_path, model_path)

    def export_best_model_parameters(self, top_checkpoint_path, task):
        print("Explorer export best model parameters")
        checkpoint = torch.load(top_checkpoint_path, map_location=torch.device("cpu"))
        with open(
            os.path.join(self.model_dir, "parameters", "best_parameters.txt"), "a+"
        ) as f:
            f.write(
                "\n\n"
                + str(self.epoch)
                + ". Epoch: Score of "
                + str(task["score"])
                + " for task "
                + str(task["id"])
                + " achieved with following parameters:"
            )
            for i in range(self.epoch):
                f.write(
                    "\n"
                    + str(checkpoint["training_params"][i])
                    + str(checkpoint["scores"][i])
                )

    def save_model_parameters(self, tasks):
        print("Explorer save model parameters")
        temp_dict = dict()
        for task in tasks:
            checkpoint = torch.load(
                task["model_path"],
                map_location=torch.device("cpu"),
            )
            checkpoint_dict = dict()
            checkpoint_dict["training_params"] = checkpoint.get("training_params", None)
            checkpoint_dict["scores"] = checkpoint.get("scores", None)
            temp_dict[task["id"]] = checkpoint_dict

        self.result_dict["parameters"][self.epoch] = temp_dict

        pickle_out = open(
            os.path.join(
                self.model_dir,
                "parameters/parameters-{:03d}-{:03d}.pickle".format(
                    self.iteration, self.epoch
                ),
            ),
            "wb",
        )
        pickle.dump(self.result_dict, pickle_out)
        pickle_out.close()

    def set_rng_states(self, rng_states):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        (
            numpy_rng_state,
            random_rng_state,
            torch_cpu_rng_state,
            torch_gpu_rng_state,
        ) = rng_states
        self.random_state.set_state(numpy_rng_state)
        random.setstate(random_rng_state)
        device = "cpu" if self.gpu_id == "cpu" else 0
        torch.cuda.set_rng_state(torch_gpu_rng_state, device=device)
        torch.random.set_rng_state(torch_cpu_rng_state)

    def generate_random_states(self):
        random_seed = self.random_state.randint(low=1, high=2 ** 32 - 1)
        random_state = np.random.RandomState(random_seed)
        numpy_rng_state = random_state.get_state()
        random.seed(random_seed)
        random_rng_state = random.getstate()
        torch.cuda.manual_seed(random_seed)
        torch.random.manual_seed(random_seed)
        torch_cpu_rng_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            torch_gpu_rng_state = torch.cuda.get_rng_state()
        else:
            torch_gpu_rng_state = torch.random.get_rng_state()

        return [
            numpy_rng_state,
            random_rng_state,
            torch_cpu_rng_state,
            torch_gpu_rng_state,
        ]

    def exploit_and_explore_step(self, tasks):
        cutoff = int(np.ceil(self.cutoff * len(tasks)))
        tops = tasks[:cutoff]
        bottoms = tasks[len(tasks) - cutoff :]

        for bottom in bottoms:
            top = self.random_state.choice(tops)
            top_checkpoint_path = top["model_path"]
            bot_checkpoint_path = bottom["model_path"]
            self.exploit_and_explore_func(
                top_checkpoint_path=top_checkpoint_path,
                bot_checkpoint_path=bot_checkpoint_path,
                random_state=self.random_state,
            )

    def prepare_new_epoch(self, tasks):
        for task in tasks:
            score = task.get("score", -1)
            print("Put task", task["id"], "in queue with score", score)
            self.train_queue.put(task)

        torch.cuda.empty_cache()
        self.epoch_start_time = time.time()

    def export_best_model_and_cleanup(self, tasks):
        trainer = self.trainer_class(device=self.device, random_state=self.random_state)
        trainer.load_checkpoint(self.current_best_model_path)
        trainer.export_best_model(
            os.path.join(self.model_dir, "bestmodels/model.pth"),
            dataset=self.dataset,
        )
        for task in tasks:
            task.pop("random_states", None)
            self.finished_queue.put(task)
        time.sleep(1)

    def process_tasks_and_export_measures(self, tasks):
        tasks = sorted(tasks, key=lambda x: x["score"], reverse=True)
        print("Total #tasks:", len(tasks))
        print("Best score on", tasks[0]["id"], "is", tasks[0]["score"])
        print("Worst score on", tasks[-1]["id"], "is", tasks[-1]["score"])
        best_model_path = tasks[0]["model_path"]
        best_score = tasks[0]["score"]
        try:
            self.export_scores(tasks=tasks)
            self.export_best_model(best_model_path, best_score)
            self.save_model_parameters(tasks=tasks)
            # self.export_best_model_parameters(best_model_path, tasks[0])

        except RuntimeError as err:
            print("Runtime Error in Explorer:", err)
            torch.cuda.empty_cache()
            return None

        if not self.big_init_population_factor == 1:
            old_population_size = self.population_size
            new_population_size = old_population_size // self.big_init_population_factor
            tasks = tasks[:new_population_size]
            self.population_size = new_population_size
            self.big_init_population_factor = 1

        return tasks


@gin.configurable(
    whitelist=[
        "trainer_class",
        "exploit_and_explore_func",
        "cutoff",
        "dataset_path",
        "max_epoch",
    ]
)
class ClassicExplorer(GeneralExplorer):
    def __init__(self, max_epoch=gin.REQUIRED, **kwargs):
        super().__init__(**kwargs)
        self.max_epoch = max_epoch

    @torch.no_grad()
    def main_loop(self):
        if (
            self.train_queue.empty()
            and self.score_queue.empty()
            and self.finished_queue.qsize() == self.population_size
        ):
            print("One epoch took", time.time() - self.epoch_start_time, "seconds")
            time.sleep(1)  # Bug of not having all tasks in finished_queue
            tasks = []
            while not self.finished_queue.empty():
                tasks.append(self.finished_queue.get())

            return self.exploit_and_explore(tasks)

        else:
            print(
                "Already finished:",
                self.finished_queue.qsize(),
                "remaining to train:",
                str(self.train_queue.qsize()),
                "to score:",
                self.score_queue.qsize(),
            )
            time.sleep(20)
            return 0

    def exploit_and_explore(self, tasks):
        print("Exploit and explore")

        tasks = self.process_tasks_and_export_measures(tasks)

        if self.epoch > self.max_epoch:
            print("Explorer: Reached exit condition")
            self.export_best_model_and_cleanup(tasks)
            self.is_stop_requested.value = True
            return 1
        else:
            self.exploit_and_explore_step(tasks)
            self.epoch += 1
            print("New epoch: ", self.epoch)
            self.prepare_new_epoch(tasks)
            return 0


@gin.configurable()
class RPUVAEExplorer(GeneralExplorer):
    def __init__(
        self,
        is_x_run=gin.REQUIRED,
        x_peak=gin.REQUIRED,
        x_model_path=gin.REQUIRED,
        score_threshold=gin.REQUIRED,
        peak_side_cutoff=gin.REQUIRED,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.is_x_run = is_x_run
        self.x_peak = x_peak
        self.x_model_path = x_model_path
        self.score_threshold = score_threshold
        self.peak_side_cutoff = peak_side_cutoff

        assert self.peak_side_cutoff < 0.5
        assert self.peak_side_cutoff >= 0

    def run(self):
        print(
            "Running in loop of explorer in epoch ", self.epoch, "on gpu", self.gpu_id
        )
        gin.external_configurable(Adam, module="torch")
        gin.parse_config(self.gin_config)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with nullcontext() if self.device == torch.device("cpu") else self.device:
            print("Explorer init dataset")

            self.dataset = (
                np.load(self.dataset_path)["images"].astype(np.float32) / 255.0
            )

            if len(self.dataset.shape) == 3:
                self.dataset = np.expand_dims(self.dataset, axis=1)

            if self.is_x_run:
                self.dataset_path = os.path.join(
                    self.model_dir, "datasets", "dataset_iteration-000.npz"
                )
                np.savez_compressed(
                    self.dataset_path, images=(self.dataset * 255.0).astype(np.uint8)
                )
            else:
                self.refresh_dataset(self.x_model_path)
                print("encoding dataset from previous model")

            print("Explorer dataset initialized")
            while not self.finished_queue.empty():
                task = self.finished_queue.get()
                if self.label_path is not None:
                    task["label_path"] = self.label_path
                self.train_queue.put(task)
            self.epoch_start_time = time.time()
            while True:
                status = self.main_loop()
                if status != 0:
                    break

        print("Explorer finishing")
        return

    @torch.no_grad()
    def main_loop(self):
        if (
            self.train_queue.empty()
            and self.score_queue.empty()
            and self.finished_queue.qsize() == self.population_size
        ):
            print("One epoch took", time.time() - self.epoch_start_time, "seconds")
            time.sleep(1)  # Bug of not having all tasks in finished_queue
            tasks = []
            while not self.finished_queue.empty():
                tasks.append(self.finished_queue.get())
            return self.exploit_and_explore(tasks)

        else:
            print(
                "Already finished:",
                self.finished_queue.qsize(),
                "remaining to train:",
                self.train_queue.qsize(),
                "to score:",
                self.score_queue.qsize(),
            )
            time.sleep(20)
            return 0

    def exploit_and_explore(self, tasks):
        print("Exploit and explore")
        tasks = self.process_tasks_and_export_measures(tasks)

        if self.end_condition(tasks):
            print("Explorer: Reached exit condition")
            self.export_best_model_and_cleanup(tasks)
            self.is_stop_requested.value = True
            return 1

        else:
            if self.next_iteration_condition(tasks):
                if self.is_x_run:
                    print("Encoding run finished, cleaning up")
                    self.export_best_model_and_cleanup(tasks)
                    self.is_stop_requested.value = True
                    return 1
                else:
                    export_model_path = self.extract_iteration_best_model()
                    self.iteration += 1
                    self.epoch = 0
                    if self.refresh_dataset(export_model_path) != 0:
                        print("Explorer: Reached exit condition from dataset reduction")
                        self.export_best_model_and_cleanup(tasks)
                        self.is_stop_requested.value = True
                        return 1
                    self.update_task_paths(tasks)
                    self.reset_population(tasks)
                    print("Explorer: New Iteration", self.iteration)

            else:
                self.exploit_and_explore_step(tasks)
                self.epoch += 1
                print("New epoch: ", self.epoch)

            self.prepare_new_epoch(tasks)
            return 0

    def extract_iteration_best_model(self):
        trainer = self.trainer_class(device=self.device, random_state=self.random_state)
        trainer.load_checkpoint(self.current_best_model_path)
        export_model_path = os.path.join(
            self.model_dir,
            "bestmodels",
            "model_iteration-{:03d}.pth".format(self.iteration),
        )
        trainer.export_best_model(export_model_path, dataset=self.dataset)
        return export_model_path

    def update_task_paths(self, tasks):
        for task in tasks:
            task["dataset_path"] = self.dataset_path

    def refresh_dataset(self, model_path):
        labels = self.label_dataset(model_path)
        return self.reduce_dataset(labels)

    def end_condition(self, tasks):
        tasks = sorted(tasks, key=lambda x: x["score"], reverse=True)
        return tasks[0]["score"] < 0.1

    def next_iteration_condition(self, tasks):
        try:
            diffs = []
            for epoch in range(self.epoch - 5, self.epoch + 1):
                diffs.append(
                    np.abs(
                        self.result_dict["scores"][epoch][0]["score"]
                        - self.result_dict["scores"][epoch - 1][0]["score"]
                    )
                )
            criteria = np.array(diffs) < 0.005
            return criteria.all()
        except (IndexError, KeyError) as e:
            print("Could not find scores at epoch: ", self.epoch)
            return False

    def label_dataset(self, model_path):
        model = VAE(device=self.device_id)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        model_function = partial(_representation_function, model=model)

        batch_size = 32
        train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        num_samples = len(self.dataset)
        labels = np.zeros((num_samples, model.z_dim))

        kl_divs = np.zeros((len(train_loader), model.z_dim))

        for i, x in enumerate(train_loader):
            means, kl = model_function(x)
            labels[i * batch_size : (i + 1) * batch_size] = means
            kl_divs[i] = kl
        print("")
        kl_means = np.mean(kl_divs, axis=0)
        print("Latent KL divs: ", kl_means)

        if np.max(kl_means) > 1:
            labels = labels[:, kl_means > 1]
        else:
            print("No latent with KL > 1 found")
            labels = None

        if labels is not None and len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=-1)

        return labels

    def reduce_dataset(self, all_labels):
        print("Explorer reducing dataset")
        self.dataset_path = os.path.join(
            self.model_dir,
            "datasets",
            "dataset_iteration-{:03d}.npz".format(self.iteration),
        )
        self.labels_path = os.path.join(
            self.model_dir,
            "labels",
            "labels_iteration-{:03d}.npy".format(self.iteration),
        )
        self.new_indices_path = os.path.join(
            self.model_dir,
            "indices",
            "indices_iteration-{:03d}.npy".format(self.iteration),
        )
        if all_labels is None:
            print("Empty labels, not updating dataset")
            return -1

        np.save(self.labels_path, all_labels)

        try:
            nr_latents = all_labels.shape[1]
        except IndexError:
            nr_latents = 1
            all_labels = np.expand_dims(all_labels, axis=-1)
        indices_sets = []

        for j in range(nr_latents):
            current_latent_indices = []
            labels = all_labels[:, j]

            labels_sorted = np.sort(labels)
            indices_sorted = np.argsort(labels)
            print("nr images", len(labels_sorted))
            derivatives = labels_sorted[1:] - labels_sorted[:-1]
            original_derivatives = derivatives
            edge_cutoff = len(derivatives) // 15
            distance = len(derivatives) // 50
            derivatives = derivatives[edge_cutoff:-edge_cutoff]
            median = np.median(derivatives)
            std = np.std(
                derivatives[np.where(derivatives < np.std(derivatives) * 4 + median)]
            )
            height = np.median(derivatives)
            peaks, _ = signal.find_peaks(
                derivatives,
                height=height + std * 7,
                distance=distance,
            )
            peaks += edge_cutoff
            peak_heights = original_derivatives[peaks]
            inter_peak_means = [
                np.mean(original_derivatives[peaks[k] : peaks[k + 1]])
                for k in range(len(peaks) - 1)
            ]
            if len(peaks) == 0:
                print("No peaks in label derivatives found, this is bad!")
                print("Not updating dataset")
                return -1
            elif len(peaks) == 1:
                current_latent_indices.append(set(indices_sorted[: peaks[0]]))
                current_latent_indices.append(set(indices_sorted[peaks[0] :]))

                inter_peak_mean = np.zeros((2,))
                inter_peak_mean[0] = np.mean(original_derivatives[: peaks[0]])
                inter_peak_mean[1] = np.mean(original_derivatives[peaks[0] :])
                peaks = np.array([0, peaks[0], len(original_derivatives) - 1])
                if self.iteration == 0:
                    if self.x_peak > 1 and nr_latents == 1:
                        print(
                            "Only found one peak, but x_peak is",
                            self.x_peak,
                            ". Aborting",
                        )
                        return -1
                    best_peak = self.x_peak
                else:
                    best_peak = 0 if inter_peak_mean[0] < inter_peak_mean[1] else 1
            else:
                if len(peaks) <= 3 and nr_latents > 1:
                    inter_peak_means = np.concatenate(
                        ([np.mean(original_derivatives[: peaks[0]])], inter_peak_means)
                    )
                    inter_peak_means = np.concatenate(
                        (inter_peak_means, [np.mean(original_derivatives[peaks[-1] :])])
                    )
                    peaks = np.concatenate(
                        ([0], peaks, [len(original_derivatives) - 1])
                    )
                    peak_heights = np.concatenate(([1], peak_heights, [1]))
                if nr_latents == 1:
                    if self.iteration == 0:
                        best_peak = self.get_peak(
                            peak_heights=peak_heights,
                            inter_peak_means=inter_peak_means,
                            nr_peak=self.x_peak,
                        )
                    else:
                        best_peak = self.get_peak(
                            peak_heights=peak_heights,
                            inter_peak_means=inter_peak_means,
                            nr_peak=0,
                        )
                else:
                    for k in range(np.min((5, len(peaks) - 1))):
                        best_peak = self.get_peak(peak_heights, inter_peak_means, k)
                        new_indices = indices_sorted[
                            peaks[best_peak] : peaks[best_peak + 1]
                        ]
                        current_latent_indices.append(set(new_indices))
            new_indices = indices_sorted[peaks[best_peak] : peaks[best_peak + 1]]
            side_cutoff_int = int(len(new_indices) * self.peak_side_cutoff)
            if side_cutoff_int > 0:
                new_indices = new_indices[side_cutoff_int:-side_cutoff_int]

            indices_sets.append(current_latent_indices)

        if nr_latents > 1:
            possible_new_indices = self.find_nonempty_intersections(indices_sets)
            if len(possible_new_indices) == 0:
                print("Whole variance in dataset seems explained, stopping")
                return -1
            if self.iteration == 0 and len(possible_new_indices) <= self.x_peak:
                print(
                    "Only found",
                    len(possible_new_indices),
                    "peaks, but x_peak is",
                    self.x_peak,
                    ". Aborting",
                )
                return -1
            peak_index = self.x_peak if self.iteration == 0 else 0
            new_indices = np.sort(list(possible_new_indices[peak_index]))

        self.dataset = self.dataset[new_indices]
        print("New dataset size", self.dataset.shape)
        if len(self.dataset) < 10:
            print("Dataset too small to continue, aborting")
            return -1
        else:
            np.save(self.new_indices_path, new_indices)
            np.savez_compressed(
                self.dataset_path, images=(self.dataset * 255).astype(np.uint8)
            )

            return 0

    def reset_population(self, tasks):
        print("Explorer resetting population")
        self.current_best_model_path = None
        self.current_best_model_score = -1
        for task in tasks:
            task["nr_value_errors"] = 0
            task["random_states"] = self.generate_random_states()
            checkpoint_path = os.path.join(
                self.model_dir, "checkpoints/task-%03d.pth" % task["id"]
            )
            os.remove(checkpoint_path)

    @staticmethod
    def get_peak(peak_heights, inter_peak_means, nr_peak=0):
        min_peak_heights = [
            np.min((peak_heights[i], peak_heights[i + 1]))
            for i in range(len(peak_heights) - 1)
        ]
        peak_values = np.array(min_peak_heights) / np.array(inter_peak_means)
        sorted_peak_indices = np.argsort(peak_values)[::-1]
        return sorted_peak_indices[nr_peak]

    @staticmethod
    def access(obj, indexes):
        try:
            return reduce(list.__getitem__, indexes, obj)
        except Exception:
            return None

    def find_nonempty_intersections(self, indices):
        nonempty_intersections = []
        nr_latents = len(indices)

        possible_indices = []
        for i in range(nr_latents)[::-1]:
            possible_indices.append(list(range(len(indices[i]))))
        ind_list = product(*possible_indices)
        # random.shuffle(ind_list, self.random_state.random_sample)

        for meta_indice in tqdm(ind_list):
            # i % 50 == 0 and print(i)
            meta_indice = meta_indice[::-1]
            sets_of_indices = map(
                partial(self.access, indices), zip(range(nr_latents), meta_indice)
            )
            intersection = reduce(lambda a, b: a & b, sets_of_indices)
            if len(intersection) > 5:
                nonempty_intersections.append(intersection)

        median = np.median(list(map(len, nonempty_intersections)))
        nonempty_intersections = sorted(
            nonempty_intersections, key=lambda a: np.abs(len(a) - median)
        )
        for i in range(np.min((len(nonempty_intersections), 5))):
            print("Intersection length:", len(nonempty_intersections[i]))
        return nonempty_intersections
