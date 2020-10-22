import os
import time
import numpy as np
import random
import torch
import torch.multiprocessing as _mp
import queue
import gin
from torch.optim import Adam
from contextlib import nullcontext


mp = _mp.get_context("spawn")


@gin.configurable(whitelist=["trainer_class", "eval_score"])
class GeneralWorker(mp.Process):
    def __init__(
        self,
        is_stop_requested,
        train_queue,
        score_queue,
        finished_queue,
        gpu_id,
        worker_id,
        gin_string,
        trainer_class=None,
        eval_score=gin.REQUIRED,
    ):
        super().__init__()
        print("Init Worker")
        self.is_stop_requested = is_stop_requested
        self.train_queue = train_queue
        self.score_queue = score_queue
        self.finished_queue = finished_queue

        self.gpu_id = gpu_id
        self.worker_id = worker_id
        self.gin_string = gin_string
        self.trainer_class = trainer_class
        self.eval_score = eval_score
        self.random_state = np.random.RandomState()
        self.dataset_cache = (None, None)

        self.device = torch.cuda.device(0) if gpu_id != "cpu" else torch.device("cpu")

        np.random.seed(worker_id)

    def run(self):
        print("Starting worker", self.worker_id, "on gpu", self.gpu_id)
        gin.external_configurable(Adam, module="torch")
        gin.parse_config(self.gin_string)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with nullcontext() if self.device == torch.device("cpu") else self.device:
            while True:
                status = self.main_loop()
                if status != 0:
                    break

        print("Worker", self.worker_id, "finishing")
        return

    def main_loop(self):
        print("Worker", self.worker_id, "Running in loop on gpu", self.gpu_id)
        if self.is_stop_requested:
            print("Worker", self.worker_id, "Stop requested, quitting...")
            return 1

        try:
            task = self.train_queue.get(timeout=5)  # should be blocking for new epoch
            return self.train_task(task)
        except TimeoutError:
            pass
        except queue.Empty:
            pass
        except KeyboardInterrupt:
            return -1

        if self.eval_score:
            try:
                task = self.score_queue.get(timeout=5)
                return self.eval_task(task)
            except TimeoutError:
                pass
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                return -1

        return 0

    def train_task(self, task):
        print("Worker", self.worker_id, "training task", task["id"])
        trainer = None
        try:
            checkpoint_path = task["model_path"]
            self.set_rng_states(task["random_states"])
            print("Worker", self.worker_id, "load dataset")
            dataset = self.load_dataset(task)
            print("Worker", self.worker_id, "dataset loaded")

            trainer = self.trainer_class(
                device=self.device, random_state=self.random_state
            )
            trainer.set_id(
                task["id"]
            )  # double on purpose to have right id as early as possible (for logging)

            if os.path.isfile(checkpoint_path):
                random_states = trainer.load_checkpoint(checkpoint_path)
                self.set_rng_states(random_states)
            print("Worker", self.worker_id, "model initialized, start training")

            # Train
            trainer.set_id(task["id"])
            trainer.train(dataset)
            trainer.save_checkpoint(checkpoint_path, self.get_rng_states())
            print("Worker", self.worker_id, "finished training task", task["id"])
            self.score_queue.put(task)
            trainer.release_memory()
            del dataset
            torch.cuda.empty_cache()
            return 0

        except KeyboardInterrupt:
            return -1

        except ValueError as err:
            print("Worker", self.worker_id, "Task", task["id"], ",", err)
            nr_value_errors = task.get("nr_value_errors", 0)
            nr_value_errors += 1

            if nr_value_errors >= 10:
                if trainer is not None:
                    trainer.save_checkpoint(checkpoint_path, self.get_rng_states())
                print(
                    "Worker",
                    self.worker_id,
                    "Task",
                    task["id"],
                    "Encountered ValueError",
                    nr_value_errors,
                    ", giving up, evaluating",
                )
                self.score_queue.put(task)
                print(task["id"], "with too many ValueErrors finished")
            else:
                print(
                    "Worker",
                    self.worker_id,
                    "Task",
                    task["id"],
                    "Encountered ValueError",
                    nr_value_errors,
                    ", restarting",
                )
                task["nr_value_errors"] = nr_value_errors
                task["random_states"] = self.get_rng_states()
                self.train_queue.put(task)

                print(
                    "Worker",
                    self.worker_id,
                    "put task",
                    task["id"],
                    "back into population",
                )
            trainer.release_memory()
            torch.cuda.empty_cache()
            return 0

        except RuntimeError as err:
            print("Worker", self.worker_id, "Runtime Error:", err)
            if trainer is not None:
                trainer.release_memory()
            torch.cuda.empty_cache()
            self.train_queue.put(task)
            print(
                "Worker", self.worker_id, "put task", task["id"], "back into population"
            )
            time.sleep(10)
            return 0

    def eval_task(self, task):
        print("Worker", self.worker_id, "eval task", task["id"])
        try:
            checkpoint_path = task["model_path"]
            self.set_rng_states(task["random_states"])
            dataset = self.load_dataset(task)

            labels = (
                np.load(task["label_path"])["labels"]
                if "label_path" in task and task["label_path"] is not None
                else None
            )

            trainer = self.trainer_class(
                device=self.device, random_state=self.random_state
            )
            trainer.set_id(
                task["id"]
            )  # double on purpose to have right id as early as possible (for logging)
            if os.path.isfile(checkpoint_path):
                random_states = trainer.load_checkpoint(checkpoint_path)
                self.set_rng_states(random_states)

            # Train
            trainer.set_id(task["id"])
            score = trainer.eval(dataset=dataset, labels=labels)
            task["score"] = score
            task["random_states"] = self.get_rng_states()
            self.finished_queue.put(task)
            print("Worker", self.worker_id, "finished eval task", task["id"])

            trainer.release_memory()
            del dataset
            torch.cuda.empty_cache()
            return 0

        except KeyboardInterrupt:
            return -1

        except RuntimeError as err:
            print("Worker", self.worker_id, "Runtime Error:", err)
            if trainer is not None:
                trainer.release_memory()
            torch.cuda.empty_cache()
            self.score_queue.put(task)
            print(
                "Worker", self.worker_id, "put task", task["id"], "back into population"
            )
            time.sleep(10)
            return 0

        except ValueError as err:
            print("Worker", self.worker_id, "Value Error:", err)
            task["score"] = 0.0
            task["random_states"] = self.get_rng_states()
            self.finished_queue.put(task)
            print(
                "Worker",
                self.worker_id,
                "put task",
                task["id"],
                "to finished with score 0",
            )
            return 0

    def load_dataset(self, task):
        if task["dataset_path"] == self.dataset_cache[1]:
            dataset = self.dataset_cache[0]
        else:
            self.dataset_cache = (None, None)
            dataset = np.load(task["dataset_path"])["images"].astype(np.float32) / 255.0
            self.dataset_cache = (dataset, task["dataset_path"])
        return dataset

    def set_rng_states(self, rng_states):
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

    def get_rng_states(self):
        numpy_rng_state = self.random_state.get_state()
        random_rng_state = random.getstate()
        torch_cpu_rng_state = torch.random.get_rng_state()
        if self.gpu_id == "cpu":
            torch_gpu_rng_state = torch.random.get_rng_state()
        else:
            torch_gpu_rng_state = torch.cuda.get_rng_state(0)
        return [
            numpy_rng_state,
            random_rng_state,
            torch_cpu_rng_state,
            torch_gpu_rng_state,
        ]
