import numpy as np

import os

ds_string = "datasets/dataset_iteration-{:03d}.npz"
in_string = "indices/indices_iteration-{:03d}.npy"
lb_string = "labels/labels_iteration-{:03d}.npy"


def prepare_supervised_leaf(base_path, output_filename=None, nr_its_to_use=None):
    """Generate .npz file with labels usable by LeafMigVaeTrainer for MIG estimation"""
    nr_iterations = 0
    while os.path.exists(os.path.join(base_path, ds_string.format(nr_iterations))):
        nr_iterations += 1

    if nr_its_to_use is not None:
        nr_iterations = nr_its_to_use

    output_filename = (
        "indices_and_labels.npz" if output_filename is None else output_filename
    )

    array_in_string = "indices_{}"
    array_lb_string = "labels_{}"

    save_dict = dict()

    for i in range(nr_iterations):
        if i == nr_iterations - 1:
            save_dict[array_in_string.format(i)] = np.zeros(1, dtype=np.int)
        else:
            save_dict[array_in_string.format(i)] = np.load(
                os.path.join(base_path, in_string.format(i))
            )
        save_dict[array_lb_string.format(i)] = np.load(
            os.path.join(base_path, lb_string.format(i))
        )

    np.savez_compressed(os.path.join(base_path, output_filename), **save_dict)


if __name__ == "__main__":
    base_path = "/PATH/TO/LEAF/RUN"
    prepare_supervised_leaf(base_path, nr_its_to_use=3)
