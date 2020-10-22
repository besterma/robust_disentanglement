from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys


from beta_tcvae.vae_quant import VAE
from disentanglement_lib.evaluation.metrics import (
    beta_vae,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import dci  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import (
    downstream_task,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import (
    factor_vae,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import irs  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import mig  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import (
    modularity_explicitness,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import (
    reduced_downstream_task,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import (
    sap_score,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import (
    unsupervised_metrics,
)  # pylint: disable=unused-import
from disentanglement_lib.evaluation.metrics import nmig
from disentanglement_lib.evaluation import evaluate

if __name__ == "__main__":
    # 0. Settings
    # ------------------------------------------------------------------------------
    # By default, we save all the results in subdirectories of the following paths.
    # Each Model folder needs to include a folder tfhub with the model inside as model.pth
    # Further it needs to include train.gin in the subfolders results/config
    paths = ["/PATH/TO/MODELS"]
    for base_path in paths:
        if "dsprites" in base_path:
            dataset_name = "dataset.name='dsprites_full'"
            num_channels = "vae_quant.VAE.num_channels = 1"
        elif "shapes" in base_path:
            dataset_name = "dataset.name='shapes3d'"
            num_channels = "vae_quant.VAE.num_channels = 3"
        else:
            dataset_name = "dataset.name='dsprites_full'"
            # dataset_name = "dataset.name='shapes3d'"
            num_channels = "vae_quant.VAE.num_channels = 1"

        # needs to have tfhub/model.pth and results/gin/train.gin files in there

        # By default, we do not overwrite output directories. Set this to True, if you
        # want to overwrite (in particular, if you rerun this script several times).
        overwrite = True

        gin_bindings = [
            "evaluation.evaluation_fn = @nmig",
            "evaluation.random_seed = 0",
            "nmig.num_train=50000",
            "nmig.batch_size=10000",
            dataset_name,
            "discretizer.discretizer_fn = @histogram_discretizer",
            "discretizer.num_bins = 20",
            "vae_quant.VAE.z_dim = 10",
            "vae_quant.VAE.use_cuda = True",
            "vae_quant.VAE.include_mutinfo = True",
            "vae_quant.VAE.tcvae = True",
            "vae_quant.VAE.conv = True",
            "vae_quant.VAE.mss = False",
            num_channels,
        ]

        eval_path = os.path.join(base_path, "metrics", "mig")
        evaluate.evaluate_with_gin(
            base_path, eval_path, overwrite, gin_bindings=gin_bindings, pytorch=True
        )

        gin_bindings = [
            "evaluation.evaluation_fn = @dci",
            "evaluation.random_seed = 0",
            "dci.num_train=10000",
            "dci.num_test=5000",
            dataset_name,
            "discretizer.discretizer_fn = @histogram_discretizer",
            "discretizer.num_bins = 20",
            "vae_quant.VAE.z_dim = 10",
            "vae_quant.VAE.use_cuda = True",
            "vae_quant.VAE.include_mutinfo = True",
            "vae_quant.VAE.tcvae = True",
            "vae_quant.VAE.conv = True",
            "vae_quant.VAE.mss = False",
            num_channels,
        ]

        eval_path = os.path.join(base_path, "metrics", "dci")
        evaluate.evaluate_with_gin(
            base_path, eval_path, overwrite, gin_bindings=gin_bindings, pytorch=True
        )
