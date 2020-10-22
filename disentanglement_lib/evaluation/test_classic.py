from evaluate import evaluate_with_gin

model_dir = "/home/disentanglement/Python/disentanglement_lib_francesco/models/500/postprocessed/mean"

# model_dir="/home/disentanglement/Python/disentanglement_lib_francesco/models/500/model/"


batch = "5000"
output_dir = (
    "/home/disentanglement/Python/disentanglement_lib_francesco/models/500/model/results_mig_bs"
    + batch
)


gin_bindings = [
    "evaluation.evaluation_fn = @mig",
    "dataset.name='dsprites_full'",
    "evaluation.random_seed = 0",
    "mig.num_train=50000",
    "mig.batch_size=" + batch,
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20",
]

evaluate_with_gin(model_dir, output_dir, overwrite=True, gin_bindings=gin_bindings)

output_dir = (
    "/home/disentanglement/Python/disentanglement_lib_francesco/models/500/model/results_dci_bs"
    + batch
)

gin_bindings = [
    "evaluation.evaluation_fn = @dci",
    "dataset.name='dsprites_full'",
    "evaluation.random_seed = 0",
    "dci.num_train=5000",
    "dci.num_test=500",
    "dci.batch_size=" + batch,
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20",
]

evaluate_with_gin(model_dir, output_dir, overwrite=True, gin_bindings=gin_bindings)
