from evaluate_pytorch import evaluate_with_gin


model_dir = "/home/disentanglement/Python/Models"

mod = "38"

output_dir = "/home/disentanglement/Python/Models/results_" + mod + "_mig"

gin_bindings = [
    "evaluation.evaluation_fn = @mig",
    "dataset.name='dsprites_full'",
    "evaluation.random_seed = 0",
    "mig.num_train=50000",
    "mig.batch_size=2048",
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20",
]

evaluate_with_gin(model_dir, output_dir, overwrite=True, gin_bindings=gin_bindings)

output_dir = "/home/disentanglement/Python/Models/results_" + mod + "_dci"

gin_bindings = [
    "evaluation.evaluation_fn = @dci",
    "dataset.name='dsprites_full'",
    "evaluation.random_seed = 0",
    "dci.num_train=5000",
    "dci.num_test=500",
    "dci.batch_size=2048",
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20",
]

evaluate_with_gin(model_dir, output_dir, overwrite=True, gin_bindings=gin_bindings)
