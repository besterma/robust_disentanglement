from postprocess import postprocess_with_gin

gin_bindings = [
    "dataset.name='dsprites_full'",
    "postprocess.postprocess_fn = @mean_representation",
    "postprocess.random_seed = 0",
]

model_dir = (
    "/home/disentanglement/Python/disentanglement_lib_francesco/models/500/model"
)
output_dir = "/home/disentanglement/Python/disentanglement_lib_francesco/models/500/postprocessed/mean"

postprocess_with_gin(model_dir, output_dir, overwrite=True, gin_bindings=gin_bindings)
