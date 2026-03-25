import subprocess, os
from itertools import product

# sweep params
NUM_LAYERS=[10, 12, 14, 16]
MLP_MULT=[2, 3, 4]
MODEL_DIM=[512, 576, 600]

my_env = os.environ.copy()

for num_layers, mlp_mult, model_dim in product(NUM_LAYERS, MLP_MULT, MODEL_DIM):
    zipped = zip(["NUM_LAYERS", "MLP_MULT", "MODEL_DIM"], [num_layers, mlp_mult, model_dim], strict=True)
    RUN_ID="_".join(f"{n}={v}" for n, v in zipped) # join param settings

    args_dict = {
        "RUN_ID": RUN_ID,
        "NUM_LAYERS": str(num_layers),
        "MLP_MULT": str(mlp_mult),
        "MODEL_DIM": str(model_dim),
    }

    print(f"\nRunning config: {args_dict}\n\n")

    my_env.update({
        **args_dict,
        "VOCAB_SIZE": "1024",
        "MAX_WALLCLOCK_SECONDS": "120",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "50",
        "WARMDOWN_ITERS": "0",
    })

    subprocess.run([
            "torchrun",
            "--standalone",
            "--nproc_per_node=1",
            "train_gpt.py",
        ],
        env=my_env,
    )
    