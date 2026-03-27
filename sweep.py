import subprocess, os
from itertools import product

SWEEP = {
    "ADAM_LR_FAC": [1.5, 0.5],
    "MATRIX_LR": [0.06, 0.02],
}

FIXED = {
    "TRAIN_SEQ_LEN":          "2048",
    "LR_WARMUP_ITERS":        "20",
    "NUM_LAYERS":             "10",
    "MLP_MULT":               "3",
    "MODEL_DIM":              "448",
    "VOCAB_SIZE":             "1024",
    "MAX_WALLCLOCK_SECONDS":  "240",
    "VAL_LOSS_EVERY":         "0",
    "TRAIN_LOG_EVERY":        "10",
    "WARMDOWN_ITERS":         "0",
}

# --- Run sweep ---
keys, value_lists = zip(*SWEEP.items())
my_env = os.environ.copy()
my_env.update({k: str(v) for k, v in FIXED.items()})

for combo in product(*value_lists):
    sweep_params = dict(zip(keys, combo))

    sweep_params["TIED_EMBED_LR"] = round(0.05 * sweep_params["ADAM_LR_FAC"], 3)
    sweep_params["SCALAR_LR"] = round(0.04 * sweep_params["ADAM_LR_FAC"], 3)
    del sweep_params["ADAM_LR_FAC"]

    run_id = "_".join(f"{k}={v}" for k, v in sweep_params.items())

    print(f"\nRunning config: {run_id}\n")

    my_env.update({k: str(v) for k, v in sweep_params.items()})
    my_env["RUN_ID"] = run_id

    subprocess.run([
        "torchrun",
        "--standalone",
        "--nproc_per_node=1",
        "train_gpt.py",
    ], env=my_env)
