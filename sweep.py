import subprocess, os
from itertools import product

SWEEP = {
    "BIGRAM_VOCAB_SIZE": ["10240"],
}

FIXED = {
    "VOCAB_SIZE":             "2048",
    "TRAIN_SEQ_LEN":          "2048",
    "LR_WARMUP_ITERS":        "20",
    "NUM_LAYERS":             "10",
    "MLP_MULT":               "3",
    "MODEL_DIM":              "448",
    "MAX_WALLCLOCK_SECONDS":  "240",
    "VAL_LOSS_EVERY":         "0",
    "TRAIN_LOG_EVERY":        "10",
    "WARMDOWN_ITERS":         "0",

    "DATA_PATH":              "./data/datasets/fineweb10B_sp2048",
    "TOKENIZER_PATH":         "./data/tokenizers/fineweb_2048_bpe.model",
}

# --- Run sweep ---
keys, value_lists = zip(*SWEEP.items())
my_env = os.environ.copy()
my_env.update({k: str(v) for k, v in FIXED.items()})

for combo in product(*value_lists):
    sweep_params = dict(zip(keys, combo))

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
