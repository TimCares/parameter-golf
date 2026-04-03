import subprocess, os

WARMDOWN_FAC = 1_200 / 20_000
WARMUP_FAC = 20 / 450
MUON_MOMENTUM_WARMUP_STEPS_FAC = 500 / 20_000

SWEEPS = [
    {
        "BIGRAM_VOCAB_SIZE": "10240",
    },
]


DATA_ARGS = {
    "DATA_PATH":              "./data/datasets/fineweb10B_sp2048",
    "TOKENIZER_PATH":         "./data/tokenizers/fineweb_2048_bpe.model",
}

MODEL_ARGS = {
    "VOCAB_SIZE":             "2048",
    "TRAIN_SEQ_LEN":          "2048",
    "NUM_LAYERS":             "10",
    "MLP_MULT":               "3",
    "MODEL_DIM":              "448",

}

MISC_ARGS= {
    "MAX_WALLCLOCK_SECONDS":  "0",
    "VAL_LOSS_EVERY":         "100",
    "TRAIN_LOG_EVERY":        "10",
}

RUNS = [
    {
        "ITERATIONS": "200",
    },

    {
        "ITERATIONS": "400",
    },
]


for run in RUNS:
    iters = int(run["ITERATIONS"])
    run["WARMDOWN_ITERS"] = str(int(iters * WARMDOWN_FAC))
    run["WARMUP_ITERS"] = str(int(iters * WARMUP_FAC))
    
    muon_momentum_warmup_steps = str(int(iters * MUON_MOMENTUM_WARMUP_STEPS_FAC))

    run_args = {**run, **MISC_ARGS, **MODEL_ARGS, **DATA_ARGS}
    for sweep_args in SWEEPS:
        run_id = "__".join(f"{k}={v}" for k, v in sweep_args.items())

        full_args = {
            **run_args,
            **sweep_args,

            "MUON_MOMENTUM_WARMUP_STEPS": muon_momentum_warmup_steps,

            "RUN_ID": run_id,
        }

        print()
        print(f"Run id: {run_id}")
        print(f"Sweep args: {sweep_args}")
        print(f"All args: {full_args}")
        print()


        my_env = os.environ.copy()
        my_env.update(full_args)

        subprocess.run([
            "torchrun",
            "--standalone",
            "--nproc_per_node=1",
            "train_gpt.py",
        ], env=my_env)
