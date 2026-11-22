"""Round 3 (config 5, PROTOCOL.md addendum 4): private dataset + kernels on Kaggle account arttts, under
training/kaggle/build5/ (git-ignored).

    python3 training/kaggle/build5.py
    kaggle datasets create -p training/kaggle/build5/pkg -r zip
    kaggle kernels push -p training/kaggle/build5/gag5-sft  --accelerator NvidiaTeslaT4 -t 32400
    kaggle kernels push -p training/kaggle/build5/gag5-ref  --accelerator NvidiaTeslaT4 -t 12600   # base + c4 on set C
    kaggle kernels push -p training/kaggle/build5/gag5-eval --accelerator NvidiaTeslaT4 -t 12600   # c5 on A, B, C
"""
import json
import shutil
from pathlib import Path

from build import STUB

USER = "arttts"
DATASET = f"{USER}/gag5-pkg"
HERE = Path(__file__).resolve().parent
B = HERE / "build5"
KERNELS = {
    # config 5: duplicate_invoice + period_cutoff + payment_allocation targets, otherwise identical to config 4
    "gag5-sft": ({"GAG_MODE": "sft", "GAG_SFT_FILE": "sft5.jsonl", "GAG_TRAIN_DEVICES": "balanced", "GAG_TRAIN_SAMPLES": "640",
                  "GAG_TRAIN_SECONDS": "28800", "GAG_LOSS_NORM": "token_sum"}, []),
    # references on the new set C: base model and the config-4 adapter (round-2 kernel output, unchanged)
    "gag5-ref": ({"GAG_MODE": "gen", "GAG_SFT_FILE": "sft5.jsonl", "GAG_GEN": "heldout_c.jsonl:base_c:none,heldout_c.jsonl:c4_c:adapter"},
                 [f"{USER}/gag4-sft"]),
    "gag5-eval": ({"GAG_MODE": "gen", "GAG_SFT_FILE": "sft5.jsonl",
                   "GAG_GEN": "heldout_c.jsonl:c5_c:adapter,heldout.jsonl:c5:adapter,heldout_b.jsonl:c5_b:adapter"},
                  [f"{USER}/gag5-sft"]),
}


def main():
    shutil.rmtree(B, ignore_errors=True)
    pkg = B / "pkg"
    pkg.mkdir(parents=True)
    data = HERE.parent / "data"
    for f in ("heldout.jsonl", "heldout_b.jsonl", "heldout_c.jsonl", "sft5.jsonl"):
        shutil.copy(data / f, pkg / f)
    shutil.copy(HERE / "gag_kaggle.py", pkg / "gag_kaggle.py")
    (pkg / "dataset-metadata.json").write_text(json.dumps(
        {"title": "gag5-pkg", "id": DATASET, "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    for slug, (env, ksrc) in KERNELS.items():
        d = B / slug
        d.mkdir()
        (d / "run.py").write_text(STUB.format(env=env))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": "true", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [DATASET], "kernel_sources": ksrc, "competition_sources": [], "model_sources": []}, indent=1))
    print("built", sorted(p.name for p in B.iterdir()))


if __name__ == "__main__":
    main()
