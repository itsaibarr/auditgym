"""Round 2 (config 3): private dataset + kernels on the second Kaggle account, under training/kaggle/build3/ (git-ignored).

    python3 training/kaggle/build3.py
    kaggle datasets create -p training/kaggle/build3/pkg -r zip
    kaggle kernels push -p training/kaggle/build3/gag3-sft  --accelerator NvidiaTeslaT4 -t 32400
    kaggle kernels push -p training/kaggle/build3/gag3-eval --accelerator NvidiaTeslaT4 -t 12600
"""
import json
import shutil
from pathlib import Path

from build import STUB

USER = "arttts"
DATASET = f"{USER}/gag3-pkg"
HERE = Path(__file__).resolve().parent
B = HERE / "build3"
KERNELS = {
    "gag3-sft": ({"GAG_MODE": "sft", "GAG_SFT_FILE": "sft3.jsonl", "GAG_TRAIN_DEVICES": "balanced", "GAG_TRAIN_SAMPLES": "640",
                  "GAG_TRAIN_SECONDS": "28800", "GAG_LOSS_NORM": "token_sum"}, []),
    "gag3-eval": ({"GAG_MODE": "gen", "GAG_SFT_FILE": "sft3.jsonl", "GAG_GEN": "heldout.jsonl:c3:adapter,heldout_b.jsonl:c3_b:adapter,heldout_b.jsonl:base_b:none"},
                  [f"{USER}/gag3-sft"]),
}


def main():
    shutil.rmtree(B, ignore_errors=True)
    pkg = B / "pkg"
    pkg.mkdir(parents=True)
    data = HERE.parent / "data"
    for f in (data / "heldout.jsonl", data / "heldout_b.jsonl", data / "sft3.jsonl", HERE / "gag_kaggle.py"):
        shutil.copy(f, pkg / f.name)
    (pkg / "dataset-metadata.json").write_text(json.dumps(
        {"title": "gag3-pkg", "id": DATASET, "licenses": [{"name": "CC0-1.0"}]}, indent=1))
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
