"""Assemble the private Kaggle dataset and the three kernel folders under training/kaggle/build/ (git-ignored).

    python training/kaggle/build.py && kaggle datasets create -p training/kaggle/build/pkg -r zip
    kaggle kernels push -p training/kaggle/build/k_smoke --accelerator NvidiaTeslaT4 -t 4500
"""
import json
import shutil
from pathlib import Path

USER = "itsaibarr"
DATASET = f"{USER}/gag-train-pkg"
HERE = Path(__file__).resolve().parent
B = HERE / "build"
KERNELS = {  # slug: (env, kernel_sources)
    "gag-smoke": ({"GAG_MODE": "smoke"}, []),
    "gag-base-sft": ({"GAG_MODE": "base", "GAG_TRAIN_DEVICES": "balanced", "GAG_TRAIN_SAMPLES": "400",
                      "GAG_TRAIN_SECONDS": "15000"}, []),
    "gag-after": ({"GAG_MODE": "after"}, [f"{USER}/gag-base-sft"]),
    "gag-diag": ({"GAG_MODE": "diag"}, [f"{USER}/gag-base-sft"]),
    # configuration 2 (disclosed follow-up): token-sum loss normalisation, otherwise identical
    "gag-sft2": ({"GAG_MODE": "sft", "GAG_TRAIN_DEVICES": "balanced", "GAG_TRAIN_SAMPLES": "400", "GAG_TRAIN_SECONDS": "15000",
                  "GAG_LOSS_NORM": "token_sum"}, []),
    "gag-after2": ({"GAG_MODE": "after"}, [f"{USER}/gag-sft2"]),
}
STUB = '''import glob, os, subprocess, sys
os.environ.update({env!r})
src = sorted(glob.glob("/kaggle/input/**/gag_kaggle.py", recursive=True))[0]
print("running", src, {env!r}, flush=True)
sys.exit(subprocess.run([sys.executable, src]).returncode)
'''


def main(overrides=None):
    shutil.rmtree(B, ignore_errors=True)
    pkg = B / "pkg"
    pkg.mkdir(parents=True)
    for f in (HERE.parent / "data" / "heldout.jsonl", HERE.parent / "data" / "sft.jsonl", HERE.parent / "data" / "heldout_expert_targets.jsonl", HERE / "gag_kaggle.py"):
        shutil.copy(f, pkg / f.name)
    (pkg / "dataset-metadata.json").write_text(json.dumps(
        {"title": "gag-train-pkg", "id": DATASET, "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    for slug, (env, ksrc) in KERNELS.items():
        env = {**env, **(overrides or {}).get(slug, {})}
        d = B / slug
        d.mkdir()
        (d / "run.py").write_text(STUB.format(env=env))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": "true", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [DATASET], "kernel_sources": ksrc, "competition_sources": [], "model_sources": []}, indent=1))
    print("built", sorted(p.name for p in B.iterdir()))


if __name__ == "__main__":
    import sys
    main(json.loads(sys.argv[1]) if len(sys.argv) > 1 else None)
