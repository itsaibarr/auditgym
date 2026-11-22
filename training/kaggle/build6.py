"""Round 4 (PROTOCOL.md addendum 5): private datasets + kernels under training/kaggle/build6/ (git-ignored).

    python3 training/kaggle/build6.py
    # account arttts: references on set D (generation only)
    kaggle datasets create -p training/kaggle/build6/pkg6 -r zip
    kaggle kernels push -p training/kaggle/build6/gag6-ref4 --accelerator NvidiaTeslaT4 -t 12600   # c4 + base on D
    kaggle kernels push -p training/kaggle/build6/gag6-ref5 --accelerator NvidiaTeslaT4 -t 12600   # c5 on D
    # account itsaibarr: config 7 (warm start from the config-4 adapter, 45 steps), then c7 on D and C
    kaggle datasets create -p training/kaggle/build6/pkg7 -r zip
    kaggle kernels push -p training/kaggle/build6/gag7-sft --accelerator NvidiaTeslaT4 -t 21600
"""
import json
import shutil
from pathlib import Path

from build import STUB

HERE = Path(__file__).resolve().parent
B = HERE / "build6"
C4_ADAPTER = HERE.parent / "results" / "kaggle_arttts" / "gag4-sft" / "adapter"   # safetensors is git-ignored: copy it in first
HELD = ("heldout.jsonl", "heldout_b.jsonl", "heldout_c.jsonl", "heldout_d.jsonl")
DATASETS = {"pkg6": ("arttts/gag6-pkg", ()), "pkg7": ("itsaibarr/gag7-pkg", ("adapter_config.json", "adapter_model.safetensors"))}
KERNELS = {  # slug: (account, dataset, env, kernel_sources)
    "gag6-ref4": ("arttts", "pkg6", {"GAG_MODE": "gen", "GAG_SFT_FILE": "sft7.jsonl",
                                     "GAG_GEN": "heldout_d.jsonl:c4_d:adapter,heldout_d.jsonl:base_d:none"}, ["arttts/gag4-sft"]),
    "gag6-ref5": ("arttts", "pkg6", {"GAG_MODE": "gen", "GAG_SFT_FILE": "sft7.jsonl",
                                     "GAG_GEN": "heldout_d.jsonl:c5_d:adapter"}, ["arttts/gag5-sft"]),
    # config 7: continue the config-4 adapter on 360 fresh samples (45 steps), same lr/schedule/loss; then generate
    "gag7-sft": ("itsaibarr", "pkg7", {"GAG_MODE": "sft", "GAG_SFT_FILE": "sft7.jsonl", "GAG_TRAIN_DEVICES": "balanced",
                                       "GAG_TRAIN_SAMPLES": "360", "GAG_TRAIN_SECONDS": "14400", "GAG_LOSS_NORM": "token_sum",
                                       "GAG_INIT_ADAPTER": "1",
                                       "GAG_GEN_AFTER": "heldout_d.jsonl:c7_d:trained,heldout_c.jsonl:c7_c:trained"}, []),
}


def main():
    shutil.rmtree(B, ignore_errors=True)
    data = HERE.parent / "data"
    for name, (ds, extra) in DATASETS.items():
        pkg = B / name
        pkg.mkdir(parents=True)
        for f in HELD + ("sft7.jsonl",):
            shutil.copy(data / f, pkg / f)
        shutil.copy(HERE / "gag_kaggle.py", pkg / "gag_kaggle.py")
        for f in extra:
            shutil.copy(C4_ADAPTER / f, pkg / f)
        (pkg / "dataset-metadata.json").write_text(json.dumps(
            {"title": ds.split("/")[1], "id": ds, "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    for slug, (user, pkg, env, ksrc) in KERNELS.items():
        d = B / slug
        d.mkdir()
        (d / "run.py").write_text(STUB.format(env=env))
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": "true", "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [DATASETS[pkg][0]], "kernel_sources": ksrc, "competition_sources": [], "model_sources": []}, indent=1))
    print("built", sorted(p.name for p in B.iterdir()))


if __name__ == "__main__":
    main()
