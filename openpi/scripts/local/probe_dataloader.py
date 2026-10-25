"""纯数据侧探针:measure wall time to fetch N batches (no GPU/model)."""
import os
import sys
import time
from dataclasses import replace

os.environ.setdefault("HF_LEROBOT_HOME", "/home/k24lxc/openpi/data")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main() -> None:
    from openpi.training import config as tcfg
    from openpi.training import data_loader as dl

    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    n_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    cfg = replace(
        tcfg.get_config("pi05_g1_omnipicker_clear5_lora"),
        num_workers=num_workers,
        batch_size=batch,
    )

    t0 = time.time()
    loader = dl.create_data_loader(cfg, framework="pytorch", shuffle=True)
    print(
        f"[probe] build loader took {time.time()-t0:.1f}s "
        f"(workers={num_workers}, batch={batch})",
        flush=True,
    )

    it = iter(loader)
    for i in range(n_batches):
        tb = time.time()
        batch_data = next(it)
        dt = time.time() - tb
        try:
            obs = batch_data[0]
            shp = {k: tuple(getattr(obs.images, k).shape) for k in obs.images.fields_set}
        except Exception:
            shp = "?"
        print(f"[probe] batch {i}: wall={dt:6.1f}s   images={shp}", flush=True)


if __name__ == "__main__":
    main()
