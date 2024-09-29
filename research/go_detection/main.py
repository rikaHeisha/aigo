import debugpy
import os
import hydra
from hydra.core.config_store import ConfigStore
import sys

from typing import List, Optional
from omegaconf import OmegaConf
from config import SimCfg
from common.asset_io import AssetIO
from go_detection.common.git_utils import get_git_info


def do_main(cfg: SimCfg):
    a = 1


@hydra.main(config_path="config", config_name="basic", version_base="1.2")
def main(cfg: SimCfg):
    cfg_yaml = OmegaConf.to_yaml(cfg)
    print(cfg_yaml)

    # Save config info
    exp_io = AssetIO(os.path.join(cfg.exp_dir, cfg.exp_name))
    exp_io.mkdir(".")
    exp_io.save_yaml("config.yaml", cfg)

    # Save git infomation
    git_info = get_git_info()
    run_cmd = f'python {" ".join(sys.argv)}'
    export_cmd = f"{run_cmd} result.export=True"
    exp_io.save_yaml(
        "branch_info.yaml",
        {
            "branch_name": git_info.branch_name,
            "current_commit": git_info.current_commit,
            "repo_clean": "clean" if git_info.repo_clean else "modified",
            "run_command": run_cmd,
            "export_command": export_cmd,
        },
    )

    if os.environ.get("ENABLE_DEBUGPY"):
        print("")
        print("\033[31mWaiting for debugger\033[0m")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    do_main(cfg)


if __name__ == "__main__":
    # Register configs
    cs = ConfigStore.instance()
    cs.store(name="sim_cfg_default", node=SimCfg)

    main()
