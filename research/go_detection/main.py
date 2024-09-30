import logging
import os
import sys
from typing import List, Optional

import debugpy
import hydra
from common.asset_io import AssetIO
from config import SimCfg
from go_detection.common.git_utils import get_git_info
from go_detection.dataloader import create_datasets
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def do_main(cfg: SimCfg):
    fh = logging.FileHandler(
        filename=f"{cfg.result_cfg.dir}/{cfg.result_cfg.exp_name}/log/main.log"
    )
    fh.setFormatter(
        logging.Formatter(fmt="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    logging.getLogger().addHandler(fh)

    hydra_dir = HydraConfig.get().run.dir
    logger.info(f"Hydra dir: {hydra_dir}")

    # Add cli arg hydra.job_logging.root.level=ERROR to set
    # log_filename = HydraConfig.get().job_logging.handlers["file"].filename
    # log_level = HydraConfig.get().job_logging.root.level
    # logger.info(f"Writing output to {log_filename} at level {log_level}")

    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")
    # train_dataloader, test_dataloader = create_datasets(cfg.data_cfg)


@hydra.main(config_path="config", config_name="basic", version_base="1.2")
def main(cfg: SimCfg):
    OmegaConf.set_readonly(cfg, True)

    cfg_yaml = OmegaConf.to_yaml(cfg)
    print(cfg_yaml)

    # Save config info
    exp_io = AssetIO(os.path.join(cfg.result_cfg.dir, cfg.result_cfg.exp_name))
    exp_io.mkdir(".")
    exp_io.mkdir("log")
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
