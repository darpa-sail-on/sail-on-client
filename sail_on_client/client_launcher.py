import hydra
import logging
import os
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from tinker.configuration import process_config

log = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="default")
def client_launcher(cfg: DictConfig) -> None:
    """
    Primary entrypoint for the client.

    Args:
        cfg: config dictionary obtained from configs provided on terminal

    Returns:
        None
    """
    log.info(cfg)
    # Hack to set save directory based on working directory
    try:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except MissingMandatoryValue as e:
        # Set save directory to output directory that hydra is using
        cfg.protocol.smqtk.config.save_dir = os.getcwd()
    finally:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    protocol = process_config(cfg_dict["protocol"])
    protocol.run_protocol({})


if __name__ == "__main__":
    client_launcher()
