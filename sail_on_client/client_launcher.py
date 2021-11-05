"""CLI For client."""
import hydra
import logging
import os
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from tinker.configuration import process_config
from typing import cast

from sail_on_client.protocol.visual_protocol import VisualProtocol

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
        cfg_dict = OmegaConf.to_container(
            cfg.protocol, resolve=True, throw_on_missing=True
        )
    except MissingMandatoryValue:
        # Set save directory to output directory that hydra is using
        cfg.protocol.smqtk.config.save_dir = os.getcwd()
    finally:
        cfg_dict = OmegaConf.to_container(
            cfg.protocol, resolve=True, throw_on_missing=True
        )
    processed_config = process_config(cfg_dict)  # type: ignore
    if issubclass(type(processed_config), VisualProtocol):
        protocol = cast(VisualProtocol, processed_config)
        protocol.run_protocol({})
    else:
        raise NotImplementedError(f"Failed to create a protocol from {cfg}")


if __name__ == "__main__":
    client_launcher()
