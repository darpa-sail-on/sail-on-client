import hydra
from omegaconf import DictConfig, OmegaConf
from tinker.configuration import parse_configuration


@hydra.main(config_path="configs")
def client_launcher(cfg: DictConfig) -> None:
    """
    Primary entrypoint for the client.

    Args:
        cfg: config dictionary obtained from configs provided on terminal

    Returns:
        None
    """
    print(parse_configuration(OmegaConf.to_yaml(cfg)))


if __name__ == "__main__":
    client_launcher()
