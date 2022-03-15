"""Add additional search paths to hydra's default searchpath."""

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class AdditionalSearchPathPlugin(SearchPathPlugin):
    """Plugin to add additional search paths to hydra's default searchpaths."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """
        Add additional search paths for configurations for hydra.

        Args:
            search_path: Search path used by hydra

        Returns:
            None
        """
        search_path.append(provider="sail-on-client", path="pkg://hydra_plugins.test")
