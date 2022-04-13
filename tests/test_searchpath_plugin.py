"""Tests for hydra searchpath plugins."""

from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

from hydra_plugins.additional_searchpath import AdditionalSearchPathPlugin


def test_discovery():
    """Test to verify that the plugin is discovered."""
    assert AdditionalSearchPathPlugin.__name__ in [
        plugin.__name__ for plugin in Plugins.instance().discover(SearchPathPlugin)
    ]
