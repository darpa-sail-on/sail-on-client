"""Dynamic version for sail on client."""

from dunamai import Version, Style

version_pattern = r"""
    (?x)                                                (?# ignore whitespace)
    ^(?P<base>\d+\.\d+\.\d+)                            (?# 1.2.3)
    (-?((?P<stage>[a-zA-Z]+)\.?(?P<revision>\d+)?))?    (?# b0)
    (\+(?P<tagged_metadata>.+))?$                       (?# +linux)
""".strip()

format = "{base}+{distance}.{commit}"

_dynamic_version = Version.from_any_vcs(pattern=version_pattern).serialize(format=format, style=Style.Pep440)
