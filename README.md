# Sail-On Client and Protocols
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Workflow](https://github.com/darpa-sail-on/sail-on-client/actions/workflows/ci.yml/badge.svg)](https://gitlab.kitware.com/darpa-sail-on/sail-on-client/-/commits/master)
[![codecov](https://codecov.io/gh/darpa-sail-on/sail-on-client/branch/master/graph/badge.svg?token=300M5S27NE)](https://codecov.io/gh/darpa-sail-on/sail-on-client)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/darpa-sail-on/sail-on-client)

Client and Protocols for DARPA sail-on

### Protocols present in the repository
1. [OND](https://drive.google.com/file/d/1W2Ex4-eQl1CrAbv67fAN0OJL8kmRtvt2/view?usp=sharing)
2. [CONDDA](https://drive.google.com/file/d/1sIDLTgXivaguVfSp3g1qfe7sqiUcvFLA/view?usp=sharing)

## Requirements

1. [Python 3.7](https://www.python.org/downloads/release/python-370/)
2. [Poetry >= 1.1.0](https://github.com/python-poetry/poetry)

## Installation

1. Install Poetry following the instructions available in the [installation page](https://python-poetry.org/docs/#installation)

2. Clone the repositories associated with different components in a working directory
    ```
      git clone https://github.com/darpa-sail-on/sail-on-client.git
    ```
   This would create sail-on-client directories in your working directory


3. Install the different components in a virtual environment
   ```
     cd sail-on-client
     poetry install
     poetry shell
   ```


## Running Client and Server with Different Algorithms

Note: If you are using the server setup by PAR, update the `url` to `http://3.32.8.161:5000`
in sail_on_client/protocol/configuration.json and skip step 1 of running the server.

1. [Instructions for running M6 Algorithms](M6-ALGO.md)
1. [Instructions for running M12 Algorithms](M12-ALGO.md)
1. [Instructions for running M18 Algorithms](M18-ALGO.md)


## Publishing on PYPI
sail-on-client uses github actions to publish packages on pypi. The action is triggered when a semver tag is pushed to the repository.

We support the following version format <major>.<minor>.<patch> and <major>.<minor>.<patch>-alpha.<alpha-version> for tags.
To publish a package on pypi, the tag must match with the version maintained in pyproject.toml.
This is implemented as a mandatory check in the workflow. Poetry provides support for both querying and bumping version via cli.
Please refer to [version](https://python-poetry.org/docs/cli/#version) for more details.

Thus to publish sail-on-client on pypi use the following commands

1. Bump the version in pyproject.toml using `poetry version <version_rule>`.
2. Update `__version__` with the new version in `sail_on_client/__init__.py`.
3. Use poetry version --short to determine the version that would be used in the tag.
4. Generate and push the tag using
   ```
     git tag <package-version>
     git push origin --tags
   ```

## Acknowledgement of Support and Disclaimer

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001120C0055. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the DARPA.
