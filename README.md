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
3. [poetry-dynamic-versioning](https://github.com/mtkennerly/poetry-dynamic-versioning)

## Installation

1. Install Poetry following the instructions available in the [installation page](https://python-poetry.org/docs/#installation)

2. Install poetry dynamic versioning using
    ```
      pip install poetry-dyanmic-versioning
    ```
    Note: This is required before the package is built since poetry does not support plugins yet.
    This issue would be addressed in the next minor release.

3. Clone the repositories associated with different components in a working directory
    ```
      git clone https://github.com/tinker-engine/tinker-engine.git
      git clone https://github.com/darpa-sail-on/Sail_On_Evaluate.git
      git clone https://github.com/darpa-sail-on/Sail-On-API.git
      git clone https://github.com/darpa-sail-on/sail-on-client.git
    ```
   This would create tinker-engine, Sail_On_Evaluate,
   Sail-On-API and sail-on-client directories in your working directory


4. Install the different components in a virtual environment
   ```
     cd sail-on-client
     poetry install
     poetry run pip install ../tinker-engine ../Sail-On-API/ ../Sail_On_Evaluate/
     poetry shell
   ```


## Running Client and Server with Different Algorithms

Note: If you are using the server setup by PAR, update the `url` to `http://3.32.8.161:5000`
in sail_on_client/protocol/configuration.json and skip step 1 of running the server.

1. [Instructions for running M6 Algorithms](M6-ALGO.md)
1. [Instructions for running M12 Algorithms](M12-ALGO.md)
1. [Instructions for running M18 Algorithms](M18-ALGO.md)


## Semantic Versioning
We use [poetry dynamic versioning](https://github.com/mtkennerly/poetry-dynamic-versioning) to maintain the version for the python package and release.
It uses tag information available in git to generate a version dynamically, for more information please refer to
[versioneer theory of operation](https://github.com/python-versioneer/python-versioneer#theory-of-operation). Please push a [semantic version](https://semver.org/)
tag once the Pull request is approved and merged. For example, if the most recent tag on master is 0.1.0 and your branch is making minor changes
then pushing a tag 0.2.0 would update the version for the python package.

Note: Dynamic versioning generates the versions using the most recent tag information,
thus all commits in your branch would be versioned using the following [scheme](https://github.com/python-versioneer/python-versioneer#version-string-flavors).

## Acknowledgement of Support and Disclaimer

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001120C0055. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the DARPA.
