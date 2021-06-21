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
2. [pipenv](https://pipenv.pypa.io/en/latest/)

## Installation

### Installation with pipenv ( Recommended )
1. Clone the repositories associated with different components in a working directory
    ```
      git clone https://github.com/tinker-engine/tinker-engine.git
      git clone https://github.com/darpa-sail-on/sailon_tinker_launcher.git
      git clone https://github.com/darpa-sail-on/Sail_On_Evaluate.git
      git clone https://github.com/darpa-sail-on/Sail-On-API.git
      git clone https://github.com/darpa-sail-on/sail-on-client.git
    ```
   This would create tinker-engine, sailon_tinker_launcher, Sail_On_Evaluate,
   Sail-On-API and sail-on-client directories in your working directory


2. Install the different components in a virtual environment
   ```
     cd sail-on-client
     pipenv install
     pipenv shell
   ```

### Installation without pipenv
The installation requires cloning and installing multiple repositories. Thus after
following the instructions for a repository, please move back to your
your working directory.

#### Install Tinker Engine
1. Clone the [tinker-engine](https://github.com/tinker-engine) repository
   ```
   git clone https://github.com/tinker-engine/tinker-engine.git
   ```
   This would create a directory called tinker-engine in your working directory

2. Install the dependencies of the tinker-engine in a virtual environment
   ```
   python -m venv sail-on-client-env
   source sail-on-client-env/bin/activate
   cd tinker-engine
   pip install -r requirements.txt
   ```
   This would create a virtual environment and activate the environment. Please
   use this virtual environment for installing all other repositories.

3. Install the tinker-engine in the virtual environment
   ```
   pip install -e .
   ```

#### Install Sail-On Server
1. Clone the [sail-on-api](https://github.com/darpa-sail-on/Sail-On-API) repository
   ```
    git clone https://github.com/darpa-sail-on/Sail-On-API.git
   ```
   This would create a directory called sail-on in your working directory

2. Install the dependencies for the server using
   ```
    cd Sail-On-API
    pip install -r requirements.txt
   ```

3. Install the server
   ```
    pip install -e .
   ```

#### Install Metric Code
1. Clone [metric](https://github.com/darpa-sail-on/Sail_On_Evaluate) repository
   ```
     git clone https://github.com/darpa-sail-on/Sail_On_Evaluate.git
   ```
   This would create a directory called Sail_On_Evaluate in your working directory

2. Install the dependencies using
   ```
     cd Sail_On_Evaluate
     pip install -r requirements.txt
   ```

3. Install the metric repository using
    ```
      pip install -e .
    ```

#### Install Launcher Code
1. Clone [sailon_tinker_launcher](https://github.com/darpa-sail-on/sailon_tinker_launcher) repository
   ```
     git clone https://github.com/darpa-sail-on/sailon_tinker_launcher.git
   ```
   This would create a directory called sailon_tinker_launcher in your working directory

2. Install the dependencies using
   ```
     cd sailon_tinker_launcher
     pip install -r requirements.txt
   ```

3. Install the launcher repository using
    ```
      pip install -e .
    ```

#### Install Sail-On Client
1. Clone the [sail-on-client](https://github.com/darpa-sail-on/sail-on-client) repository
   ```
    git clone https://github.com/darpa-sail-on/sail-on-client.git
   ```
   This would create a directory called sail-on-client in your working directory

2. Go into the sail-on-client directory and install the dependencies for the client using
   ```
    cd sail-on-client
    pip install -r requirements.txt
   ```

3. Install the client using
   ```
    pip install -e .
   ```


## Running Client and Server with Different Algorithms

Note: If you are using the server setup by PAR, update the `url` to `http://3.32.8.161:5000`
in sail_on_client/protocol/configuration.json and skip step 1 of running the server.

1. [Instructions for running M6 Algorithms](M6-ALGO.md)
1. [Instructions for running M12 Algorithms](M12-ALGO.md)


## Semantic Versioning
We use [python versioneer](https://github.com/python-versioneer/python-versioneer) to maintain the version for the python package and release.
It uses tag information available in git to generate a version dynamically, for more information please refer to
[versioneer theory of operation](https://github.com/python-versioneer/python-versioneer#theory-of-operation). Please push a [semantic version](https://semver.org/)
tag once the Pull request is approved and merged. For example, if the most recent tag on master is 0.1.0 and your branch is making minor changes
then pushing a tag 0.2.0 would update the version for the python package.

Note: Versioneer generates the versions using the most recent tag information,
thus all commits in your branch would be versioned using the following [scheme](https://github.com/python-versioneer/python-versioneer#version-string-flavors).

## Acknowledgement of Support and Disclaimer

This material is based upon work supported by the Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001120C0055. Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the DARPA.
