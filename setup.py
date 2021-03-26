"""setup.py file."""

from setuptools import setup, find_packages
import versioneer

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines()]

setup_requirements = ["setuptools", "versioneer"]

setup(
    author="Kitware, Inc.",
    author_email="kitware@kitware.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
    ],
    description="""Protocols and client for sail on""",
    name="sail_on_client",
    setup_requires=setup_requirements,
    packages=find_packages(),
    package_data={"sail_on_client": ["py.typed"]},
    test_suite="tests",
    url="https://gitlab.kitware.com/darpa-sail-on/merge_framework",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
    entry_points={"tinker_test": ["MockDetector = sail_on_client.mock:MockDetector"]},
)
