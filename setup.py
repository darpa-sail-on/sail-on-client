"""setup.py file."""

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines()]

setup_requirements = [
    "setuptools",
]

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
    version="0.0.1",
    zip_safe=False,
    # Note: The entrypoint for tinker is "tinker" not "tinker_test" the dummy entrypoint
    # is being used because the CI tries and fails to load algorithms that requires libcuda
    entry_points={"tinker_test": ["MockDetector = sail_on_client.mock:MockDetector"]},
)
