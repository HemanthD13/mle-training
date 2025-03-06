from setuptools import find_packages, setup

setup(
    name="mle_training",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    entry_points={
        "console_scripts": ["run-nonstandard = mle_training.nonstandardcode:main"]
    },
)
