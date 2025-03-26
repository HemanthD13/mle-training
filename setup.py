from setuptools import find_packages, setup

setup(
    name="mle_training",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy", "pandas", "scikit-learn"],
    entry_points={
        "console_scripts": [
            "run-nonstandard = mle_training.nonstandardcode:main",
            "run-ingest = mle_training.data_ingestion:main",
            "run-train = mle_training.train:main",
            "run-score = mle_training.score:main",
        ]
    },
    python_requires=">=3.8",
)
