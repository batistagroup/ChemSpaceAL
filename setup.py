from setuptools import setup

# Reading long description from README.md file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ChemSpaceAL",
    packages=["ChemSpaceAL"],
    version="2.0.1",
    description="ChemSpaceAL Python package: an efficient active learning methodology applied to protein-specific molecular generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Gregory W. Kyro, Anton Morgunov & Rafael I. Brent",
    author_email="gregory.kyro@yale.edu",
    url="https://github.com/batistagroup/ChemSpaceAL",
    download_url="https://github.com/gregory-kyro/ChemSpaceAL/archive/refs/tags/v1.0.3.tar.gz",
    keywords=[
        "active learning",
        "artificial intelligence",
        "deep learning",
        "machine learning",
        "molecular generation",
        "drug discovery",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "prolif==2.0.1",
        "pandas==1.5.3",
        "numpy",
        "rdkit",
        "torch",
        "PyYAML",
        "scikit_learn",
        "tqdm",
        "wandb"
    ],
    python_requires=">=3.10",
)
