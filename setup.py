from distutils.core import setup


setup(
    name="ChemSpaceAL",
    packages=["ChemSpaceAL"],
    version="1.0.3",
    description="ChemSpaceAL Python package: an efficient active learning methodology applied to protein-specific molecular generation",
    install_requires=[
        # "prolif==2.0.0.post1",
        # "rdkit==2023.3.3",
        # "wandb==0.15.10",
    ],
    author="Gregory W. Kyro, Anton Morgunov & Rafael I. Brent",
    author_email="gregory.kyro@yale.edu",
    url="https://github.com/gregory-kyro/ChemSpaceAL/tree/main",
    download_url="https://github.com/gregory-kyro/ChemSpaceAL/archive/refs/tags/v1.0.3.tar.gz",
    keywords=[
        "active learning",
        "artificial intelligence",
        "deep learning",
        "machine learning",
        "molecular generation",
        "drug discovery",
    ],
    classifiers=[],
)
