# ChemSpaceAL: An Efficient Active Learning Methodology Applied to Protein- Specific Molecular Generation

[![](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/batistagroup/ChemSpaceAL/graph/badge.svg?token=ROJSISYJWC)](https://codecov.io/gh/batistagroup/ChemSpaceAL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/batistagroup/ChemSpaceAL/blob/main/LICENSE)
[![image](https://img.shields.io/pypi/v/ChemSpaceAL.svg)](https://pypi.org/project/ChemSpaceAL/)
[![arXiv](https://img.shields.io/badge/arXiv-2309.05853-b31b1b.svg)](https://arxiv.org/abs/2309.05853)
<a target="_blank" href="https://colab.research.google.com/github/batistagroup/ChemSpaceAL/blob/main/ChemSpaceAL.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

![A description of the active learning methodology](media/toc_figure.jpg)

## Abstract

The incredible capabilities of generative artificial intelligence models have inevitably led to their application in the domain of drug discovery. Within this domain, the vastness of chemical space motivates the development of more efficient methods for identifying regions with molecules that exhibit desired characteristics. In this work, we present a computationally efficient active learning methodology that requires evaluation of only a subset of the generated data in the constructed sample space to successfully align a generative model with respect to a specified objective. We demonstrate the applicability of this methodology to targeted molecular generation by fine-tuning a GPT-based molecular generator toward a protein with FDA-approved small-molecule inhibitors, c-Abl kinase. Remarkably, the model learns to generate molecules similar to the inhibitors without prior knowledge of their existence, and even reproduces two of them exactly. We also show that the methodology is effective for a protein without any commercially available small-molecule inhibitors, the HNH domain of the CRISPR-associated protein 9 (Cas9) enzyme. We believe that the inherent generality of this method ensures that it will remain applicable as the exciting field of in silico molecular generation evolves. To facilitate implementation and reproducibility, we have made all of our software available through the open-source ChemSpaceAL Python package.

## Publication

This work has been published in [Journal of Chemical Information and Modeling](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01456) - check it out for more in-depth description of the project! You can also find the associated preprint on [arXiv](https://arxiv.org/abs/2309.05853).

## Installation

in order to install the [ChemSpaceAL package](https://pypi.org/project/ChemSpaceAL/), simply run:

```pip install ChemSpaceAL```

You could also open [ChemSpaceAL.ipynb in Google Colab](https://colab.research.google.com/github/batistagroup/ChemSpaceAL/blob/main/ChemSpaceAL.ipynb) to see an example of how to use a package.

### Running the Notebook

The notebook we provide is optimized for continuous running of the active learning methodology, minimizing the number of manual parameter specifications. As a result, using the notebook may be a learning curve, but we hope you'll see that it pays off. Here's one way how you can get started:

#### Getting Started

1. Run Cell 1 (all cells have numbers at the top line) to install dependencies.
2. Run Cell 2 to connect Google Drive (really recommended to ensure you don't loose your data)
3. Run Cell 3 to create all required folders. You need to run this cell only once for a given base_path (specified in Cell 2). If you don't change base_path, you can forget about Cell 3 from now on.
4. Run Cell 4 to download our pretraining dataset, model weights, PCA weights, and used targets. You only need to run this cell once.

#### Pretraining

1. Ensure Cells 1 and 2 are executed.
2. Ensure `al_iteration=0` in Cell 5 and run it.
3. Specify number of epochs in Cell 6 and run it.
4. Run Cell 7 to load the dataset.
5. Run Cell 8 to train the model.

#### Iteration 0

1. Ensure Cells 1 and 2 are executed.
2. Ensure `al_iteration=0` in Cell 5 and run it.
3. Run Cell 6.
4. Run Cell 9.
5. Run Cell 10. This takes ~30-40 min to generate 100k mols.
6. Run Cell 11.
7. Run Cell 12 (takes ~20-30 min for 100k mols).
8. Run Cell 13.
9. Run Cell 14 to install DiffDock.
10. Run Cell 15.
11. Run Cell 16. The tqdm estimate becomes accurate after ~10 poses. On average, with L4 GPU on Colab it takes 60s per ligand, or ~18 hours for 1000 mols.

Most likely you come back after a day and find your Colab runtime disconnected. So, we proceed by:

1. Ensuring Cells 1 and 2 are executed.
2. Ensure `al_iteration=0` in Cell 5 and run it.
3. Run Cells 6, 9, 11, 15. These are config cells and are required to ensure all parameters are properly set.
4. Run Cell 17 (takes 4-5 min for 1000 mols)
5. Run Cell 18.
6. Run Cell 19.
7. Run Cell 20.

At this point, you have to go back to Cell 5, change `al_iteration=1`. Then:

1. Run Cells 6, 9, 11, 15, 19.
2. Run Cell 21. The AL set loaded should have al0, i.e. 1 smaller than what's now set in Cell 5 as `al_iteration=1`.
3. Run Cell 22. As a sanity check, make sure that the weights loaded will be from previous iteration, and saved to the currently set iteration.
4. Run Cell 23.

This concludes 0th iteration of the methodology. From this point, you can:

1. Ensure Cells 1 and 2 are executed.
2. Ensure `al_iteration=1` in Cell 5 and run it.
3. Run Cell 6, 9.
4. Now you can run Cell 10 to start generating new set of molecules. This step is equivalent to Step 5 at the beginning of this section and this closes the cycle.

## Contact

Please feel free to reach out to us through either of the following emails if you have any questions or need any additional files:

- <gregory.kyro@yale.edu>
- <anton.morgunov@yale.edu>
- <rafi.brent@yale.edu>
