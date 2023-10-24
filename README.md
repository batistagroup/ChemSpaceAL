# ChemSpaceAL: An Efficient Active Learning Methodology Applied to Protein- Specific Molecular Generation

[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/batistagroup/ChemSpaceAL/graph/badge.svg?token=ROJSISYJWC)](https://codecov.io/gh/batistagroup/ChemSpaceAL)

![A description of the active learning methodology](media/toc_figure.jpg)

## Abstract

The incredible capabilities of generative artificial intelligence models have inevitably led to their application in the domain of drug discovery. It is therefore of tremendous interest to develop methodologies that enhance the abilities and applicability of these powerful tools. In this work, we present a novel and efficient semi-supervised active learning methodology that allows for the fine-tuning of a generative model with respect to an objective function by strategically operating within a constructed representation of the sample space. In the context of targeted molecular generation, we demonstrate the ability to fine-tune a GPT-based molecular generator with respect to an attractive interaction-based scoring function by strategically operating within a chemical space proxy, thereby maximizing attractive interactions between the generated molecules and a protein target. Importantly, our approach does not require the individual evaluation of all data points that are used for fine-tuning, enabling the incorporation of computationally expensive metrics. We are hopeful that the inherent generality of this methodology ensures that it will remain applicable as this exciting field evolves. To facilitate implementation and reproducibility, we have made all of our software available through the open-source ChemSpaceAL Python package.

## Preprint

Associated preprint can be found on [arXiv](https://arxiv.org/pdf/2309.05853.pdf)

## Installation

in order to install the ChemSpaceAL package, simply run:

```pip install ChemSpaceAL```
