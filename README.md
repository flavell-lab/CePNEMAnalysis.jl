# CePNEMAnalysis.jl

[![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://flavell-lab.github.io/CePNEMAnalysis.jl/stable/

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://flavell-lab.github.io/CePNEMAnalysis.jl/dev/ 

This package provides a collection of tools for interpreting and visualizing CePNEM model fits. It comes with four companion notebooks (available in the `notebook` directory in this package):

- `CePNEM-analysis.ipynb`: Loads raw CePNEM fit data and computes metrics such as neural encoding, encoding change, variability, and more.
- `CePNEM-plots.ipynb`: Presents the most common plots used to visualize CePNEM fits, as well as a guide to exploring neural encodings generated from the `CePNEM-analysis.ipynb` notebook. This notebook can also be used by downloading our preprocessed data from wormwideweb.org and examining it here.
- `CePNEM-UMAP.ipynb`: Demonstrates how to use UMAP to visualize CePNEM fits.
- `CePNEM-auxiliary.ipynb`: Presents less-commonly used plots and functions, such as model validation metrics, decoder training, and more.

## Citation
To cite this work, please refer to [this article](https://github.com/flavell-lab/AtanasKim-Cell2023/tree/main#citation).

## Data download
To download our preprocessed data compatible with loading into our notebooks, please see the `.jld2` files [here](https://zenodo.org/record/8185377).
