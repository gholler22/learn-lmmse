# Learning the linear minimum mean squared error (LMMSE) estimator

This repository provides the source code for the paper ["How many samples are needed to reliably approximate the best linear estimator for a linear inverse problem?"](https://arxiv.org/abs/2107.00215).


## Requirements

The code is written for Python 3. It was tested in a virtual environment with Python 3.6.9 and the following Python packages installed:

- [numpy](https://pypi.org/project/numpy/)
- [scipy](https://pypi.org/project/scipy/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [tensorflow](https://pypi.org/project/tensorflow/)
- [notebook](https://pypi.org/project/notebook/)

A list of installed packages is given in [requirements.txt](requirements.txt).

## Installing

The development version is available here on github:

``` bash
git clone https://github.com/gholler22/learn-lmmse.git
```

## Usage

A demo for a Gaussian inverse problem can be found in [gaussian_learn_lmmse.ipynb](gaussian_learn_lmmse.ipynb).

A demo for a denoising inverse problem can be found in [denoising_learn_lmmse.ipynb](denoising_learn_lmmse.ipynb).

## Author

- Gernot Holler gernot.holler@gmail.com

## Publications

* G. Holler,  ["How many samples are needed to reliably approximate the best linear estimator for a linear inverse problem?"](https://arxiv.org/abs/2107.00215), arXiv preprint arXiv:2107.00215, 2021.

## Acknowledgments

The development of this code was partially supported by the International Research Training Group IGDK 1754 „Optimization and Numerical Analysis for Partial Differential Equations with Nonsmooth Structures“, funded by the German Research Council (DFG) and the Austrian Science Fund (FWF):[W 1244-N18].



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.