# EVO - Evolutionary Variational Optimization of Generative Models
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)

This repository contains the implementation of the EBSC and ES3C algorithms described in the paper [1]. Examples how to setup and run the algorithms can be found [here](/examples). You may also check out our `tvo` package which features PyTorch implementations of the algorithms, and which is available [here](https://github.com/tvlearn/tvo).



## Installation

We recommend [Anaconda](https://www.anaconda.com/) to manage the installation. Create a new environment to bundle the packages required:

```bash
$ conda create -c conda-forge -c anaconda -n evo python=2.7.15 pip pytables imageio
```

You can also install a Python 3 version instead, but note that the results described in [1] for EBSC and ES3C were produced by running in Python 2.

The implementations of the algorithms are parallelized using `mpi4py`, which requires a system level installtion of MPI. Run one of the following commands, depending on your system, to install: 

```
$ brew install mpich  # MacOS
$ sudo apt install mpich  # Ubuntu
```

Please consult the official documentation of [mpi4py](https://mpi4py.readthedocs.io/en/stable/install.html) and [MPICH](https://www.mpich.org/documentation/guides/) if you need help.

Once MPICH is installed, you are ready to install mpi4py and further required packages by running:

```bash
$ pip install -r requirements.txt
```

Running the examples furthermore requires the `tvutil` package, for which installation instructions can be found [here](https://github.com/tvlearn/tvutil).

Finally, you are ready to install `evo`:

```bash
$ python setup.py install
```

__Remark__:

For code formatting and analysis, we use `black` and `pylama`. These tools can be installed via:

```bash
$ pip3 install black[python2] pylama
```


## Reference

[1] Jakob Drefs, Enrico Guiraud, Jörg Lücke. Evolutionary Variational Optimization of Generative Models. _Journal of Machine Learning Research_ 23(21):1-51, 2022. [(online access)](https://www.jmlr.org/papers/v23/20-233.html)
