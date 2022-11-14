# Feature Learning


## Requirements
To run this example, make sure to have completed the installation instructions [described here](../../README.md) and to have the `evo` environment activated.

```bash
conda activate evo
```


## Get started
To start the experiment, run `python main.py`.

By default, the example uses randomly extracted image patches from the standard Barbara image as exemplary training data set. To use your own data set, modify the `--data_file` argument accordingly. Further options can be display via:

```bash
$ python main.py -h           
usage: main.py [-h] [--data_file DATA_FILE] [--output_directory OUTPUT_DIRECTORY] [--model {bsc,sssc}] [-H H] [--Ksize KSIZE] [--parent_selection {fit,rand}]
               [--mutation_algorithm {randflip,sparseflip,cross,cross_randflip,cross_sparseflip}] [--no_parents NO_PARENTS] [--no_children NO_CHILDREN]
               [--no_generations NO_GENERATIONS] [--bitflip_prob BITFLIP_PROB] [--no_epochs NO_EPOCHS] [--sort_gfs]

Feature Learning

options:
  -h, --help            show this help message and exit
  --data_file DATA_FILE
                        .npz file with training data set (default: ./data/barbara-2k-patches.npz)
  --output_directory OUTPUT_DIRECTORY
                        Directory to write training output and visualizations to (will be output/<TIMESTAMP> if not specified) (default: None)
  --model {bsc,sssc}    Generative Model (default: bsc)
  -H H                  Number of generative fields to learn (default: 100)
  --Ksize KSIZE         Size of the K sets (i.e., S=|K]) (default: 15)
  --parent_selection {fit,rand}
                        Selection operator (default: fit)
  --mutation_algorithm {randflip,sparseflip,cross,cross_randflip,cross_sparseflip}
                        Mutation strategy (default: randflip)
  --no_parents NO_PARENTS
                        Number of parental states to select per generation (default: 5)
  --no_children NO_CHILDREN
                        Number of children to evolve per generation (default: 2)
  --no_generations NO_GENERATIONS
                        Number of generations to evolve (default: 1)
  --bitflip_prob BITFLIP_PROB
                        Bitflip probability (only relevant for sparseflip-based mutation algorithms) (default: None)
  --no_epochs NO_EPOCHS
                        Number of epochs to train (default: 200)
  --sort_gfs            Whether to visualize learned generative fields according to prior activation (default: False)
```

For distributed execution on multiple CPU cores (requires MPI to be installed), run with `mpirun -n <n_proc> python ...`. For example, to use four cores, run:

```bash
mpirun -n 4 python main.py
```


## Reference
[1] Evolutionary Variational Optimization of Generative Models. Jakob Drefs, Enrico Guiraud, Jörg Lücke. _Journal of Machine Learning Research_ 23(21):1-51, 2022. [(online access)](https://www.jmlr.org/papers/v23/20-233.html)
