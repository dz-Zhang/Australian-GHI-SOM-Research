Code used to obtain the SOM results in results.pdf. The underlying SOM uses the [Torch SOM](https://opensource.michelin.io/TorchSOM/) library.

# Installation
```
git clone https://github.com/dz-Zhang/UNSW-Climate-Research
```

Install dependencies. Requires the main scientific computing packages as well as PyTorch.
```
pip install -r requirements.txt
```
Installing the library
```
pip install .
```

# Examples
- [SOM Grid Search](examples/som_grid_search.py): Run SOM on various grid sizes and compare final quantisation, topographic errors.
- [Example SOM Run](examples/som_sequence.ipynb): An example running the full SOM pipeline from loading data, training SOM to visualising results.
