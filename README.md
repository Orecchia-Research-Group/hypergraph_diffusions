# Hypergraph Diffusions

## Step 1: Create a dataset

Inside the `data` directory, run the following command

```python
python hypergraphs2hmetis.py -vv
```

If you want to build only one example dataset you can use

```python
python hypergraphs2hmetis.py -i Paper_datasets -o Paper_datasets -n covertype -vv -t 4 5 --suffix 45
```

## Step 2: Run the animation

From the top level you can run this command

```python
python animate_diffusion.py -g data/fauci_email.hmetis -l data/fauci_email.label -f infinity -v
```

# Manifold learning experiments

No setup action necessary, see notebook for demo.

To generate the figures in the ICML paper, run

```python
python ICML2024_Figure_Generation.py
```

# Fast C++ implementation

For using the fast C++ implementation you will first need to install the Eigen and cxxopts libraries

```
sudo apt install libeigen3-dev
sudo apt install libcxxopts-dev
```

Or download Eigen through [here](https://eigen.tuxfamily.org/dox/GettingStarted.html) and cxxopt through [here](https://github.com/jarro2783/cxxopts/blob/master/INSTALL). Make sure that both are available in `/usr/include`.

Install `pybind11`

```
python3 -m pip install pybind11
```

Compile the python wrapper

```
python diffusion_wrapper_setup.py build_ext --inplace
```

Opening a `python` terminal you can run a hypergraph diffusion as follows

```python
import reading
import diffusion


n, m, node_weights, hypergraph, weights, center_id, hypergraph_node_weights = reading.read_hypergraph('data/Paper_datasets/zoo.hmetis')
# Turn hypergraph into list of lists
hypergraph = [list(e) for e in hypergraph]
label_names, labels = reading.read_labels('data/Paper_datasets/zoo.label')
gs = diffusion.GraphSolver(n, m, node_weights, hypergraph, len(label_names), labels, 0)
gs.run_diffusions("zoo", 1, 100, 1, 0.1, 20, 5, 50, 0)
```

Alternatively, you can compile and use the C++ driver

```bash
g++ -std=c++20 -Wall -Wextra -O3 -o fast_alg diffusion.cpp fast_alg.cpp
```

An example call is

```bash
./fast_alg --graph_filename data/Paper_datasets/covertype45.hmetis --label_filename data/Paper_datasets/covertype45.label -T 3000 --lambda 1 -h 0.4 --minimum_revealed 20 --step 20 --maximum_revealed 200 -r 5 -e 40 --schedule 0 -v 2> covertype45_all.txt
```
