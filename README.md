# Hypergraph Diffusions

Code for different diffusions on hypergraphs.

## Step 1: Create a dataset

Inside the `data` directory, run the following command

```python
python hypergraphs2hmetis.py -vv
```

## Step 2: Run the animation

From the top level you can run this command

```python
python animate_diffusion.py -g data/fauci_email.hmetis -l data/fauci_email.label -v
```

# Manifold learning experiments

No setup action necessary, see notebook for demo.

# Restaurant Clustering

Step 1: download publiclly available data (https://www.yelp.com/dataset) and add to data/yelp.

Step 2: run build_restaurant_hypergraph from yelp_restaurant_clustering.py. This constructs and saves a hypergraph with data from all reviews. You can later process this hypergraph (e.g. delete hyperedges with few nodes, etc.)