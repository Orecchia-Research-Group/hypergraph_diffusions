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
