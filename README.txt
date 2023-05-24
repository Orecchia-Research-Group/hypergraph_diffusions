Catalog:

Hypergraphs/[name].hmetis:
    These files contain the hypergraphs in the same format as hMETIS.
    The first line contains three integeres, `m`, `n` and `fmt`.
    `m` is the number of hyperedges, `n` the number of nodes and `fmt` describes what is present in the rest of the file and is of the form `xyz`.
    All of `x`, `y` and `z` are 0 or 1. `x` = 1 notes that the hyperedges are weighted.
    `y` = 1 means that node volume is also provided.
    `z` is not used, but is intended to describe how much each node participates in the hyperedge.
    In the next `m` lines the hyperedges are provided. If they are weighted, then the first number provides their weight, the rest of the numbers are integers describing the participating nodes.
    If node volume is provided, in the next `n` lines there is a single real number, the volume of node `u`.
    

hypergraph2hmetis.py:
    Create hypergraphs from raw data. For finished hypergraphs see the `Hypergraphs` directory.

    Examples:
        python hypergraphs2hmetis.py -vv -f -n grid -i "10 20"

        python hypergraphs2hmetis.py -h
        usage: hypergraphs2hmetis.py [-h] [-i INPUT_DIRECTORY] [-o OUTPUT_DIRECTORY]
                                     [-n {fauci_email,grid,nodeGrid,clique} [{fauci_email,grid,nodeGrid,clique} ...]]
                                     [-v] [-f]

        Convert the varying hypergraph formats into the uniform hMETIS format.

        optional arguments:
          -h, --help            show this help message and exit
          -i INPUT_DIRECTORY, --input_directory INPUT_DIRECTORY
                                Directory where the raw data is stored.
          -o OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                                Directory to write the processed results.
          -n {fauci_email,grid,nodeGrid,clique} [{fauci_email,grid,nodeGrid,clique} ...], --names {fauci_email,grid,nodeGrid,clique} [{fauci_email,grid,nodeGrid,clique} ...]
                                Choose which hypergraphs you want to process.
          -v, --verbose         Verbose mode. Prints out useful information. Higher
                                levels print more information, including basic
                                hypergraph stats.
          -f, --force           Force reprocessing even if .hmetis file is present.

          Konstantinos Ameranis, University of Chicago 2023


produce_hypergraph_extractedDblp.py:
    Creates the DBLP dataset from the JSON containing papers.

    Example:
        python produce_hypergraph_extractedDblp.py

make_hypergraphs.py:
    Used to turn KONECT bipartite graphs into hypergraphs.

make_random_geometric.py:
    Create a random geometric (hyper)graph with k-NN and run PageRank and Personalized PageRank on both the graph and hypergraph.
    Results are shown in Figure 3 in the appendix of our paper.

    Examples:
        python make_random_geometric.py 100 -k 5 -s 5 -l 0.02

        python make_random_geometric.py -h
        usage: make_random_geometric.py [-h] [-k K] [-d DIRECTORY] [-s SEED]
                                        [-l LAMDA]
                                        n

        positional arguments:
          n                     Number of nodes

        optional arguments:
          -h, --help            show this help message and exit
          -k K                  Number of neighbors
          -d DIRECTORY, --directory DIRECTORY
                                Directory to save (hyper)graph
          -s SEED, --seed SEED  Random seed
          -l LAMDA, --lamda LAMDA
                                Lamda value for PPR

        Konstantinos Ameranis, University of Chicago 2023


reading.py:
    Functions to read all relevant files. Requires dblp.json.gz from https://projects.csail.mit.edu/dnd/DBLP/dblp.json.gz

diffusion_functions.py:
    Contains the infinity, quadratic and linear diffusion functions, degree and clique regularizer,
    `diffusion` that takes all the arguments and functions to find sweep cuts.

animate_functions.py
    Produces animations for different settings. From fixed positions to diffusion values being the positions, all can be found here

    Example:
        python animate_diffusion.py -g Hypergraphs/hyperExtractedDblp.hmetis -v --step-size 1 -e 0.000001 -l Hypergraphs/hyperExtractedDblp.labels -r 17 --no-save -f infinity --dimensions 6 -T 30 -s 0.5 --screenshots 11 --confusion -x 0

        python animate_diffusion.py -h
        usage: animate_diffusion.py [-h] -g HYPERGRAPH [--step-size STEP_SIZE]
                                    [-s SEED] [-l LABELS] [-p POSITION]
                                    [-f {quadratic,linear,infinity}] [-r RANDOM_SEED]
                                    [-e EPSILON] [-x X] [--no-plot] [--no-save]
                                    [--save-folder SAVE_FOLDER] [-d DIMENSIONS]
                                    [--screenshots SCREENSHOTS] [-T ITERATIONS]
                                    [--confusion] [-v]

        Animate an electrical flow diffusion.

        optional arguments:
          -h, --help            show this help message and exit
          -g HYPERGRAPH, --hypergraph HYPERGRAPH
                                Filename of hypergraph to use.
          --step-size STEP_SIZE
                                Step size value.
          -s SEED, --seed SEED  Filename storing the seed vectors for each node.
          -l LABELS, --labels LABELS
                                Filename containing the groundtruth communities
          -p POSITION, --position POSITION
                                Filename containing positions
          -f {quadratic,linear,infinity}, --function {quadratic,linear,infinity}
                                Which diffusion function to use.
          -r RANDOM_SEED, --random-seed RANDOM_SEED
                                Random seed to use for initialization.
          -e EPSILON, --epsilon EPSILON
                                Epsilon used for convergence criterion.
          -x X                  Filename to read initial x_0 from. Ignores dimensions.
          --no-plot             Skip plotting to focus with classification.
          --no-save             Disable saving the animation. Results in faster
                                completion time.
          --save-folder SAVE_FOLDER
                                Folder to save pictures.
          -d DIMENSIONS, --dimensions DIMENSIONS
                                Number of embedding dimensions.
          --screenshots SCREENSHOTS
                                How many screenshots of the animation to save.
          -T ITERATIONS, --iterations ITERATIONS
                                Maximum iterations for diffusion.
          --confusion           Produce a confusion matrix.
          -v, --verbose         Verbose mode. Prints out useful information. Higher
                                levels print more information.

        Konstantinos Ameranis, University of Chicago 2023

ikeda.py:
    Following the implementation of Ikeda et al. and augmenting with our approach of returning the average of all iterates instead of last iterate.

    Examples:
        python ikeda.py -g Hypergraphs/dbpedia_genre.hmetis -f infinity -d 20 -T 20 -v -r 42 -l 0.04 --no-sweep --regularizer degree

        python ikeda.py -h
        usage: Ikeda Practical Evaluation [-h] -g HYPERGRAPH [--step-size STEP_SIZE]
                                          [-f {quadratic,linear,infinity}]
                                          [-r RANDOM_SEED] [-d DIMENSIONS]
                                          [-T ITERATIONS] [--lamda LAMDA] [-e ETA]
                                          [--regularizer {degree,clique}]
                                          [--save-folder SAVE_FOLDER] [--no-sweep]
                                          [--write-values] [-v]

        Run practical experiments using the method described in Ikeda et al.

        optional arguments:
          -h, --help            show this help message and exit
          -g HYPERGRAPH, --hypergraph HYPERGRAPH
                                Filename of hypergraph to use.
          --step-size STEP_SIZE
                                Step size value.
          -f {quadratic,linear,infinity}, --function {quadratic,linear,infinity}
                                Which diffusion function to use.
          -r RANDOM_SEED, --random-seed RANDOM_SEED
                                Random seed to use for initialization.
          -d DIMENSIONS, --dimensions DIMENSIONS
                                Number of embedding dimensions.
          -T ITERATIONS, --iterations ITERATIONS
                                Maximum iterations for diffusion.
          --lamda LAMDA, -l LAMDA
                                Parameter used in personalized pagerank.
          -e ETA, --eta ETA     Exponential averaging parameter.
          --regularizer {degree,clique}
                                Preconditioner for hypergraph diffusion
          --save-folder SAVE_FOLDER
                                Folder to save pictures.
          --no-sweep            Disable doing sweep cuts.
          --write-values        Save all values in a pickle file based on the
                                arguments in the save folder.
          -v, --verbose         Verbose mode. Prints out useful information. Higher
                                levels print more information.

        Konstantinos Ameranis, University of Chicago 2023

get_data_from_pickles.py:
    Quickly load all results from the pickle files in paper_results.

    Example:
        results = data_from_pickle(directory='paper_results/', regularizer='degree')


