"""
Animate Hypergraph Diffusion

Given a hypergraph and one of the available diffusions
animate an electrical flow diffusion.
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

from diffusion_functions import diffusion_functions, diffusion
import reading


STEP_SIZE = 1e-1
EPS = 1e-6


def animate_diffusion(graph_name, diffusion_function, degree, x, fx, label_names, labels, screenshots, pos=None):
    """Animate the results of a diffusion"""
    fig = plt.figure()
    graph_ax = plt.subplot2grid((5, 1), (0, 0), rowspan=4)
    func_ax = plt.subplot2grid((5, 1), (4, 0))

    # Get PCA of x
    T, n, dimensions = x.shape
    if dimensions > 2:
        pca = PCA(n_components=2)
        x_pca = np.zeros([T, n, 2])
        for i, embedding in enumerate(x):
            x_pca[i, :, :] = pca.fit_transform(embedding)
    else:
        x_pca = x

    # Plot hypergraph
    x_plot = x[0, :, 0] if pos is None else pos[:, 0]
    y_plot = x[0, :, 1] if pos is None else pos[:, 1]
    c_plot = labels if pos is None else np.linalg.norm(x[0, :, :], axis=1)
    node_collection = graph_ax.scatter(x_plot, y_plot, s=np.sqrt(degree), c=c_plot, alpha=0.4, cmap='coolwarm', clim=(0, 1))
    graph_ax.xaxis.tick_top()
    # graph_ax.set_title('Gradient Flow in the Fauci Email dataset')
    handles, _ = node_collection.legend_elements()
    graph_ax.legend(handles, label_names, loc='lower left')

    # Plot function
    x_axis = np.arange(len(fx))
    func_ax.fill_between(x_axis, fx, alpha=0.3, color='k')
    vline = func_ax.axvline(x=0, color='r', linewidth=0.5)
    point, = func_ax.plot(0, fx[0], 'ro')
    func_ax.set_yscale('log')
    func_ax.set_xlim([0, len(fx)])
    func_ax.set_xlabel('Steps')
    func_ax.set_ylabel(r'$f\left(x\left(t\right)\right)$')

    # Animation function
    #
    # Move points and zoom
    # Change scanning line
    animation_running = True
    frame = 0

    def generate_frame():
        nonlocal frame
        nonlocal T
        frame = 0
        while frame < T:
            yield frame
            frame += 1

    def animate(fr):
        nonlocal frame
        frame = fr
        if pos is None:
            node_update = list(zip(x[frame, :, 0], x[frame, :, 1]))
            node_collection.set_offsets(node_update)
            x_min, x_max = x[frame, :, 0].min(), x[frame, :, 0].max()
            y_min, y_max = x[frame, :, 1].min(), x[frame, :, 1].max()
            graph_ax.set_xlim([x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min)])
            graph_ax.set_ylim([y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min)])
        else:
            color_update = np.linalg.norm(x[frame, :, :], axis=1)
            node_collection.set_color(node_collection.to_rgba(color_update))
        vline.set_xdata(frame)
        point_update = (np.array([frame]), np.array([fx[frame]]))
        point.set_data(point_update)

        return node_collection, vline, point

    def onClick(event):
        nonlocal frame
        nonlocal T
        if event.key.isspace():
            nonlocal animation_running
            animation_running ^= True
            if animation_running:
                ani.event_source.start()
            else:
                ani.event_source.stop()
        elif event.key.lower() == 'n':
            frame = (frame + 1) % T
            animate(frame)
            plt.draw()
        elif event.key.lower() == 'p':
            frame = (frame - 1) % T
            animate(frame)
            plt.draw()

    frame_saves = np.linspace(T - 1, 0, screenshots).astype(int)[::-1]
    for i, fr in enumerate(frame_saves):
        screenshot_filename = f'{graph_name}_{diffusion_function}_diffusion_{i:03d}.png'
        animate(fr)
        fig.savefig(screenshot_filename, bbox_inches='tight', dpi=500)

    fig.canvas.mpl_connect('key_press_event', onClick)
    ani = FuncAnimation(fig, animate, frames=generate_frame, interval=40, repeat=True, repeat_delay=1500, save_count=T)
    return ani


def train(x, y, label_names, verbose=0):
    """Use diffusion results to train a model."""
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    clf = RidgeClassifier(normalize=True, class_weight='balanced')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # conf_matrix = confusion_matrix(label_names[y_test], label_names[y_pred])
    if verbose > 0:
        print(classification_report(y_test, y_pred))
        fig = plt.Figure()
        plot_confusion_matrix(clf, x_test, y_test, display_labels=label_names, values_format='d')
        plt.show()


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Animate an electrical flow diffusion.')
    parser.add_argument('-g', '--hypergraph', help='Filename of hypergraph to use.', type=str, required=True)
    parser.add_argument('--step-size', help='Step size value.', type=float, default=STEP_SIZE)
    parser.add_argument('-s', '--seed', help='Filename storing the seed vectors for each node.', type=str, default=None)
    parser.add_argument('-l', '--labels', help='Filename containing the groundtruth communities', type=str, default=None)
    parser.add_argument('-p', '--position', help='Filename containing positions', type=str, default=None)
    parser.add_argument('-f', '--function', help='Which diffusion function to use.', choices=diffusion_functions.keys(), default=list(diffusion_functions.keys())[0])
    parser.add_argument('-r', '--random-seed', help='Random seed to use for initialization.', type=int, default=None)
    parser.add_argument('-e', '--epsilon', help='Epsilon used for convergence criterion.', type=float, default=EPS)
    parser.add_argument('-x', help='Filename to read initial x_0 from. Ignores dimensions.', type=str, default=None)
    parser.add_argument('--no-plot', help='Skip plotting to focus with classification.', action='store_true')
    parser.add_argument('--no-save', help='Disable saving the animation. Results in faster completion time.', action='store_true')
    parser.add_argument('-d', '--dimensions', help='Number of embedding dimensions.', type=int, default=2)
    parser.add_argument('--screenshots', help='How many screenshots of the animation to save.', default=0, type=int)
    parser.add_argument('-T', '--iterations', help='Maximum iterations for diffusion.', type=int, default=None)
    parser.add_argument('--confusion', help='Produce a confusion matrix.', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose mode. Prints out useful information. Higher levels print more information.', action='count', default=0)
    args = parser.parse_args()
    return args


def main():
    """
    Main controlling function

    Process arguments
    Read hypergraph
    Compute diffusion
    Animate and show
    """
    args = parse_args()
    graph_name = os.path.basename(os.path.splitext(args.hypergraph)[0])
    if args.verbose > 0:
        print(f'Reading hypergraph from file {args.hypergraph}')
    n, m, node_weights, hypergraph, weights, center_id, hypergraph_node_weights = reading.read_hypergraph(args.hypergraph)
    if args.random_seed is None:
        args.random_seed = np.random.randint(1000000)
    np.random.seed(args.random_seed)
    x0 = reading.read_positions(args.x)
    if x0 is not None:
        args.dimensions = len(x0[0])
    else:
        x0 = np.random.rand(n, args.dimensions)
    label_names, labels = reading.read_labels(args.labels)
    if len(label_names) == 0:
        label_names = ['Nodes']
        labels = [0] * n
    pos = reading.read_positions(args.position)
    s = reading.read_seed(args.seed, labels, args.dimensions, node_weights)

    if args.verbose > 0:
        print(f'Performing diffusion on hypergraph with {n} nodes and {m} hyperedges.')
        print(f'Random seed = {args.random_seed}')
    x, _, fx = diffusion(x0, n, m, node_weights, hypergraph,
                         weights, center_id, hypergraph_node_weights,
                         diffusion_functions[args.function],
                         s=s, h=args.step_size, T=args.iterations,
                         verbose=args.verbose, eps=args.epsilon)

    if args.confusion:
        train(x[-1], labels, label_names, verbose=args.verbose)
    if not args.no_plot:
        ani = animate_diffusion(graph_name, args.function, node_weights, x, fx, label_names, labels, args.screenshots, pos=pos)
        if not args.no_save:
            ani.save(f'{graph_name}_{args.function}_diffusion.gif', writer='imagemagick', fps=10)
        plt.show()


if __name__ == '__main__':
    main()
