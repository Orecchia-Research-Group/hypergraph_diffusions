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
from sklearn.metrics import confusion_matrix, classification_report     # , plot_confusion_matrix

from diffusion_functions import diffusion_functions, diffusion
import reading


STEP_SIZE = 1e-1
EPS = 1e-6
SAVE_FOLDER = 'results'
COST_OPTIONS = ['fx', 'variance']


def animate_diffusion(graph_name, diffusion_function, degree, x, fx, label_names, labels, screenshots, save_folder=SAVE_FOLDER, pos=None, cost='fx'):
    """Animate the results of a diffusion"""
    color_list = ['b', 'r', 'y', 'g']

    if cost not in COST_OPTIONS:
        raise ValueError(f'Unknown cost option specified. Must be on of {COST_OPTIONS}, instead got {cost}.')

    fig = plt.figure()
    number_of_hypergraphs = len(graph_name)
    graph_ax = [plt.subplot2grid((5, number_of_hypergraphs), (0, i), rowspan=4) for i in range(number_of_hypergraphs)]
    func_ax = plt.subplot2grid((5, 1), (4, 0))

    node_collection = []
    vline = []
    point = []
    cost_values = []

    # Get PCA of x
    for i, hyper_graph_name, hyper_x, hyper_fx, hyper_labels, hyper_pos in zip(range(number_of_hypergraphs), graph_name, x, fx, labels, pos):
        T, n, dimensions = hyper_x.shape
        if dimensions > 2:
            pca = PCA(n_components=2)
            x_pca = np.zeros([T, n, 2])
            for i, embedding in enumerate(hyper_x):
                x_pca[i, :, :] = pca.fit_transform(embedding)
        else:
            x_pca = hyper_x

        # Plot hypergraph
        if hyper_pos is not None:
            x_plot = hyper_pos[:, 0]
            y_plot = hyper_pos[:, 1] if hyper_pos.shape[1] > 1 else hyper_x[0, :, 0]
        else:
            x_plot = hyper_x[0, :, 0]
            y_plot = hyper_x[0, :, 1]
        c_plot = hyper_labels if hyper_pos is None else np.linalg.norm(hyper_x[0, :, :], axis=1)
        node_collection.append(graph_ax[i].scatter(x_plot, y_plot, s=20, c=c_plot, alpha=1, cmap='coolwarm', clim=(0, 1)))
        graph_ax[i].xaxis.tick_top()
        # graph_ax.set_title('Gradient Flow in the Fauci Email dataset')
        handles, _ = node_collection[i].legend_elements()
        # graph_ax.legend(handles, label_names, loc='lower left')
        graph_ax[i].set_aspect('equal')

        # Plot function
        if cost == 'fx':
            cost_values.append(hyper_fx.sum(axis=1))
        elif cost == 'variance':
            x_bar = np.einsum('i,ij->j', degree[i], hyper_x[0, :, :]) * (1 / sum(degree[i]))
            dx = hyper_x - x_bar
            cost_values.append(np.einsum('i,kij,kij->k', degree[i], dx, dx))
        x_axis = np.arange(len(cost_values[i]))
        func_ax.fill_between(x_axis, cost_values[i].reshape(-1), alpha=0.3, color='k')
        vline.append(func_ax.axvline(x=0, color=color_list[i], linewidth=0.5))
        p, = func_ax.plot(0, cost_values[i][0], 'o', color=color_list[i])
        point.append(p)
        func_ax.set_yscale('log')
        func_ax.set_xlim([0, len(cost_values[i])])
        func_ax.set_xlabel('Steps')
        func_ax.set_ylabel(cost)

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
        for i in range(number_of_hypergraphs):
            local_frame = min(frame, x[i].shape[0] - 1)
            if pos[i] is None:
                node_update = list(zip(x[i][local_frame, :, 0], x[i][local_frame, :, 1]))
                node_collection[i].set_offsets(node_update)
                x_min, x_max = x[i][local_frame, :, 0].min(), x[i][local_frame, :, 0].max()
                y_min, y_max = x[i][local_frame, :, 1].min(), x[i][local_frame, :, 1].max()
                graph_ax[i].set_xlim([x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min)])
                graph_ax[i].set_ylim([y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min)])
            else:
                color_update = np.linalg.norm(x[i][local_frame, :, :], axis=1)
                node_collection[i].set_color(node_collection[i].to_rgba(color_update))
                if pos[i].shape[1] == 1:
                    node_update = list(zip(pos[i][:, 0], x[i][local_frame, :, 0]))
                    node_collection[i].set_offsets(node_update)
            vline[i].set_xdata(local_frame)
            point_update = (np.array([local_frame]), np.array([cost_values[i][local_frame]]))
            point[i].set_data(point_update)

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
        elif event.key.lower() in ['n', 'right']:
            frame = (frame + 1) % T
            animate(frame)
            plt.draw()
        elif event.key.lower() in ['p', 'left']:
            frame = (frame - 1) % T
            animate(frame)
            plt.draw()
        elif '0' <= event.key <= '9':
            frame = int(event.key) * T // 10
            animate(frame)
            plt.draw()
        elif event.key in ['up', 'home']:
            frame = 0
            animate(frame)
            plt.draw()
        elif event.key in ['down', 'end']:
            frame = T - 1
            animate(frame)
            plt.draw()

    frame_saves = np.linspace(T - 1, 0, screenshots).astype(int)[::-1]
    for i, fr in enumerate(frame_saves):
        screenshot_filename = f'{graph_name}_{diffusion_function}_diffusion_{i:03d}.png'
        animate(fr)
        fig.savefig(os.path.join(save_folder, screenshot_filename), bbox_inches='tight', dpi=500)

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
    return clf, x_test, y_test, y_pred


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description='Animate an electrical flow diffusion.')
    parser.add_argument('-g', '--hypergraph', help='Filename of hypergraph to use.', type=str, required=True, nargs='+')
    parser.add_argument('--step-size', help='Step size value.', type=float, default=STEP_SIZE)
    parser.add_argument('-s', '--seed', help='Filename storing the seed vectors for each node.', type=str, default=None, nargs='+')
    parser.add_argument('-l', '--labels', help='Filename containing the groundtruth communities', type=str, default=None, nargs='+')
    parser.add_argument('-p', '--position', help='Filename containing positions', type=str, default=None, nargs='+')
    parser.add_argument('-f', '--function', help='Which diffusion function to use.', choices=diffusion_functions.keys(), default=list(diffusion_functions.keys())[0])
    parser.add_argument('-r', '--random-seed', help='Random seed to use for initialization.', type=int, default=None)
    parser.add_argument('-e', '--epsilon', help='Epsilon used for convergence criterion.', type=float, default=EPS)
    parser.add_argument('-x', help='Filename to read initial x_0 from. Ignores dimensions.', type=str, default=None, nargs='+')
    parser.add_argument('--no-plot', help='Skip plotting to focus with classification.', action='store_true')
    parser.add_argument('--no-save', help='Disable saving the animation. Results in faster completion time.', action='store_true')
    parser.add_argument('--save-folder', help='Folder to save pictures.', default=SAVE_FOLDER)
    parser.add_argument('-d', '--dimensions', help='Number of embedding dimensions.', type=int, default=2)
    parser.add_argument('--screenshots', help='How many screenshots of the animation to save.', default=0, type=int)
    parser.add_argument('-T', '--iterations', help='Maximum iterations for diffusion.', type=int, default=None)
    parser.add_argument('--confusion', help='Produce a confusion matrix.', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose mode. Prints out useful information. Higher levels print more information.', action='count', default=0)
    parser.add_argument('-c', '--cost', help='Cost function to plot under animation.', default=COST_OPTIONS[0], choices=COST_OPTIONS)
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
    print(args)
    assert args.seed is None or len(args.hypergraph) == len(args.seed), 'Number of seeds must be equal to number of hypergraphs.'
    assert args.labels is None or len(args.hypergraph) == len(args.labels), 'Number of labels must be equal to number of hypergraphs.'
    assert args.position is None or len(args.hypergraph) == len(args.position), 'Number of positions must be equal to number of hypergraphs.'
    t, x, fx, node_weights, graph_name, pos, labels, label_names = [], [], [], [], [], [], [], []
    for i, hyper in enumerate(args.hypergraph):
        hyper_graph_name = os.path.basename(os.path.splitext(hyper)[0])
        graph_name.append(hyper_graph_name)
        if args.verbose > 0:
            print(f'Reading hypergraph from file {hyper}')
        n, m, hyper_node_weights, hypergraph, weights, center_id, hypergraph_node_weights = reading.read_hypergraph(hyper)
        node_weights.append(hyper_node_weights)
        if args.random_seed is None:
            args.random_seed = np.random.randint(1000000)
        np.random.seed(args.random_seed)
        hyper_x = None if args.x is None else args.x[i]
        x0 = reading.read_positions(hyper_x, n, args.dimensions)
        if x0 is not None:
            args.dimensions = len(x0[0])
        else:
            x0 = np.random.rand(n, args.dimensions)
        hyper_labels = None if args.labels is None else args.labels[i]
        hyper_label_names, hyper_label = reading.read_labels(hyper_labels)
        if len(hyper_label_names) == 0:
            hyper_label_names = ['Nodes']
            hyper_label = [0] * n
        label_names.append(hyper_label_names)
        labels.append(hyper_label)
        hyper_pos = None if args.position is None else args.position[i]
        pos.append(reading.read_positions(hyper_pos, n, args.dimensions))
        hyper_seed = None if args.seed is None else args.seed[i]
        s = reading.read_seed(hyper_seed, labels[i], args.dimensions, node_weights[i])

        if args.verbose > 0:
            print(f'Performing diffusion on hypergraph with {n} nodes and {m} hyperedges.')
            print(f'Random seed = {args.random_seed}')
        hyper_t, hyper_x, _, hyper_fx = diffusion(x0, n, m, node_weights[i], hypergraph,
                                                  weights, center_id=center_id,
                                                  hypergraph_node_weights=hypergraph_node_weights,
                                                  func=diffusion_functions[args.function],
                                                  s=s, h=args.step_size, T=args.iterations,
                                                  verbose=args.verbose, eps=args.epsilon)
        t.append(hyper_t)
        x.append(hyper_x)
        fx.append(hyper_fx)

    # if args.confusion:
    #     train(x[-1], labels, label_names, verbose=args.verbose)
    if not args.no_plot:
        ani = animate_diffusion(graph_name, args.function, node_weights, x, fx, label_names, labels,
                                args.screenshots, pos=pos, save_folder=args.save_folder, cost=args.cost)
        if not args.no_save:
            ani.save(os.path.join(args.save_folder, f'{graph_name}_{args.function}_diffusion.gif'), writer='imagemagick', fps=10)
        plt.show()


if __name__ == '__main__':
    main()
