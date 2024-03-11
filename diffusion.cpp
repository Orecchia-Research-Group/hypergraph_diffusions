#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <iterator>
#include <chrono>
#include <ctime>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include "diffusion.h"


MatrixReplacement::Index MatrixReplacement::rows() const { return mp_mat->rows(); }
MatrixReplacement::Index MatrixReplacement::cols() const { return mp_mat->cols(); }

template<typename Rhs>
Eigen::VectorXd MatrixReplacement::operator*(const Eigen::MatrixBase<Rhs>& x) const {
    Eigen::VectorXd y = (*mp_mat) * x;
    double x_sum = x.sum();
    for(int i = 0; i < y.size(); i++)
        y(i) += x_sum;
    return y;
}

void MatrixReplacement::attachMyMatrix(const Eigen::SparseMatrix<double> &mat) {
    mp_mat = &mat;
}

const Eigen::SparseMatrix<double> MatrixReplacement::my_matrix() const { return *mp_mat; }


void GraphSolver::read_hypergraph(std::string filename) {
    int fmt;
    std::string line;
    std::ifstream input_file;
    input_file.open(filename);
    input_file >> this->m >> this->n >> fmt;
    getline(input_file, line);

    // Read hyperedges
    for(int i = 0; i < m; i++) {
        std::vector<int> hyperedge;
        int node;
        getline(input_file, line);
        std::istringstream iss(line);
        while(iss >> node) {
            hyperedge.push_back(node - 1);
        }
        this->hypergraph.push_back(hyperedge);
    }

    // Read degrees
    for(int i = 0; i < n; i++) {
        double d;
        input_file >> d;
        this->degree.push_back(d);
    }
    input_file.close();
}

void GraphSolver::read_labels(std::string filename) {
    std::ifstream label_file;
    label_file.open(filename);
    std::string line;
    getline(label_file, line);
    std::istringstream iss(line);
    int label;
    std::map<int, int> label_map;
    for(this->label_count = 0; iss >> label; this->label_count++) {
        label_map[label] = this->label_count;
    }
    for(int i = 0; i < this->n; i++) {
        label_file >> label;
        this->labels.push_back(label_map[label]);
    }
}

inline double GraphSolver::fmax(double a, double b) {
    return a > b ? a : b;
}

inline double GraphSolver::fmin(double a, double b) {
    return a < b ? a : b;
}

Eigen::SparseMatrix<double> GraphSolver::create_laplacian() {
    Eigen::SparseMatrix<double> laplacian(n+m, n+m);
    for(int j = 0; j < m; j++) {
        auto h = hypergraph[j].size();
        laplacian.coeffRef(n+j, n+j) = h;
        for(auto it = hypergraph[j].begin(); it < hypergraph[j].end(); it++) {
            auto v = *it;
            laplacian.coeffRef(v, v) += 1;
            laplacian.coeffRef(v, n+j) = -1;
            laplacian.coeffRef(n+j, v) = -1;
        }
    }
    return laplacian;
}

GraphSolver::GraphSolver(std::string graph_filename, std::string label_filename, std::string preconditioner) {
    read_hypergraph(graph_filename);
    if(!label_filename.empty()) read_labels(label_filename);
    if(!preconditioner.empty() || preconditioner.compare("degree") == 0) preconditionerType = 0;
    else if(preconditioner.compare("star") == 0) {
        preconditionerType = 1;
        starLaplacian = create_laplacian();
        L.attachMyMatrix(starLaplacian);
        solver.compute(L);
    }
    else {
        perror("Unknown type of preconditioner.");
        exit(1);
    }
    // std::cerr << "Constructed hypergraph with " << n << " nodes and " << m << " hyperedges" << std::endl;
}

GraphSolver::GraphSolver(int n, int m, std::vector<double> degree, std::vector<std::vector<int>> hypergraph, int label_count, std::vector<int> labels):
    n(n), m(m), degree(degree), hypergraph(hypergraph), label_count(label_count), labels(labels) {};

Eigen::VectorXd GraphSolver::infinity_subgradient(Eigen::VectorXd x) {
    bool * is_max = new bool[n];
    bool * is_min = new bool[n];
    Eigen::VectorXd gradient(n);
    gradient.setZero();
    for(int j = 0; j < m; j++) {
        if(hypergraph[j].size() == 0)
            continue;
        double ymin = INFINITY;
        double ymax = -INFINITY;
        for(auto it = hypergraph[j].begin(); it != hypergraph[j].end(); it++) {
            ymin = fmin(ymin, x(*it));
            ymax = fmax(ymax, x(*it));
        }
        double u = ymin + (ymax - ymin) / 2;
        double total_min_degree = 0;
        double total_max_degree = 0;
        for(auto it = hypergraph[j].begin(); it != hypergraph[j].end(); it++) {
            is_min[*it] = (ymin == x(*it));
            is_max[*it] = (ymax == x(*it));
            total_min_degree += is_min[*it] * degree[*it];
            total_max_degree += is_max[*it] * degree[*it];
        }
        for(long unsigned int i = 0; i < hypergraph[j].size(); i++) {
            int node = hypergraph[j][i];
            gradient(node) += (x(node) - u) * degree[node] * (is_min[node] / total_min_degree + is_max[node] / total_max_degree);
        }
    }
    delete[] is_max;
    delete[] is_min;
    return gradient;
}

Eigen::VectorXd GraphSolver::diffusion(const Eigen::VectorXd s, int T, double lambda, double h) {
    Eigen::VectorXd x(n);
    Eigen::VectorXd dx(n);
    Eigen::VectorXd solution(n);
    x.setZero();
    solution.setZero();
    for(int t = 0; t < T; t++) {
        Eigen::VectorXd gradient = infinity_subgradient(x);
        for(int i = 0; i < n; i++) {
            gradient(i) += lambda * degree[i] * x(i) - s(i);
        }
        switch(this->preconditionerType) {
            case 0:
                for(int i = 0; i < n; i++) dx(i)  = gradient(i) / degree[i];
                break;
            case 1:
                Eigen::VectorXd augmented(n+m);
                Eigen::VectorXd conditioned(n+m);
                augmented.setZero();
                augmented(Eigen::seq(0, n-1)) = gradient;
                conditioned = solver.solve(augmented);
                dx = conditioned(Eigen::seq(0, n-1));
                // std::cout << t << ' ';
                break;
        }
        x -= h * dx;
        solution += x;
    }
    for(int i = 0; i < n; i++) {
        solution(i) /= T;
    }
    return solution;
}

double GraphSolver::compute_fx(Eigen::VectorXd x, Eigen::VectorXd s, double lambda) {
    double fx = 0;
    for(int j = 0; j < m; j++) {
        if(hypergraph[j].size() == 0)
            continue;
        double ymin = INFINITY;
        double ymax = -INFINITY;
        for(auto it = hypergraph[j].begin(); it != hypergraph[j].end(); it++) {
            ymin = fmin(ymin, x(*it));
            ymax = fmax(ymax, x(*it));
        }
        fx += (ymax - ymin) * (ymax - ymin);
    }
    for(int i = 0; i < n; i++) {
        fx += lambda * degree[i] * (x(i) - s(i) / lambda / degree[i]) * (x(i) - s(i) / lambda / degree[i]);
    }
    return fx;
}

void GraphSolver::run_diffusions(std::string graph_name, int repeats, int T, double lambda, double h, int minimum_revealed, int step, int maximum_revealed) {
    Eigen::VectorXd seed(n);
    double * max_sol = new double[n];
    double fx;
    double * predicted_labels = new double[n];

    int * order = new int[n];
    for(int i = 0; i < n; i++) {
        order[i] = i;
    }

    srand(unsigned(time(0)));

    // Multiple repeats
    for(int repeat = 0; repeat < repeats; repeat++) {
        seed.setZero();
        // Run for different number of revealed
        std::random_shuffle(order, order+n);
        for(int revealed = minimum_revealed; revealed <= maximum_revealed; revealed += step) {
            const auto start{std::chrono::steady_clock::now()};
            fx = 0;
            for(int i = 0; i < n; i++) {
                predicted_labels[i] = 0;
                max_sol[i] = -INFINITY;
            }
            for(int r = 0; r < label_count; r++) {
                for(int i = 0; i < revealed; i++) {
                    int node = order[i];
                    seed(node) = lambda * (2 * (labels[node] == r) - 1);
                }
                auto solution = diffusion(seed, T, lambda, h);
                for(int i = 0; i < n; i++)
                    if(max_sol[i] < solution(i)) {
                        predicted_labels[i] = r;
                        max_sol[i] = solution(i);
                    }
                fx += compute_fx(solution, seed, lambda);
            }
            const auto end{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> time{end - start};
            double error = 0;
            for(int i = 0; i < n; i++)
                error += (predicted_labels[i] != labels[i]);
            // for(int t=0; t < T; t++)
            //    cout << t+1 << " " << fx[t] << endl;
            std::cout << graph_name << ",C++," << repeat << "," << revealed << "," << lambda << "," << time.count() << "," << error / n << "," << fx / label_count << "," << h << std::endl;
        }
    }
    delete[] max_sol;
    delete[] predicted_labels;
    delete[] order;
}
