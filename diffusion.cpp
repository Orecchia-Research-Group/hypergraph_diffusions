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

    this->degree = Eigen::VectorXd(n);

    // Read degrees
    for(int i = 0; i < n; i++) {
        double d;
        input_file >> d;
        this->degree(i) = d;
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

GraphSolver::GraphSolver(std::string graph_filename, std::string label_filename, std::string preconditioner, int verbose):
        graph_name(graph_filename), early_stopping(-1), verbose(verbose) {
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

GraphSolver::GraphSolver(int n, int m, Eigen::VectorXd degree, std::vector<std::vector<int>> hypergraph, int label_count, std::vector<int> labels, int verbose):
    n(n), m(m), graph_name(""), degree(degree), hypergraph(hypergraph), label_count(label_count), labels(labels), early_stopping(-1), verbose(verbose) {};

Eigen::MatrixXd GraphSolver::infinity_subgradient(Eigen::MatrixXd x) {
    size_t d = x.rows();
    Eigen::MatrixXd gradient(d, n);
    gradient.setZero();
    for(int k = 0; k < x.rows(); k++) {
        bool * is_max = new bool[n];
        bool * is_min = new bool[n];
        for(int j = 0; j < m; j++) {
            if(hypergraph[j].size() == 0)
                continue;
            double ymin = INFINITY;
            double ymax = -INFINITY;
            for(auto it = hypergraph[j].begin(); it != hypergraph[j].end(); it++) {
                ymin = fmin(ymin, x(k, *it));
                ymax = fmax(ymax, x(k, *it));
            }
            double u = ymin + (ymax - ymin) / 2;
            double total_min_degree = 0;
            double total_max_degree = 0;
            for(auto it = hypergraph[j].begin(); it != hypergraph[j].end(); it++) {
                is_min[*it] = (ymin == x(k, *it));
                is_max[*it] = (ymax == x(k, *it));
                total_min_degree += is_min[*it] * degree(*it);
                total_max_degree += is_max[*it] * degree(*it);
            }
            for(long unsigned int i = 0; i < hypergraph[j].size(); i++) {
                int node = hypergraph[j][i];
                gradient(k, node) += (x(k, node) - u) * degree(node) * (is_min[node] / total_min_degree + is_max[node] / total_max_degree);
            }
        }
        delete[] is_max;
        delete[] is_min;
    }
    return gradient;
}

Eigen::MatrixXd GraphSolver::diffusion(const Eigen::SparseMatrix<double> s, int T, double lambda, double h, int schedule) {
    // schedule:
    //      0 --> h
    //      1 --> h / sqrt(t + 1)
    //      2 --> constant, divided by sqrt(2) at early stopping
    //      3 --> h / sqrt(t + 1) where h is divided by sqrt(2) every early stopping
    const auto start{std::chrono::steady_clock::now()};
    int function_stopping = early_stopping;
    int best_t = 1;
    int d = s.rows();
    int t;
    double step = h;
    double best_fx = INFINITY;
    int best_fx_unchanged = 0;
    Eigen::MatrixXd best_solution(d, n);
    Eigen::MatrixXd x(d, n);
    Eigen::MatrixXd dx(d, n);
    Eigen::MatrixXd solution(d, n);
    x.setZero();
    solution.setZero();
    for(t = 0; t < T; t++) {
        Eigen::MatrixXd gradient = infinity_subgradient(x);
        for(int j = 0; j < d; j++)
            for(int i = 0; i < n; i++) 
                gradient(j, i) += lambda * degree(i) * x(j, i) - s.coeff(j, i);
        switch(this->preconditionerType) {
            case 0:
                // dx = gradient.rowwise() / degree.transpose();
                for(int j = 0; j < d; j++)
                    for(int i = 0; i < n; i++)
                        dx(j, i) = gradient(j, i) / degree(i);
                break;
            case 1:
                for(int j = 0; j < d; j++) {
                    Eigen::VectorXd augmented(n+m);
                    Eigen::VectorXd conditioned(n+m);
                    augmented.setZero();
                    augmented(Eigen::seq(0, n-1)) = gradient.row(j);
                    conditioned = solver.solve(augmented);
                    dx.row(j) = conditioned(Eigen::seq(0, n-1));
                }
                break;
        }
        switch(schedule % 2) {
            case 0:
                break;
            case 1:
                step = h / sqrt(t + 1);
        }
        x -= step * dx;
        solution += x;
        
        double current_fx = this->compute_fx(x, s, lambda);
        double current_error = this->compute_error(x);
    
        double solution_fx = this->compute_fx(solution, s, lambda, t+1);
        double solution_error = this->compute_error(x);

        if(this->verbose > 0) {
            const auto current{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> time{current - start};
            std::cerr << graph_name << ",C++ x," << repeat << "," << revealed << "," << lambda << "," << time.count() << "," << t+1 << "," << schedule << "," << current_error << "," << current_fx / label_count << "," << h << std::endl;

            std::cerr << graph_name << ",C++ sol," << repeat << "," << revealed << "," << lambda << "," << time.count() << "," << t+1 << ","  << schedule << "," << solution_error << "," << solution_fx / label_count << "," << h << std::endl;
        }
        
        if(solution_fx < best_fx) {
            best_fx = solution_fx;
            best_solution = solution;
            best_fx_unchanged = 0;
            best_t = t + 1;
        }

        if((function_stopping > 0) && (best_fx_unchanged > function_stopping)) {
            if((schedule / 2) % 2) {        // Second LSB is 1
                best_fx_unchanged = 0;
                function_stopping *= sqrt(2);
                if(h < 1e-2)
                    break;
                h /= sqrt(2);
            }
            else
                break;
        }
        best_fx_unchanged++;
    }
    for(int j = 0; j < d; j++) {
        best_solution.row(j) /= best_t;
    }
    return best_solution;
}

double GraphSolver::compute_fx(Eigen::MatrixXd x, Eigen::SparseMatrix<double> s, double lambda, int t) {
    double fx = 0;
    for(int k = 0; k < s.rows(); k++) {
        for(int j = 0; j < m; j++) {
            if(hypergraph[j].size() == 0)
                continue;
            double ymin = INFINITY;
            double ymax = -INFINITY;
            for(auto it = hypergraph[j].begin(); it != hypergraph[j].end(); it++) {
                ymin = fmin(ymin, x(k, *it));
                ymax = fmax(ymax, x(k, *it));
            }
            fx += (ymax - ymin) * (ymax - ymin) / (t * t);
        }
        for(int i = 0; i < n; i++) {
            fx += lambda * degree(i) * (x(k, i) / t - s.coeff(k, i) / lambda / degree(i)) * (x(k, i) / t - s.coeff(k, i) / lambda / degree(i));
        }
    }
    return fx;
}

double GraphSolver::compute_error(Eigen::MatrixXd x) {
    double error = 0;
    double * max_sol = new double[n];
    double * predicted_labels = new double[n];
    for(int i = 0; i < n; i++) {
        predicted_labels[i] = 0;
        max_sol[i] = -INFINITY;
    }
    for(int r = 0; r < label_count; r++)
        for(int i = 0; i < n; i++)
            if(max_sol[i] < x(r, i)) {
                predicted_labels[i] = r;
                max_sol[i] = x(r, i);
            }
    for(int i = 0; i < n; i++)
        error += (predicted_labels[i] != labels[i]);
    delete[] predicted_labels;
    delete[] max_sol;
    return error / n;
}

void GraphSolver::run_diffusions(std::string graph_name, int repeats, int T, double lambda, double h, int minimum_revealed, int step, int maximum_revealed, int schedule) {
    Eigen::SparseMatrix<double> seed(label_count, n);
    double fx;

    // if(this->verbose > 0)
    //     std::cerr << "Graph Name,Method,repeat,seeds,lambda,time,iteration,error,fx,h" << std::endl;

    int * order = new int[n];
    for(int i = 0; i < n; i++) {
        order[i] = i;
    }

    srand(unsigned(time(0)));

    // Multiple repeats
    for(repeat = 0; repeat < repeats; repeat++) {
        seed.setZero();
        // Run for different number of revealed
        std::random_shuffle(order, order+n);
        for(revealed = minimum_revealed; revealed <= maximum_revealed; revealed += step) {
            const auto start{std::chrono::steady_clock::now()};
            for(int r = 0; r < label_count; r++) {
                for(int i = 0; i < revealed; i++) {
                    int node = order[i];
                    seed.coeffRef(r, node) = lambda * (2 * (labels[node] == r) - 1);
                }
            }
            auto solution = diffusion(seed, T, lambda, h, schedule);
            fx = compute_fx(solution, seed, lambda);
            const auto end{std::chrono::steady_clock::now()};
            const std::chrono::duration<double> time{end - start};
            double error = compute_error(solution);
            // for(int t=0; t < T; t++)
            //    cout << t+1 << " " << fx[t] << endl;
            std::cout << graph_name << ",C++," << repeat << "," << revealed << "," << lambda << "," << time.count() << "," << error << "," << fx / label_count << "," << h << std::endl;
        }
    }
    delete[] order;
}
