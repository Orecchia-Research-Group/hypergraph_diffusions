#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <iterator>
#include <chrono>
#include <ctime>

#include <Eigen/Dense>
#include <Eigen/Sparse>

class GraphSolver {
    private:
        void read_hypergraph(char filename[]) {
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

        void read_labels(char filename[]) {
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

        inline double fmax(double a, double b) {
            return a > b ? a : b;
        }

        inline double fmin(double a, double b) {
            return a < b ? a : b;
        }

    public:
        int n;                                      // Number of nodes
        int m;                                      // Number of (hyper)edges
        std::vector<double> degree;                 // Node weights
        std::vector<std::vector<int>> hypergraph;   // Hypergraph edges
        int label_count;                            // Number of labels
        std::vector<int> labels;                    // Label for each node
        double lambda;                              // Balancing term for L2 regularizer
        double h;                                   // Step size

        Eigen::VectorXd solution;                   // Most recent solution

        //GraphSolver(char graph_filename[], char label_filename[]=NULL);
        //GraphSolver(int n, int m, std::vector<double> degree, std::vector<std::vector<int>> hypergraph, int label_count=0, std::vector<int> labels=std::vector<int>());


        GraphSolver(char graph_filename[], char label_filename[]=NULL) {
            read_hypergraph(graph_filename);
            if(label_filename) {
                read_labels(label_filename);
            }
        }

        GraphSolver(int n, int m, std::vector<double> degree, std::vector<std::vector<int>> hypergraph, int label_count=0, std::vector<int> labels=std::vector<int>()): 
            n(n), m(m), degree(degree), hypergraph(hypergraph), label_count(label_count), labels(labels) {};

        Eigen::VectorXd infinity_subgradient(Eigen::VectorXd x) {
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

        Eigen::VectorXd diffusion(const Eigen::VectorXd s, int T, double lambda, double h) {
            Eigen::VectorXd x(n);
            Eigen::VectorXd solution(n);
            x.setZero();
            solution.setZero();
            for(int t = 0; t < T; t++) {
                Eigen::VectorXd gradient = infinity_subgradient(x);
                for(int i = 0; i < n; i++) {
                    gradient(i) += lambda * degree[i] * x(i) - s(i);
                    x(i) -= h * gradient(i) / degree[i];
                    solution(i) += x(i);
                }
            }
            for(int i = 0; i < n; i++) {
                solution(i) /= T;
            }
            return solution;
        }

        double compute_fx(Eigen::VectorXd x, Eigen::VectorXd s, double lambda) {
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

        void run_diffusions(std::string graph_name, int repeats, int T, double lambda, double h, int minimum_revealed, int step, int maximum_revealed) {
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
                    std::cout << graph_name << ",C++," << repeat << "," << revealed << "," << lambda << "," << time.count() << "," << error / n << "," << fx / label_count << std::endl;
                }
            }
            delete[] max_sol;
            delete[] predicted_labels;
            delete[] order;
        }
};

int main(int argc, char* argv[]) {
    if(argc != 10) {
        std::cerr << "Wrong number of arguments" << std::endl;
        return -1;
    }
    GraphSolver G(argv[1], argv[2]);    
    int T = std::stoi(argv[3]);
    double lambda = std::stod(argv[4]);
    double h = std::stod(argv[5]);
    int minimum_revealed = std::stoi(argv[6]);
    int step = std::stoi(argv[7]);
    int maximum_revealed = std::stoi(argv[8]);
    if(maximum_revealed > G.n) {
        perror("Cannot reveal more nodes than total number of nodes.");
        exit(1);
    }
    int repeats = std::stoi(argv[9]);
    G.run_diffusions(argv[1], repeats, T, lambda, h, minimum_revealed, step, maximum_revealed);

    return 0;
}
