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


using namespace std;

void read_hypergraph(char filename[], int &n, int &m, vector<double> &degree, vector<vector<int> > &hypergraph) {
    int fmt;
    string line;
    ifstream input_file;
    input_file.open(filename);
    input_file >> m >> n >> fmt;
    getline(input_file, line);

    // Read hyperedges
    for(int i = 0; i < m; i++) {
        vector<int> hyperedge;
        int node;
        getline(input_file, line);
        istringstream iss(line);
        while(iss >> node) {
            hyperedge.push_back(node - 1);
        }
        hypergraph.push_back(hyperedge);
    }

    // Read degrees
    for(int i = 0; i < n; i++) {
        double d;
        input_file >> d;
        degree.push_back(d);
    }
    input_file.close();
}

int read_labels(char filename[], int n, vector<int> &labels) {
    ifstream label_file;
    label_file.open(filename);
    string line;
    getline(label_file, line);
    istringstream iss(line);
    int label;
    int label_count;
    map<int, int> label_map;
    for(label_count = 0; iss >> label; label_count++) {
        label_map[label] = label_count;
    }
    for(int i = 0; i < n; i++) {
        label_file >> label;
        labels.push_back(label_map[label]);
    }
    return label_count;
}


inline double fmax(double a, double b) {
    return a > b ? a : b;
}

inline double fmin(double a, double b) {
    return a < b ? a : b;
}

void diffusion(int n, int m, const vector<double> degree, const vector<vector<int>> hypergraph, double x[], double s[], int T, double lambda, double h, double solution[]) {
    double gradient[n];
    for(int t = 0; t < T; t++) {
        for(int i = 0; i < n; i++)
            gradient[i] = 0;
        for(int j = 0; j < m; j++) {
            if(hypergraph[j].size() == 0)
                continue;
            double ymin = INFINITY;
            double ymax = -INFINITY;
            for(auto it = hypergraph[j].begin(); it != hypergraph[j].end(); it++) {
                ymin = fmin(ymin, x[*it]);
                ymax = fmax(ymax, x[*it]);
            }
            double u = ymin + (ymax - ymin) / 2;
            double total_min_degree = 0;
            double total_max_degree = 0;
            bool is_max[n];
            bool is_min[n];
            for(auto it = hypergraph[j].begin(); it != hypergraph[j].end(); it++) {
                is_min[*it] = (ymin == x[*it]);
                is_max[*it] = (ymax == x[*it]);
                total_min_degree += is_min[*it] * degree[*it];
                total_max_degree += is_max[*it] * degree[*it];
            }
            for(long unsigned int i = 0; i < hypergraph[j].size(); i++) {
                int node = hypergraph[j][i];
                gradient[node] += (x[node] - u) * degree[node] * (is_min[node] / total_min_degree + is_max[node] / total_max_degree);
            }
        }
        for(int i = 0; i < n; i++) {
            gradient[i] += lambda * degree[i] * x[i] - s[i];
            x[i] -= h * gradient[i] / degree[i];
            solution[i] += x[i];
        }
    }
    for(int i = 0; i < n; i++) {
        solution[i] /= T;
    }
}

double compute_fx(int n, int m, vector<double> degree, vector<vector<int>> hypergraph, double x[], double s[], double lambda) {
    double fx = 0;
    for(int j = 0; j < m; j++) { 
        if(hypergraph[j].size() == 0)
            continue;
        double ymin = INFINITY;
        double ymax = -INFINITY;
        for(auto it = hypergraph[j].begin(); it != hypergraph[j].end(); it++) {
            ymin = fmin(ymin, x[*it]);
            ymax = fmax(ymax, x[*it]);
        }
        fx += (ymax - ymin) * (ymax - ymin);
    }
    for(int i = 0; i < n; i++) {
        fx += lambda * degree[i] * (x[i] - s[i] / lambda / degree[i]) * (x[i] - s[i] / lambda / degree[i]);
    }
    return fx;
}



int main(int argc, char* argv[]) {
    int n;
    int m;
    vector<vector<int>> hypergraph;
    vector<double> degree;
    vector<int> labels;
    if(argc != 10) {
        cerr << "Wrong number of arguments" << endl;
        return -1;
    }

    read_hypergraph(argv[1], n, m, degree, hypergraph);
    int label_count = read_labels(argv[2], n, labels);
    int T = stoi(argv[3]);
    double lambda = stod(argv[4]);
    double h = stod(argv[5]);
    int minimum_revealed = stoi(argv[6]);
    int step = stoi(argv[7]);
    int maximum_revealed = stoi(argv[8]);
    int repeats = stoi(argv[9]);

    double seed[label_count][n+2];
    double x[label_count][n+2];
    double solution[label_count][n+2];
    double fx;
    int predicted_labels[n+2];

    int order[n];
    for(int i = 0; i < n; i++) {
        order[i] = i;
    }

    srand(unsigned(time(0)));
    for(int r = 0; r < label_count; r++) {
        for(int i = 0; i < n; i++) {
            x[r][i] = seed[r][i] = solution[r][i] = 0;
        }
    }

    // Multiple repeats
    for(int repeat = 0; repeat < repeats; repeat++) {
        // Run for different number of revealed
        random_shuffle(order, order+n);
        for(int revealed = minimum_revealed; revealed <= maximum_revealed; revealed += step) {
            const auto start{chrono::steady_clock::now()};
            fx = 0;
            for(int i = 0; i < n; i++)
                predicted_labels[i] = 0;
            for(int r = 0; r < label_count; r++) {
                for(int i = 0; i < n; i++) {
                    x[r][i] = seed[r][i] = solution[r][i] = 0;
                }
                for(int i = 0; i < revealed; i++) {
                    int node = order[i];
                    seed[r][node] = lambda * (2 * (labels[node] == r) - 1);
                }
                diffusion(n, m, degree, hypergraph, x[r], seed[r], T, lambda, h, solution[r]);
                for(int i = 0; i < n; i++)
                    if(solution[predicted_labels[i]][i] < solution[r][i])
                        predicted_labels[i] = r;
                fx += compute_fx(n, m, degree, hypergraph, solution[r], seed[r], lambda);
            }
            const auto end{chrono::steady_clock::now()};
            const chrono::duration<double> time{end - start};
            double error = 0;
            for(int i = 0; i < n; i++)
                error += (predicted_labels[i] != labels[i]);
            // for(int t=0; t < T; t++)
            //    cout << t+1 << " " << fx[t] << endl;
            cout << argv[1] << ",C++," << repeat << "," << revealed << "," << lambda << "," << time.count() << "," << error / n << "," << fx / label_count << endl;
        }
    }
    return 0;
}
