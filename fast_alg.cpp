#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cxxopts.hpp>

#include "diffusion.h"

int main(int argc, char* argv[]) {
    cxxopts::Options options(argv[0], "Run SemiSupervised Learning (SSL) Hypergraph Graph experiments.");
    options.add_options()
        ("f,graph_filename", "Hypergraph filename", cxxopts::value<std::string>())
        ("s,label_filename", "Node label filename", cxxopts::value<std::string>())
        ("p,preconditioner", "Preconditioner choice of 'degree' and 'star'", cxxopts::value<std::string>()->default_value("degree"))
        ("T", "Number of iterations", cxxopts::value<int>()->default_value("300"))
        ("l,lambda", "Lambda value", cxxopts::value<double>()->default_value("1"))
        ("h", "Step size", cxxopts::value<double>()->default_value("0.1"))
        ("minimum_revealed", "Minimum number of labels revealed", cxxopts::value<int>())
        ("step", "Number of additional labels revealed at each step", cxxopts::value<int>())
        ("maximum_revealed", "Maximum number of labels revealed", cxxopts::value<int>())
        ("r,repeats", "Minimum number of labels revealed", cxxopts::value<int>());

    auto args = options.parse(argc, argv);
    
    std::string graph_filename = args["graph_filename"].as<std::string>();
    std::string label_filename = args["label_filename"].as<std::string>();
    std::string preconditioner = args["preconditioner"].as<std::string>();
    int T = args["T"].as<int>();
    double lambda = args["lambda"].as<double>();
    double h = args["h"].as<double>();
    int minimum_revealed = args["minimum_revealed"].as<int>();
    int step = args["step"].as<int>();
    int maximum_revealed = args["maximum_revealed"].as<int>();
    int repeats = args["repeats"].as<int>();
    
    GraphSolver G(graph_filename, label_filename, preconditioner);
    
    if(maximum_revealed > G.n) {
        perror("Cannot reveal more nodes than total number of nodes.");
        exit(1);
    }
    
    G.run_diffusions(graph_filename, repeats, T, lambda, h, minimum_revealed, step, maximum_revealed);
    return 0;
}
