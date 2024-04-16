#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cxxopts.hpp>

#include "diffusion.h"

int main(int argc, char* argv[]) {
    std::string graph_filename;
    std::string label_filename;
    std::string preconditioner;
    int T;
    double lambda;
    double h;
    int minimum_revealed;
    int step;
    int maximum_revealed;
    int repeats;
    int early_stopping;
    int schedule;
    int verbose;

    cxxopts::Options options(argv[0], "Run SemiSupervised Learning (SSL) Hypergraph Graph experiments.");
    options.add_options()
        ("f,graph_filename", "Hypergraph filename", cxxopts::value(graph_filename))
        ("s,label_filename", "Node label filename", cxxopts::value(label_filename))
        ("p,preconditioner", "Preconditioner choice of 'degree' and 'star'", cxxopts::value(preconditioner)->default_value("degree"))
        ("T", "Number of iterations", cxxopts::value(T)->default_value("300"))
        ("l,lambda", "Lambda value", cxxopts::value(lambda)->default_value("1.0"))
        ("h", "Step size", cxxopts::value(h)->default_value("0.1"))
        ("minimum_revealed", "Minimum number of labels revealed", cxxopts::value(minimum_revealed))
        ("step", "Number of additional labels revealed at each step", cxxopts::value(step))
        ("maximum_revealed", "Maximum number of labels revealed", cxxopts::value(maximum_revealed))
        ("r,repeats", "Minimum number of labels revealed", cxxopts::value(repeats))
        ("e,early_stopping", "Number of solution non-decreasing iterations before early stopping", cxxopts::value(early_stopping)->default_value("10"))
        ("schedule", "Step size schedule. 0 is for constant, 1 is for h / sqrt(t)", cxxopts::value(schedule)->default_value("0"))
        ("v,verbose", "Verbose mode. Prints out useful information");

    auto args = options.parse(argc, argv);
    verbose = args.count("verbose");
    
    GraphSolver G(graph_filename, label_filename, preconditioner, verbose);
    G.early_stopping = early_stopping;
    
    if(maximum_revealed > G.n) {
        perror("Cannot reveal more nodes than total number of nodes.");
        exit(1);
    }
    
    G.run_diffusions(graph_filename, repeats, T, lambda, h, minimum_revealed, step, maximum_revealed, schedule);
    return 0;
}
