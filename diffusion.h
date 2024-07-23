#pragma once

#include <Eigen/Sparse>

/*class MatrixReplacement;

namespace Eigen {
    namespace internal {
        template<>
        struct traits<MatrixReplacement> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> > {};
    }
}

class MatrixReplacement : public Eigen::EigenBase<MatrixReplacement> {
    public:
        // Required typedefs, constants, and method:
        typedef double Scalar;
        typedef double RealScalar;
        typedef int StorageIndex;
        enum {
            ColsAtCompileTime = Eigen::Dynamic,
            MaxColsAtCompileTime = Eigen::Dynamic,
            IsRowMajor = false
        };

        Index rows() const;
        Index cols() const;

        template<typename Rhs>
        Eigen::VectorXd operator*(const Eigen::MatrixBase<Rhs>& x) const;
        void attachMyMatrix(const Eigen::SparseMatrix<double> &mat);
        const Eigen::SparseMatrix<double> my_matrix() const;

    private:
        const Eigen::SparseMatrix<double> *mp_mat;
};*/


class GraphSolver {
private:
    int repeat;
    int revealed;
    //  Eigen::BiCGSTAB<MatrixReplacement, Eigen::IdentityPreconditioner> solver;
    // Eigen::SparseMatrix<double> starLaplacian;
    // MatrixReplacement L;

    void read_hypergraph(std::string filename);
    void read_labels(std::string filename);
    Eigen::SparseMatrix<double> create_laplacian();
    inline double fmax(double a, double b);
    inline double fmin(double a, double b);

public:
    int n;                                          // Number of nodes
    int m;                                          // Number of (hyper)edges
    std::string graph_name;
    Eigen::VectorXd degree;                         // Node weights
    std::vector<std::vector<int>> hypergraph;       // Hypergraph edges
    std::vector<double> weights;                    // Hyperedge weights (optional)
    std::vector<std::vector<double> > hypergraph_node_weights;      // Used when nodes have different weights in the hyperedge (optional)
    std::vector<int> center_id;                     // If the hyperedges have a fixed node as the center it will be stored here (optional)
    int label_count;                                // Number of labels
    std::vector<int> labels;                        // Label for each node
    int early_stopping;
    int verbose;
    // double lambda;                                  // Balancing term for L2 regularizer
    // double h;                                       // Step size
    int preconditionerType;                         // Kind of preconditioner. 0 is degree, 1 is star

    GraphSolver(std::string graph_filename, std::string label_filename, std::string preconditioner, int verbose=0);
    GraphSolver(int n, int m, Eigen::VectorXd degree, std::vector<std::vector<int>> hypergraph, int label_count=0, std::vector<int> labels=std::vector<int>(), int verbose=0);
    Eigen::MatrixXd infinity_subgradient(Eigen::MatrixXd x);
    Eigen::MatrixXd diffusion(const Eigen::SparseMatrix<double> s, int T, double lambda, double h, int schedule=0);
    double compute_fx(Eigen::MatrixXd x, Eigen::SparseMatrix<double> s, double lambda, int t=1);
    double compute_error(Eigen::MatrixXd x);
    void run_diffusions(std::string graph_name, int repeats, int T, double lambda, double h, int minimum_revealed, int step, int maximum_revealed, int schedule=0);
};

