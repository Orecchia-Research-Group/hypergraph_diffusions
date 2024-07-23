#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "diffusion.h"

namespace py = pybind11;

PYBIND11_MODULE(diffusion, m) {
    /*py::class_<MatrixReplacement>(m, "MatrixReplacement")
        .def(py::init<>())
        .def("rows", &MatrixReplacement::rows)
        .def("cols", &MatrixReplacement::cols)
        .def("attachMyMatrix", &MatrixReplacement::attachMyMatrix)
        .def("my_matrix", &MatrixReplacement::my_matrix);*/

    py::class_<GraphSolver>(m, "GraphSolver")
        .def(py::init<std::string, std::string, std::string, int>())
        .def(py::init<int, int, Eigen::VectorXd, std::vector<std::vector<int>>, int, std::vector<int>, int>())
        .def("infinity_subgradient", &GraphSolver::infinity_subgradient)
        .def("diffusion", &GraphSolver::diffusion)
        .def("compute_fx", &GraphSolver::compute_fx)
        .def("compute_error", &GraphSolver::compute_error)
        .def("run_diffusions", &GraphSolver::run_diffusions)
        .def_readwrite("degree", &GraphSolver::degree)
        .def_readwrite("hypergraph", &GraphSolver::hypergraph)
        .def_readwrite("labels", &GraphSolver::labels)
        .def_readwrite("label_count", &GraphSolver::label_count)
        .def_readwrite("weights", &GraphSolver::weights)
        .def_readwrite("hypergraph_node_weights", &GraphSolver::hypergraph_node_weights)
        .def_readwrite("center_id", &GraphSolver::center_id)
        .def_readwrite("early_stopping", &GraphSolver::early_stopping)
        .def_readwrite("verbose", &GraphSolver::verbose)
        ;
}
