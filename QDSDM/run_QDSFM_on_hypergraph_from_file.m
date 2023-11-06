clear;
%% compile different approaches
% mex QRCDM_cversion.cpp
% mex QRCDM_AP_cversion.cpp
% mex PDHG_QDSFM_cversion.cpp
% mex Subgradient_QDSFM_cversion.cpp

%% load hypergraph objects from file
% % use hypergraph file constructed from Li, He, and Milekovic's own code
% filename = "example_hypergraph_file.mat";
% % 
% use hypergraph file converted from python using our files
filename = "example_python2matlab_hypergraph.mat";


load(filename)
% File must contain:
% 'N' number of vertices, a scalar double
% 'R' number of hyperedges, a scalar double
% 'a' vector of labels, a 1xN array of doubles
% 'incidence_list' an Rx1 cell array. incidence_list{i} is an array of 
%                   doubles specifying the vertices that hyperedge i covers                   
% 'parameter_homo_list' an Rx1 cell array. parameter_homo_list{i} is a 
%                   scalar double encoding the weight of hyperedge i
% 'submodular_type' an Rx1 cell array. submodular_type{i} encodes the edge 
%                   penalty for hyperedge i. "standard hyperedges" (i.e. I
%                   think ell-infinity penalty) is encoded as 'h'


% degrees of different vertices
degree_vec = degree_stat_homo(incidence_list, parameter_homo_list, N, R);
bias_vec = a./degree_vec;

%%semisupervised learning
% For all QDSFM problems, different approaches are to solve

% min_x ||x-bias_vec||_W^2 + sum_r [f_r(x)]^2

%lambda for QDSFM problems 
lambda_QDSFM = 0.02;
%weighted matrix for norm 
W = lambda_QDSFM*degree_vec;
%number of iterations
T = 300;
record_dis = T/30;
tic;
[x_PDHG, record, final_gap] = PDHG_QDSFM_cversion(incidence_list, parameter_homo_list, submodular_type, bias_vec, W, N, R, T,record_dis);
time_PDHG = toc;
%% For some reason, their assessment method seems to assume the first community will have positive sign.
% This can result in reports of poor performance even when clustering has
% produced very strong separation, just along the wrong sign. Therefore,
% consider the clustering error for both + and - x_hat
[clustering_err, conductance, thre] = sign_invariant_performance_eval(incidence_list, parameter_homo_list, x_PDHG, degree_vec, N, R);
gap_PDHG = record;
fprintf('PDHG-based QDSFM result:\n #incorrect clustered vertices:%d\n conductance:%f\n cputime:%f\n', clustering_err, conductance, time_PDHG);


%weighted matrix for norm 
W = lambda_QDSFM*degree_vec;
%number of iterations
T = 300*R;
record_dis = T/30;
tic;
[x_QRCDM, record, final_gap] = QRCDM_cversion(incidence_list, parameter_homo_list, submodular_type, bias_vec, W, N, R, T,record_dis);
time_QRCDM = toc;
[clustering_err, conductance, thre] = sign_invariant_performance_eval(incidence_list, parameter_homo_list, x_QRCDM, degree_vec, N, R);
fprintf('RCD-based QDSFM result:\n #incorrect clustered vertices:%d\n conductance:%f\n cputime:%f\n', clustering_err, conductance, time_QRCDM);

%weighted matrix for norm
W = lambda_QDSFM*degree_vec;
%number of iterations
T = 300;
record_dis = T/30;
tic;
[x_AP, record, final_gap] = QRCDM_AP_cversion(incidence_list, parameter_homo_list, submodular_type, bias_vec, W, N, R, T,record_dis);
time_AP = toc;
[clustering_err, conductance, thre] = sign_invariant_performance_eval(incidence_list, parameter_homo_list, x_AP, degree_vec, N, R);
%[clustering_err, conductance, thre] = result_analysis_homo(incidence_list, parameter_homo_list, x_AP, degree_vec, N, R);
fprintf('AP-based QDSFM result:\n #incorrect clustered vertices:%d\n conductance:%f\n cputime:%f\n', clustering_err, conductance, time_AP);

%weighted matrix for norm
W = lambda_QDSFM*degree_vec;
%number of iterations
T = 15000;
record_dis = T/30;
tic;
[x_Subgradient, record] = Subgradient_QDSFM_cversion(incidence_list, parameter_homo_list, submodular_type, bias_vec, W, N, R, T,record_dis);
time_Subgradient = toc;
[clustering_err, conductance, thre] = sign_invariant_performance_eval(incidence_list, parameter_homo_list, x_Subgradient, degree_vec, N, R);
%[clustering_err, conductance, thre] = result_analysis_homo(incidence_list, parameter_homo_list, x_Subgradient, degree_vec, N, R);
fprintf('Subgradint-based QDSFM result:\n #incorrect clustered vertices:%d\n conductance:%f\n cputime:%f\n', clustering_err, conductance, time_Subgradient);

%lambda for clique-expansion method 
lambda_CE = 0.001;
%weighted matrix for norm
W = lambda_CE*degree_vec;
tic;
x_CE = clique_expansion(incidence_list, parameter_homo_list, a, lambda_CE, N, R);
time_CE = toc;
[clustering_err, conduction, thre] = sign_invariant_performance_eval(incidence_list, parameter_homo_list, x_CE, degree_vec, N, R);
fprintf('clique-expansion result:\n #incorrect clustered vertices:%d\n conductance:%f\n cputime:%f\n', clustering_err, conductance, time_CE);

% For some reason, their assessment method seems to assume the first community will have positive sign.
% This can result in reports of poor performance even when clustering has
% produced very strong separation, just along the wrong sign. Therefore,
% consider the clustering error for both + and - x_hat
function [clustering_err, conductance, thre] = sign_invariant_performance_eval(incidence_list, parameter_homo_list, x_hat, degree_vec, N, R)
    [pos_clustering_err, pos_conductance, pos_thre] = result_analysis_homo(incidence_list, parameter_homo_list, x_hat, degree_vec, N, R);
    [neg_clustering_err, neg_conductance, neg_thre] = result_analysis_homo(incidence_list, parameter_homo_list, -x_hat, degree_vec, N, R);
    if pos_clustering_err<neg_clustering_err
        clustering_err = pos_clustering_err;
        conductance = pos_conductance;
        thre = pos_thre;
    else
        clustering_err = neg_clustering_err;
        conductance = neg_conductance;
        thre = neg_thre;
    end
end



