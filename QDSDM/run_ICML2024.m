function [] = run_ICML2024(input_folder, output_fodler, graph_name, ...
    minimum_samples, step, maximum_samples, repeats)
%RUN_ICML2024 Run experiments on Heim et al. and Li et al. to compare.

%% Read inputs

graph_filename = fullfile(input_folder, join([graph_name, ".mat"], ""));
load(graph_filename);

degree_vec = degree_stat_homo(incidence_list, parameter_homo_list, N, R);
a = 2 * a - 1;

%%semisupervised learning
% For all QDSFM problems, different approaches are to solve

% min_x ||x-bias_vec||_W^2 + sum_r [f_r(x)]^2

%lambda for QDSFM problems 
lambda_QDSFM = 0.1;
lambda_CE = 0.001;
%weighted matrix for norm 
W = lambda_QDSFM*degree_vec;
%number of iterations
T = 300;
record_dis = T+1;
functions = {@PDHG_QDSFM_cversion, @QRCDM_cversion, @QRCDM_AP_cversion, @Subgradient_QDSFM_cversion};
function_names = {'PDHG QDSFM', 'QRCDM', 'QRCDM AP', 'Subgradient QDSFM'};
iterations = [300, 300 * R, 300, 15000];
output_file = fopen(fullfile(output_fodler, sprintf("matlab_%s.csv", graph_name)), 'w');
fprintf(output_file, "Graph Name,Method,repeat,seeds,lambda,time,error,gap,conductance\n");
%% Run stuff
for i = 1:length(functions)
    func = functions{i};
    func_name = function_names{i};
    T = iterations(i);
    for it = 1:repeats
        reveal_index = randperm(N);
        bias_vec = zeros(1, N);
        for top = minimum_samples:step:maximum_samples
            revealed = reveal_index(1:top);
            bias_vec(revealed) = a(revealed) ./ degree_vec(revealed);
            tic;
            if strcmp(func_name, 'Subgradient QDSFM')
                x = clique_expansion(incidence_list, parameter_homo_list, bias_vec, lambda_CE, N, R);
            else
                [x, ~, final_gap] = func(incidence_list, parameter_homo_list, submodular_type, bias_vec, W, N, R, T,record_dis);
            end
            time = toc;
            [~, conductance, ~] = sign_invariant_performance_eval(incidence_list, parameter_homo_list, x, degree_vec, N, R);
            clustering_err = sum((2 * (x > 0) - 1) ~= a);
               
            fx = (x - bias_vec) .^ 2 * W';
            for j=1:R
                tempmin = min(x(incidence_list{j}));
                tempmax = max(x(incidence_list{j}));
                fx = fx + (tempmax - tempmin)^2 * parameter_homo_list{j};
            end
            fprintf(output_file, '%s,%s,%d,%d,%f,%f,%f,%.12f,%.10f\n', graph_name, func_name, it, top, lambda_QDSFM, time, clustering_err / N, fx, conductance);
        end
    end
end
fclose(output_file);
end

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