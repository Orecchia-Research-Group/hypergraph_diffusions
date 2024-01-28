function [] = run_ICML2024(input_folder, output_fodler, graph_name, ...
    minimum_samples, step, maximum_samples, lambdas, repeats)
%RUN_ICML2024 Run experiments on Heim et al. and Li et al. to compare.

%% Read inputs

graph_filename = fullfile(input_folder, join([graph_name, ".mat"], ""));
load(graph_filename);

degree_vec = degree_stat_homo(incidence_list, parameter_homo_list, N, R);

% Prepare 1 vs all labels
unique_labels = unique(a);
num_labels = length(unique_labels);
true_labels = zeros(1, N);
seeds = zeros(num_labels, N);
for i=1:num_labels
    ind = (a == unique_labels(i));
    true_labels(ind) = i;
    seeds(i, :) = 2 * ind - 1;
end

%%semisupervised learning
% For all QDSFM problems, different approaches are to solve

% min_x ||x-bias_vec||_W^2 + sum_r [f_r(x)]^2

%number of iterations
functions = {@PDHG_QDSFM_cversion, @QRCDM_cversion, @QRCDM_AP_cversion, @Subgradient_QDSFM_cversion};
function_names = {'PDHG QDSFM', 'QRCDM', 'QRCDM AP', 'Subgradient QDSFM'};
iterations = [300, 300 * R, 300, 15000];
output_file = fopen(fullfile(output_fodler, sprintf("matlab_%s.csv", graph_name)), 'w');
fprintf(output_file, "Graph Name,Method,repeat,seeds,lambda,time,error,fx\n");
%% Run stuff
for lam=lambdas
    W = lam * degree_vec;
    for i = 1:length(functions)
        func = functions{i};
        func_name = function_names{i};
        T = iterations(i);
        for it = 1:repeats
            reveal_index = randperm(N);
            bias_vec = zeros(num_labels, N);
            for top = minimum_samples:step:maximum_samples
                revealed = reveal_index(1:top);
                bias_vec(:, revealed) = seeds(:, revealed) ./ degree_vec(revealed);
                x = zeros(num_labels, N);
                for c=1:num_labels
                    tic;
                    if strcmp(func_name, 'Subgradient QDSFM')
                        x(c, :) = clique_expansion(incidence_list, parameter_homo_list, bias_vec(c, :), lam, N, R);
                    else
                        [x(c, :), ~, final_gap] = func(incidence_list, parameter_homo_list, submodular_type, bias_vec(c, :), W, N, R, T,T+1);
                    end
                    time = toc;
                end
                % [~, conductance, ~] = sign_invariant_performance_eval(incidence_list, parameter_homo_list, x, degree_vec, N, R);
                [~, predicted_label] = max(x, [], 1);
                clustering_err = sum(predicted_label ~= true_labels);
                   
                fx = sum((x - bias_vec) .^ 2 * W');
                for j=1:R
                    tempmin = min(x(:, incidence_list{j}), [], 2);
                    tempmax = max(x(:, incidence_list{j}), [], 2);
                    fx = fx + sum((tempmax - tempmin).^2 * parameter_homo_list{j});
                end
                fx = fx / num_labels;
                fprintf(output_file, '%s,%s,%d,%d,%f,%f,%f,%.12f\n', graph_name, func_name, it, top, lam, time, clustering_err / N, fx);
            end
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