clear;

%% generate data
% number of vertices
N = 1000;
% for each vertex, din many hyperedges within the same cluster are generated (typically <=1) 
din = 1; 
% for each vertex, dout many hyperedges within the same cluster are generated (typically <=1) 
dout = 1;
% size of hyperedge
K = 20;
% rate of observed label 
observed_rate= 0.002;
a = zeros(1,N);
% assign observed labels
a(randperm(N/2, N/2*observed_rate)) = 1;
a(randperm(N/2, N/2*observed_rate)+N/2) = -1;
% count the number of hyperedges
count = 0;
% incidence_list{i} contains the vertices that hyperedge i covers
incidence_list = {};
% parameter_homo_list{i} contains the weight of hyperedge i
parameter_homo_list = {};
% submodular_type{i} contains the submodular type of the hyperedge i:
% standard hypergraphs use 'h'
submodular_type = {};
% generate hyperedges within the first cluster
for k = 1:N/2,
    if rand(1)> din,
        continue;
    end
    count = count + 1;
    list = randperm(N/2, K-1);
    while sum(list==k)>0,
        list = randperm(N/2, K-1);
    end
    incidence_list{count} = [k list];
    parameter_homo_list{count} = 1;
    submodular_type{count} = 'h';
end
% generate hyperedges within the second cluster
for k = 1:N/2,
    if rand(1)> din,
        continue;
    end
    count = count + 1;
    list = randperm(N/2, K-1);
    while sum(list==k)>0,
        list = randperm(N/2, K-1);
    end
    incidence_list{count} = [k list]+N/2;
    parameter_homo_list{count} = 1;
    submodular_type{count} = 'h';
end
% generate hyperedges across the two clusters
for k = 1:(N*dout),
    count = count + 1;
    list = randperm(N, K);
    while min(list) > N/2 || max(list) <= N/2,
        list = randperm(N, K);
    end
    incidence_list{count} = list;
    parameter_homo_list{count} = 1;
    submodular_type{count} = 'h';
end

% Adela note: I think this just reverses dimensions 1xn to nx1
incidence_list = incidence_list(:);
parameter_homo_list = parameter_homo_list(:);
submodular_type = submodular_type(:);

% total number of hyperedges
R =count;

% save hypergraph to file for trials
filename = "example_hypergraph_file"; %string(datetime("now"));
%% Save necessary variables
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
save(filename,'N','R','a','incidence_list','parameter_homo_list','submodular_type')

