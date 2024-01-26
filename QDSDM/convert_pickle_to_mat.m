function [] = convert_pickle_to_mat(graph_name, input_folder, output_folder)
%CONVERT_PICKLE_TO_MAT Reads a pickle file of a hypergraph and saves the
%same hypergraph as a .mat file
% File must contain:
% 'N' number of vertices, a scalar double
% 'R' number of hyperedges, a scalar double
% 'a' vector of labels, a 1xN array of doubles
% 'incidence_list' an Rx1 cell array. incidence_list{i} is an array of 
%                   doubles specifying the vertices that hyperedge i covers                   
% 'parameter_homo_list' an Rx1 cell array. parameter_homo_list{i} is a 
%                   scalar double encoding the weight of hyperedge i
% 'k' the cardinality of each hyperedge, assumed to be constant
    pickle = py.importlib.import_module('pickle');
    input_filename = fullfile(input_folder, join([graph_name, ".pickle"], ""));
    fh = py.open(input_filename, 'rb');
    P = pickle.load(fh);
    fh.close();
    
    % unpack the values and convert to MATLAB objects
    N = double(P(1));
    R = double(P(2));
    k = unpack_python_list(P(3), R, false);
    % unpack labels WITHOUT re-indexing: labels are 0-1 and not influenced by 1
    % versus 0-indexing
    a = unpack_python_list(P(4), N, false);
    
    % unpack weights WITHOUT re-indexing
    parameter_homo_array = unpack_python_list(P(6),R, false);
    % convert to an Rx1 cell array
    parameter_homo_list = {};
    for index = 1:R
        parameter_homo_list{index} = parameter_homo_array(index);
    end
    % reshape from 1xR to Rx1
    parameter_homo_list = parameter_homo_list(:);
    
    % unpack incidence lists WITH re-indexing: add 1 to each vertex index
    incidence_list = unpack_nested_python_list(P(5),R,k,true);
    % reshape from 1xN to Nx1
    incidence_list = incidence_list(:);
    
    % Assume standard edge penalties for each hyperedge
    % 'submodular_type' an Rx1 cell array. submodular_type{i} encodes the edge 
    %                   penalty for hyperedge i. "standard hyperedges" (i.e. I
    %                   think ell-infinity penalty) is encoded as 'h'
    submodular_type = {};
    for index = 1:R
        submodular_type{index} = 'h';
    end
    % reshape from 1xR to Rx1
    submodular_type = submodular_type(:);
    
    % save hypergraph to file for trials
    output_filename = fullfile(output_folder, join(graph_name, ".mat"));
    save(output_filename,'N','R','a','incidence_list','parameter_homo_list','submodular_type')
    
    % helper function for unpacking python list of integers into array
    % potentially re-index, shifting from 0-indexed to 1-indexed
    function array = unpack_python_list(python_list,list_length,reindex)
        python_list = cell(python_list);
        python_list = python_list{:};
        array = zeros(1,list_length);
        for index = 1:list_length
            if reindex
                array(index) = double(python_list(index))+1;
            elseif not(reindex)
                array(index) = double(python_list(index));
            end
        end
    end
    
    % helper function for unpacking python nested lists of integers
    % potentially re-index, shifting from 0-indexed to 1-indexed
    function nested_cell = unpack_nested_python_list(nested_list,outer_length,inner_length,reindex)
        if length(inner_length) == 1
            inner_length = ones(outer_length) * inner_length;
        end
        if length(inner_length) ~= outer_length
            error("Inner and outer length mismatch.")
        end
        nested_list = cell(nested_list);
        nested_list = nested_list{:};
        nested_cell = {};
        for index = 1:outer_length
            nested_cell{index} = unpack_python_list(nested_list(index),inner_length(index),reindex);
        end
    end



end