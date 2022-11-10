function [sources,target] = LoadSrcTar(pathDatasets,indexTarget)
%LOADSRCTAR Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) pathDatasets: the path of datasets, e.g., 'D:\AEEEM\'; 
%   (2) indexTarget:  the index of target dataset (default indexTarget=1);
% OUTPUTS:
%   (1) sources: a cell array where each element is a 2D matrix;
%   (2) target:  a 2D matrix;
%

% Default value
if ~exist('indexTarget', 'var')||isempty(indexTarget)
    indexTarget = 1; 
end

% Find all arff/csv file
files = dir(fullfile(pathDatasets, '*.arff' ));
if isempty(files)
    files = dir(fullfile(pathDatasets, '*.csv' ));
end

% traverse arff file
dataNames = cell(1, length(files));
for j = 1 : length( files)
    dataNames{j} = files(j).name;
%     temp = files( j ).name;
%     dataNames{j} = regexprep(temp, '\.arff','');
end

% Load datasets
sources = cell(1, numel(dataNames));
for i = 1:numel(dataNames)
    try % arff
        file = java.io.File([pathDatasets,dataNames{i}]);  % create a Java File object (arff file is just a text file)
        loader = weka.core.converters.ArffLoader;  % create an ArffLoader object
        loader.setFile(file);  % using ArffLoader to load data in file .arff
        insts = loader.getDataSet; % get an Instances object
        insts.setClassIndex(insts.numAttributes()-1); %  set the index of class label
        [source,featureNamesSrc,~,stringVals,relationName] = weka2matlab(insts,[]); %{false,true}-->{0,1}
        class = insts.classAttribute;
        if (class.isNominal)
            if strcmp(stringVals{end}{1}, 'Y')||strcmpi(stringVals{end}{1}, 'Yes')||strcmpi(stringVals{end}{1}, 'true')||strcmpi(stringVals{end}{1}, 'T')||strcmpi(stringVals{end}{1}, 'Buggy')||strcmpi(stringVals{end}{1}, 'Defect')
                source(source(:,end)==0, end) = -1;
                source(source(:,end)==1, end) = 0;
                source(source(:,end)==-1, end) = 1;
            end
        end
    catch % csv
        file = readtable([pathDatasets,dataNames{i}]); % 'ReadVariableNames', true
        if iscell(table2array(file(end,end))) % 
            labelTemp = table2cell(file(:,end));  % buggy/clean, N/Y, T/F, TRUE/FALSE
            label = double(categorical(labelTemp));
            if ismember('buggy',unique(labelTemp))||ismember('Buggy',unique(labelTemp))||ismember('BUGGY',unique(labelTemp))
                label(label(:,end)==1, end) = 0;
                label(label(:,end)==2, end) = 1;
                label(label(:,end)==0, end) = 2;
            end
            label = label-1; % label of each instance is 0 or 1
        else % numeric - number of defect
            label = double(categorical(table2cell(file(:,end))));
        end
        j = 0;
        while(iscell(table2array(file(1,j+1))))
            j = j+1;
        end
        mat = table2array(file(:,(j+1):end-1));
        source = [mat,label];
    end
    
    source = [source(:, 1:end-1), double(source(:, end)>0)];
    
    % Remove duplicated instances
    source = unique(source,'rows','stable');
    % Remove instances having missing values
    [idx_r idx_c] = find(isnan(source));
    source(unique(idx_r),:) = [];
    sources{i} = source;
end

target = sources{indexTarget};
sources(indexTarget) = [];

end

