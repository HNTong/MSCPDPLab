function [trainData, target] = Burakfilter(sources, target, K, boolUseLogFilter)
%BARUKFILTER Summary of this function goes here
%   Function: to obtain filtered training data from source data by using Burak's Filter
%
% INPUTS:
%   (1) sources - [S1;S2;...;SN]; each row of normalized Si corresponds to a observation and the last column is label {0,1}.
%   (2) target - normalized test data from target project; each row is
%       a observation and the last column is label {0,1}. The label will not be
%       used in this function.
%   (3) K - the number of neighbors of each test instance.
% OUTPUTS:
%   filtered_TDS - 
%
% Reference: 
% 


%% Default value
if ~exist('K', 'var')||isempty(K)
    K = 10;
end
if ~exist('boolUseLogFilter', 'var')||isempty(boolUseLogFilter)
    boolUseLogFilter = false;
end

%% Preprocessing
EPSILON = 1e-6;
targetX = target(:,1:end-1);
targetX(targetX<=EPSILON)=EPSILON;
target(:,1:end-1) = targetX;
if boolUseLogFilter
    target(:,1:end-1)=log(target(:,1:end-1)); % log, P12 'Therefore, we replace all numeric values with a "log-filter"...'
end


%% Transform cell to a mat array
src=[];
for i=1:numel(sources)
    if iscell(sources) %
        source = sources{i};
        % Take logorithm (do this only all values are positive) - either log(x) or log10(x), here we use natural logrithm.
        % 0 -> 1e-3
        
        srcX = source(:,1:end-1);
        srcX(srcX<=EPSILON)=EPSILON;
        source(:,1:end-1) = srcX;
        if boolUseLogFilter
            source(:,1:end-1)=log(source(:,1:end-1)); % log(x)£¬log10(x)
        end
        src=[src; source];
    else
        source = sources;
        srcX = source(:,1:end-1);
        srcX(srcX<=EPSILON)=EPSILON;
        source(:,1:end-1) = srcX;
        if boolUseLogFilter
            source(:,1:end-1)=log(source(:,1:end-1));
        end
        src = source;
    end
end

index=[];
if K <= size(target, 1)
    [D2, index] = pdist2(src(:, 1:end-1), target(:, 1:end-1),'euclidean','Smallest', K); % Return a K*size(target,1), i.e., each column denotes the indexes of K nearest neighbors for each target instance 
else
    [D2, index] = pdist2(src(:, 1:end-1), target(:, 1:end-1), 'euclidean', 'Smallest', size(src, 1)); %
end

trainData=src(unique(index(:)),:); % Remove the duplicated instances.
idx = randperm(size(trainData,1), round(size(trainData,1)*0.9)); 
trainData = trainData(idx,:);

% trainData=src(index(:),:); 

end


