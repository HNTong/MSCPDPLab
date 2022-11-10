function probPos = BurakMHD(sources,target,k)
%BURAKMHD Summary of this function goes here
%   Detailed explanation goes here, 
% INPUTS:
%   (1) sources - a cell array where each element is a n_i*(d_i+1) sized
%   source dataset, the last column is the label where 1 - defective and 0
%   - nondefective;
%   (2) target  - a n_t*(d_t+1) array where d_t = d_i;
%   (3) k - Numebr of the nearest neighbors (10 by default);
% OUTPUTS:
%   probPos - A column vector denotes the predicted probability of being
%   positive class of target instances.
%
% Reference: Bhat N A , Farooq S U . An Improved Method for Training Data
%    Selection for Cross-Project Defect Prediction. Arabian Journal for
%    Science and Engineering, 2021:1-16.
%

warning('off');

if ~exist('k','var')||isempty(k)
    k = 10; 
end

nFeature = size(target, 2)-1;
nGroups = 20;
tars = cell(1, numel(sources));

%% Data transformation 
% (1) Min-max Normalization
psCell = cell(1, numel(sources)); 
for i=1:numel(sources)
    src = sources{i}; %
    srcX = src(:,1:end-1);
    [temp, ps] = mapminmax(srcX'); % [-1,1]
    psCell{i} = ps;
    srcX = temp';
    sources{i} = [srcX, src(:,end)];
end

tarX = target(:,1:end-1);
[temp, ps] = mapminmax(tarX'); % [-1,1]
tarX = temp';
target = [tarX, target(:,end)];


% (2) Discretization: equal-width discretization
for i=1:numel(sources)
    
    src = sources{i};
    
    % Discretize 'src' with equal-width unsupervised discretization method by using WEKA
    wekaSrc = mat2ARFF(src, 'classification'); % self-defined function 'mat2ARFF' 
    filterDiscSrc = javaObject('weka.filters.unsupervised.attribute.Discretize'); % Do not use class label 
    filterDiscSrc.setInputFormat(wekaSrc);
    filterDiscSrc.setOptions(weka.core.Utils.splitOptions(['-R 1-', num2str(nFeature), ' -B ', num2str(nGroups)])); % B - the number of bins to divide numeric attributes into.
    wekaSrcDisc = weka.filters.Filter.useFilter(wekaSrc, filterDiscSrc);
    [matSrc,featureNames,targetNDX,stringVals,relationName1] = weka2matlab(wekaSrcDisc,[]);
    sources{i} = matSrc; 
   
end

wekaTar = mat2ARFF(target, 'classification'); % self-defined function 'mat2ARFF'
filterDiscTar = javaObject('weka.filters.unsupervised.attribute.Discretize'); % Do not use class label
filterDiscTar.setInputFormat(wekaTar);
filterDiscTar.setOptions(weka.core.Utils.splitOptions(['-R 1-', num2str(nFeature), ' -B ', num2str(nGroups)])); % B - the number of bins to divide numeric attributes into.
wekaTarDisc = weka.filters.Filter.useFilter(wekaTar, filterDiscTar);
[target,featureNames,targetNDX,stringVals,relationName1] = weka2matlab(wekaTarDisc,[]);


%% Instance filtering
trainPosData = []; % Initialize 
trainNegData = [];
for i=1:numel(sources)
    src = sources{i};
    trainNegData = [trainNegData; src(src(:,end)==0,:)];
    trainPosData = [trainPosData; src(src(:,end)==1,:)];
end

indexPos = [];
if k<= size(trainPosData, 1)
    [D1, indexPos] = pdist2(trainPosData(:, 1:end-1), target(:, 1:end-1), 'hamming', 'Smallest', k); % k nearest neighbors of each target instance 
else
    [D1, indexPos] = pdist2(trainPosData(:, 1:end-1), target(:, 1:end-1), 'hamming', 'Smallest', size(trainPosData, 1)); %
end

indexNeg = [];
if k<= size(trainNegData, 1)
    [D2, indexNeg] = pdist2(trainNegData(:, 1:end-1), target(:, 1:end-1), 'hamming', 'Smallest', k); %
else
    [D2, indexNeg] = pdist2(trainNegData(:, 1:end-1), target(:, 1:end-1), 'hamming', 'Smallest', size(trainNegData, 1)); %
end

% Combination
trainData = [trainPosData(indexPos(:), :); trainNegData(indexNeg(:), :)];


%% Training
wekaTrain = mat2ARFF(trainData, 'classification'); % self-defined function 'mat2ARFF'
model = javaObject('weka.classifiers.bayes.NaiveBayes');
model.buildClassifier(wekaTrain);

%% Prediction
wekaTar = mat2ARFF(target, 'classification'); % self-defined function 'mat2ARFF'
classProbsPred = zeros(size(target,1), 2); % 2 - two classes
for i=0:(size(target,1)-1)
    classProbsPred(i+1,:) = model.distributionForInstance(wekaTar.instance(i));
end
probPos = classProbsPred(:,2); % the probability of being positive

end

function arff = mat2ARFF(data, type)
% Summary of this function goes here: 
%   Detailed explanation goes here
% INPUTS:
%   (1) data - a n*(d+1) matrix where the last column is independent variable.
%   (2) type - a string, 'regression' or 'classification'
% OUTPUTS:
%   arff     - an ARFF file

%javaaddpath('D:\Program Files\Weka-3-8-4\weka.jar');
if ~exist('type','var')||isempty(type)
    type = 'regression';
end
label = cell(size(data,1),1);
if strcmp(type, 'classification')
    temp = data(:,end);
    for j=1:size(data,1)
        if (temp(j)==1)
            label{j} = 'true';
        else
            label{j} = 'false';
        end
    end %{0,1}--> {false, true}
else 
    label = num2cell(data(:,end));
end
featureNames = cell(size(data,2),1);
for j=1:(size(data,2)-1)
    featureNames{j} = ['X', num2str(j)];
end
featureNames{size(data,2)} = 'Defect';
arff = matlab2weka('data', featureNames, [num2cell(data(:,1:end-1)), label]);
end
