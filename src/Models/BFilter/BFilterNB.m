function probPos = BFilterNB(sources, target, K)
%BFILTERNB Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) source - a m*(d1+1) matrix where the last column is the label
%   belonging to {0,1} where 1 denotes the positive class (minority) or a
%   cell array consisting of some m*(d1+1) sized matrix;
%   (2) target - a n*(d2+1) matrix (d2>=d1) where the last column is the label
%   belonging to {0,1} where 1 denotes the positive class (minority);
% OUTPUTS:
%   probPos - A column vector denotes the predicted probability of being
%   positive class of target instances.
%
% Reference: Turhan, B., Menzies, T., Bener, A. B., & Di Stefano, J.
%            On the relative value of cross-company and within-company data
%            for defect prediction. Empirical Software Engineering, 14(5),2009:540-578.
%


warning('off');
if ~exist('K','var')||isempty(K)
    K = 10; % the number of nearest-neighbors.
end

filterB = Burakfilter(sources, target, K);

%% Training on the source data with NB
label = cell(size(filterB,1),1);
for i=1:size(filterB,1)
    if filterB(i,end)==0
        label{i} = 'No';
    else
        label{i} = 'Yes';
    end
end
feaClaNames = cell(1, size(filterB,2));
for j0=1:(size(filterB,2)-1)
    feaClaNames{j0} = ['X',num2str(j0)];
end
feaClaNames{end} = 'Defect';
sourceARFF = matlab2weka('data', feaClaNames,  [num2cell(filterB(:,1:end-1)), label]);
sourceARFF.setClassIndex(sourceARFF.numAttributes()-1);
model = javaObject('weka.classifiers.bayes.NaiveBayes');
model.buildClassifier(sourceARFF);

%% Prediction on the target data
labelTar = cell(size(target,1),1);
for i=1:size(target,1)
    if target(i,end)==0
        labelTar{i} = 'No';
    else
        labelTar{i} = 'Yes';
    end
end
targetARFF = matlab2weka('data', feaClaNames,  [num2cell(target(:,1:end-1)), labelTar]);
targetARFF.setClassIndex(targetARFF.numAttributes()-1);

% pred = zeros(m,1);
classProbsPred = zeros(size(target,1),2); % 2 - two classes
for i = 0:(size(target,1)-1)
%     pred(i+1) = model.classifyInstance(train.instance(i));
    classProbsPred(i+1,:) = model.distributionForInstance(targetARFF.instance(i));
end
probPos = classProbsPred(:,2); % the probability of being positive

end

