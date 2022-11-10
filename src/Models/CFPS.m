function probPos = CFPS(sources, target, topK, classifier, measureApp)
%CFPS Summary of this function goes here: Implement the MSCPDP model -- CFPS
%   Detailed explanation goes here, 
% INPUTS:
%   (1) sources - a cell array where each element is a n_i*(d_i+1) sized
%   source dataset, the last column is the label where 1 - defective and 0
%   - nondefective;
%   (2) target  - a n_t*(d_t+1) array where d_t = d_i;
%   (3) classifier - a string from {'NaiveBayes','functions.Logistic','trees.J48',
%                   'trees.RandomForest','functions.SMO'}.
%   (4) measureApp  -  a string belonging to {'AUC','F-measure'};
% OUTPTS:
%   probPos - A column vector denotes the predicted probability of being
%   positive class of target instances.
%
% Reference: [1] Hongbin Sun and Junqi Li and Heli Sun and Liang He, CFPS:
%     Collaborative filtering based source projects selection for cross-project
%     defect prediction. Applied Soft Computing. 2021(99): 106940.
%     https://doi.org/10.1016/j.asoc.2020.106940.
%

%% Default value used in the reference paper
if ~exist('topK','var')||isempty(topK)
   topK = 1; 
end
if ~exist('classifier','var')||isempty(classifier)
    classifier = 'RandomForest'; 
end
if ~exist('measureApp','var')||isempty(measureApp)
   measureApp = 'AUC'; 
end


%% GetSimiScores: Algo.1
M = numel(sources);
simiScores = zeros(1, M);
NP = [mean(target(:,1:end-1)), std(target(:,1:end-1))];
for i = 1:M
    src = sources{i};
    HP_i = [mean(src(:,1:end-1)), std(src(:,1:end-1))];
    simiScores(i) = 1/(1+sqrt(sum((HP_i-NP).^2)));
end


%% GetAppScores: Algo.2
appScores = zeros(M, M);

for i = 1:M
    test = sources{i}; % 
    testData = mat2ARFF(test, 'classification'); % Tranform mat into ARFF by calling the self-defined function
    
    for j=1:M
        
        if i==j
            continue; % next loop
        end
        
        wekaSrcData = mat2ARFF(sources{j}, 'classification'); % Call the self-defined function
        
        
        type = ['functions.', classifier];
        if ~strcmp(classifier, 'SMO')
            if strcmp(classifier, 'Logistics')
                type = ['functions.', classifier];
            elseif strcmp(classifier, 'J48') || strcmp(classifier, 'RandomForest')
                type = ['trees.', classifier];
            else
                type = ['bayes', classifier];
            end
        end
        wekaClassifier = javaObject(['weka.classifiers.',type]);
        wekaClassifier.buildClassifier(wekaSrcData);
        
        classProbs = [size(test, 1), 2];
        for t=0:testData.numInstances-1
            classProbs(t+1,:) = (wekaClassifier.distributionForInstance(testData.instance(t)))';
        end
        [prob,predLabel] = max(classProbs,[],2); % begin from 1
        predLabel = predLabel - 1;
        [F1, AUC] = PerformanceSimply(test(:,end), classProbs(:,2));
        
        appScores(i,j) = AUC;
        if strcmp(measureApp, 'F1')
            appScores(i,j) = F1;
        end
    end
end

%% GetRecScores: Algo.3
recScores = zeros(1,numel(sources));
for i=1:numel(simiScores)
    for j=1:numel(appScores,1)
        if j~=i
            recScores(i) = recScores(i) + simiScores(j)*appScores(j,i);
        end 
    end
end


%% Training: Sun et.al. did not specify how to use the recommended source datasets to perform CPDP. Hence, the famous CPDP method 'TCA+' is used.
trainData = [];
[~, idx] = sort(recScores, 'descend');
for i=1:topK
    trainData = [trainData; sources{idx(i)}];
end

%% Prediction
probPos = TCAplus(trainData,target);

end


function arff = mat2ARFF(data, type)
% Summary of this function goes here: 
%   Detailed explanation goes here
% INPUTS:
%   (1) data - a n*(d+1) matrix where the last column is independent variable.
%   (2) type - a string, 'regression' or 'classification'
% OUTPUTS:
%   arff     - an ARFF file


% javaaddpath('D:\Program Files\Weka-3-8-4\weka.jar');

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


function [F1, AUC] = PerformanceSimply(actual_label, probPos)
% function [ PD,PF,Precision, F1,AUC,Accuracy,G_measure,MCC ] = Performance( actual_label,predict_label, probPos)
%PERFORMANCE Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) actual_label - The actual label, a column vetor, each row is an instance's class label.
%   (2) predict_label - The predicted label, a column vetor, each row is an instance label.
%   (3) probPos - The probability of being predicted as postive class.
% OUTPUTS:
%   PF,PF,..,MCC - A total of eight performance measures.

assert(numel(unique(actual_label)) > 1, 'Please ensure that ''actual_label'' includes two or more different labels.'); % 
assert(length(actual_label)==length(probPos), 'Two input parameters must have the same size.');


predict_label = double(probPos>=0.5);

cf=confusionmat(actual_label,predict_label);
TP=cf(2,2);
TN=cf(1,1);
FP=cf(1,2);
FN=cf(2,1);

PD=TP/(TP+FN);
PF=FP/(FP+TN);
Precision=TP/(TP+FP);
F1=2*Precision*PD/(Precision+PD);
[X,Y,T,AUC]=perfcurve(actual_label, probPos, '1');% 
end

