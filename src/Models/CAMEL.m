function probPos = CAMEL(sources, target, alpha)
%CAMEL Summary of this function goes here: 
%   Detailed explanation goes here
% INPUTS:
%   (1) sources  - A K-sized cell array, each element is a N_S_i-by-(d+1) source dataset (i=1,2,...,K) where 
%                     the last column is the label (1 - defecttive and 0 - nondefective).
%   (2) target   - A n_t-by-(d+1) array where the last column is the label (1 - defecttive and 0 - nondefective).
%
% OUTPUTS:
%   probPos - A column vector denotes the predicted probability of being
%   positive class of target instances.
%
% Reference: [1] E. Kim, J. Baik, and D. Ryu, "Heterogeneous defect prediction
%     through correlation-based selection of multiple source projects and
%     ensemble learning" in 2021 IEEE 21st International Conference on
%     Software Quality, Reliability and Security (QRS), 2021, pp. 503–513.
%
% Writen by Haonan Tong (hntong@bjtu.edu.cn)
% 

warning('off');

%% Hyper-parameters: According to the setting in [1]
if ~exist('alpha', 'var')||isempty(alpha)
    alpha = 0.95; 
end
R = 0.6;
selectRatio = 0.15;

%% Select source metrics: Top X metrics based on gain ratio
for d = 1:numel(sources)
    sources{d} = gainRatioFun(sources{d}, selectRatio); % Call self-defined function 'selectRatio()'.
end

%% Matching source and target datasets
N = numel(sources); % Number of source datasets
P = cell(1,N); % Initialization
Q = cell(1,N);
Y = cell(1,N);
f = zeros(1,N);
for i=1:N
    source = sources{i};
    Y{i} = source(:,end);
    [newSrcX, newTarX, isMatched] = KSMatch(source(:,1:end-1), target(:,1:end-1)); % Call self-defined function 'KSMatch()'.
    P{i} = newSrcX;
    Q{i} = newTarX;
    if isMatched
        corScore = corScoreFun(newSrcX', newTarX'); % Call self-defined function 'corScoreFun()'.
        f(i) = alpha*corScore + (1-alpha)*log(size(newSrcX,1)/size(newTarX,1));
    else
        f(i) = 0;
    end
end
[f, idx] = sort(f, 'descend');

% Resort P,Q, and Y based on 'idx'
P = P(1, idx);
Q = Q(1, idx);
Y = Y(1, idx);

P(find(f==0)) = []; % 
Q(find(f==0)) = [];

P((round(R*numel(P)))+1:end) = []; % Keep the first R datasets
Q((round(R*numel(Q)))+1:end) = [];

%% Prediction
probPosMatr = zeros(size(target,1), numel(P));
for i=1:numel(P)
    sourceX = P{i};
    targetX = Q{i};
    model = glmfit(sourceX,Y{i},'binomial', 'link', 'logit');
    probPosMatr(:,i) = glmval(model,targetX, 'logit');
end

probPos = mean(probPosMatr,2);

end


function simSrc = gainRatioFun(source, selectRatio)
%GAINRATIOFUN Summary of this function goes here: Conduct feature selection based on gain ratio. 
%   Detailed explanation goes here
% INPUTS:
%   (1) source - A n_sample*(d+1) array, the last column is the label (1 - defecttive and 0 - nondefective).
%   (2) selectRatio - A number belonging to [0,1].
%
% OUTPUTS:
%   simSrc - A n_sampel*(d1+1) array where d1<=d.

label = cell(size(source,1),1);
temp = source(:,end);
for i=1:size(source,1) % each sample
    if (temp(i)==1)
        label{i} = 'true';
    else
        label{i} = 'false';
    end
end %{0,1}--> {false, true}
featureNames = cell(size(source,2),1);
for i=1:(size(source,2)-1) % each metrics
    featureNames{i} = ['X', num2str(i)];
end
featureNames{end}='Defect';
insts = matlab2weka('myARFF', featureNames, [num2cell(source(:,1:end-1)), label]);
gainRatio = javaObject('weka.attributeSelection.GainRatioAttributeEval'); % 'GainRatioAttributeEval','ReliefFAttributeEval'
attrSelector = javaObject('weka.attributeSelection.AttributeSelection');
searchMethod = weka.attributeSelection.Ranker(); % weka.attributeSelection.BestFirst();
% filter.setOptions(weka.core.Utils.splitOptions(['-R ',num2str(insts.numAttributes())]));
% filter.setInputFormat(insts);
attrSelector.setEvaluator(gainRatio);
attrSelector.setSearch(searchMethod);
attrSelector.SelectAttributes(insts);
selAttrIndex = attrSelector.selectedAttributes(); % 返回一个列向量(特征+因变量的索引)(注意：索引遵循weka标准，即从零开始)
source = source(:, selAttrIndex+1); % adjust the position of all features
sourceX = source(:, 1:floor((size(source,2)-1)*selectRatio));
simSrc= [sourceX, source(:,end)];
end


function corScore = corScoreFun(X_S, X_T)
%CORSCOREFUN Summary of this function goes here: 
%   Detailed explanation goes here
% INPUTS:
%   (1) X_S  - A d*n_s array without label.
%   (2) X_T  - A d*n_t array without label.
% OUTPUTS:

m = size(X_S, 1);  % number of metrics
Ns = size(X_S, 2); % number of source instances
Nt = size(X_T, 2); % number of target instances
C = zeros(1, 100);
for k=1:100  % According to the setting in [1]
    C_i = zeros(1,m);
    for i=1:m
        S_i = X_S(i,:);
        T_i = X_T(i,:);
        if Ns>Nt
            S_i = sort(S_i(1, randperm(Ns, Nt)));
            T_i = sort(T_i);
        else
            S_i = sort(S_i);
            T_i = sort(T_i(1, randperm(Nt, Ns)));
        end
        [score, pvalue] = corr(S_i', T_i', 'Type','Spearman'); %  If p-value < 0.05, it indicates rejection of the hypothesis that no correlation exists between tow vectors
        if pvalue>0.05 || isnan(score)
            score = 0;
        end
        C_i = score;
    end
    C(k) = mean(C_i);
end
corScore = mean(C);

end


function [newSrcX, newTarX, isMatched] = KSMatch(sourceX, targetX, threshold)
% KSMATCH Summary of this function goes here: 
%   Detailed explanation goes here
% INPUTS:
%   (1) sourceX  - A n_s*d_s array, d_s is the number of metrics.
%   (2) targetX  - A n_t*d_t array.
%   (3) threshold - A number belonging to [0,1].
% OUTPUTS:
%   (1) newSrcX - A n_s*d array.
%   (2) newTarX - A n_t*d array.
%
% Reference: [1] Nam J , Fu W , Kim S , et al. Heterogeneous Defect Prediction. IEEE Transactions on Software Engineering, 44(9), pp:874-896,2018.

if ~exist('threshold', 'var')||isempty(threshold)
    threshold = 0.05; % Significance level
end
isMatched = false;

% Construct metric matching matrix A
A = zeros(size(sourceX,2), size(targetX,2));
for i=1:size(sourceX,2) % x_source
    for j=1:size(targetX,2) % x_target
        [h, p] = kstest2(sourceX(:, i), targetX(:, j)); % KSAnalyzer
        % A(i, j) = p;
        if p <= eps  % too small
            A(i, j) = 0;
        else
            A(i, j) = p;
        end
    end
end

% A=[7,3,14,19,23;8,10,15,18,20;6,8,12,16,19;4,9,13,20,25;2,7,12,10,15]; % just for testing

% Find the best matches
if length(unique(A)) > 1
    isMatched = true;
    munk = py.munkres.Munkres(); % Must first install Python (note its verison) and package 'munkres'. If MATLAB 2018b is using, you must install py3.5 or 3.6.
    cost_matrix = py.munkres.make_cost_matrix(py.numpy.array(A)); % Call Python function, Must first install Python and the corresponding package.
    idxMatch = munk.compute(cost_matrix);
    idxMatch = double(py.numpy.array(idxMatch)) + 1; % For python, the index starts from 0, but it starts from 1 for MATLAB. 
    idxRow = idxMatch(:,1);
    idxCol = idxMatch(:,2);

    idxSrc = []; idxTar = [];
    for i = 1:length(idxRow) % KSAnalyzer
        [h, p] = kstest2(sourceX(:, idxRow(i)), targetX(:, idxCol(i))); 
        if p >= threshold
            idxSrc = [idxSrc, idxRow(i)];
            idxTar = [idxTar, idxCol(i)];
        end
    end
    newSrcX = sourceX(:, idxSrc);
    newTarX = targetX(:, idxTar);
else % just 
    numCommFea = min(floor(size(sourceX,2)*selectRatio), size(targetX,2));
    newSrcX = sourceX(:, 1:numCommFea); % 
    newTarX = targetX(:, 1:numCommFea); % 
end

end
