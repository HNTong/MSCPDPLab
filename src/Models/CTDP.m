function probPos = CTDP(sources, labeledTarget, unlabeledTarget)
%MJWDEL Summary of this function goes here: Implement CTDP proposed by Chen et al. [1].
%   Detailed explanation goes here
% INPUTS:
%   (1) sources         - A K-sized cell array, each element is a N_S_i-by-(d+1) source dataset (i=1,2,...,K) where 
%                     the last column is the label (1 - defecttive and 0 - nondefective).
%   (2) labeledTarget   - A N_tt-by-(d+1) array where the last column is the label (1 - defecttive and 0 - nondefective).
%   (3) unlabededtarget - A N_t-by-(d+1) array where the last column is the label (1 - defecttive and 0 - nondefective). 
% OUTPUTS:
%   probPos - A column vector denotes the predicted probability of being
%   positive class of target instances.
% 
%
% Reference: [1] J. Chen, K. Hu, Y. Yang, Y. Liu, and Q. Xuan, "Collective
%    transfer learning for defect prediction" Neurocomputing, vol. 416, pp.
%    103–116, 2020. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0925231219308720.
% 
% Writen by Haonan Tong (hntong@bjtu.edu.cn)
%

warning('off');

%% Divide the labeled target data into training target data and testing target data equally.
posLabTar = labeledTarget(labeledTarget(:,end)==1,:);
negLabTar = labeledTarget(labeledTarget(:,end)==0,:);
k1 = round(size(posLabTar,1)/2); % half number of positive instances
k2 = round(size(negLabTar,1)/2);
trainTarget = [posLabTar(1:k1,:); negLabTar(1:k2,:)];
testTarget = [posLabTar(k1+1:end,:); negLabTar(k2+1:end,:)];

target = [testTarget; unlabeledTarget];
sources{end+1} = trainTarget;

%% Build classifier set
normalizers = {'None','Min-Max','Z-score-src-tar','Z-score-src','Z-score-tar'};
classifierSet = cell(1,numel(sources)*numel(normalizers));
tarSet = cell(1,numel(classifierSet));
k = 1;
for d=1:numel(sources)
    for i=1:numel(normalizers)
        [model, tarARFF] = submodelTrain(sources{d}, target, normalizers{i});
        classifierSet{k} = model;
        tarSet{k} = tarARFF;
        k = k +1;
    end
end


%% Find the best weights with PSO based on testTarget dataset
temp = tarARFF;
filter = javaObject('weka.filters.unsupervised.instance.RemoveRange'); 
string = ['-R ', num2str(size(testTarget,1)+1), '-', num2str(size(target,1))]; % Begin from 1
filter.setOptions(weka.core.Utils.splitOptions(string));
filter.setInputFormat(temp);
testTarARFF = weka.filters.Filter.useFilter(temp, filter);
weights = SMPSO(numel(classifierSet)+1, classifierSet, testTarARFF); % Call self-defined function SMPSO()

threshold = weights(1,end);
weights = weights(1,1:end-1);

%% Prediction
probPoss = zeros(size(unlabeledTarget,1), numel(classifierSet));
for i=1:numel(classifierSet)
    
    model = classifierSet{i};
    targetARFF = tarSet{i}; % Including testTarget and unlabeled target
    filter = javaObject('weka.filters.unsupervised.instance.RemoveRange'); 
    string = ['-R ', '1-', num2str(size(testTarget,1))];
    filter.setOptions(weka.core.Utils.splitOptions(string));
    filter.setInputFormat(targetARFF);
    unlabeledTarARFF = weka.filters.Filter.useFilter(targetARFF, filter); 
    
    classProbsPred = zeros(unlabeledTarARFF.numInstances(),2); % '2' denotes two classes
    for j = 0:(unlabeledTarARFF.numInstances()-1) % For WEKA, the index starts from 0.
        classProbsPred(j+1,:) = model.distributionForInstance(unlabeledTarARFF.instance(j));
    end
    probPoss(:,i) = classProbsPred(:,2); % the probability of being positive
end

probPos = probPoss*weights';

end

function [model, tarARFF] = submodelTrain(source, target, normalizer)
%SUBMODELTRAIN Summary of this function goes here:
%   Detailed explanation goes here
% INPUTS:
%   (1) source - A n_src*(d+1) array where the last column is the label (1 - defecttive and 0 - nondefective).
%   (2) target - A n_tar*(d+1) array where the last column is the label (1 - defecttive and 0 - nondefective).
%   (3) normalizer - A string belonging to {'Min-Max','Z-score-src-tar','Z-score-src','Z-score-tar'}. 
%
% OUTPUTS:
%   (1) model - J48 in WEKA.
%   (2) tarARFF - A ARFF file.

sourceX = source(:,1:end-1);
targetX = target(:,1:end-1);
n_src = size(sourceX,1); % Number of samples
n_tar = size(targetX,1);
        
switch normalizer
    case 'None'
        normSrcX = sourceX;
        normTarX = targetX;
    case 'Min-Max'
        [temp1, ~] = mapminmax(sourceX',0,1); % Each row is a feature
        normSrcX = temp1';
        [temp2, ~] = mapminmax(targetX',0,1);
        normTarX = temp2';
    case 'Z-score-src-tar' 
        stdVal = std([sourceX; targetX],1); % Maybe zero
        stdVal(stdVal==0) = eps;            % Repalce 'zero' with a very small number
        normSrcX = (sourceX-repmat(mean([sourceX; targetX],1), n_src, 1))./repmat(stdVal, n_src, 1);
        normTarX = (targetX-repmat(mean([sourceX; targetX],1), n_tar, 1))./repmat(stdVal, n_tar, 1);
    case 'Z-score-src'
        stdVal = std(sourceX,1); % Maybe zero
        stdVal(stdVal==0) = eps; % 
        normSrcX = (sourceX-repmat(mean(sourceX,1), n_src, 1))./repmat(stdVal, n_src, 1);
        normTarX = (targetX-repmat(mean(sourceX,1), n_tar, 1))./repmat(stdVal, n_tar, 1);
    case 'Z-score-tar'
        stdVal = std(targetX,1); % Maybe zero
        stdVal(stdVal==0) = eps;
        normSrcX = (sourceX-repmat(mean(targetX,1), n_src, 1))./repmat(stdVal, n_src, 1);
        normTarX = (targetX-repmat(mean(targetX,1), n_tar, 1))./repmat(stdVal, n_tar, 1);
end

[newSrcX, newTarX, ~] = TCA(normSrcX, normTarX); % Call self-defined fucntion TCA().

srcARFF = mat2ARFF([newSrcX, source(:,end)], 'classification'); % Tranform mat into ARFF
tarARFF = mat2ARFF([newTarX, target(:,end)], 'classification'); %
model = javaObject('weka.classifiers.trees.J48'); % J48 is used by Chen [1].
model.buildClassifier(srcARFF);

end

function arff = mat2ARFF(data, type)
% Summary of this function goes here: 
%   Detailed explanation goes here
% INPUTS:
%   (1) data - a n*(d+1) matrix where the last column is independent variable.
%   (2) type - a string, 'regression' or 'classification'
% OUTPUTS:
%   arff     - an ARFF file

javaaddpath('D:\Program Files\Weka-3-8-4\weka.jar');
if ~exist('type','var')||isempty(type)
    type = 'classification';
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



function [X_src_new,X_tar_new,A] = TCA(X_src, X_tar, options)
% The is the implementation of Transfer Component Analysis.
% Reference: Sinno Pan et al. Domain Adaptation via Transfer Component Analysis. TNN 2011.
%
% Inputs: 
%%% X_src          :    source feature matrix, ns * n_feature
%%% X_tar          :    target feature matrix, nt * n_feature
%%% options        :    option struct
%%%%% lambda       :    regularization parameter
%%%%% dim          :    dimensionality after adaptation (dim <= n_feature)
%%%%% kernel_tpye  :    kernel name, choose from 'primal' | 'linear' | 'rbf'
%%%%% gamma        :    bandwidth for rbf kernel, can be missed for other kernels

% Outputs: 
%%% X_src_new      :    transformed source feature matrix, ns * dim
%%% X_tar_new      :    transformed target feature matrix, nt * dim
%%% A              :    adaptation matrix, (ns + nt) * (ns + nt)
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Default value
    if ~exist('options','var')||isempty(options)
        options.lambda = 1;
        options.dim = size(X_src,2);
        options.kernel_type = 'rbf'; 
        options.gamma = 1;
    end
	%% Set options
	lambda = options.lambda;              
	dim = options.dim;                    
	kernel_type = options.kernel_type;    
	gamma = options.gamma;                

	%% Calculate
	X = [X_src',X_tar'];
    X = X*diag(sparse(1./sqrt(sum(X.^2))));
    
    % Exception handling - Repalce NAN with eps
    X(isnan(X)) = eps;
    
	[m,n] = size(X);
	ns = size(X_src,1);
	nt = size(X_tar,1);
	e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
	M = e * e';
	M = M / norm(M,'fro');
	H = eye(n)-1/(n)*ones(n,n);
	if strcmp(kernel_type,'primal')
		[A,~] = eigs(X*M*X'+lambda*eye(m),X*H*X',dim,'SM');
		Z = A' * X;
        Z = Z * diag(sparse(1./sqrt(sum(Z.^2))));
		X_src_new = Z(:,1:ns)';
		X_tar_new = Z(:,ns+1:end)';
	else
	    K = TCA_kernel(kernel_type,X,[],gamma);
	    [A,~] = eigs(K*M*K'+lambda*eye(n),K*H*K',dim,'SM');
	    Z = A' * K;
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        X_src_new = Z(:,1:ns)';
		X_tar_new = Z(:,ns+1:end)';
	end
end


function K = TCA_kernel(ker,X,X2,gamma)
% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf','sam'
%       X:      data matrix (features * samples)
%       gamma:  bandwidth of the RBF/SAM kernel
% Output:
%       K: kernel matrix
%
% Gustavo Camps-Valls
% 2006(c)
% Jordi (jordi@uv.es), 2007
% 2007-11: if/then -> switch, and fixed RBF kernel
% Modified by Mingsheng Long
% 2013(c)
% Mingsheng Long (longmingsheng@gmail.com), 2013
    switch ker
        case 'linear'

            if isempty(X2)
                K = X'*X;
            else
                K = X'*X2;
            end

        case 'rbf'

            n1sq = sum(X.^2,1);
            n1 = size(X,2);

            if isempty(X2)
                D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            else
                n2sq = sum(X2.^2,1);
                n2 = size(X2,2);
                D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq -2*X'*X2;
            end
            K = exp(-gamma*D); 

        case 'sam'

            if isempty(X2)
                D = X'*X;
            else
                D = X'*X2;
            end
            K = exp(-gamma*acos(D).^2);

        otherwise
            error(['Unsupported kernel ' ker])
    end
end

function solution = SMPSO(dim, modelSet, testTarget)
% SMPSO Summary of this function goes here: Implement SMPSO algorithm 
%   Detailed explanation goes here
% INPUTS:
%   (1) dim - An integer denoting the dimensionality of the solution.
%   (2) mdoelSet - A cell.
%   (3) testTarget - A ARFF file. 
% OUTPUTS:
%   solution - A row vector denoting the best solution.
% Reference: Nebro, A. J. , et al. "SMPSO: A new PSO-based metaheuristic
%        for multi-objective optimization." Computational intelligence in
%        miulti-criteria decision-making, 2009. mcdm '09. ieee symposium on IEEE,2009.
%


%% 参数初始化
%粒子群算法中的两个参数,[1.5, 2.5]
c1 = 1.5; 
c2 = 1.5;
w = 1;

maxGen=50;   % 进化次数  
sizePop=50;  % 种群规模

Vmax=1;
Vmin=-1;
popMax=1;
popMin=0;

% popmax=5;
% popmin=-5;
% weightsMax = 1;
% weightsMin = 0;
% thresholdMax = 1;
% thresholdMin = 0;

%% 产生初始粒子和速度
pop = zeros(sizePop, dim); % Initialization
V = zeros(sizePop, dim);
fitness = zeros(1, sizePop);
for i=1:sizePop
    %随机产生一个种群
    pop(i,:) = rand(1,dim); %初始种群
    V(i,:)=rand(1,dim);     %初始化速度
    %计算适应度
    fitness(i)=fitnessFun((pop(i,1:end-1))', pop(i,end), modelSet, testTarget);   %染色体的适应度
end

%% 个体极值和群体极值
[bestfitness, bestindex] = max(fitness);
globalBest = pop(bestindex,:);   %全局最佳粒子
fitnessGBest = bestfitness;      %全局最佳适应度值
indiBest = pop;              %个体最佳
fitnessIndBest = fitness;    %个体最佳适应度值


%% 迭代寻优
for i=1:maxGen
    for j=1:sizePop
        
        %速度更新
        V(j,:) = w*V(j,:) + c1*rand*(indiBest(j,:) - pop(j,:)) + c2*rand*(globalBest - pop(j,:));
        % V(j,find(V(j,:)>Vmax))=Vmax;
        % V(j,find(V(j,:)<Vmin))=Vmin;
        delta = (Vmax-Vmin)/2;
        V(j,find(V(j,:)>delta))=delta;
        V(j,find(V(j,:)<=-delta))=-delta;
        
        %种群更新
        pop(j,:)=pop(j,:)+V(j,:);
        pop(j,find(pop(j,:)>popMax))=popMax;
        pop(j,find(pop(j,:)<popMin))=popMin;
        
        % 变异
        if rand > 0.8  %      
            pop(j,:) = rand(1,dim);
        end
    
        %适应度值
        fitness(j) = fitnessFun((pop(j,1:end-1))', pop(j,end), modelSet, testTarget); 
    end
    
    
    for j=1:sizePop
        
        pop(j,1:end-1) = pop(j,1:end-1)/sum(pop(j,1:end-1)); % 归一化
        
        %个体最优更新
        if fitness(j) > fitnessIndBest(j)
            indiBest(j,:) = pop(j,:);
            fitnessIndBest(j) = fitness(j);
        end
        
        %群体最优更新
        if fitness(j) > fitnessGBest
            globalBest = pop(j,:);
            fitnessGBest = fitness(j);
        end
    end   

end
solution = globalBest;

end

function fitness = fitnessFun(weights, threshold, modelSet, testTarget)
% Summary of this function goes here: 
%   Detailed explanation goes here
% INPUTS:
%   (1) weights    - A column vector;
%   (2) threshold  -
%   (3) modelSet   - A cell having the same size as 'weights';
%   (4) testTarget - A ARFF dataset;
% OUTPUTS:
%   fitness     - F-1 score of modelSet with weights on testTarget.

weights = weights/sum(weights);
n_instances = testTarget.numInstances();
probPos = zeros(n_instances, numel(modelSet));
for i=1:numel(modelSet)
    model = modelSet{i};
    classProbsPred = zeros(n_instances,2); % '2' denotes two classes
    for j = 0:(n_instances-1) % For WEKA, the index starts from 0.
        classProbsPred(j+1,:) = model.distributionForInstance(testTarget.instance(j));
    end
    probPos(:,i) = classProbsPred(:,2); % the probability of being positive
end
[mat,featureNames,~,stringVals,relationName] = weka2matlab(testTarget,[]);
try
    [ PD,PF,Precision,F1,AUC,Accuracy,G_measure,MCC, Balance] = Performance(mat(:,end), probPos*weights, threshold); % Call self-defined function Performance()
    fitness = F1;
catch
    fitness = 0;
end

end

