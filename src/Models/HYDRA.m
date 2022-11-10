function probPos = HYDRA(sources, trainTarget, target, indexLOC, PopSize, MaxGen, PC, PM, K)
%HYDRA Summary of this function goes here
%   Detailed explanation goes here, 
% INPUTS:
%   (1) sources     -  
%   (2) traintarget -  
%   (3) target      -  
%
%
% Reference: X. Xia, D. Lo, S. J. Pan, N. Nagappan, and X. Wang, “HYDRA:
%    Massively compositional model for cross-project defect prediction,”
%    IEEE Trans. Softw. Eng., vol. 42, no. 10, pp. 977–998, Oct. 2016.
%
% Written by Haonan Tong (hntong@bjtu.edu.cn)
%

% Define gloabal variables
global trainDatasets; % Used in fitness() (HYDRA_Train.m --> GAfunc.m --> fitness.m)
% global trainTarget;   % Used in fitness()
global M;             % Assign value in fitness.m and being used in EL phase as follows.  
global idxLOC;

% Assign value to global variables
trainDatasets = [sources,{trainTarget}]; % 
idxLOC = indexLOC; 

warning('off');
%% Default value used in the reference paper
if ~exist('PopSize','var')||isempty(PopSize)
    PopSize = 50; % 500
end
if ~exist('MaxGen','var')||isempty(MaxGen)
    MaxGen = 10; % 200
end
if ~exist('PC','var')||isempty(PC)
    PC = 0.35; 
end
if ~exist('PM','var')||isempty(PM)
    PM = 0.08; 
end
if ~exist('K','var')||isempty(K)
    K = 50; % 100
end

%% Calculate learning rate βs
n_s = 0;
for i=1:numel(sources)
    n_s = n_s + size(sources{i},1);
end
n_t = size(trainTarget,1);
n = n_s + n_t;
beta_s = (1/2)*log(1+sqrt(2*(log(n_s/K)))); % Line 9 in Algo.2

%% Initialize distribution weights of each source and trainTarget datasets
Ws = cell(1, numel(sources)); % 
Wtt = repmat(1/n, n_t, 1);    % Line 10 in Algo.2
for i=1:numel(sources)
    Ws{i} = repmat(1/n,size(sources{i},1),1); % Line 10 in Algo.2
end

%% EL phase of HYDRA
sumBetaK = 0;
GA = cell(1, K);
for i = 1:K
    % Normalize the distribution weights such that the summation of all
    % the weights equals to 1
    weightSum = 0;
    for j=1:numel(Ws)
        weightSum = weightSum + sum(Ws{j});
    end
    weightSum = weightSum + sum(Wtt);
    for j=1:numel(Ws)
        Ws{j} = Ws{j} / weightSum; % Normalize the distribution weights of source data
    end
    Wtt = Wtt / weightSum; % Normalize the distribution weights of trainTarget data
    
    %% Conduct GA phase: (Algorithm 1 in the paper):the first (N+1)
    %   doubles represent the weights of N+1 classifiers trained on each of
    %   N+1 datasets (S1,...SN, trainTarget), and the last double represents
    %   the threshold whose value ranges from 0 to N+1.
    bestParams = GAfunc(numel(sources)+2, PopSize, MaxGen, PC, PM);% Call GAfunc() to find the best parameters according to F1 on the trainTarget dataset. 
    
    
    %% Best parameters
    weights = bestParams(1,1:end-1);
    % weights = weights / sum(weights); % Normalization
    threshold = bestParams(1,end);
    
    %% Prediction on trainTarget Data
    posiProbs = zeros(size(trainTarget,1), numel(M));
    for j=1:numel(M)
        posiProbs(:,j) = glmval(M{j},trainTarget(:,1:end-1), 'logit'); 
    end
    
    %% Predicted Label
    Comp = sum(repmat(weights,(size(trainTarget,1)),1) .* posiProbs, 2) ./ trainTarget(:,indexLOC); % Eq.(1) in Page 4
    preLabel = double(Comp>=threshold);
    
    %% Classification error
    epsilon_k = sum(Wtt.*(abs(preLabel-trainTarget(:,end)))) / sum(Wtt); % See Eq.(2) 
    
    %% Update distribution weights
    if (epsilon_k<=1/2)
        beta_k = epsilon_k / (1 - epsilon_k); % Line 16 in Algo.2
        
        for j=1:numel(sources)
            dataset = sources{j};
            posiProb = glmval(M{j},dataset(:,1:end-1), 'logit');
            Ws{j} = Ws{j}.* exp(-beta_s * abs(double(posiProb>=0.5)-dataset(:,end)));
        end
        posiProb = glmval(M{end},trainTarget(:,1:end-1), 'logit');
        Wtt = Wtt .* exp(-beta_k * abs(double(posiProb>=0.5)-trainTarget(:,end)));
    else
        beta_k = 0;
    end  
    
    % Record
    GA{i} = {M, bestParams, beta_k};
    % GA{i} = struct('models',M,'alpha',P,'beta_k',beta_k);
    sumBetaK = sumBetaK + beta_k;
end
hydra = GA;

%% Prediction
probPos = zeros(size(target,1),1);
preLabels = [];
betaKs = [];
count = 1;
for i= 1:numel(hydra)
    
    myGA = hydra{i};
    if myGA{3} > 0
        params = myGA{2};
        weights = params(1,1:end-1);
        % weights = weights / sum(weights); % Normalization
        threshold = params(end);
        betaKs(1,count) = myGA{3};
        
        % Prediction on trainTarget Data
        posiProbs02 = zeros(size(target,1), numel(myGA{1}));
        for j=1:numel(myGA{1})
            posiProbs02(:,j) = glmval(myGA{1}{j},target(:,1:end-1), 'logit');
        end
        
        
        % Calculat Comp which is defined in the paper
%         Comp = sum(repmat(weights,(size(target,1)),1) .* posiProbs02, 2);
%         posiProb = posiProb + Comp * myGA{3}/sumBetaK;
        Comp = sum(repmat(weights,(size(target,1)),1) .* posiProbs02, 2) ./ target(:,indexLOC);
        preLabels(:,count) = double(Comp>=threshold);
        
        
        count = count + 1;
    end
end
% betaKs = betaKs / sum(betaKs);
% posiProb = sum(repmat(betaKs,(size(target,1)),1) .* preLabels, 2);
probPos = preLabel;


end

function params = GAfunc(numParams, PopSize, MaxGen, PC, PM)
%GAFUNC Summary of this function goes here
%   Detailed explanation goes here:% According to F1 on trainTarget dataset in fitness().

% 500-200-0.08-0.35
% Do not use @crossoversinglepoint when you have linear constraints.-- ga help Documentation

% For new and old version MATLAB
try
    options = optimoptions('ga','PopulationSize', PopSize,'MaxGenerations',MaxGen,'MutationFcn', {@mutationuniform, PM},'CrossoverFcn', {@crossoverintermediate, PC},'display','off'); % @mutationadaptfeasible @mutationuniform
    
    % Find the best parameters to minimize fitness
    [params,fval,~,~] = ga(@fitness,numParams,[],[],[],[],zeros(numParams,1),[ones(numParams-1,1); numParams-1],[],[],options); % the value of 'lb' and 'ub' are stated in Page 5
%     [P,fval,~,~] = ga(@fitness,dim,[],[],[],[],zeros(dim,1)+0.01,ones(dim,1),[],[],options); % Call self-defined function fitness().
catch 
    options = gaoptimset('PopulationSize',PopSize,'Generations',MaxGen,'MutationFcn', {@mutationuniform, PM},'CrossoverFcn', {@crossoverintermediate, PC},'display','off');
    [params,fval,~,~] = ga(@fitness,numParams,[],[],[],[],zeros(numParams,1),[ones(numParams-1,1); numParams-1],[],[],options); % Call self-defined function fitness().
end
end

function negativeF1 = fitness(x)
%FITNESS Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%
%

warning('off');

% global trainTarget
global trainDatasets
global M
global idxLOC

% Ensure x is a row column
if (size(x,2)==1)
    x = x';
end

% Divide into weights and threshold
weights = x(1,1:end-1);
% weights = weights / sum(weights); % normalization
threshold = x(1,end);

Ms = cell(1,numel(weights));
trainTargetData = trainDatasets{end};
if ~isempty(setdiff(unique(trainTargetData(:,end)), [0,1]'))
    error('The label does not belong to {0,1}!');
end
posiProbs = zeros(size(trainTargetData, 1), numel(trainDatasets)); % the probability of being positive class.

for i=1:numel(trainDatasets)
    
    dataset = trainDatasets{i};
    
    % Method-1: Training with weka
%     model = javaObiect('weka.classifiers.functions.Logistic');
%     model.buildClassifier(dataset);
%     
%     % Prediction
%     classProbs = [];
%     for j=0:trainTarget.numInstances -1  
%        classProbs(j+1,:) = (classifier.distributionForInstance(trainTarget.instance(j)))'; % classProbs(:,2) is the probability of being positive class.
%     end
%     posiProbs(:,i) = classProbs(:,2);
    
    % Method-2: Training with built-in logistic regression model in MATLAB
    model =glmfit(dataset(:,1:end-1),dataset(:,end),'binomial', 'link', 'logit'); % glmfit对于二分类要求y的元素为{0,1}，不能是{-1,1}
    posiProbs(:,i) = glmval(model,trainTargetData(:,1:end-1), 'logit');           % get fitted probabilities for scores

    % Save trained model
    Ms{i} = model; 
    
end

M = Ms;

% Predicted Label
Comp = sum(repmat(weights,(size(trainTargetData,1)),1) .* posiProbs, 2) ./ trainTargetData(:,idxLOC); % See Eq.(1) in Page 4; sum(A,2) - Calculate row sum of matrix A.
preLabel = double(Comp>=threshold);

% Confusion Matrix
cfm = confusionmat(trainTargetData(:,end), preLabel); % return a 2*2 matrix - [TN,FP;FN,TP]
TP=cfm(2,2);
TN=cfm(1,1);
FP=cfm(1,2);
FN=cfm(2,1);

% Calculate negative F1
PD=TP/(TP+FN);
precision=TP/(TP+FP);
F1=2*precision*PD/(precision+PD);
negativeF1 = - F1;

end

