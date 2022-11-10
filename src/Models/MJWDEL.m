function probPos = MJWDEL(sources, trainTarget, target, lambda)
%MJWDEL Summary of this function goes here: Implement MJWDEL proposed by Zou et al. [1].
%   Detailed explanation goes here
% INPUTS:
%   (1) sources     - A K-sized cell array, each element is a N_S_i-by-(d+1) source dataset (i=1,2,...,K) where 
%                     the last column is the label (1 - defecttive and 0 - nondefective).
%   (2) trainTarget - A N_tt-by-(d+1) array where the last column is the label (1 - defecttive and 0 - nondefective).
%   (3) target      - A N_t-by-(d+1) array where the last column is the label (1 - defecttive and 0 - nondefective). 
%   (3) lambda      - A real number belonging to {0,0.1,...,1}
% OUTPUTS:
%   probPos - A column vector denotes the predicted probability of being
%   positive class of target instances.
%
% Reference: [1] Zou, Q., Lu, L., Yang, Z., & Xu, H.Multi-source Cross Project Defect Prediction with 
%            Joint Wasserstein. Distance and Ensemble Learning. In 2021 IEEE 32nd International Symposium
%            on Software Reliability Engineering (ISSRE),2021,pp. 57-68.
% 
% Writen by Haonan Tong (hntong@bjtu.edu.cn)
%

warning('off');

%% 
if ~exist('lambda','var')||isempty(lambda)
    lambda = 0.4; % 
end

targetX = target(:,1:end-1); % All features

fEL = cell(1,numel(sources));
pseudoProbPos = zeros(size(target,1), numel(fEL));
e = zeros(numel(sources),1);

%% Training and predicting
for k=2:numel(sources) %
    src = sources{k};
    trainData = [src; trainTarget]; % According to Section IV-E in [1] 
    [X_new_S, X_new_T] = TCA(trainData(:,1:end-1), targetX);
    LR = glmfit(X_new_S, trainData(:,end), 'binomial', 'link', 'logit'); % logical regression model
    fEL{k} = LR;
    pseudoProbPos(:,k) = glmval(LR, X_new_T, 'logit'); % Probability of being positive class (i.e., defectiveness)
    pseudoLabel = double(pseudoProbPos(:,k)>0.5);
    
    src_new = [X_new_S, trainData(:,end)];
    rho_k = margWassersteinDist(src_new, X_new_T);
    try
        rho_pos = margWassersteinDist(src_new(src_new(:,end)==1,:), X_new_T(pseudoLabel==1,:));
    catch
        rho_pos = 0;
    end
    try
        rho_neg = margWassersteinDist(src_new(src_new(:,end)==0,:), X_new_T(pseudoLabel==0,:));
    catch
        rho_neg = 0;
    end
    eta_k = (rho_neg+rho_pos)/2;
    JWD_k = lambda*rho_k + (1-lambda)*eta_k;
    e(k) = -(JWD_k)^2; 
end

%% Evaluate predicting performance
W = softmax(e); % i.e., exp(e)./sum(exp(e))
probPos = pseudoProbPos*W;

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


function rho = margWassersteinDist(source, targetX)
%MARGWASSERSTEINDIST Summary of this function goes here: Calculate the marginal Wasserstein distance between two datasets
%   Detailed explanation goes here
% INPUTS:
%	(1) source - A N_S*(d+1) matrix where the last column is the label. 
%	(2) targetX - A N_t*d matrix.
% OUTPUTS:
%	rho  - Return the marginal Wasserstein distance between sourceX and targetX
%

if isempty(targetX)||isempty(source)
    rho = inf;
    return
end

%% Discretization
% javaaddpath('D:\Program Files\Weka-3-8-4\weka.jar'); % Make sure that WEKA can be used

% Supervised discretization for source dataset
wekaSrc = mat2ARFF(source, 'classification'); % self-defined function: .mat to .ARFF 
unsupSrcDisc = javaObject('weka.filters.unsupervised.attribute.Discretize'); % Do not use class label
unsupSrcDisc.setInputFormat(wekaSrc);
wekaSrcDisc = weka.filters.Filter.useFilter(wekaSrc, unsupSrcDisc); 
[matSrc,featureNames,targetNDX,stringVals,relationName1] = weka2matlab(wekaSrcDisc,[]); % norminal to number (starting from 0)
matSrcX = matSrc(:,1:end-1);

% % Supervised discretization for source dataset (Cannot be used for the conditional Wasserstein distance!!!)
% wekaSrc = mat2ARFF(source, 'classification'); % self-defined function: .mat to .ARFF 
% unsupSrcDisc.setInputFormat(wekaSrc);
% supDisc = javaObject('weka.filters.supervised.attribute.Discretize');  % Fayyad & Irani's MDL method
% supDisc.setInputFormat(wekaSrc);
% supDisc.setOptions(weka.core.Utils.splitOptions('-R first-last')); % All features but not the label
% wekaSrcDisc = weka.filters.Filter.useFilter(wekaSrc, supDisc); 
% [matSrc,featureNames,targetNDX,stringVals,relationName1] = weka2matlab(wekaSrcDisc,[]); % norminal to number (starting from 0)
% matSrcX = matSrc(:,1:end-1);

% Unsupervised discretization for targetX
indexDelFea = [];
for j=1:size(targetX,2)
    if (numel(unique(roundn(targetX(:,j),-5)))==1)
        indexDelFea = [indexDelFea,j];
    end
end
matSrcX(:,indexDelFea) = [];
source(:,indexDelFea) = [];
targetX(:,indexDelFea) = []; % Must delete the feature having only one unique value

target = [targetX, zeros(size(targetX,1), 1)]; % 
wekaTar = mat2ARFF(target, 'classification');
unsupDisc = javaObject('weka.filters.unsupervised.attribute.Discretize'); % Do not use class label
unsupDisc.setInputFormat(wekaTar);
wekaTarDisc = weka.filters.Filter.useFilter(wekaTar, unsupDisc);
[matTar,featureNames,targetNDX,stringVals,relationName1] = weka2matlab(wekaTarDisc,[]);
matTarX = matTar(:,1:end-1);

%% Calculate instance distribution of source and target datasets, respectively
probSrcFeatures = zeros(size(matSrcX));
for i=1:size(matSrcX,2) % each column/feature
    uniqueValue = unique(matSrcX(:,i));  
    for j=1:numel(uniqueValue) 
        boolUnique = matSrcX(:,i)==uniqueValue(j);
        probSrcFeatures(boolUnique,i) = sum(boolUnique)/size(matSrcX,1);
    end
end

probTarFeatures = zeros(size(matTarX));
for i=1:size(matTarX,2) % each column/feature
    uniqueValue = unique(matTarX(:,i));  
    for j=1:numel(uniqueValue) 
        boolUnique = matTarX(:,i)==uniqueValue(j);
        probTarFeatures(boolUnique,i) = sum(boolUnique)/size(matTarX,1);
    end
end

probSrcInstances = mean(probSrcFeatures,2);
probSrcInstances = probSrcInstances/sum(probSrcInstances); % normalization - sum of all elements is 1
probTarInstances = mean(probTarFeatures,2);
probTarInstances = probTarInstances/sum(probTarInstances); % 

%% Calculate the marginal Wasserstein distance rho
psi = (pdist2(source(:,1:end-1), targetX)).^2; % According to [1]
theta = probSrcInstances*probTarInstances';
rho = sum(psi.*theta, 'all'); % Frobenius dot(or inner) product of psi and theta

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


