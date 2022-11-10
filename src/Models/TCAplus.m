function probPos = TCAplus(source,target)
%TCA+ Summary of this function goes here
%   Detailed explanation goes here:
% Input:
%   (1) source - each row is a observation, the last column is the label {0,1}
%   (2) target - each row is a observation, the last column is the label {0,1}
% Output:
%   probPos - A column vector denotes the predicted probability of being
%   positive class of target instances.
%
% Reference: Nam, Jaechang, Sinno Jialin Pan, and Sunghun Kim. "Transfer
%     defect learning." 2013 35th international conference on software
%     engineering (ICSE). IEEE, 2013.  

% Trabsfer cell to mat
source1=[];
if strcmp('cell',class(source))
    for i=1:numel(source)
        source1 = [source1;source{i}];
    end
    clear source;
    source = source1;
end

% The number of instances
[numRow1,col1] = size(source);
[numRow2,col2]  =size(target);

source_x = source(:,1:end-1);
target_x = target(:,1:end-1);

DIST_source = [];
DIST_target = [];

DCV_source = zeros(6,1); % [min, max, mean, medium, std, number of Rows] 
DCV_target = zeros(6,1);

%% Sep1:Calculate DIST for source data and target data, respectively.
k=1;
for i=1:numRow1-1 % each of the first (numRow1-1) instances
    for j=i+1:numRow1 % each of the last (numRow1-i) instances
        DIST_source(k) = norm(source_x(i,:)-source_x(j,:),2);
        k=k+1;
    end
end
k=1;
for i=1:numRow2-1
    for j=i+1:numRow2
        DIST_target(k) = norm(target_x(i,:)-target_x(j,:),2);
        k=k+1;
    end
end


%% Step2: Calaulate DCV 
DCV_source = [min(DIST_source),max(DIST_source),mean(DIST_source),median(DIST_source),std(DIST_source),numRow1];
DCV_target = [min(DIST_target),max(DIST_target),mean(DIST_target),median(DIST_target),std(DIST_target),numRow2];


%% Normalization
if (DCV_source(3)*0.9<DCV_target(3)<=DCV_source(3)*1.1)&&(DCV_source(5)*0.9<DCV_target(5)<=DCV_source(5)*1.1)
    rule = 1; % no normalization
elseif ((DCV_target(1)<DCV_source(1)*0.4)&&(DCV_target(2)<DCV_source(2)*0.4)&&(DCV_target(6)<DCV_source(6)*0.4))||((DCV_source(1)*1.6<DCV_target(1))&&(DCV_source(2)*1.6<DCV_target(2))&&(DCV_source(6)*1.6<DCV_target(6)))
    rule = 2; % Perform max-min normalization
    [t1,ps1] = mapminmax(source_x');
    source_x  =t1';
    [t2,ps2] = mapminmax(target_x');
    target_x  =t2';
elseif ((DCV_source(5)*1.6<DCV_target(5))&&(numRow2<numRow1))||((DCV_target(5)<DCV_source(5)*0.4)&&(numRow2>numRow1)) % Error:Attempted to access DCV_source(5); index out of bounds because numel(DCV_source)=4.
    rule =3; % Only perform normalization on source data
    source_x = zscore(source_x);
elseif ((DCV_source(5)*1.6<DCV_target(5))&&(numRow2>numRow1))||((DCV_target(5)<DCV_source(5)*0.4)&&(numRow2<numRow1))
    rule = 4; % Only perform normalization on target data
    target_x = zscore(target_x);
else
    rule = 5; % Z-score normalization
    source_x = zscore(source_x);
    target_x  = zscore(target_x);
end


%% Calculate the components 
[source_new, target_new] = TCA(source_x, target_x); % Call TCA()
% [source_new, target_new] = fun_TCA(source_x', target_x', numRow1,numRow2); % Call TCA()

%% Training and Prediction
% Component based source data and target data
tca_source = [source_new,source(:,end)];
tca_target = [target_new,target(:,end)];

model =glmfit(tca_source(:,1:end-1),tca_source(:,end),'binomial', 'link', 'logit');% glmfit要求二分类的取值为{0,1},通常令1为正样本,0为负样本.

probPos = glmval(model,tca_target(:,1:end-1), 'logit');%p - the probability of predicted as postive

end

% function [source_new, target_new] = fun_TCA(source_new, target_new, source_total, target_total)
% %FUN_TCA 
% %
% % Input:
% %	source_new - each column is a observation, each row is a feature; note that labels is not included. 
% %	target_new - each column is a observation, each row is a feature; note that labels is not included.
% %	source_total - the number of observations, namely the number of columns of source_new.
% %	target_total - the number of observations, namely the number of columns of target_new.
% % Output:
% %	source_new - each column is a observation
% %	target_new - 
% 
% % Reference: 《Domain Adaptation via Transfer Component Analysis》
% 
% % % Reduce memory usage: double -> single
% % source_new = single(source_new);
% % target_new = single(target_new);
% 
% % Case1: if memory is enough
% % construct the kernel matrix
% rbf = sum(var(source_new')) ;% var - variance; rbf = sum(var(source_new')) * 1;
% Dss = EuDist2(source_new',source_new',0);% a matrix
% Dtt = EuDist2(target_new',target_new',0);%
% Dst = EuDist2(source_new',target_new',0);%
% 
% Kss = exp(-Dss / rbf);  % 
% Ktt = exp(-Dtt / rbf);
% Kst = exp(-Dst / rbf);
% Kts = Kst';
% K = [Kss, Kst; Kts, Ktt];
% clear Dss Dtt Dst Kss Ktt Kst Kts;
% 
% % construct L matrix
% L = -1.0 / (source_total*target_total)*ones(source_total+target_total);
% L(1:source_total,1:source_total) = 1.0 / (source_total^2) * ones(source_total);
% L(source_total+1:end, source_total+1:end) = 1.0 / (target_total^2) * ones(target_total);
% 
% % construct H matrix
% H = eye(source_total+target_total) - 1.0 / (source_total+target_total) * ones(source_total+target_total,1)*ones(1,source_total+target_total);
% 
% % compute W matrix
% u = 1;
% 
% ss = pinv(u*eye(source_total+target_total) + K*L*K) * K*H*K; %  
% 
% rr = rank(ss);
% [V_ss, D_ss] = eig(ss);
% [dd_value, dd_site] = sort(abs(D_ss), 'descend');
% W = V_ss(:,dd_site(1:rr));
% 
% K_new = W'*K;
% clear K W;
% source_new = K_new(:,1:source_total);% 
% target_new = K_new(:,source_total+1:end);
% clear K_new
% 
% end

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

