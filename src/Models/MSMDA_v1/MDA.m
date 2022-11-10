function probPos = MDA(Xs,Ys,options)

% number of source instances
[~,ns] = size(Xs);
options.Ys = Ys;

% get variables
Xl = options.Xl; Yl = options.Yl; % training target data
Xu = options.Xu; Yu = options.Yu; % test target data
Yltr = options.Yltr; % training data
Ylva = options.Ylva; % validation data

Xt = [Xl,Xu]; % target data
[dt,nt] = size(Xt);

% block data
X = blkdiag(Xs,Xt); % (ds+dt) x (ns+nt)

doTraining = options.doTraining;
if doTraining == 1 % training
    Y = [Ys,Yltr];
    Y(Y==0) = 2;
    Y = [Y,zeros(1,length(Ylva)),zeros(1,length(Yu))];
    Y = Y';
else % test
    Y = [Ys,Yl];
    Y(Y==0) = 2;
    Y = [Y,zeros(1,length(Yu))];
    Y = Y';
end

% source-target discrepancy reducing term 
Lr11 = eye(ns)/ns;
Lr22 = eye(nt)/nt;
Lr12 = -ones(ns,nt)/(ns*nt);
Lr21 = Lr12';
Lr = [Lr11,Lr12; Lr21,Lr22];
Lr = Lr/norm(Lr,'fro');
Fr = X*Lr*X';
Fr = max(Fr,Fr');

% locality preserving term
Ws = options.Ws;
if isempty(Ws)
    Ws = KNNGraph(Xs',10);
end
Wt = KNNGraph(Xt',10);
W  = blkdiag(Ws,Wt); % (ns+nt) x (ns+nt)
D = sum(W,2); 
L = diag(D) - W; % geometry Laplacian

% class discriminant term
Wcs = repmat(Y,1,length(Y)) == repmat(Y,1,length(Y))'; Wcs(Y == 0,:) = 0; Wcs(:,Y == 0) = 0; Wcs = double(Wcs);
Wcd = repmat(Y,1,length(Y)) ~= repmat(Y,1,length(Y))'; Wcd(Y == 0,:) = 0; Wcd(:,Y == 0) = 0; Wcd = double(Wcd);

Wcs = Wcs + eye(size(Wcs,1));
Wcd = Wcd + eye(size(Wcd,1));

Sw = sum(sum(W));

Swcs = sum(sum(Wcs));
Wcs = Wcs/Swcs*Sw;

Swcd = sum(sum(Wcd));
Wcd = Wcd/Swcd*Sw;

Dcs = sum(Wcs,2); 
Lcs = diag(Dcs) - Wcs;
Dcd = sum(Wcd,2); 
Lcd = diag(Dcd) - Wcd;

% parameters for tuning: set to 1 for simplicity
alpha = 1;
beta = 1;
gamma = 1;

% generalized eigenproblem
Fl = X*L*X'; Fl = max(Fl,Fl');
Fd = X*(Lcs-gamma*Lcd)*X'; Fd = max(Fd,Fd');
AA = Fr+alpha*Fl+beta*Fd; 

try
    [P,DD] = eig(AA);
catch
    lambda = 1e-6;
    I = eye(size(X,1));
    Fcs = X*Lcs*X';
    Fcd = X*Lcd*X';
    AA = Fr + alpha*Fl + beta*Fcs + lambda*I;
    BB = Fcd + lambda*I;
    [P,DD] = eig(AA,BB);
end
diagD = diag(DD);
[sD,sidx] = sort(diagD);
P = P(:,sidx);

% projected dimension
dim = ceil(dt*0.15);
P = P(:,1:dim);
Z = P'*X;

% projected data
train_new = Z(:,1:ns);
test_new = Z(:,ns+1:end);

nl = length(Yl);
tar_train_new = test_new(:,1:nl); % training target data

if doTraining == 1
    nltr = length(Yltr);
    tar_train_new_tr = tar_train_new(:,1:nltr);
    tar_train_new_va = tar_train_new(:,nltr+1:end);
    
    temp1 = tar_train_new_va(:,Ylva==1); % defective class
    temp2 = tar_train_new_va(:,Ylva==0); % non-defective class
    tar_train_new_va = [temp1,temp2];
    Ylva = [ones(1,size(temp1,2)), zeros(1,size(temp2,2))];
    
    % LR 
    model = train([Ys,Yltr]', sparse([train_new,tar_train_new_tr]'),'-s 0 -c 1 -B -1 -q'); % num * fec
    [~, ~, prob_estimates] = predict(Ylva', sparse(tar_train_new_va'), model, '-b 1');
    score = prob_estimates(:,1)';
    
    % evaluation
    measure = performanceMeasure(Ylva,score);
else
    tar_test_new = test_new(:,nl+1:end);
    
    model = train([Ys,Yl]', sparse([train_new,tar_train_new]'),'-s 0 -c 1 -B -1 -q'); % num * fec
    [~, ~, prob_estimates] = predict(Yu', sparse(tar_test_new'), model, '-b 1');
    score = prob_estimates(:,1)';
    
    % evaluation
    measure = performanceMeasure(Yu,score);
    
    probPos = score';
end

