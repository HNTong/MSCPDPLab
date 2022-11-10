function [Xl,Yl,Xu,Yu] = normN2_target(target,idx,ratio)
%
%
%
%


% normalization for target
dt = size(target,1);

% get target training data, test data
Y = target(dt,:); % the last row is the label
Y(Y>1) = 1;
posind = find(Y == 1);
negind = find(Y == 0);
temp1 = target(:,posind);
temp2 = target(:,negind);
target = [temp1, temp2];

% ratio = 0.1;
tar_trIdxPos = idx(1:ceil(ratio*length(posind)));
testIdxPos = setdiff(idx(1:length(posind)),tar_trIdxPos);
tar_trIdxNeg = idx(length(posind)+1:length(posind)+ceil(ratio*length(negind)));
testIdxNeg = setdiff(idx(length(posind)+1:end),tar_trIdxNeg);

tar_trIdx = [tar_trIdxPos,tar_trIdxNeg];
Xl = target(1:dt-1,tar_trIdx);
Yl = target(dt,tar_trIdx);
Yl(Yl>1) = 1;


testIdx = [testIdxPos,testIdxNeg];
Xu = target(1:dt-1,testIdx);
Yu = target(dt,testIdx);
Yu(Yu>1) = 1;

clear posind negind temp1 temp2

% N2 normalization 
Xl = zscore(Xl,0,2); % 0 - zscore scales X using the sample standard deviation, with n - 1 in the denominator; 2 - zscore uses the means and standard deviations along the rows of X
Xu = zscore(Xu,0,2);   
