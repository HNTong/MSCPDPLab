function [Xs,Ys] = normN2_source(source)

Xs = source(1:end-1,:);
Ys = source(end,:);
Ys(Ys>1) = 1;

temp1 = Xs(:,Ys==1); % defective class
temp2 = Xs(:,Ys==0); % non-defective class
Xs = [temp1,temp2];
Xs = zscore(Xs,0,2);

Ys = [ones(1,size(temp1,2)), zeros(1,size(temp2,2))];