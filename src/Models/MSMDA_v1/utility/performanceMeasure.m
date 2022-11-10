function measure = performanceMeasure(test_label, predict_label)

% True Positive: the number of defective modules predicted as defective
% False Negative: the number of defective modules predicted as non-defective
% False Positive: the number of non-defective modules predictied as defective
% Ture Nagative: the number of non-defective modules predictied as non-defective

% confusion matrix
%  
%                            defective             non-defective (true class)
%  ------------------------------------------------------
%   predicted   defective        TP                  FP (Type I error)
%    class      ---------------------------------------------
%               non-defective    FN (Type II error)  TN
%  -------------------------------------------------------
% 
% Pd = recall = TP / (TP+FN)
% 
% Pf = FP / (FP+TN)
% 

% [X_coordi,Y_coordi,T_thre,AUC] = perfcurve(test_label',predict_label',1);
[AUC,~] = calcAUC(test_label,predict_label,1,0);

predict_label(predict_label>0.5) = 1;
predict_label(predict_label<=0.5) = 0;

test_total = length(test_label);
posNum = sum(test_label == 1); % defective
negNum = test_total - posNum;

pos_index = test_label == 1;
pos = test_label(pos_index);
pos_predicted = predict_label(pos_index);
FN = sum(pos ~= pos_predicted); % Type II error

neg_index = test_label == 0; % defective free
neg = test_label(neg_index);
neg_predicted = predict_label(neg_index);
FP = sum(neg ~= neg_predicted);  % Type I error

TP = posNum-FN;
TN = negNum-FP;

pd_recall = 0;
pf = 0;

acc = (TP+TN)/test_total;

if TP+FN ~= 0
    pd_recall = TP/(TP+FN); % 1.0-error1/posNum;
end

if FP+TN ~= 0
    pf = FP/(FP+TN); % negNum
end

if pd_recall+1-pf ~= 0
    g_measure = 2*pd_recall*(1-pf)/(pd_recall+1-pf);
else
    g_measure = 0;
end

%% MCC
PD = pd_recall;
PF = pf;
MCC = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));
Balance = 1 - sqrt(((0-PF)^2+(1-PD)^2)/2);
Precision=TP/(TP+FP);
F1=2*Precision*PD/(Precision+PD);

measure = [AUC, g_measure, acc, MCC, Balance, PD, PF, Precision, F1];
