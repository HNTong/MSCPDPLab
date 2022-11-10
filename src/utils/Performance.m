function perfs = Performance(actLabel, probPos, LOC, threshold)
% function [ PD,PF,Precision, F1,AUC,Accuracy,G_measure,MCC, Balance] = Performance(actual_label, probPos, LOC, threshold)
%PERFORMANCE Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) actLabel - The actual label, a column vetor, each row is an instance's class label;
%   (2) probPos - A column vector, the probability of being predicted as positive class;
%   (3) LOC - a column vector where each element denotes lines of code of each test instance;
%   (4) threshold - A number in [0,1], by default threshold=0.5.
% OUTPUTS:
%   PF,PF,..,MCC.

% Default value
if ~exist('threshold','var')||isempty(threshold)
    threshold = 0.5;
end
if ~exist('LOC','var')||isempty(LOC)
    LOC = [];
    perfs.Popt20 = nan; perfs.cost_effectiveness = nan;
end

% if numel(unique(actual_label)) < 1
%     error('Please make sure that the true label ''actual_label'' must has at least two different kinds of values.');
% end

assert(numel(unique(actLabel)) > 1, 'Please ensure that ''actual_label'' includes two or more different labels.'); % 
assert(length(actLabel)==length(probPos), 'Two input parameters must have the same size.');


predict_label = double(probPos>=threshold);

cf=confusionmat(actLabel,predict_label);
TP=cf(2,2);
TN=cf(1,1);
FP=cf(1,2);
FN=cf(2,1);

Accuracy = (TP+TN)/(FP+FN+TP+TN);
PD=TP/(TP+FN);
PF=FP/(FP+TN);
Precision=TP/(TP+FP);
F1=2*Precision*PD/(Precision+PD);
[X,Y,T,AUC]=perfcurve(actLabel, probPos, '1');% 
G_measure = (2*PD*(1-PF))/(PD+1-PF);
MCC = (TP*TN-FP*FN)/(sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)));
Balance = 1 - sqrt(((0-PF)^2+(1-PD)^2)/2);

nMCC = (MCC+1)/2; % See "An Improved Method for Training Data Selection for Cross-Project Defect Prediction"
G_Mean = sqrt(PD*(1-PF)); 
Popt20 = CalculatePopt(actLabel, predict_label, LOC);
cost_effectiveness = costEffectiveness(actLabel, probPos, LOC);
% IFA = CalculateIFA(actual_label, probPos);

perfs.PD = PD; perfs.PF = PF; perfs.Precision = Precision; perfs.F1 = F1; perfs.AUC = AUC; perfs.Accuracy= Accuracy; perfs.G_Measure = G_measure; 
perfs.G_Mean = G_Mean; perfs.MCC = MCC; perfs.Balance = Balance; perfs.nMCC = nMCC; perfs.Popt20 = Popt20; perfs.cost_effectiveness = cost_effectiveness;

end

