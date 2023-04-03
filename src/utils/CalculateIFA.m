function IFA = CalculateIFA(actLabel, probPos, LA_LD)
%IFAFUN Summary of this function goes ahereL: Number of Initial False Alarms encountered before we find the first defect.
%   Detailed explanation goes here
% INPUTS:
%   (1) actLabel - A column vector where 1 and 0 denote the defective class and non-defective class, respectively. 
%   (2) probPos  - A column vector (the predicted probability being positive/predictive class) having the same size as actLabel.
%   (3) LA_LD - A column vector where each element is the sum of LA and LD which are two metrics for just-in-time software defect prediction.
% OUTPUTS:
%  
%  
% Reference: [1] Q. Huang, X. Xia and D. Lo, "Supervised vs Unsupervised
%       Models: A Holistic Look at Effort-Aware Just-in-Time Defect Prediction,"
%       2017 IEEE International Conference on Software Maintenance and Evolution
%      (ICSME), 2017, pp. 159-170, doi: 10.1109/ICSME.2017.51.
% [2] C. Ni, X. Xia, D. Lo, X. Chen, and Q. Gu, "Revisiting supervised
% and unsupervised methods for effort-aware cross-project defect
% prediction", IEEE Transactions on Software Engineering, vol. 48,
% no. 3, pp. 786¡§C802, 2022.
%


if ~exist('LA_LD','var')||isempty(LA_LD)
    [~, idx] = sort(probPos, 'descend');  % Ni et al. [2]
    temp = actLabel(idx);
else
    [~, idx] = sort(probPos, 'descend');
    sort_actLabel = actLabel(idx);
    sort_LA_LD = LA_LD(idx);
    [~, idx1] = sort(sort_LA_LD, 'ascend'); % Huang et al. [1]
    temp = sort_actLabel(idx1);
end
index = find(temp==1);
if ~isempty(index)
    IFA = index(1)-1;
else
    IFA = length(index);
end



end

