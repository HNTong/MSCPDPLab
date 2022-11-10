function IFA = CalculateIFA(actLabel, probPos)
%IFAFUN Summary of this function goes ahereL: Number of Initial False Alarms encountered before we find the first defect.
%   Detailed explanation goes here
% INPUTS:
%   (1) actLabel - A column vector where 1 and 0 denote the defective class and non-defective class, respectively. 
%   (2) probPos  - A column vector having the same size as actLabel.
% OUTPUTS:
%   IFA  
%  
% Reference:  
%
% [2] C. Ni, X. Xia, D. Lo, X. Chen, and Q. Gu, "Revisiting supervised
% and unsupervised methods for effort-aware cross-project defect
% prediction", IEEE Transactions on Software Engineering, vol. 48,
% no. 3, pp. 786¨C802, 2022.


[val, idx] = sort(probPos, 'descend'); 
temp = actLabel(idx);
index = find(temp==1);
IFA = index(1);


end

