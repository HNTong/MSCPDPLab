function pofB20 = costEffectiveness(actLabel, probPos, LOC)
%COSTEFFECTIVENESS Summary of this function goes here
%   Detailed explanation goes here, 
% INPUTS:
%   (1) actLabel: true label of testing instances, a column vector;
%   (2) probPos: probability of being positive class (i.e., defective) of testing instances;
%   (3) LOC: line-of-code of testing instances;
% OUTPUTS:
%   pofB20: a real number.
%
% Reference: X. Xia, D. Lo, S. J. Pan, N. Nagappan, and X. Wang, "Hydra:
%    Massively compositional model for cross-project defect prediction," IEEE Transactions on Software Engineering, vol. 42,
%    no. 10, pp. 977¨C998, 2016.
%

% Sort probPos in descending order
[~, idx] = sort(probPos, 'descend');

actLabelSorted = actLabel(idx);
LOCSorted = LOC(idx);

% Calculate the cumulative sum
cumLOCSorted  =cumsum(LOCSorted);
cumactLabelSorted = cumsum(actLabelSorted);

% Calculate the percentage
Xs = cumLOCSorted/cumLOCSorted(end);
Ys = cumactLabelSorted/(cumactLabelSorted(end));

firstPercent = 0.2;
idx = find(Xs<=firstPercent);
if isempty(idx) % It indicates that the percentage of 1st module's size is larger than 0.2
    pofB20 = firstPercent*Ys(1); 
else
    if Xs(idx(end))==firstPercent % Just equal
        pofB20 = Ys(idx(end)); % 
    else
        pofB20 = Ys(idx(end)) + (Ys(idx(end)+1)-Ys(idx(end)))*(firstPercent-Xs(idx(end)))/(Xs(idx(end)+1)-Xs(idx(end)));
    end
end

end

