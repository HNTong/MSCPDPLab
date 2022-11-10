function PoptValue = CalculatePopt(actY, predY, LOC, firstPercent)
%CALCULATEPOPT Summary of this function goes ahere
%   Detailed explanation goes here
% INPUTS:
%   (1) actY  - A n_samples sized column vector denoting the actual # of defects or label (0 or 1, 1 denotes the defective proneness);
%   (2) predY - A n_samples sized column vector denoting the predicted # of defects/label;
%   (3) LOC   - A n_samples sized column denoting the line-of-code of each module. 
%   (4) firstPercent - A real number belonging to [0,1], 0.2 by default.
% OUTPUTS:
%   PoptValue        - return the Popt value.
%


% Set default value
if ~exist('firstPercent', 'var')||isempty(firstPercent)
    firstPercent = 0.2; 
end
if firstPercent<0||firstPercent>1
    error('Parameter value is out of range!');
end

data = [actY, predY, LOC];

% if min(data(:,2))<0
%     error('The 2nd column of data cannot include negative value!');
% end
% 
% if min(data(:,3))<=0
%     error('LOC must be larger than zero!');
% end

% Step1: Calculate actual defect density (NOTE:)
defDensity = (actY+1)./(LOC+1); % 8/8/2021:(actY+1)./(LOC+1); 1 is used to avoid the item being zero.
data = [data,defDensity]; % Add a column, i.e., the actual defect sensity

% Step2: Sorting in 'descend' order by PredDefects(or ProbPos)
% predDefFensity = (data(:,2)+1)./(data(:,3));
% data = [data, predDefFensity];
% data = sortrows(data,-5); % For 2nd parameter, positive - ascend order, negative - descend order

% Step3: Calculate area of optimal model, actual model, and worst model.
areaActual = CalculateArea(data, firstPercent, 'actual');
areaOptimal = CalculateArea(data, firstPercent, 'opt');
areaWorst   = CalculateArea(data, firstPercent, 'worst');

% % Step3: Calculate area of optimal model, actual model, and worst model.
% area_m       = CalculateArea(sortrows(data,-2), firstPercent); % area_m = CalculateArea(data, firstPercent)
% area_optimal = CalculateArea(sortrows(data, [-4, 3]), firstPercent); % sortrows(A,[m,-n]) - m and n denote the m-th abd n-th columns of A, sort A in ascending order by A(:,m), 
%                                                                      %    if there are same values in A(:,m),then sort them in descending order by A(:,n).
% area_worst   = CalculateArea(sortrows(data, [4, -3]), firstPercent);


% Step4: Calculate Popt 
try
    PoptValue = 1 - (areaOptimal-areaActual)/(areaOptimal-areaWorst);
%     PoptValue = 1 - (area_optimal-area_m)/(area_optimal-0);
catch
    PoptValue = nan; % If denominator is zero.
end


end

function area = CalculateArea(data, firstPercent, type)
%CALCULATEAREA Summary of this function goes ahere
%   Detailed explanation goes here
% INPUTS:
%   (1) data - n*4 matrix, where 4 columns are RealDefects(or RealLabel), PredDefects(or ProbPos), LOC, defectDensity, respectively.
%   (2) firstPercent - [0,1], e.g., 0.2 denotes the first 20% LOC.
%   (3) type - {'opt','worst','actual'}
% OUTPUTS:
%   area - 'Approximate' area under a curve.

%%
if size(data,2)~=4
    error('Not enough columns in data!');
end

if strcmp(type, 'opt')
    % Sorting by actual defect density (4th column) in descending order firstly and then by LOC (3rd column) in ascending order for the samples having the same defect density 
    tempData = sortrows(data, [-4, 3]);
    
    % Calculate the cumulative sum of LOC and actual defects
    cumLOC = cumsum(tempData(:,3));
    cumDefects = cumsum(tempData(:,1));
    
    % Calculate proportion
    Xs = cumLOC/cumLOC(end);
    Ys = cumDefects/cumDefects(end);
  
elseif strcmp(type,'worst')
    % Sorting by actual defect density (4th column) in 'ascending' order firstly and then by LOC (3rd column) in 'descending' order for the samples having the same defect density
    tempData = sortrows(data, [4, -3]);
    
    cumLOC = cumsum(tempData(:,3));
    cumDefects = cumsum(tempData(:,1));
    
    % Calculate proportion
    Xs = cumLOC/cumLOC(end);
    Ys = cumDefects/cumDefects(end);
    
else 
    % Add a column - predcited defect density
    predDefDensity = (data(:,2)+1)./(data(:,3)+1); % 8/8/2021: (data(:,2)+1)./(data(:,3)+1)
    
    data = [data, predDefDensity];%
    
    % Sorting by predicted defect density (5th column) in descending order firstly, for the replicated samples further sort in ascending order by LOC 
    data = sortrows(data, [-5, 3]);

    % Calculate the cumulative sum of LOC
    cumLOC = cumsum(data(:,3));
      
    % Calculate proportion
    Xs = cumLOC/cumLOC(end);
    
    
    cumDefects = cumsum(data(:,1));
    Ys = cumDefects/cumDefects(end);
    
end

% Identify the index of element which is nearest with 'firstPercent'
idx = find(Xs<=firstPercent);
if isempty(idx) % 
    subArea = 0.5*firstPercent*(firstPercent*Ys(1)/Xs(1)); 
else
    if Xs(idx(end))==firstPercent % Just equal
        subArea = zeros(idx(end),1); % Initialization                                                    % Initialization
        subArea(1) = 0.5*Xs(1)*Ys(1); % Area of the first triangle
        for i=2:idx(end)
            subArea(i) = (Ys(i)+Ys(i-1))*(Xs(i)-Xs(i-1))/2; % Area of each trapezoid
        end
    else
        subArea = zeros(idx(end)+1,1);                                                    % Initialization
        subArea(1) = 0.5*Xs(1)*Ys(1);                                                 % area of a triangle
        for i=2:idx(end)
            subArea(i) = (Ys(i)+Ys(i-1))*(Xs(i)-Xs(i-1))/2; % Area of each trapezoid
        end
        subArea(idx(end)+1) = 0.5 * (2*Ys(idx(end)) + (((Ys(idx(end)+1)-Ys(idx(end)))*(firstPercent-Xs(idx(end)))) / (Xs(idx(end)+1)-Xs(idx(end))))) * (firstPercent-Xs(idx(end))); % the area of a small trapezoid (the last trapezoid)
    end
end


% Calculate the sum of areas of all small regions
area = sum(subArea);

end