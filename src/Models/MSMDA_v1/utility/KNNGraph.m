function W = KNNGraph(data,k)
%-------------------------------------------------------------------
% Build a knn graph
%
% W = KNNGraph(data,k);
%
% Input:
%    - data : n X d, the input data matrix
%    - k    : number of neighbors
%
% Output:
%    - W    : the weight graph
%
%-------------------------------------------------------------------

dist = EuDist2(data);
dist = dist.*(1-eye(size(dist))); % force 0 on the diagonal

if k >= size(dist,1)-1 % k large than the number of samples 
    dist(:,:) = 1; 
    dist = dist.*(1-eye(size(dist)));
else 
    [A,IX] = sort(dist,1);

    % Gives 1 to the k nearest neighbors and 0 to the others
    for j=1:size(dist,2)
        dist(IX(2:k+1,j),j) = 1; 
        dist(IX(k+2:end,j),j) = 0;
    end
end

W = max(dist,dist');


