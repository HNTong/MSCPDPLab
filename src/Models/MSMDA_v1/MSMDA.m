function probPos = MSMDA(data,options,th)
% number of sources
v = size(data,1);

% firstly, select the best source for the target, then the better, and so on ...
bestSource = data{1,1};
[Xs,Ys] = normN2_source(bestSource);
Xs = Xs*diag(1./sqrt(sum(Xs.^2)));
Ws = KNNGraph(Xs',10);

for i=2:v
    currXs = Xs;
    currYs = Ys;
    currWs = Ws;
    
    source = data{i,1};
    [xsi,ysi] = normN2_source(source);
    xsi = xsi*diag(1./sqrt(sum(xsi.^2)));
    
    currXs = blkdiag(currXs,xsi);
    currYs = [currYs,ysi];
    
    wsi = KNNGraph(xsi',10);
    currWs = blkdiag(currWs,wsi);
    
    options.Ws = currWs;
    mea = MDA(currXs,currYs,options);
    if mea(2) > th
        th = mea(2);
        Xs = currXs;
        Ys = currYs;
        Ws = currWs;
    end
end

options.Ws = Ws;
options.doTraining = 2; % test
probPos = MDA(Xs,Ys,options);
