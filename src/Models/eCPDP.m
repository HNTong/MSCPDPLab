function probPos = eCPDP(sources,target)
%ECPDP Summary of this function goes here
%   Detailed explanation goes here, 
% INPUTS:
%   (1) sources - a cell array where each element is a n_i*(d_i+1) sized
%   source dataset, the last column is the label where 1 - defective and 0
%   - nondefective;
%   (2) target  - a n_t*(d_t+1) array where d_t = d_i;
%
% OUTPTS:
%   probPos - A column vector denotes the predicted probability of being
%   positive class of target instances.
%
% Reference: [1] S. Kwon, D. Ryu and J. Baik, "eCPDP: Early Cross-Project
%     Defect Prediction," 2021 IEEE 21st International Conference on Software
%     Quality, Reliability and Security (QRS), 2021, pp. 470-481, doi:
%     10.1109/QRS54544.2021.00058.
%
%
% @INPROCEEDINGS{Kwon2021eCPDP,
% 	author={Kwon, Sunjae and Ryu, Duksan and Baik, Jongmoon},
% 	booktitle={2021 IEEE 21st International Conference on Software Quality, Reliability and Security (QRS)}, 
% 	title={eCPDP: Early Cross-Project Defect Prediction}, 
% 	year={2021},
% 	volume={},
% 	number={},
% 	pages={470-481},
% 	doi={10.1109/QRS54544.2021.00058}}


% sourcesOri = sources;

%% Z-score
NormSrcs = cell(1, numel(sources));
mu = zeros(numel(sources),size(sources{1},2)-1);
sigma = zeros(numel(sources),size(sources{1},2)-1);
for i=1:numel(sources)
    src = sources{i};
    [Z, mu(i,:), sigma(i,:)] = zscore(src(:,1:end-1)); 
    sources{i} = [Z, src(:,end)]; 
end
sigma(sigma==0) = eps; % Some feature may be constant.

targetX = target(:,1:end-1);
normTarX = cell(1, numel(sources));
for i=1:numel(sources)
    normTarX{i} = (targetX-repmat(mu(i,:),size(target,1),1))./repmat(sigma(i,:),size(target,1),1);
end



%% Alignment
for i=1:numel(sources)
    src = sources{i};
    try
        srcX = src(:,1:end-1);
        [U, S, V] = svd(srcX'); % see Fig.1 in [1]
        sources{i} = [(S*V)', src(:,end)];
        normTarX{i} = (U'*(normTarX{i})')';
    catch
    end
end

%% Training
for i=1:numel(sources)
    src = sources{i};
    srcX = src(:,1:end-1);
    srcY = src(:,end);
    % FS = py.sklearn.feature_selection.chi2(py.numpy.array(srcX), py.numpy.array(srcY).reshape(int64(-1),int64(1))); % Chi2 just can be used for non-negative feature
    FS = py.sklearn.feature_selection.f_classif(py.numpy.array(srcX), py.numpy.array(srcY).reshape(int64(-1),int64(1)));
    temp = py.numpy.array(FS(2));
    pValues = temp.data.double;
    [val, idx] = sort(pValues,'ascend');
    selectedFeaIdx = [];
    if sum(val<0.05)>0
        selectedFeaIdx = idx(val<0.05);
    else
        selectedFeaIdx = idx(1:round(numel(idx)*0.2));
    end
    sources{i} = [srcX(:,selectedFeaIdx), srcY];
    normTarX{i} = normTarX{i}(:,selectedFeaIdx);
    
end

%% Prediction
probPosMat = zeros(size(target,1), numel(sources));
for i=1:numel(sources)
    src = sources{i};
    model = py.sklearn.linear_model.LogisticRegressionCV(pyargs('cv',int64(5), 'solver', 'lbfgs', 'class_weight','balanced', 'max_iter',int64(500)));
    if size(src,2)==2 %
        trainX = py.numpy.array(src(:,1:end-1));
        trainY = py.numpy.array(src(:,end));
        model.fit(trainX.reshape(int64(-1),int64(1)), trainY.reshape(int64(-1),int64(1)));
        testX = py.numpy.array(normTarX{i});
        predLabel = model.predict(testX.reshape(int64(-1),int64(1))); %
        predPro = model.predict_proba(testX.reshape(int64(-1),int64(1)));
        predPro = predPro.data.double;
        probPosMat(:,i) = predPro(:,2);
    else
        model.fit(src(:,1:end-1), src(:,end));
        predLabel = model.predict(normTarX{i}); %
        predPro = model.predict_proba(normTarX{i});
        predPro = predPro.data.double;
        probPosMat(:,i) = predPro(:,2);
    end
    
end
probPos = mean(probPosMat,2);

end

