
%% An example of usage of USMSCPDP model

path = 'E:\ReLink\'; % Please first define the path of dataset folder
[sources,target]=LoadSrcTar(path,1); 
probPos1=CFPS(sources, target);
perfs1=Performance(target(:,end), probPos1, target(:,11));


%% An example of usage of SSMSCPDP model
[sources,target]=LoadSrcTar('E:\ReLink\',1);
n=size(target,1);
trainTarget=target(randperm(n,floor(0.1*n)),:);
testData=target(find(ismember(target,trainTarget,'rows')==0),:);
probPos3=CTDP(sources, trainTarget,testData);
perfs3=Performance(testData(:,end), probPos3, testData(:,11));