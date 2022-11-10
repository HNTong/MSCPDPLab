function [SRDO_data,LOC] = SRDO(orgData)
% the attribute values of sensitive and class are unchanged in the privatized dataset
LOC = orgData(:,1);

data = orgData(:,1:end-1);
[~,dim] = size(data);

label = orgData(:,end);
label(label>1)=1;

c = 2;
SRDO_data = [];
a = 0.15; b = 0.35;

for i=1:c
    x = data(label==i-1,:); % within calss instances
    normx = data(label==i-1,:);
    for j=1:c
        if j~=i
            z = data(label==j-1,:); % between class instances
            normz = data(label==j-1,:);
        end
    end
    
    % unit L2 norm
    normx = normx./( repmat(sqrt(sum(normx.*normx,2)), [1,dim]) );
    normz = normz./( repmat(sqrt(sum(normz.*normz,2)), [1,dim]) );
    
    % generate random number
    rnum = 2*rand(size(x,1),1)-1; % [-1,1]
    r = rnum;
    r(rnum<0) = (b-a)*rnum(rnum<0)-a;
    r(rnum>0) = (b-a)*rnum(rnum>0)+a;
    sgnr = sign(r);
    
    % find the nearest similar neighbor (NSN) by sparse representation
    lambda = 1e-3; % tune
    I = eye(dim);
    nw = size(normx,1);
    temp1 = [];
    for ii = 1:nw
        B = normx([1:ii-1, ii+1:nw], :);
        A = [B; I]; 
        currentX = normx(ii,:);
        
        % call l1_ls function
        coef = l1_ls_nonneg(A',currentX',lambda);
        coef = coef(1:nw);
        
        tempcoef = ones(nw,1)*(-Inf);
        tempcoef(1:ii-1) = coef(1:ii-1);
        tempcoef(ii+1:nw) = coef(ii:nw-1);
        [wc,idxw] = max(tempcoef,[],1);
        
        temp1 = [temp1; r(ii).*(x(ii)-x(idxw,:))];
    end
       
    % generate random number
    rnum = 2*rand(size(x,1),1)-1; % [-1,1]
    r = rnum;
    r(rnum<0) = (b-a)*rnum(rnum<0)-a;
    r(rnum>0) = (b-a)*rnum(rnum>0)+a;
   
    r = abs(r);
    r = -sgnr.*r;

    % find the nearest unlike neighbor (NUN) by sparse representation
    nb = size(normz,1);
    temp2 = [];
    for ii = 1:nw
        B = normz;
        A = [B; I]; 
        currentX = normx(ii,:);

        coef = l1_ls_nonneg(A',currentX',lambda);
        coef = coef(1:nb);
        [bc,idxb] = max(coef,[],1);
        temp2 = [temp2; r(ii).*(x(ii)-z(idxb,:))];
    end
    
    obfuscated = x+temp1+temp2;  
    SRDO_data = [SRDO_data; obfuscated];
end

SRDO_data = [LOC, SRDO_data, label]; % add LOC and class label
