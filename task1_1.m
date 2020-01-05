clear;
clc;
% hard maigin with kernel x1T*x2
load train.mat
load test.mat
stdtr = std(train_data,0,2);%standard deviation
stdte = std(test_data,0,2);
mte = mean(test_data,2);
mtr = mean(train_data,2);
train_data = (train_data-mtr)./stdtr;% preprocess
test_data = (test_data-mte)./stdte;
[m,n] = size(train_data);
A = [];
b = [];
H = zeros(n,n);
beq = 0;
for i = 1 : n
    for j = 1 : n
        xi = train_data(:,i);
        xj = train_data(:,j);
        di = train_label(i);
        dj = train_label(j);
        H(i,j) = di*dj*xi'*xj;
    end
end
f = -ones(2000,1);
Aeq = train_label';
Beq = 0;
lb = zeros(n,1);
ub = ones(n,1)*10^6;% hard margin c = 10^6
x0 = [];
options = optimset('LargeScale','off','MaxIter',1000);
alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub,x0,options);
for i = 1 : 2000
    if alpha(i)<=1e-4
        alpha(i) = 0;
    end
end
phi = 0;
for i = 1 : 2000
    phi = phi + 1;
    if alpha(i) ~= 0
        break
    end 
end    
w0 = 0;
for i =  1 : 2000
    w0 = w0 + alpha(i)*train_label(i)*train_data(:,i);
end
b = train_label(phi) - w0'*train_data(:,phi);
beta = 0;
for t = 1 : 2000
    ftr = w0'*train_data(:,t)+b;
    if sign(ftr) == train_label(t)
        beta = beta + 1;
    end
end
tr_acu = beta/2000;
sigma = 0;
for t = 1 : 1536
    fte = w0'*test_data(:,t)+b;
    if sign(fte) == test_label(t)
        sigma = sigma + 1;
    end
end
te_acu = sigma/1536;