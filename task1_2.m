clear;
clc;
% hard maigin with kernel x1T*x2
load train.mat
load test.mat
stdtr = std(train_data,0,2);%standard deviation
%stdte = std(test_data,0,2);
%mte = mean(test_data,2);
mtr = mean(train_data,2);
train_data = (train_data-mtr)./stdtr;% preprocess
test_data = (test_data-mtr)./stdtr;
[m,n] = size(train_data);
A = [];
b = [];
H = zeros(n,n);
beq = 0;
p = 5;
for i = 1 : n
    for j = 1 : n
        xi = train_data(:,i);
        xj = train_data(:,j);
        di = train_label(i);
        dj = train_label(j);
        H(i,j) = di*dj*(xi'*xj+1)^p;
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
% for i = 1 : 2000
%     if alpha(i)<=1e-4
%         alpha(i) = 0;
%     end
% end
lambda = 0;
phi = zeros(1,2000);
for i = 1 : 2000
    if alpha(i) > 1e-4
        phi(i) = i;
        lambda = lambda + 1;%number of support vector
    end
end
phi(phi==0) = [];
%bm = zeros(1,lambda);
%for t = 1 : lambda
    g = 0;
    for i = 1 : 2000
    g = g + alpha(i)*train_label(i)*(train_data(:,phi(1))'*train_data(:,i)+1)^p;
    end
    b = train_label(phi(1)) - g;
%end
%b = mean(bm);
beta = 0;
for t = 1 : 2000
    ftr = 0;
    for i = 1 : 2000
        ftr = ftr + alpha(i)*train_label(i)*(train_data(:,t)'*train_data(:,i)+1)^p;
    end
    ftr = ftr + b;
    if sign(ftr) == train_label(t)
        beta = beta + 1;
    end
end
tracu = beta/2000;
sigma = 0;
for t = 1 : 1536
    fte = 0;
    for i = 1 : 2000
        fte = fte + alpha(i)*train_label(i)*(test_data(:,t)'*train_data(:,i)+1)^p;
    end
    fte = fte + b;
    if sign(fte) == test_label(t)
        sigma = sigma + 1;
    end
end
teacu = sigma/1536;