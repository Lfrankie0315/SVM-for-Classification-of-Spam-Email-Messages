load train.mat
%load test.mat
% eval.mat
stdtr = std(train_data,0,2);%standard deviation
mtr = mean(train_data,2);
train_data = (train_data-mtr)./stdtr;% preprocess
%test_data = (test_data-mtr)./stdtr;
eval_data = (eavl_data-mtr)./stdtr;
[m,n] = size(train_data);
A = [];
b = [];
H = zeros(n,n);
beq = 0;
sigma = 10;
for i = 1 : n
    for j = 1 : n
        xi = train_data(:,i);
        xj = train_data(:,j);
        di = train_label(i);
        dj = train_label(j);
        H(i,j) = di*dj*exp(-norm(xi-xj)^2/sigma^2); %new kernel
    end
end
f = -ones(2000,1);
Aeq = train_label';
Beq = 0;
lb = zeros(n,1);
ub = ones(n,1)*1.1;% soft margin c
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
bm = zeros(1,lambda);
for t = 1 : lambda
    g = 0;
    for i = 1 : 2000
        g = g + alpha(i)*train_label(i)*exp(-norm(train_data(:,phi(t))-train_data(:,i))^2/sigma^2);
    end
    bm(t) = train_label(phi(t)) - g;
end
b = mean(bm);
beta = 0;
for t = 1 : 2000
    ftr = 0;
    for i = 1 : 2000
        ftr = ftr + alpha(i)*train_label(i)*exp(-norm(train_data(:,t)-train_data(:,i))^2/sigma^2);
    end
    ftr = ftr + b;
    if sign(ftr) == train_label(t)
        beta = beta + 1;
    end
end
tracu = beta/2000;%train accuracy
eval_predicted = zeros(600,1);
gama = 0;
for t = 1 : 600 
    fte = 0;
    for i = 1 : 2000
        fte = fte + alpha(i)*train_label(i)*exp(-norm(eval_data(:,t)-train_data(:,i))^2/sigma^2);
    end
    fte = fte + b;
    eval_predicted(t) = sign(fte);
    if sign(fte) == eval_label(t)
        gama = gama + 1;
    end
end
teacu = gama/600;%test accuracy
