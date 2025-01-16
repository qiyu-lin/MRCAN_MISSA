%_________________________________________________________________________%
% 麻雀优化算法             %
%_________________________________________________________________________%
function [Best_pos,Best_score,curve]=MRCAN_MISSA(src3,src4,pop,Max_iter,lb,ub,dim,fobj)

ST = 0.6;%预警值
PD = 0.7;%发现者（探索者/生产者）的比列，剩下的是加入者（跟随者/拾荒者）
SD = 0.2;%意识到有危险麻雀的比重，（侦查者/警戒者/防御者）

PDNumber = round(pop*PD); %发现者数量
SDNumber = round(pop*SD); %意识到有危险麻雀数量

if(max(size(ub)) == 1)
   ub = ub.*ones(1,dim);
   lb = lb.*ones(1,dim);  
end

%% 种群初始化,引入TC混沌映射
X0=TC_initialization(pop,dim,ub,lb);
X = X0;

%% 计算初始适应度值
fitness = zeros(1,pop);
for i = 1:pop
   fitness(i) =  fobj(src3,src4,X(i,:));
end

%% 适应度值排序
[fitness, index]= sort(fitness,'descend');%排序
BestF = fitness(1);
WorstF = fitness(end);
GBestF = fitness(1);%全局最优适应度值

%% 根据排序后的索引更新种群位置
for i = 1:pop
    X(i,:) = X0(index(i),:);
end
curve=zeros(1,Max_iter);
GBestX = X(1,:);%全局最优位置
X_new = X;

%% 更新规则
for i = 1: Max_iter
    
    BestF = fitness(1);
    WorstF = fitness(end);   
    R2 = rand(1);
    
    % 发现者更新，引入改进搜索因子和正余弦策略
    alpha = 1;
    eta = 1;
    r1_prime = alpha * (1 - (i / Max_iter) ^ eta) ^ (1 / eta);
    r1_double_prime = (exp(i / Max_iter) - 1) / (exp(1) - 1);
    r2 = 2 * pi * rand(1);
    r3 = 2 * rand(1);
   for j = 1:PDNumber 
      if(R2<ST)
          X_new(j,:) = r1_double_prime * X(j,:) + r1_prime * sin(r2) * abs(r3 * GBestX - X(j,:));
      else
          X_new(j,:) = r1_double_prime * X(j,:) + r1_prime * cos(r2) * abs(r3 * GBestX - X(j,:));
      end     
   end
   
   % 加入者更新，引入柯西高斯变异和可变螺旋搜索策略
   for j = PDNumber+1:pop
        a2 = -1 + i * ((-1) / Max_iter);
        l = (a2 - 1) * rand + 1; 
        z = exp(5* cos(pi - (1 - (i / Max_iter))));
%       if(j>(pop/2))
        if(j>(pop - PDNumber)/2 + PDNumber)
          Gauss_noise = normrnd(0, 1, size(GBestX));
          Cauchy_noise = tan(pi * (randn(size(GBestX)) - 0.5));
          zeta1 = (Max_iter - i + 1) / Max_iter;
          zeta2 = 1 - zeta1;
          X_new(j, :) = GBestX .* (1 + zeta1 * Gauss_noise + zeta2 * Cauchy_noise);
       else
          %产生-1，1的随机数
          A = ones(1,dim);
          for a = 1:dim
            if(rand()>0.5)
                A(a) = -1;
            end
          end 
          AA = A'*inv(A*A');     
          X_new(j, :) = X(1,:) + abs(X(j, :) -X(1,:)) .* AA' .* exp(z*l) .* cos(2*pi*l);
       end
   end
   
   % 预警更新
   Temp = randperm(pop);
   SDchooseIndex = Temp(1:SDNumber); 
   for j = 1:SDNumber
       if(fitness(SDchooseIndex(j))<BestF) % 最小化适应度函数，“S>B”；最大化适应度函数，“S<B”。
           X_new(SDchooseIndex(j),:) = X(1,:) + randn().*abs(X(SDchooseIndex(j),:) - X(1,:));
       elseif(fitness(SDchooseIndex(j))== BestF)
           K = 2*rand() -1;
           X_new(SDchooseIndex(j),:) = X(SDchooseIndex(j),:) + K.*(abs( X(SDchooseIndex(j),:) - X(end,:))./(fitness(SDchooseIndex(j)) - fitness(end) + 10^-8));
       end
   end
   
   %边界控制
   for j = 1:pop
       for a = 1: dim
           if(X_new(j,a)>ub)
               X_new(j,a) =ub(a);
           end
           if(X_new(j,a)<lb)
               X_new(j,a) =lb(a);
           end
       end
   end 
   
   %更新位置
   for j=1:pop
    fitness_new(j) = fobj(src3,src4,X_new(j,:));
   end
   for j = 1:pop
    if(fitness_new(j) > GBestF)  % 最小化适应度函数，“f<G”；最大化适应度函数，“f>G”。
       GBestF = fitness_new(j);
       GBestX = X_new(j,:);   
    end
   end
   X = X_new;
   fitness = fitness_new;
   
   % 引入自适应t分布变异和反向学习策略
    T = Max_iter; 
    t = i; 
    M = rand();
    % 如果M小于0.5，使用自适应t分布变异
    if M < 0.5
        df = i; % 自由度参数设置为当前迭代次数
        t_noise = trnd(df, size(GBestX)); % 生成t分布随机数

        X_best1 = GBestX +GBestX .* t_noise;

        % 应用边界控制
        X_best1 = max(min(X_best1, ub), lb);

        % 更新全局最优位置
        fitness_best1 = fobj(src3, src4, X_best1);
        if fitness_best1 > GBestF
            GBestX = X_best1;
            GBestF = fitness_best1;
        end

    % 如果M大于等于0.5，则使用反向学习
    else
        xi = (T - t / T) ^ t;
        r = rand(1, dim);

        % 计算X_best2
        X_best2 = ub + r .^ (lb - GBestX);

        % 更新最优个体位置
        X_best1 = X_best2 + xi .^ (GBestX - X_best2);

        % 应用边界控制
        X_best1 = max(min(X_best1, ub), lb);

        % 更新全局最优位置
        fitness_best1 = fobj(src3, src4, X_best1);
        if fitness_best1 > GBestF
            GBestX = X_best1;
            GBestF = fitness_best1;
        end
    end
   
   %排序更新
   [fitness, index]= sort(fitness,'descend'); % 降序
   BestF = fitness(1);
   WorstF = fitness(end);
   for j = 1:pop
      X(j,:) = X(index(j),:);
   end
   curve(i) = GBestF;
   % 输出每轮的最优适应度值
   disp(['Iteration ' num2str(i) ': Best Score = ' num2str(GBestF)]);
end
Best_pos =GBestX;
Best_score = curve(end);
end



