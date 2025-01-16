%_________________________________________________________________________%
% 麻雀优化算法             %
%_________________________________________________________________________%
function [Best_pos,Best_score,curve]=ASFSSA(src3,src4,pop,Max_iter,lb,ub,dim,fobj)

ST = 0.6;%预警值
PD = 0.7;%发现者（探索者/生产者）的比列，剩下的是加入者（跟随者/拾荒者）
SD = 0.2;%意识到有危险麻雀的比重，（侦查者/警戒者/防御者）

PDNumber = round(pop*PD); %发现者数量
SDNumber = round(pop*SD); %意识到有危险麻雀数量

if(max(size(ub)) == 1)
   ub = ub.*ones(1,dim);
   lb = lb.*ones(1,dim);  
end

%% 种群初始化
X0=Tent_initialization(pop,dim,ub,lb);
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
    
    % 发现者更新,引入自适应加权策略
    w = 0.2 * cos(pi/2 * (1-(i/Max_iter)));
    for j = 1:PDNumber
       if(R2<ST)
           X_new(j,:) = w *X(j,:).*exp(-j/(rand(1)*Max_iter));
       else
           X_new(j,:) = w *X(j,:) + randn()*ones(1,dim);
      end     
    end
    % 采用莱维飞行策略对发现者位置再次更新
    for j = 1:PDNumber
        X_new(j,:) = func_levy(X_new(j,:), GBestX);
        X_new(j,:) = Bounds(X_new(j,:), lb, ub);
    end
   
   % 加入者更新,引入可变螺旋搜索策略
   a2 = -1 + i * ((-1) / Max_iter);
   l = (a2 - 1) * rand + 1; 
   z = exp(5* cos(pi* (1 - (i / Max_iter))));
   for j = PDNumber+1:pop
%        if(j>(pop/2))
        if(j>(pop - PDNumber)/2 + PDNumber)
          X_new(j,:)= randn().*exp((X(end,:) - X(j,:))/j^2) * exp(z*l) * cos(2*pi*l);
       else
          %产生-1，1的随机数
          A = ones(1,dim);
          for a = 1:dim
            if(rand()>0.5)
                A(a) = -1;
            end
          end 
          AA = A'*inv(A*A');     
          X_new(j,:)= X(1,:) + abs(X(j,:) - X(1,:)).*AA' * exp(z*l) * cos(2*pi*l);
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

% Application of simple limits/bounds
function s = Bounds( s, Lb, Ub)
  % Apply the lower bound vector
  temp = s;
  I = temp < Lb;
  temp(I) = Lb(I);
  
  % Apply the upper bound vector 
  J = temp > Ub;
  temp(J) = Ub(J);
  % Update this new move 
  s = temp;
end



