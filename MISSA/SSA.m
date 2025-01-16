%_________________________________________________________________________%
% ��ȸ�Ż��㷨             %
%_________________________________________________________________________%
function [Best_pos,Best_score,curve]=SSA(src3,src4,pop,Max_iter,lb,ub,dim,fobj)

ST = 0.6;%Ԥ��ֵ
PD = 0.7;%�����ߣ�̽����/�����ߣ��ı��У�ʣ�µ��Ǽ����ߣ�������/ʰ���ߣ�
SD = 0.2;%��ʶ����Σ����ȸ�ı��أ��������/������/�����ߣ�

PDNumber = round(pop*PD); %����������
SDNumber = round(pop*SD); %��ʶ����Σ����ȸ����

if(max(size(ub)) == 1)
   ub = ub.*ones(1,dim);
   lb = lb.*ones(1,dim);  
end

%% ��Ⱥ��ʼ��
X0=initialization(pop,dim,ub,lb);
X = X0;

%% �����ʼ��Ӧ��ֵ
fitness = zeros(1,pop);
for i = 1:pop
   fitness(i) =  fobj(src3,src4,X(i,:));
end

%% ��Ӧ��ֵ����
[fitness, index]= sort(fitness,'descend');%����
BestF = fitness(1);
WorstF = fitness(end);
GBestF = fitness(1);%ȫ��������Ӧ��ֵ

%% ��������������������Ⱥλ��
for i = 1:pop
    X(i,:) = X0(index(i),:);
end
curve=zeros(1,Max_iter);
GBestX = X(1,:);%ȫ������λ��
X_new = X;

%% ���¹���
for i = 1: Max_iter
    
    BestF = fitness(1);
    WorstF = fitness(end);   
    R2 = rand(1);
    
    % �����߸���
   for j = 1:PDNumber
      if(R2<ST)
          X_new(j,:) = X(j,:).*exp(-j/(rand(1)*Max_iter));
      else
          X_new(j,:) = X(j,:) + randn()*ones(1,dim);
      end     
   end
   
   % �����߸���
   for j = PDNumber+1:pop
%        if(j>(pop/2))
        if(j>(pop - PDNumber)/2 + PDNumber)
          X_new(j,:)= randn().*exp((X(end,:) - X(j,:))/j^2);
       else
          %����-1��1�������
          A = ones(1,dim);
          for a = 1:dim
            if(rand()>0.5)
                A(a) = -1;
            end
          end 
          AA = A'*inv(A*A');     
          X_new(j,:)= X(1,:) + abs(X(j,:) - X(1,:)).*AA';
       end
   end
   
   % Ԥ������
   Temp = randperm(pop);
   SDchooseIndex = Temp(1:SDNumber); 
   for j = 1:SDNumber
       if(fitness(SDchooseIndex(j))<BestF) % ��С����Ӧ�Ⱥ�������S>B���������Ӧ�Ⱥ�������S<B����
           X_new(SDchooseIndex(j),:) = X(1,:) + randn().*abs(X(SDchooseIndex(j),:) - X(1,:));
       elseif(fitness(SDchooseIndex(j))== BestF)
           K = 2*rand() -1;
           X_new(SDchooseIndex(j),:) = X(SDchooseIndex(j),:) + K.*(abs( X(SDchooseIndex(j),:) - X(end,:))./(fitness(SDchooseIndex(j)) - fitness(end) + 10^-8));
       end
   end
   
   %�߽����
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
   
   %����λ��
   for j=1:pop
    fitness_new(j) = fobj(src3,src4,X_new(j,:));
   end
   for j = 1:pop
    if(fitness_new(j) > GBestF)  % ��С����Ӧ�Ⱥ�������f<G���������Ӧ�Ⱥ�������f>G����
       GBestF = fitness_new(j);
       GBestX = X_new(j,:);   
    end
   end
   X = X_new;
   fitness = fitness_new;
   
   %�������
   [fitness, index]= sort(fitness,'descend'); % ����
   BestF = fitness(1);
   WorstF = fitness(end);
   for j = 1:pop
      X(j,:) = X(index(j),:);
   end
   curve(i) = GBestF;
   % ���ÿ�ֵ�������Ӧ��ֵ
   disp(['Iteration ' num2str(i) ': Best Score = ' num2str(GBestF)]);
end
Best_pos =GBestX;
Best_score = curve(end);
end



