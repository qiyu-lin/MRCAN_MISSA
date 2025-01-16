% ʹ�÷���
%__________________________________________
% fobj = @YourCostFunction        �趨��Ӧ�Ⱥ���
% dim = number of your variables   �趨ά��
% Max_iteration = maximum number of generations �趨����������
% SearchAgents_no = number of search agents   ��Ⱥ����
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n  �����±߽�
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n   �����ϱ߽�
% If all the variables have equal lower bound you can just
% define lb and ub as two single number numbers

% To run SSA: [Best_pos,Best_score,curve]=SSA(pop,Max_iter,lb,ub,dim,fobj)
%__________________________________________

clear all 
clc

%% ��ȡSRͼ��
I1 = imread('D:/JWLprocess/MRCAN_MISSA/Wafer_match/Wafer095/00_MRCAN.png');
I2 = imread('D:/JWLprocess/MRCAN_MISSA/Wafer_match/Wafer095/05_MRCAN.png');

%% ��ȡHRͼ��
I3 = imread('D:/JWLprocess/MRCAN_MISSA/Wafer_match/Wafer095/00.png');
I4 = imread('D:/JWLprocess/MRCAN_MISSA/Wafer_match/Wafer095/05.png');

%% �ҶȻ�
if size(I1, 3) == 3               % SR1
    I1_2 = rgb2gray(I1);
else
    I1_2 = I1;
end
if size(I2, 3) == 3               % SR2
    I2_2 = rgb2gray(I2);
else
    I2_2 = I2;
end
if size(I3, 3) == 3               % HR1
    I3_2 = rgb2gray(I3);
else
    I3_2 = I3;
end
if size(I4, 3) == 3               % HR2
    I4_2 = rgb2gray(I4);
else
    I4_2 = I4;
end

%% ��ȡ��ͼ
src1 = I1_2(1:40, 1:40, :);       % SR1
src2 = I2_2(1:40, 1:40, :);       % SR2
src3 = I3_2(1:40, 1:40, :);       % HR1
src4 = I4_2(1:40, 1:40, :);       % HR2

%% ��ͼ�Ŵ�10��
srcc1 = imresize(src1, 50, 'bicubic');  % SR1
srcc2 = imresize(src2, 50, 'bicubic');  % SR2
srcc3 = imresize(src3, 50, 'bicubic');  % HR1
srcc4 = imresize(src4, 50, 'bicubic');  % HR2

%% �Ż�
rng('default');

SearchAgents_no=50; % ��Ⱥ����

Function_name='F24'; % �趨��Ӧ�Ⱥ���

Max_iteration=50; % �趨����������

[lb,ub,dim,fobj]=Get_Functions_details(Function_name);  %�趨�߽��Լ��Ż�����

[Best_pose_MRCAN_MISSA,Best_score_MRCAN_MISSA,MRCAN_MISSA_curve]=MRCAN_MISSA(srcc1,srcc2,SearchAgents_no,Max_iteration,lb,ub,dim,fobj); 
[Best_pose_SSA,Best_score_SSA,SSA_curve]=SSA(srcc3,srcc4,SearchAgents_no,Max_iteration,lb,ub,dim,fobj); 
[Best_score_GWO,Best_pos_GWO,GWO_cg_curve]=GWO(srcc3,srcc4,SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[GlobalBest, BestCost]=PSO(srcc3,srcc4,SearchAgents_no,Max_iteration,lb,ub,dim,fobj);
[AfMax,AbestX,ASFSSA_curve]=ASFSSA(srcc3,srcc4,SearchAgents_no,Max_iteration,lb,ub,dim,fobj);  

%% ���չʾ
semilogy(MRCAN_MISSA_curve,'LineWidth',1.5,'LineStyle','-','Color','r')   %��ɫʵ��
hold on
semilogy(SSA_curve,'LineWidth',1.5,'LineStyle',':','Color','b')     %��ɫ���� 
hold on
semilogy(GWO_cg_curve,'LineWidth',1.5,'LineStyle','-.','Color','k') %��ɫ�㻮��
hold on
semilogy(BestCost,'LineWidth',1.5,'LineStyle','-.','Color','m');    %Ʒ��ɫ�㻮��
hold on
semilogy(ASFSSA_curve,'LineWidth',1.5,'LineStyle','--','Color','g')  %��ɫ����
title('Wafer095')
xlabel('Iteration');
ylabel('MI');

axis tight
grid on
box on
legend('Proposed','SSA','GWO','PSO','ASFSSA')

display(['MRCAN_MISSA����λ�� : ', num2str(Best_pose_MRCAN_MISSA)]);
display(['MRCAN_MISSA������Ӧ��ֵ : ', num2str(Best_score_MRCAN_MISSA)]);

display(['SSA����λ�� : ', num2str(Best_pose_SSA)]);
display(['SSA������Ӧ��ֵ : ', num2str(Best_score_SSA)]);

display(['GWO����λ�� : ', num2str(Best_pos_GWO)]);
display(['GWO������Ӧ��ֵ : ', num2str(Best_score_GWO)]);

display(['PSO����λ�� : ', num2str(GlobalBest.Position)]);
display(['PSO������Ӧ��ֵ : ', num2str(GlobalBest.Cost)]);

display(['ASFSSA����λ�� : ', num2str(AfMax)]);
display(['ASFSSA������Ӧ��ֵ : ', num2str(AbestX)]);  



