% ��MATLAB�����д����е��ã�calculateRMSE('data.xlsx')

function rmse = calculateRMSE(filename)
    % ��ȡExcel�ļ��е�����
    data = xlsread(filename);
    
    % ��ȡʵ��ֵ��Ԥ��ֵ
    actual = data(:, 1); % ��һ����ʵ��ֵ
    predicted = data(:, 2); % �ڶ�����Ԥ��ֵ
    
    % ����������
    mse = mean((actual - predicted).^2);
    
    % ������������
    rmse = sqrt(mse);
    
    % ��ʾ���
    fprintf('��������� (RMSE): %f\n', rmse);
end
