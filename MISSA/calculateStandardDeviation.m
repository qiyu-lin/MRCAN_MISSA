% ��MATLAB�����д����е��ã�calculateStandardDeviation('data.xlsx')

function std_dev = calculateStandardDeviation(filename)
    % ��ȡExcel�ļ��е�����
    data = xlsread(filename);
    
    % �����ֵ������һ�У�ʵ��ֵ����ȥ�ڶ��У�����ֵ��
    differences = data(:, 1) - data(:, 2);

    % ����ֵ��ӵ����ݾ����У���Ϊ������
    data = [data, differences];
    xlswrite(filename, data);

    % �����ֵ�ı�׼����
    std_dev = std(differences);

    % ��ʾ���
    fprintf('��ֵ�ı�׼����: %f\n', std_dev);
end
