% 在MATLAB命令行窗口中调用：calculateStandardDeviation('data.xlsx')

function std_dev = calculateStandardDeviation(filename)
    % 读取Excel文件中的数据
    data = xlsread(filename);
    
    % 计算差值，即第一列（实际值）减去第二列（计算值）
    differences = data(:, 1) - data(:, 2);

    % 将差值添加到数据矩阵中，作为第三列
    data = [data, differences];
    xlswrite(filename, data);

    % 计算差值的标准方差
    std_dev = std(differences);

    % 显示结果
    fprintf('差值的标准方差: %f\n', std_dev);
end
