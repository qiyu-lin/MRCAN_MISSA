% 在MATLAB命令行窗口中调用：calculateRMSE('data.xlsx')

function rmse = calculateRMSE(filename)
    % 读取Excel文件中的数据
    data = xlsread(filename);
    
    % 提取实际值和预测值
    actual = data(:, 1); % 第一列是实际值
    predicted = data(:, 2); % 第二列是预测值
    
    % 计算均方误差
    mse = mean((actual - predicted).^2);
    
    % 计算均方根误差
    rmse = sqrt(mse);
    
    % 显示结果
    fprintf('均方根误差 (RMSE): %f\n', rmse);
end
