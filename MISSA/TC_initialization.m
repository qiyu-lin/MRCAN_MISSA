function Positions=TC_initialization(SearchAgents_no,dim,ub,lb)
    Boundary_no= size(ub,2); % numnber of boundaries
    
    % 初始化Tent-Cubic混沌序列
    z = zeros(1, SearchAgents_no*dim);
    alpha = rand();
    z(1) = rand();
    r = 3.6;
    
    % 生成Tent-Cubic混沌序列
    for k=1:(SearchAgents_no*dim-1)
        if z(k) < alpha
            z(k+1)=mod(2*r*z(k)*(1-z(k)*z(k))+(8-r)*z(k)/2,1);
        else
            z(k+1) = mod(2*r*z(k)*(1-z(k)*z(k))+(8-r)*(1-z(k))/2,1);
        end
    end
    
    % 如果所有变量的界限都相等
    if Boundary_no==1
        Positions = z(1:SearchAgents_no*dim)'.*(ub-lb)+lb;
    end
    
    % 如果每个变量有不同的lb和ub
    if Boundary_no>1
        idx = 1;
        for i=1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:,i) = z(idx:idx+SearchAgents_no-1)'.*(ub_i-lb_i)+lb_i;
            idx = idx + SearchAgents_no;
        end
    end
end
