%% 淘个代码 %%
% 2023/10/04 %
%微信公众号搜索：淘个代码，获取更多免费代码
%禁止倒卖转售，违者必究！！！！！
%唯一官方店铺：https://mbd.pub/o/author-amqYmHBs/work，其他途径都是骗子！
function [ result ] = func_levy( x,bestX )
% Levy flights 
beta = 1.5 ;
[N,D] = size(x) ;
sigma_u = (gamma(1+beta)*sin(pi*beta/2)/(beta*gamma((1+beta)/2)*2^((beta-1)/2)))^(1/beta) ;
sigma_v = 1 ;
u = normrnd(0,sigma_u,N,D) ;
v = normrnd(0,sigma_v,N,D) ;
step = u./(abs(v).^(1/beta)) ;
l = 0.01 * ( x  - bestX); 
result = x + l .* step ;
end