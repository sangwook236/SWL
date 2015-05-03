function retval = mse(data)
%
% mean squared error
%

retval = mean((data - mean(data)).^2);
