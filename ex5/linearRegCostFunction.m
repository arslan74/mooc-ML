function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h =  X * theta;
error_vector = h - y;
error_square = error_vector .^ 2;
error_sum = sum(error_square);
cost = error_sum/ (2*m);
J = cost;

grad = (X' * error_vector) / m;

theta(1) = 0;

theta_squ = sum(theta .^ 2) * (lambda / (2*m));
J = cost + theta_squ;

grad = grad + theta * (lambda / m);










% =========================================================================

grad = grad(:);

end
