function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sigmoid_z = sigmoid(X * theta); 
sigmoid_log = log(sigmoid_z);
y_inverse = y';

y_1 = (-y_inverse) * sigmoid_log;

y_0 = 1 - sigmoid_z;
y_0 = log(y_0);
subtract_y_1 = (1 - y_inverse);
y_0 = subtract_y_1 * y_0; 

j = y_1 - y_0;
J = j / m;

% regularization
theta(1) = 0;
squared_theta = (theta') * theta;
regular = lambda / (2 * m);
regular =  regular * squared_theta;
J = J + regular;

% theta values

start = (sigmoid_z - y)';
start = start * X;
grad = (start / m);

grad_regular = theta * (lambda / m);
grad = grad + grad_regular';

% =============================================================

end
