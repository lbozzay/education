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

%fprintf("X size: %f\n", size(X)); % 12 x 2
%fprintf("y size: %f\n", size(y)); % 12 x 1
%fprintf("theta size: %f\n", size(theta)); % 2 x 1
%fprintf("lambda: %f\n", lambda); % 1


h = X * theta;

J =  1 / (2 * m) * sum((h - y).^2);

theta(1) = 0;

J = J + (lambda / (2 * m) * sum(theta.^2));


%fprintf("J: %f\n", J);
%fprintf("J: %f\n", size(J));
grad_0 = 1/m * (X(:,1)' * (h-y));
%  fprintf("grad_non_reg: %f\n", grad_non_reg);
%  fprintf("grad_non_reg size: %f\n", size(grad_non_reg));
grad_rest = (1/m * (X(:,2:end)' * (h-y))) + ((lambda / m) * sum(theta.^2));
%  fprintf("grad_reg: %f\n", grad_reg);
%  fprintf("grad_reg size: %f\n", size(grad_reg));

grad(1) = grad_0;
grad(2:end) = grad_rest;










% =========================================================================

grad = grad(:);

end
