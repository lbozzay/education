function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations.
%
% Hint: When computing the gradient of the regularized cost function,
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta;
%           temp(1) = 0;   % because we don't add anything for j = 0
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
  fprintf("size X: %f \n", size(X));
  fprintf("size theta: %f \n", size(theta));
  fprintf("\n");
  fprintf("theta: %f\n", theta);
  fprintf("thetaT: %f", theta');
  h = sigmoid(X * theta);
  fprintf("\n");
  fprintf("h: %f\n", h);
  J_p = (-y .* log(h))-((1-y) .* log(1-h));
  fprintf("J_p: %f\n", J_p);
  J_nonreg = 1/m .* sum(J_p);
  fprintf("J_nonreg: %f\n", J_nonreg);

  J_reg = lambda / (2 * m) .* sum(theta(2:end).^2);
  fprintf("J_reg: %f\n", J_reg);

  J = J_nonreg + J_reg;
  fprintf("J: %f\n", J);

  grad_non_reg = 1/m .* (X' * (h-y));
  fprintf("grad_non_reg: %f\n", grad_non_reg);
  fprintf("grad_non_reg size: %f\n", size(grad_non_reg));
  grad_reg = grad_non_reg + ((lambda / m) * sum(theta(2:end).^2));
  fprintf("grad_reg: %f\n", grad_reg);
  fprintf("grad_reg size: %f\n", size(grad_reg));

  grad(1) = grad_non_reg(1);
  grad(2:end) = grad_non_reg(2:end) + ((lambda / m) * (theta(2:end)));






% =============================================================

grad = grad(:);

end
