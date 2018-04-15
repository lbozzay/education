function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% y labels should be 10 dimensional vectors containing only 0,1 values representing the value of label
%fprintf("num_labels %f\n", num_labels);
yi = zeros(m,num_labels);
for i = 1:m
  yi(i,y(i)) = 1;
endfor

%fprintf("yi %f\n", size(yi));

% add column of 1s to X
X1 = ones(m,1);
X = [X1 X]; % 5000 x 401

%compute J
%Theta1 25 x 401
%Theta2 10 x 26
z2 = X * Theta1';
a2 = sigmoid(z2);

aones = ones(m,1);
a2 = [aones a2]; % 5000 x 26
%fprintf("a2 %f\n", size(a2));

z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3; % 5000 x 10
%a2 = zeros(5000,25);

%for i=1:m
  %Perform forward propagation and backpropagation using example (x(i),y(i))

%  a2(i,:) = sigmoid(X(i,:) * Theta1');


  % yi 5000 x 10
  %J = J + sum((-yi(i,:) .* log(h(i,:)))-(1-yi(i,:) .* log(1-h(i,:))));
  %fprintf("J: %f\n", J);
%endfor
%fprintf("z2 %f\n", z2(1,:));
%J = 1/m * J;
J = (1/m)*sum(sum((-yi).*log(h) - (1-yi).*log(1-h), 2));
%fprintf('h %f', size(h));

regularized = (lambda / (2 * m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

J = J + regularized;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

D = 0;

    delta3 = a3 - yi; %delta3

    delta2 = (delta3 * Theta2); % 5000 x 26
    %fprintf("delta2 size before colum wihtdrawal %f \n", size(delta2));

    %fprintf("delta2 size after colum wihtdrawal %f \n", size(delta2));
    %fprintf("z2 size %f \n", size(z2)); % 5000 x 25
    delta2 = delta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2]); %delta2
    delta2 = delta2(:,2:end); % 5000 x 25
    Theta2_grad = delta3;
    Theta1_grad = delta2;

    Delta_1 = delta2'*X;
    Delta_2 = delta3'*a2;

    Theta1_grad = Delta_1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
    Theta2_grad = Delta_2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
