# RNN

Recurrent Neural Networks (RNNs) are a type of neural network architecture designed to handle sequential data. Unlike traditional feedforward neural networks, RNNs have connections that form cycles, allowing them to maintain a hidden state that captures information from previous inputs in the sequence. This makes them particularly well-suited for tasks like time series prediction, language modeling, and sequence generation.



Input Sequence: Let x_1, x_2, ..., x_T be an input sequence, where each x_t is a vector representing the input at time step t. The sequence length is T.

Hidden State: The hidden state h_t at time step t is a vector that captures information from the current input x_t and the previous hidden state h_(t-1). The hidden state is updated using the following equation:
h_t = tanh(W_xh * x_t + W_hh * h_(t-1) + b_h)
where:
W_xh is the weight matrix for the input to hidden connection.
W_hh is the weight matrix for the hidden to hidden connection.
b_h is the bias vector for the hidden state.
tanh is the hyperbolic tangent activation function, which squashes the values to be within the range [-1, 1].

Output: The output y_t at time step t is computed from the hidden state h_t using the following equation:
y_t = W_hy * h_t + b_y
where:
W_hy is the weight matrix for the hidden to output connection.
b_y is the bias vector for the output.
Sequence Processing: The RNN processes the input sequence step by step, updating the hidden state h_t at each time step t. The initial hidden state h_0 is typically initialized to a vector of zeros:
h_0 = 0 (or some other initialization method)
The final output sequence is y_1, y_2, ..., y_T.


Backpropagation Through Time (BPTT): Training an RNN involves minimizing a loss function (e.g., mean squared error) over the entire sequence. The gradients of the loss with respect to the weights are computed using a method called Backpropagation Through Time (BPTT), which is an extension of the standard backpropagation algorithm to handle the temporal dependencies in the data.

During BPTT, the gradients are computed for each time step t, and they are backpropagated through the network over time. The weight updates are performed as follows:
W_xh <- W_xh - learning_rate * dL/dW_xh
W_hh <- W_hh - learning_rate * dL/dW_hh
W_hy <- W_hy - learning_rate * dL/dW_hy
b_h <- b_h - learning_rate * dL/db_h
b_y <- b_y - learning_rate * dL/db_y
where dL/dW_xh, dL/dW_hh, dL/dW_hy, dL/db_h, and dL/db_y are the gradients of the loss L with respect to the weights and biases.

Gradient Clipping: RNNs can suffer from the problem of exploding gradients, where the gradients grow exponentially during backpropagation. To mitigate this, gradient clipping is often used to limit the magnitude of the gradients:
clip(dparam, -threshold, threshold)
where threshold is a predefined value.


Initialization: Initialize the hidden state h_0 and the weight matrices W_xh, W_hh, and W_hy, as well as the bias vectors b_h and b_y.

Forward Pass:
For each time step t in the sequence:
Compute the hidden state h_t using the current input x_t and the previous hidden state h_(t-1).
Compute the output y_t from the hidden state h_t.

Backward Pass (BPTT):
Compute the gradients of the loss with respect to the weights and biases.
Backpropagate the gradients through time, updating the weights and biases.

Training:
Iterate over the training dataset, performing forward and backward passes for each sequence to minimize the loss.
