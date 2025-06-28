import torch
torch.manual_seed(5387)  # For reproducibility

# Create a random tensor of shape (1, 10)
x = torch.randn(1,10, requires_grad=True)

prev_h = torch.randn(1, 20, requires_grad=True)  # Previous hidden state
W_h = torch.randn(20, 20, requires_grad=True)  # Weight matrix for hidden state
W_x = torch.randn(20, 10, requires_grad=True)  # Weight matrix for input

i2h = torch.mm(W_x, x.t())  # Input to hidden
h2h = torch.mm(W_h, prev_h.t())  # Hidden to hidden

next_h = torch.tanh(i2h + h2h)  # New hidden state
loss = next_h.sum()  # Example loss function
loss.backward()  # Backpropagation. This computes gradients for W_x, W_h, x, and prev_h so that we can update them later to minimize the loss.
# Print gradients
print("Gradients:")
print("W_x:", W_x.grad)
print("W_h:", W_h.grad)
print("x:", x.grad)
print("prev_h:", prev_h.grad)

print(next_h)