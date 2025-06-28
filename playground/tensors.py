# Learning pytorch syntax
import torch

# Create a tensor
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # Specify data type

# create a random tensor. 
# this is often used for initializing weights in neural networks
torch.manual_seed(5387)  # For reproducibility
y = torch.rand(2, 2, dtype=torch.float32)  # Random tensor of shape (2, 3)
# Print the tensor
print(y)


# tensors with similar shapes can be added together, multiplied, etc.
print(x + y)  # Element-wise addition
print(x * y)  # Element-wise multiplication

print(x @ y)  # Matrix multiplication
print(x @ y.T)  # Matrix multiplication with transpose

print(x*2)  # Scalar multiplication
print(x + 2)  # Scalar addition

print(x.shape)
print(x.size()) # this is the same as shape
print(x.dtype)
print(x.device)  # Device information (CPU or GPU)

# to change range of rand
z = (torch.rand(2, 2, dtype=torch.float32) - 0.5) * 10  # Random tensor of shape (2, 3) with values in [-5, 5]
print(z)

#.abs()
#.asin() inverse sine
#det() determinant
#.svd() singular value decomposition
#.eig() eigenvalues and eigenvectors
#.inverse() matrix inverse
#.diag() diagonal matrix
#.std_meanUn() standard deviation and mean
#.norm() vector norm
#.sort() sort elements


x = torch.tensor([[1, 2, 3], 
                  [3, 4, 5],
                  [5, 6, 7]], dtype=torch.float32)

