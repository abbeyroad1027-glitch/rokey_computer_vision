import torch
data = [[10,20],[30,40],[50,60]]

x = torch.tensor(data)

print(f"Tensor Dtype: {x.dtype}")
print(f"Tensor Shape: {x.shape}")

a = torch.tensor([[1,2], [3,4]])
b = torch.tensor([[10,20],[30,40]])

element_wise_mul = a*b
matrix_mul = a@b

print('='*50)

x = torch.tensor(4.0, requires_grad=True)
y = 3*x**2 + 2
y.backward()
print(x.grad)

print('='*50)

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
z = 4*a+5*b**2
z.backward()
print(a.grad)
print(b.grad)

