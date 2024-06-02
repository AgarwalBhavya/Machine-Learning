import numpy as np

matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])

result_addition = matrix_a + matrix_b# Addition
print(result_addition)
result_subtraction = matrix_a - matrix_b #Substraction
print(result_subtraction)
result_elementwise_multiply = matrix_a * matrix_b# Element-wise multiplication
print(result_elementwise_multiply)
result_matrix_multiply = np.dot(matrix_a, matrix_b)# Matrix multiplication (dot product)
print(result_matrix_multiply)
matrix_a_transpose = np.transpose(matrix_a)# Transpose of a matrix
print(matrix_a_transpose)
matrix_a_inverse = np.linalg.inv(matrix_a)# Inverse of a matrix (assuming it's invertible)
print(matrix_a_inverse)
matrix_a_determinant = np.linalg.det(matrix_a)# Determinant of a matrix
print(matrix_a_determinant)
eigenvalues, eigenvectors = np.linalg.eig(matrix_a)# Eigenvalues and eigenvectors
print(eigenvalues, eigenvectors)
matrix_trace = np.trace(matrix_a)# Matrix trace (sum of diagonal elements)
print(matrix_trace)
matrix_rank = np.linalg.matrix_rank(matrix_a)# Matrix rank
print(matrix_rank)
A = np.array([[2, 1], [1, 3]])# Solving linear equations (Ax = b)
b = np.array([4, 5])# Given Ax = b, find x where A and b are known
x = np.linalg.solve(A, b)
print(x)
identity_matrix = np.identity(3)  # Creates a 3x3 identity matrix
print(identity_matrix)
zero_matrix = np.zeros((2, 3))  # Creates a 2x3 matrix filled with zeros
print(zero_matrix)
ones_matrix = np.ones((3, 2))  # Creates a 3x2 matrix filled with ones
print(ones_matrix)
random_matrix = np.random.rand(2, 2)  # Creates a 2x2 matrix with random values between 0 and 1
print(random_matrix)
custom_identity_matrix = np.full((4, 4), 5)# Identity matrix with a custom value (e.g., 5)
print(custom_identity_matrix)
matrix_to_reshape = np.array([1, 2, 3, 4, 5, 6])# Reshaping a matrix
reshaped_matrix = matrix_to_reshape.reshape(2, 3)
print(reshaped_matrix)
matrix_shape = matrix_a.shape# Getting the dimensions of a matrix
print(matrix_shape)
element = matrix_a[0, 1]  # Accesses the element in the first row, second column in a matrix
print(element)
sliced_matrix = matrix_a[0:2, 0:2]  # Slices a 2x2 submatrix from the top-left corner from a matrix
print(sliced_matrix)
concatenated_matrix = np.concatenate((matrix_a, matrix_b), axis=1) # Concatenating matrices Along columns
print(concatenated_matrix)
sqrt_matrix_a = np.sqrt(matrix_a)# Element-wise functions (e.g., square root of each element)
print(sqrt_matrix_a)
exp_matrix_a = np.exp(matrix_a)# Element-wise exponentiation
print(exp_matrix_a)
log_matrix_a = np.log(matrix_a)# Element-wise logarithm
print(log_matrix_a)
sin_matrix_a = np.sin(matrix_a)# Element-wise sine
print(sin_matrix_a)
cos_matrix_a = np.cos(matrix_a)# Element-wise cosine
print(cos_matrix_a)