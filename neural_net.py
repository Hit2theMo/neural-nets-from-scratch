import numpy as np
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

layer_op = []
for i in range(len(weights)):
    dot = np.dot(weights[i], inputs)+biases[i]
    layer_op.append(dot)
print(layer_op)

'''
#Using ZIP
layer_op = []
for i in range(len(weights)):
    nn = 0
    for a, b in zip(inputs, weights[i]):
        nn += (a * b)
    layer_op.append(nn+biases[i])
print(layer_op)
'''

