import matplotlib.pyplot as plt
import numpy as np
import neurolab as nl
# enseñar a sumar dos valores entre -0.5 y 0.5
# train samples
input = np.random.uniform(-0.5, 0.5, (10, 2)) # 10 entradas de dos en dos
target = (input[:, 0] + input[:, 1]).reshape(10, 1) # array con la suma de los pares de entradas

# network with 2 inputs, 5 neurons in input layer and 1 in output layer
net = nl.net.newff([[-0.5, 0.5], [-0.5, 0.5]], [5, 1])
# train process
err = net.train(input, target, show=20)

result = net.sim([[0.2, 0.1], [0.2, 0.1]]) # 0.2 + 0.1
print(result)

plt.figure()
plt.plot(err)
plt.xlabel('numero de epocas')
plt.ylabel('error de entrenamiento')
plt.grid()
plt.show()