import matplotlib.pyplot as plt
import numpy as np
import neurolab as nl

# train examples
x = np.linspace(-7, 7, 150)
y = np.sin(x) * 0.5

size = len(x)
input = x.reshape(size, 1) # para separar cada input
target = y.reshape(size, 1)
plt.subplot(211)
plt.plot(target, color='#5a7d9a', marker='o', label='target')

# red con dos capas, capa de entrada con 5 neuronas y capa de salida con 1
net = nl.net.newff([[-7, 7]], [5, 1])

# entrenar red
error = net.train(input, target, epochs=500, show=100, goal=0.02)

#simular la red
out = net.sim(input)
plt.plot(out, color='#adad3b', marker='p', label='output')
plt.legend()

plt.subplot(212)
plt.plot(error)
plt.xlabel('numero de epocas')
plt.ylabel('error de entrenamiento')
plt.grid()

plt.show()