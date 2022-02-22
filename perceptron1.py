import matplotlib.pyplot as plt
import neurolab as nb

input = [[0, 0], [0, 1], [1, 0], [1, 1]]
clasificacion = [[0], [0], [0], [1]]

net = nb.net.newp([[0, 1], [0, 1]], 1)
error_progreso = net.train(input, clasificacion, epochs=500, show=20, lr=0.1)

plt.figure()
plt.plot(error_progreso)
plt.xlabel('numero de epocas')
plt.ylabel('error de entrenamiento')
plt.grid()
plt.show()
