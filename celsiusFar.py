import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([capa])

model.compile(
     optimizer=tf.keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
)

print("Comenzando entrenamiento...")

historial = model.fit(celsius, fahrenheit, epochs=500, verbose=False)

plt.xlabel('# Epoca')
plt.ylabel('Magnitud de pérdida')
plt.plot(historial.history["loss"])
plt.show()

print("Modelo entrenado!")
print("-------------!")
print("Hagamos una prediccion!")

resultado = model.predict([100.0])

print("El resultado es " + str(resultado) + " fahrenheit!")
