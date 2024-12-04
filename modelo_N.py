import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

# Ruta del archivo de datos
ruta_archivo = "datos_reemplazados_no_estesi-categ.csv"
datos = pd.read_csv(ruta_archivo)

# Separar características y objetivos
caracteristicas = datos.drop(columns=['PUNT_INGLES', 'PUNT_MATEMATICAS', 'PUNT_SOCIALES_CIUDADANAS',
                               'PUNT_C_NATURALES', 'PUNT_LECTURA_CRITICA', 'PUNT_GLOBAL'])
objetivos = datos[['PUNT_INGLES', 'PUNT_MATEMATICAS', 'PUNT_SOCIALES_CIUDADANAS',
                'PUNT_C_NATURALES', 'PUNT_LECTURA_CRITICA', 'PUNT_GLOBAL']]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(caracteristicas, objetivos, test_size=0.2, random_state=42)

# Escalar las características
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

# Definir el modelo de red neuronal
modelo = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_escalado.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(6, activation='linear')
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])

# Entrenar el modelo
historial = modelo.fit(
    X_train_escalado, y_train,
    validation_data=(X_test_escalado, y_test),
    epochs=50,
    batch_size=32,
    verbose=1)

# Evaluar el modelo
perdida, mae = modelo.evaluate(X_test_escalado, y_test)
print(f'Pérdida en prueba: {perdida}, MAE: {mae}')

# Guardar el modelo
modelo.save('modelo_multitarea.h5')

# Cargar el modelo y hacer una predicción de ejemplo
modelo = tf.keras.models.load_model('modelo_multitarea.h5')

# Ejemplo de entrada para predicción
entrada_ejemplo = np.array([0, 1, 6, 1, 0, 1, 2, 3, 2, 1, 1, 1, 1]).reshape(1, -1)
entrada_escalada = escalador.transform(entrada_ejemplo)

# Predecir puntajes
puntajes_predichos = modelo.predict(entrada_escalada)

# Imprimir puntajes predichos
print("Puntajes predichos:")
print(f"Inglés: {puntajes_predichos[0][0]:.2f}")
print(f"Matemáticas: {puntajes_predichos[0][1]:.2f}")
print(f"Sociales y Ciudadanas: {puntajes_predichos[0][2]:.2f}")
print(f"Ciencias Naturales: {puntajes_predichos[0][3]:.2f}")
print(f"Lectura Crítica: {puntajes_predichos[0][4]:.2f}")
print(f"Global: {puntajes_predichos[0][5]:.2f}")
