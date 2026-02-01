import numpy as np

# Importo el archivo donde limpio los datos

from limpiardatos import limpiardatos

class PerceptronSimple:
    def __init__(self, n_features, learning_rate=0.01, epochs=100):
        # Inicializamos pesos en cero o valores aleatorios pequeños
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, z):
        # Función Sigmoide para devolver valores entre 0 y 1
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        # El paso de predicción: X * W + b
        z = np.dot(X, self.weights) + self.bias
        return self.activation(z)

    def train(self, X, y):
        # Ciclo de entrenamiento (épocas)
        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                # 1. Predicción (Forward pass)
                y_pred = self.predict(X[i])
                
                # 2. Calcular el error
                error = y[i] - y_pred
                
                # 3. Actualizar pesos y sesgo (Gradiente descendente simplificado)
                # Peso_nuevo = Peso_actual + (LR * Error * Entrada)
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

df = limpiardatos()

X_data = df.drop(columns=['Status', 'Days'], errors='ignore').to_numpy()
y_data = df['Status'].to_numpy()

# Crear la neurona
mi_neurona = PerceptronSimple(n_features=X_data.shape[1], learning_rate=0.1, epochs=50)

# Entrenar
mi_neurona.train(X_data, y_data)

# Probar con un dato nuevo
prediccion = mi_neurona.predict(X_data[0])
print(f"Probabilidad de riesgo: {round(prediccion, 2)}")