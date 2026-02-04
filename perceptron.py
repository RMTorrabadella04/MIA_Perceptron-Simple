from limpiardatos import limpiardatos
import numpy as np
import pandas as pd

class perceptron():
    
    # Constructor
    
    def __init__(self, eta=0.1, n_iter=50, bias=1.0):
        self.eta = eta
        self.n_iter = n_iter
        self.bias = bias
        self.w = None

    def fit(self):
        # Recogo los datos de entrenamiento
        
        x_train, y_train = recogerDatos(1)
        
        self.w = np.zeros(x_train.shape[1])
        
        for epoch in range(self.n_iter):
            for xi, target in zip(x_train.values, y_train.values):
                # 1. Calculamos predicción (Valor neto y función escalón)
                prediction = self.predict(xi)
                
                # 2. Calculamos la actualización (Delta w) 
                update = self.eta * (target - prediction)
                
                # 3. Actualizamos pesos y bias: w = w + (delta_w * x)
                self.w += update * xi
                self.bias += update
        return self
    
    def new_input(self, x):
        return np.dot(x, self.w) + self.bias
    
    def predict(self, X):
        return np.where(self.new_input(X) >= 0.0, 1, 0)
    
    def score(self, X, y):
        predicciones = self.predict(X)
        
        # Comparamos cuántos son iguales y sacamos la media (aciertos / total)
        
        aciertos = np.sum(predicciones == y)
        total = len(y)
        accuracy = (aciertos / total) * 100
        return accuracy


# Pillo los datos del otro archivo y los voy separando aquí para evitar errores de filtraciones

# Esta parte la hice con pandas para poder quitar la columna Status con más facilidad,
# luego de ser necesario pasaremos a numpy

def recogerDatos(queDato=0):
    datos = limpiardatos()
    
    # Randomizamos los datos, con semilla para que se repita siempre el orden y salga el mismo resultado
    
    datos = datos.sample(frac=1, random_state=41).reset_index(drop=True)
    
    # Cortamos 80-20
    
    corte = int(len(datos) * 0.8)
    
    train = datos.iloc[:corte]
    test = datos.iloc[corte:]
    
    if queDato == 0:
        return datos # Le paso todo (Seguramente no se use, pero por si acaso)

    # X son todas las columnas excepto "Status"
    # Y es solo la columna "Status"

    elif queDato == 1:
        X_train = train.drop(columns=["Status"]) 
        y_train = train["Status"]
        return X_train, y_train
        
    elif queDato == 2:
        X_test = test.drop(columns=["Status"])
        y_test = test["Status"]
        return X_test, y_test



if __name__ == "__main__":
    modelo = perceptron(eta=0.01, n_iter=20)

    # Entrenamos 
    print("Entrenando el modelo...")
    modelo.fit() 

    # Probamos
    
    X_test, y_test = recogerDatos(2)
    
    precision = modelo.score(X_test, y_test)
    
    print(f"\n--- RESULTADOS ---")
    print(f"Pesos finales: {modelo.w}")
    print(f"Bias final: {modelo.bias}")
    print(f"Precisión del modelo (Accuracy): {precision:.2f}%")