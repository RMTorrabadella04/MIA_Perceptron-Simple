import pandas as pd
import numpy as np

# Sklearn lo utilizaremos solo para escalar los datos

from sklearn.preprocessing import MinMaxScaler

def limpiardatos():
    df = pd.read_csv("cirrhosis.csv")
    
    # Primero quitare los nulos
    
    df= df.dropna()
    
    # Eliminamos la columna ID
    
    df = df.drop(["ID"], axis=1)
    
    # Luego convertire en binario TOTALMENTE el status que actualmente tiene 3 estados:
    # D = Muerto
    # C = Censusado/Perdido (Por que se ha curado, seguramente)
    # CL = Censurado/Perdido por trasplante de higado
    # Convertiremos el CL en C poniendo el trasplante en una nueva columna
    
    df["Transplante_Higado"] = df["Status"].apply(lambda x: 1 if x == "CL" else 0)
    
    df["Status"] = df["Status"].apply(lambda x: "C" if x == "CL" else x)
    
    # Ahora corregiremos la edad (esta en dias vamos a ponerla en anyos)
    # Cabe aclarar que primero arreglaremos todos los datos y luego ya los escalaremos
    
    df["Age"] = round(df["Age"]/365, 2)
    
    # El numero de dias tambien lo pasaremos a anyos
    
    df["N_Days"] = round(df["N_Days"]/365, 2)
    
    # Seguiremos con el resto menos con Edema, estos los convertiremos en 0 y 1
    
    df["Status"] = df["Status"].apply(lambda x: 0 if x == "D" else 1)
    df["Sex"] = df["Sex"].apply(lambda x: 0 if x == "F" else 1)
    df["Ascites"] = df["Ascites"].apply(lambda x: 0 if x == "N" else 1)
    df["Hepatomegaly"] = df["Hepatomegaly"].apply(lambda x: 0 if x == "N" else 1)
    df["Spiders"] = df["Spiders"].apply(lambda x: 0 if x == "N" else 1)
    df["Drug"] = df["Drug"].apply(lambda x: 0 if x == "Placebo" else 1)
    
    # Como edema tiene N, S y Y pondremos respectivamente 0, 0.5, 1
    
    df["Edema"] = df["Edema"].apply(lambda x: 0 if x == "N" else (0.5 if x == "S" else 1))
    
    # Ahora escalaremos los datos 
    
    columnas_a_escalar = [
        "Age", "N_Days",
        "Bilirubin", "Cholesterol", "Albumin", "Copper", 
        "Alk_Phos", "SGOT", "Tryglicerides", "Platelets", 
        "Prothrombin", "Stage"
    ]

    scaler = MinMaxScaler()

    # Aplicamos el scaler al df
    
    df[columnas_a_escalar] = scaler.fit_transform(df[columnas_a_escalar])
    
    df[columnas_a_escalar] = df[columnas_a_escalar].round(2)
    
    return df

