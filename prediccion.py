import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

# Lo he puesto porque daba un warning de que en futuras versiones de pandas no se iba a poder hacer lo que se estaba haciendo
warnings.filterwarnings("ignore", category=FutureWarning)

def predecir_partidos(csv_fase):
    # Lista de nombres de archivos CSV
    archivos_csv = ["2017-2018.csv", "2018-2019.csv", "2019-2020.csv", "2020-2021.csv", "2021-2022.csv", "2022-2023.csv", "2023-2024.csv"]

    # Lista para almacenar cada DataFrame
    dataframes = []

    # Leer cada archivo CSV y almacenar su contenido en la lista de DataFrames
    for archivo in archivos_csv:
        df = pd.read_csv(archivo)
        dataframes.append(df)

    # Concatenar todos los DataFrames en uno solo
    datos_historicos = pd.concat(dataframes, ignore_index=True)

    # Eliminar filas con valores faltantes en goles_local y goles_visitante
    datos_historicos = datos_historicos.dropna(subset=['goles_local', 'goles_visitante'])

    # Dividir el conjunto de datos en características (X) y etiquetas (y)
    X = datos_historicos[['fase', 'equipo_local', 'equipo_visitante']]
    y = datos_historicos[['goles_local', 'goles_visitante']]

    # Codificar las características categóricas
    X = pd.get_dummies(X)

    # Dividir los datos en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)

    # Entrenar el modelo de Random Forest
    modelo = RandomForestRegressor(n_estimators=100, random_state=33)
    modelo.fit(X_train, y_train)

    # Predecir los valores faltantes en los partidos incompletos
    partidos_sin_resultado = pd.read_csv(csv_fase)

    # Asegurar que las características de los datos de prueba coincidan con las del conjunto de entrenamiento
    X_octavos = partidos_sin_resultado[['fase', 'equipo_local', 'equipo_visitante']]
    X_octavos_encoded = pd.get_dummies(X_octavos)

    # Asegurar que las características de los datos de prueba coincidan con las del conjunto de entrenamiento
    X_test = pd.DataFrame(columns=X_train.columns, data=np.zeros((X_octavos_encoded.shape[0], X_train.shape[1])))
    for col in X_octavos_encoded.columns:
        if col in X_test.columns:
            X_test[col] = X_octavos_encoded[col]

    # Predecir los goles para los partidos incompletos
    predicciones = modelo.predict(X_test)

    # Rellenar los valores faltantes con las predicciones
    partidos_sin_resultado.loc[:, ['goles_local', 'goles_visitante']] = np.round(predicciones)

    # Guardar los datos actualizados
    partidos_sin_resultado.to_csv(csv_fase, index=False)

'''
predecir_partidos("partidos_octavos.csv")
predecir_partidos("partidos_cuartos.csv")
predecir_partidos("partidos_semifinal.csv")
'''
predecir_partidos("partidos_final.csv")