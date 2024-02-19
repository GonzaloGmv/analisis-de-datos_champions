# analisis-de-datos

EL link al repositorio es: [github](https://github.com/GonzaloGmv/analisis-de-datos)


Antes que nada hay que tener en cuenta que para hacer buenas predicciones faltan muchos datos para saber el estado de forma de los equipos, como partidos de sus ligas o estado de forma de sus jugadores. De esta forma, las predicciones que hecho yo con datos más limitados estará mucho más limitada.

### Funcionamiento

Para realizar las predicciones constaba de todos los partidos de las champions de los 6 años anteriores y de la fase de grupos de este año

Sabiendo los cruces de octavos he entrenado un modelo de Random Forest utilizando los datos anteriores para predecir estos partidos.

Una vez tenía estos partidos veía qué equipo había ganado, en caso de empate hacía sorteo, ya que ese partido se decidiría en una tanda de penaltis que al fin y al cabo tiene un factor aleatorio muy importante (cabe destacar la existencia de un posible sesgo a la hora de hacer los sorteos para desempatar, debido a que la persona que ha hecho estos sorteos pueda ser más afín o menos a según que equipo).

Posteriormente hacía los sorteos de los enfrentamientos (bolas calientes). Una vez tenía hechos los enfrentamientos de la siguiente fase, volvía a ejecutar el random forest, y así hasta acabar.

Finalmente me sale como ganador el Real Madrid ganando la final por 2-1 ante el Arsenal

![image](https://github.com/GonzaloGmv/analisis-de-datos/assets/91721237/9c494349-e231-492e-8450-05823ecced60)


Este es el código final:
```
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
    archivos_csv = ["datos/2017-2018.csv", "datos/2018-2019.csv", "datos/2019-2020.csv", "datos/2020-2021.csv", "datos/2021-2022.csv", "datos/2022-2023.csv", "datos/2023-2024.csv"]

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
predecir_partidos("predicciones/partidos_final.csv")
```
