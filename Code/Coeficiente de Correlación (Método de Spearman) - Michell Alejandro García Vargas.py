"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#  Nombre: Michell Alejandro García Vargas
#  Expediente: 259663
#  Materia: Procesos Estocásticos
#  Tema: Coeficiente de Correlación (Método de Spearman)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Importar librerías a usar
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

os.system('cls') # Limpiar consola

print("""
=====================================================================================================================================================
#  1. Cargar datos del dataset y comprobaciones de lectura
=====================================================================================================================================================
""")

# Cargar datos del dataset (Cambiar por la ruta del archivo, del script actual)
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "Dataset/imdb_top_5000_tv_shows_clean_up.csv"))

print(f"Cantidad de series: {data.shape[0]} registros") # Número de filas
print(f"Cantidad de columnas: {data.shape[1]} columnas") # Número de columnas
print("\nColumnas disponibles:", data.columns.tolist()) # Nombres de las columnas

print("""
=====================================================================================================================================================
#  2. Preparar datos para el análisis
=====================================================================================================================================================
""")

# Conservar solo las columnas necesarias para el análisis
columnasAnalisis = ["primaryTitle", "averageRating", "numVotes", "genres1"]
newData = data[columnasAnalisis].copy()

# Seleccionar los 3 generos más recurrentes del dataset para analizarlos
topGenres = newData["genres1"].value_counts().head(4).index.tolist()

# Conservar solo los registros que cumplan con el genero a analizar
newData = newData[newData["genres1"].isin(topGenres)].copy()

print("Géneros más recurrentes:", topGenres) # Géneros más recurrentes 
print("\nConteo de series por género:")
print(newData["genres1"].value_counts().reset_index()) # Mostrar cuántas series se tienen por género

print(f"\nTotal de series a analizar: {len(newData)}") # Total de series
print("\nData después de haberla transformado:") # Imprimir data despues de haberla transformado
print(newData.head()) # 5 Primeros registros

print("""
=====================================================================================================================================================
#  3. Calcular coeficiente de correlación de Spearman para cada género
=====================================================================================================================================================
""")

# Función para calculo manual de spearman
def coeficienteSpearman(x, y):
    # Se calculan los rangos para la variable X & Y
    rangoX = x.rank()
    rangoY = y.rank()
    # Se calculan las medias de los rangos
    promedioX = rangoX.mean()
    promedioY = rangoY.mean()
    # Se calcula el numerador
    numerador = ((rangoX - promedioX) * (rangoY - promedioY)).sum()
    # Se calcula el denominador
    denominador = np.sqrt(((rangoX - promedioX) ** 2).sum() * ((rangoY - promedioY) ** 2).sum())
    # Se calcula el coeficiente de correlación de Spearman
    rho = numerador / denominador
    return rho

# Comedy =============================================================================================================================================

comediaData = newData[newData["genres1"] == "Comedy"].copy() # Copia del dataset filtrado por "Comedy"

votosComedia = comediaData["numVotes"] # Número de votos (variable X)
valoracionesComedia = comediaData["averageRating"] # Valoración promedio (variable Y)

rhoComediaManual = coeficienteSpearman(votosComedia, valoracionesComedia) # Se calcula el coeficiente de Spearman manualmente
rhoComedia, _ = spearmanr(votosComedia, valoracionesComedia) # Se calcula el coeficiente de Spearman con SciPy

print(f"Número de series de Comedia: {len(comediaData)}") # Número de series de comedia
print("Variable X: Numero de Votos") # Variable X
print("Variable Y: Raiting Promedio") # Variable Y
print(f"Coeficiente de Spearman para Comedia (Manual): {rhoComediaManual:.4f}") # Coeficiente de Spearman manual
print(f"Coeficiente de Spearman para Comedia (SciPy): {rhoComedia:.4f}\n") # Coeficiente de Spearman con SciPy

# Action =============================================================================================================================================

accionData = newData[newData["genres1"] == "Action"].copy() # Copia del dataset filtrado por "Action"

votosAccion = accionData["numVotes"] # Número de votos (variable X)
valoracionesAccion = accionData["averageRating"] # Valoración promedio (variable Y)

rhoAccionManual = coeficienteSpearman(votosAccion, valoracionesAccion) # Se calcula el coeficiente de Spearman manualmente
rhoAccion, _ = spearmanr(votosAccion, valoracionesAccion) # Se calcula el coeficiente de Spearman con SciPy

print(f"Número de series de Acción: {len(accionData)}") # Número de series de acción
print("Variable X: Numero de Votos") # Variable X
print("Variable Y: Raiting Promedio") # Variable Y
print(f"Coeficiente de Spearman para Acción (Manual): {rhoAccionManual:.4f}") # Coeficiente de Spearman manual
print(f"Coeficiente de Spearman para Acción (SciPy): {rhoAccion:.4f}\n") # Coeficiente de Spearman con SciPy

# Drama =============================================================================================================================================

dramaData = newData[newData["genres1"] == "Drama"].copy() # Copia del dataset filtrado por "Drama"

votosDrama = dramaData["numVotes"] # Número de votos (variable X)
valoracionesDrama = dramaData["averageRating"] # Valoración promedio (variable Y)

rhoDramaManual = coeficienteSpearman(votosDrama, valoracionesDrama) # Se calcula el coeficiente de Spearman manualmente
rhoDrama, _ = spearmanr(votosDrama, valoracionesDrama) # Se calcula el coeficiente de Spearman con SciPy

print(f"Número de series de Drama: {len(dramaData)}") # Número de series de drama
print("Variable X: Numero de Votos") # Variable X
print("Variable Y: Raiting Promedio") # Variable Y
print(f"Coeficiente de Spearman para Drama (Manual): {rhoDramaManual:.4f}") # Coeficiente de Spearman manual
print(f"Coeficiente de Spearman para Drama (SciPy): {rhoDrama:.4f}\n") # Coeficiente de Spearman con SciPy

# Crime =============================================================================================================================================

crimeData = newData[newData["genres1"] == "Crime"].copy() # Copia del dataset filtrado por "Crime"

votosCrime = crimeData["numVotes"] # Número de votos (variable X)
valoracionesCrime = crimeData["averageRating"] # Valoración promedio (variable Y)

rhoCrimeManual = coeficienteSpearman(votosCrime, valoracionesCrime) # Se calcula el coeficiente de Spearman manualmente
rhoCrime, _ = spearmanr(votosCrime, valoracionesCrime)

print(f"Número de series de Crimen: {len(crimeData)}") # Número de series de crimen
print("Variable X: Numero de Votos") # Variable X
print("Variable Y: Raiting Promedio") # Variable Y
print(f"Coeficiente de Spearman para Crimen (Manual): {rhoCrimeManual:.4f}") # Coeficiente de Spearman manual
print(f"Coeficiente de Spearman para Crimen (SciPy): {rhoCrime:.4f}\n") # Coeficiente de Spearman con SciPy

# Caso Especial: Family ============================================================================================================================================

familyData = data[data["genres1"] == "Family"].copy() # Copia del dataset filtrado por "Family"

votosFamily = familyData["numVotes"] # Número de votos (variable X)
valoracionesFamily = familyData["averageRating"] # Valoración promedio (variable Y)

rhoFamilyManual = coeficienteSpearman(votosFamily, valoracionesFamily) # Se calcula el coeficiente de Spearman manualmente
rhoFamily, _ = spearmanr(votosFamily, valoracionesFamily) # Se calcula el coeficiente de Spearman con SciPy

print(f"Número de series de Family: {len(familyData)}") # Número de series de family
print("Variable X: Numero de Votos") # Variable X
print("Variable Y: Raiting Promedio") # Variable Y
print(f"Coeficiente de Spearman para Familia (Manual): {rhoFamilyManual:.4f}") # Coeficiente de Spearman manual
print(f"Coeficiente de Spearman para Familia (SciPy): {rhoFamily:.4f}\n") # Coeficiente de Spearman con SciPy

print("""
=====================================================================================================================================================
# 4. Gráficas de dispersión por género
=====================================================================================================================================================
""")

# Comedy ============================================================================================================================================

# Se preparan los puntos para la Tendencia Suavizada por Series de Comedia
datosCurvaComedia = pd.DataFrame({
    "x": votosComedia,
    "y": valoracionesComedia
}).sort_values("x")
datosCurvaComedia["ySuave"] = datosCurvaComedia["y"].rolling(
    window=len(datosCurvaComedia)//3,
    center=True,
    min_periods=1
).mean()

# Gráfica de Dispersión por Series de Comedia
plt.figure(figsize=(9, 6))
plt.scatter(
    votosComedia,
    valoracionesComedia,
    alpha=0.3,
    label="Series de Comedia"
)
plt.xscale("log")
plt.xticks(
    [10_000, 100_000, 1_000_000],
    ["10 mil", "100 mil", "1 millón"]
)

# Se gráfica la Tendencia Suavizada (Representación de la Asociación Monotónica)
plt.plot(
    datosCurvaComedia["x"], datosCurvaComedia["ySuave"],
    color="red", linewidth=2.5, label="Tendencia Suavizada"
)

# Se establecen las etiquetas y el título, y propiedades de la gráfica
plt.xlabel("Número de Votos (X)")
plt.ylabel("Rating Promedio (Y)")
plt.title(f"Comedy: Votos vs. Rating\n(Spearman = {rhoComediaManual:.4f})")
plt.legend()
plt.grid(True)

# Rangos para graficar Comedia
rangosVotosComedia = votosComedia.rank()
rangosValoracionesComedia = valoracionesComedia.rank()

# Se preparan los puntos para la Tendencia Suavizada por Series de Comedia por Rangos
datosCurvaComedia = pd.DataFrame({
    "x": rangosVotosComedia,
    "y": rangosValoracionesComedia
}).sort_values("x")
datosCurvaComedia["ySuave"] = datosCurvaComedia["y"].rolling(
    window=len(datosCurvaComedia)//3,
    center=True,
    min_periods=1
).mean()

# Gráfica de Dispersión por Series de Comedia por Rangos
plt.figure(figsize=(9, 6))
plt.scatter(
    rangosVotosComedia,
    rangosValoracionesComedia,
    alpha=0.3,
    label="Series de Comedia"
)
plt.ylim(-100, 700)

# Se gráfica la Tendencia Suavizada (Representación de la Asociación Monotónica)
plt.plot(
    datosCurvaComedia["x"], datosCurvaComedia["ySuave"],
    color="red", linewidth=2.5, label="Tendencia Suavizada"
)

# Se establecen las etiquetas y el título, y propiedades de la gráfica
plt.xlabel("Rango del Número de Votos (X)")
plt.ylabel("Rango del Rating Promedio (Y)")
plt.title(f"Comedy: Asociación entre Rangos\n(Spearman = {rhoComediaManual:.4f})")
plt.legend()
plt.grid(True)
plt.show()

# Action ============================================================================================================================================

# Se preparan los puntos para la Tendencia Suavizada por Series de Acción
datosCurvaAccion = pd.DataFrame({
    "x": votosAccion,
    "y": valoracionesAccion
}).sort_values("x")
datosCurvaAccion["ySuave"] = datosCurvaAccion["y"].rolling(
    window=len(datosCurvaAccion)//3,
    center=True,
    min_periods=1
).mean()

# Gráfica de Dispersión por Series de Acción
plt.figure(figsize=(9, 6))
plt.scatter(
    votosAccion,
    valoracionesAccion,
    alpha=0.3,
    label="Series de Acción"
)
plt.xscale("log")
plt.xticks(
    [10_000, 100_000, 1_000_000],
    ["10 mil", "100 mil", "1 millón"]
)

# Se gráfica la Tendencia Suavizada (Representación de la Asociación Monotónica)
plt.plot(
    datosCurvaAccion["x"], datosCurvaAccion["ySuave"],
    color="red", linewidth=2.5, label="Tendencia Suavizada"
)

# Se establecen las etiquetas y el título, y propiedades de la gráfica
plt.xlabel("Número de Votos (X)")
plt.ylabel("Rating Promedio (Y)")
plt.title(f"Action: Votos vs. Rating\n(Spearman = {rhoAccionManual:.4f})")
plt.legend()
plt.grid(True)

# Rangos para graficar Acción
rangosVotosAccion = votosAccion.rank()
rangosValoracionesAccion = valoracionesAccion.rank()

# Se preparan los puntos para la Tendencia Suavizada por Series de Acción por Rangos
datosCurvaAccion = pd.DataFrame({
    "x": rangosVotosAccion,
    "y": rangosValoracionesAccion
}).sort_values("x")
datosCurvaAccion["ySuave"] = datosCurvaAccion["y"].rolling(
    window=len(datosCurvaAccion)//3,
    center=True,
    min_periods=1
).mean()

# Gráfica de Dispersión por Series de Acción por Rangos
plt.figure(figsize=(9, 6))
plt.scatter(
    rangosVotosAccion,
    rangosValoracionesAccion,
    alpha=0.3,
    label="Series de Acción"
)
plt.ylim(-100, 700)

# Se gráfica la Tendencia Suavizada (Representación de la Asociación Monotónica)
plt.plot(
    datosCurvaAccion["x"], datosCurvaAccion["ySuave"],
    color="red", linewidth=2.5, label="Tendencia Suavizada"
)

# Se establecen las etiquetas y el título, y propiedades de la gráfica
plt.xlabel("Rango del Número de Votos (X)")
plt.ylabel("Rango del Rating Promedio (Y)")
plt.title(f"Action: Asociación entre Rangos\n(Spearman = {rhoAccionManual:.4f})")
plt.legend()
plt.grid(True)
plt.show()

# Drama =============================================================================================================================================

# Se preparan los puntos para la Tendencia Suavizada por Series de Drama
datosCurvaDrama = pd.DataFrame({
    "x": votosDrama,
    "y": valoracionesDrama
}).sort_values("x")
datosCurvaDrama["ySuave"] = datosCurvaDrama["y"].rolling(
    window=len(datosCurvaDrama)//3,
    center=True,
    min_periods=1
).mean()

# Gráfica de Dispersión por Series de Drama
plt.figure(figsize=(9, 6))
plt.scatter(
    votosDrama,
    valoracionesDrama,
    alpha=0.3,
    label="Series de Drama"
)
plt.xscale("log")
plt.xticks(
    [10_000, 100_000, 1_000_000],
    ["10 mil", "100 mil", "1 millón"]
)

# Se gráfica la Tendencia Suavizada (Representación de la Asociación Monotónica)
plt.plot(
    datosCurvaDrama["x"], datosCurvaDrama["ySuave"],
    color="red", linewidth=2.5, label="Tendencia Suavizada"
)

# Se establecen las etiquetas y el título, y propiedades de la gráfica
plt.xlabel("Número de Votos (X)")
plt.ylabel("Rating Promedio (Y)")
plt.title(f"Drama: Votos vs. Rating\n(Spearman = {rhoDramaManual:.4f})")
plt.legend()
plt.grid(True)

# Rangos para graficar Drama
rangosVotosDrama = votosDrama.rank()
rangosValoracionesDrama = valoracionesDrama.rank()

# Se preparan los puntos para la Tendencia Suavizada por Series de Drama por Rangos
datosCurvaDrama = pd.DataFrame({
    "x": rangosVotosDrama,
    "y": rangosValoracionesDrama
}).sort_values("x")
datosCurvaDrama["ySuave"] = datosCurvaDrama["y"].rolling(
    window=len(datosCurvaDrama)//3,
    center=True,
    min_periods=1
).mean()

# Gráfica de Dispersión por Series de Drama por Rangos
plt.figure(figsize=(9, 6))
plt.scatter(
    rangosVotosDrama,
    rangosValoracionesDrama,
    alpha=0.3,
    label="Series de Drama"
)
plt.ylim(-100, 700)

# Se gráfica la Tendencia Suavizada (Representación de la Asociación Monotónica)
plt.plot(
    datosCurvaDrama["x"], datosCurvaDrama["ySuave"],
    color="red", linewidth=2.5, label="Tendencia Suavizada"
)

# Se establecen las etiquetas y el título, y propiedades de la gráfica
plt.xlabel("Rango del Número de Votos (X)")
plt.ylabel("Rango del Rating Promedio (Y)")
plt.title(f"Drama: Asociación entre Rangos\n(Spearman = {rhoDrama:.4f})")
plt.legend()
plt.grid(True)
plt.show()

# Crime =============================================================================================================================================

# Se preparan los puntos para la Tendencia Suavizada por Series de Crimen
datosCurvaCrime = pd.DataFrame({
    "x": votosCrime,
    "y": valoracionesCrime
}).sort_values("x")
datosCurvaCrime["ySuave"] = datosCurvaCrime["y"].rolling(
    window=len(datosCurvaCrime)//3, 
    center=True, 
    min_periods=1
).mean()

# Gráfica de Dispersión por Series de Crimen
plt.figure(figsize=(9, 6))
plt.scatter(
    votosCrime, 
    valoracionesCrime, 
    alpha=0.3, 
    label="Series de Crimen"
)
plt.xscale("log")
plt.xticks(
    [10_000, 100_000, 1_000_000],
    ["10 mil", "100 mil", "1 millón"]
)

# Se gráfica la Tendencia Suavizada (Representación de la Asociación Monotónica)
plt.plot(
    datosCurvaCrime["x"], datosCurvaCrime["ySuave"],
    color="red", linewidth=2.5, label="Tendencia Suavizada"
)

# Se establecen las etiquetas y el título, y propiedades de la gráfica
plt.xlabel("Número de Votos (X)")
plt.ylabel("Rating Promedio (Y)")
plt.title(f"Crime: Votos vs. Rating\n(Spearman = {rhoCrimeManual:.4f})")
plt.legend()
plt.grid(True)

# Rangos para graficar Crimen
rangosVotosCrime = votosCrime.rank()
rangosValoracionesCrime = valoracionesCrime.rank()

# Se preparan los puntos para la Tendencia Suavizada por Series de Crimen por Rangos
datosCurvaCrime = pd.DataFrame({
    "x": rangosVotosCrime,
    "y": rangosValoracionesCrime
}).sort_values("x")
datosCurvaCrime["ySuave"] = datosCurvaCrime["y"].rolling(
    window=len(datosCurvaCrime)//3,
    center=True,
    min_periods=1
).mean()

# Gráfica de Dispersión por Series de Crimen por Rangos
plt.figure(figsize=(9, 6))
plt.scatter(
    rangosVotosCrime, 
    rangosValoracionesCrime, 
    alpha=0.3, 
    label="Series de Crimen"
)
plt.ylim(-100, 700)

# Se gráfica la Tendencia Suavizada (Representación de la Asociación Monotónica)
plt.plot(
    datosCurvaCrime["x"], datosCurvaCrime["ySuave"],
    color="red", linewidth=2.5, label="Tendencia Suavizada"
)

# Se establecen las etiquetas y el título, y propiedades de la gráfica
plt.xlabel("Rango del Número de Votos (X)")
plt.ylabel("Rango del Rating Promedio (Y)")
plt.title(f"Crime: Asociación entre Rangos\n(Spearman = {rhoCrimeManual:.4f})")
plt.legend()
plt.grid(True)
plt.show()

# Family ============================================================================================================================================

# Se preparan los puntos para la Tendencia Suavizada por Series de Familia
datosCurvaFamily = pd.DataFrame({
    "x": votosFamily,
    "y": valoracionesFamily
}).sort_values("x")
datosCurvaFamily["ySuave"] = datosCurvaFamily["y"].rolling(
    window=3,
    center=True,
    min_periods=1
).mean()

# Gráfica de Dispersión por Series de Familia
plt.figure(figsize=(9, 6))
plt.scatter(
    votosFamily,
    valoracionesFamily,
    alpha=0.3,
    label="Series de Familia"
)
plt.xscale("log")

# Se gráfica la Tendencia Suavizada (Representación de la Asociación Monotónica)
plt.plot(
    datosCurvaFamily["x"], datosCurvaFamily["ySuave"],
    color="red", linewidth=2.5, label="Tendencia Suavizada"
)

# Se establecen las etiquetas y el título, y propiedades de la gráfica
plt.xlabel("Número de Votos (X)")
plt.ylabel("Rating Promedio (Y)")
plt.title(f"Family: Votos vs. Rating\n(Spearman = {rhoFamilyManual:.4f})")
plt.legend()
plt.grid(True)

# Rangos para graficar Familia
rangosVotosFamily = votosFamily.rank()
rangosValoracionesFamily = valoracionesFamily.rank()

# Se preparan los puntos para la Tendencia Suavizada por Series de Familia por Rangos
datosCurvaFamily = pd.DataFrame({
    "x": rangosVotosFamily,
    "y": rangosValoracionesFamily
}).sort_values("x")
datosCurvaFamily["ySuave"] = datosCurvaFamily["y"].rolling(
    window=3,
    center=True,
    min_periods=1
).mean()

# Gráfica de Dispersión por Series de Familia por Rangos
plt.figure(figsize=(9, 6))
plt.scatter(
    rangosVotosFamily,
    rangosValoracionesFamily,
    alpha=0.3,
    label="Series de Familia"
)

# Se gráfica la Tendencia Suavizada (Representación de la Asociación Monotónica)
plt.plot(
    datosCurvaFamily["x"], datosCurvaFamily["ySuave"],
    color="red", linewidth=2.5, label="Tendencia Suavizada"
)

# Se establecen las etiquetas y el título, y propiedades de la gráfica
plt.xlabel("Rango del Número de Votos (X)")
plt.ylabel("Rango del Rating Promedio (Y)")
plt.title(f"Family: Asociación entre Rangos\n(Spearman = {rhoFamilyManual:.4f})")
plt.legend()
plt.grid(True)
plt.show()
