import os
import sys

# 1. ENRUTAMIENTO CORRECTO AL NUEVO MOTOR R 4.6.0
r_home = r"C:\Program Files\R\R-4.6.0"
r_bin = r"C:\Program Files\R\R-4.6.0\bin\x64"

os.environ["R_HOME"] = r_home

# Inyectamos la ruta binaria en el PATH de esta sesión de Python
if r_bin not in os.environ["PATH"]:
    os.environ["PATH"] = r_bin + os.pathsep + os.environ["PATH"]

# Registramos las DLLs a nivel de núcleo de Python en Windows
if sys.platform == "win32":
    try:
        os.add_dll_directory(r_bin)
        os.add_dll_directory(r"C:\Program Files\R\R-4.6.0\library\stats\libs\x64")
    except AttributeError:
        pass
#_____________________________________________________________
import rpy2.robjects as ro
from rpy2.robjects import conversion, default_converter
from rpy2.robjects.pandas2ri import converter as pandas_converter
from rpy2.robjects.packages import importr

# 3. ESTABLECEMOS TU RUTA REAL DE LIBRERÍA PERSONAL (La de AppData)
mi_ruta_real = "C:/Users/victo/AppData/Local/R/win-library/4.6"
ro.r(f'.libPaths(c("{mi_ruta_real}", .libPaths()))')

## 4. Activamos el conversor de DataFrames de Pandas <-> R
pandas_R_converter = default_converter + pandas_converter

## 5. Cargamos las librerías desde tu ruta real
print("Cargando librerías desde AppData/Local...")
ggplot2 = importr('ggplot2')
caret = importr('caret')
e1071 = importr('e1071')
randomForest = importr('randomForest')
gbm = importr('gbm')

print("--- ¡ÉXITO ROTUNDO! Entorno conectado, sincronizado y listo para trabajar ---")

#_____________________________________________________________________

### Vamos a emplear rpy2 para ejecutar código de R
### Queremos emplear distintas librerías para la parte de regresión



# %%

import os
# Cambia la ruta a tu librería personal de R

import rpy2.robjects as ro
from rpy2.robjects import conversion, default_converter
from rpy2.robjects.pandas2ri import converter as pandas_converter
from rpy2.robjects.packages import importr

ruta_libreria_personal = "C:/Users/victo/AppData/Local/R/win-library/4.6"

ro.r(f'.libPaths(c("{ruta_libreria_personal}", .libPaths()))')
## Conversor DF de pandas a DF de R automáticamente y viceversa
pandas_R_converter = default_converter + pandas_converter

## Cargamos las librerias que vamos a necesitar

caret = importr('caret')
e1071 = importr('e1071')
randomForest = importr('randomForest')
gbm = importr('gbm')

## A continuación vamos a inyectar los conjuntos que hemos definido
## previamente dentro de nuestro código de R.

with conversion.localconverter(pandas_R_converter):
#Conjuntos de entrenamiento y test normalizados + OHE
    ro.globalenv['X_train_proc'] = X_train_proc
    ro.globalenv['X_test_proc'] = X_test_proc

    #Conjuntos de entrenamiento y test OHE
    ro.globalenv['X_train_proc_tree'] = X_train_proc_tree
    ro.globalenv['X_test_proc_tree'] = X_test_proc_tree

    #Variable objetivo
    ro.globalenv['y_train_Reg'] = y_train_Reg
    ro.globalenv['y_test_Reg'] = y_test_Reg

ro.r("""
    # Unimos X e y en un solo df (preferido para caret)
    
    df_train <- X_train_proc
    df_train <- as.numeric(y_train_Reg)
           
    df_test <- X_test_proc
    df_test <- as.numeric(y_test_Reg)
           
    df_train_tree <- X_train_proc_tree
    df_train_tree <- as.numeric(y_train_Reg)
           
    df_test_tree <- X_test_proc_tree
    df_test_tree <- as.numeric(y_test_Reg)
           
    # Eliminamos de la memoria las variables temp
    rm(X_train_proc, X_test_proc, X_train_proc_tree, X_test_proc_tree, y_train_Reg, y_test_Reg)
""")

df_prueba = ro.globalenv['df_train']
print(df_prueba.head())