import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline

def load_selected_data(dataset_name):
    data = pd.read_parquet(dataset_name)
    cols_to_category = ['Origin_Port', 'Destination_Port', 'Transport_Mode',
                        'Product_Category', 'Weather_Condition', 'Disruption_Occurred']
    for col in cols_to_category:
        data[col] = data[col].astype('category')
    data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    return data

def preproc_data(df, drop_col, res_col_cont, res_col_reg):
    X = df.drop(columns=drop_col)
    y1 = df[res_col_cont]
    y2 = df[res_col_reg]
    X_train, X_test, y_train_clas, y_test_clas, y_train_reg, y_test_reg = train_test_split(
        X,
        y1,
        y2,
        random_state=42,
        test_size=0.2
    )
    
    return X_train, X_test, y_train_clas, y_test_clas, y_train_reg, y_test_reg

data = load_selected_data("https://raw.githubusercontent.com/vicsilnieR/GlobalSupply/main/datos/supplies_data.parquet")
random_state_value = 42


X_categ_cols = ['Origin_Port', 'Destination_Port', 'Transport_Mode',
            'Product_Category', 'Weather_Condition']
X_cont_cols = ['Distance_km', 'Weight_MT', 'Fuel_Price_Index', 'Geopolitical_Risk_Score',
            'Carrier_Reliability_Score']

## Conjunto de entrenamiento y test. Variable objetivo para clasificación/regresión

X_train, X_test, y_train_Clas, y_test_Clas, y_train_Reg, y_test_Reg = preproc_data(
    data, 
    ['Date', 'Lead_Time_Days', 'Disruption_Occurred','Year','Month','Day'],
    ['Disruption_Occurred'],
    ['Lead_Time_Days']
)

##Apareció en consola: recomendaba hacer ravel() para mejor funcionamiento al entrenar
y_train_Clas = y_train_Clas.values.ravel()
y_test_Clas = y_test_Clas.values.ravel()


## Preprocesado KNN, SVM y Regresión Logística (Escalados)

ct = make_column_transformer(
    (StandardScaler(), X_cont_cols),
    (OneHotEncoder(sparse_output=False), X_categ_cols)
)
ct.set_output(transform='pandas')

## Preprocesado Bosques aleatorios y Gradient Boosting (NO escalados)

ct_tree = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), X_categ_cols),
    remainder='passthrough' 
)


## Configuración multimodelo y directorio

directorio_modelos = 'modelos'
os.makedirs(directorio_modelos, exist_ok=True)

## Definimos los modelos base 

#Escalados
knn = KNeighborsClassifier(n_neighbors=9)
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=random_state_value, probability=True)
lr = LogisticRegression(max_iter=1000, random_state=random_state_value)
#NO escalados
rf = RandomForestClassifier(n_estimators=150, random_state=random_state_value, n_jobs=-1,
                            min_samples_split=5, max_depth=10)
gb = GradientBoostingClassifier(n_estimators=50, random_state=random_state_value)

## Diccionarios para iterar

models_scaled = {
    'K_vecinos_mas_cercanos': knn,
    'Máquinas_de_soporte_vectorial': svm,
    'Regresión_logistica': lr
}

models = {
    'Bosques_aleatorios': rf,
    'Gradient_boosting': gb
}

results_data = {}

## Entrenamiento y evaluación (bucle)

#Escalado
for model_name, model in models_scaled.items():
    
    #Pipeline preprocesado + model (Para no transformar en main.py)
    pipeline_proc_model = Pipeline([
        ('Preprocesado', ct),
        ('Modelo', model)
    ])


    #entrenamiento
    pipeline_proc_model.fit(X_train, y_train_Clas)

    #predicciones y probs
    y_pred = pipeline_proc_model.predict(X_test)
    y_pred_proba = pipeline_proc_model.predict_proba(X_test)[:, 1]

    #calcular métricas
    accuracy = accuracy_score(y_test_Clas, y_pred)
    precision = precision_score(y_test_Clas, y_pred, zero_division=0)
    recall = recall_score(y_test_Clas, y_pred, zero_division=0)
    f1 = f1_score(y_test_Clas, y_pred, zero_division=0)

    #Guardamos en el diccionario de resultados
    results_data[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

    # Guardamos matriz de confusión
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test_Clas, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negativo (0)', 'Positivo (1)'],
                yticklabels=['Negativo (0)', 'Positivo (1)'])
    plt.title(f'Matriz de Confusión - {model_name.replace("_", " ")}')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.savefig(f'{directorio_modelos}/{model_name}_cm.png')
    plt.close()

    #Guardamos la curva ROC
    fpr, tpr, thresholds = roc_curve(y_test_Clas, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title(f'Curva ROC - {model_name.replace("_", " ")}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'{directorio_modelos}/{model_name}_roc.png')
    plt.close()

    #Guardamos modelo(.pkl)
    joblib.dump(pipeline_proc_model, f'{directorio_modelos}/{model_name}.pkl')

#NO escalado

for model_name, model in models.items():
    
    #Pipeline preprocesado + model (Para no transformar en main.py)
    pipeline_proc_model = Pipeline([
        ('Preprocesado', ct_tree),
        ('Modelo', model)
    ])

    #entrenamiento
    pipeline_proc_model.fit(X_train, y_train_Clas)

    #predicciones y probs
    y_pred = pipeline_proc_model.predict(X_test)
    y_pred_proba = pipeline_proc_model.predict_proba(X_test)[:, 1]

    #calcular métricas
    accuracy = accuracy_score(y_test_Clas, y_pred)
    precision = precision_score(y_test_Clas, y_pred, zero_division=0)
    recall = recall_score(y_test_Clas, y_pred, zero_division=0)
    f1 = f1_score(y_test_Clas, y_pred, zero_division=0)

    #Guardamos en el diccionario de resultados
    results_data[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }

    # Guardamos matriz de confusión
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test_Clas, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negativo (0)', 'Positivo (1)'],
                yticklabels=['Negativo (0)', 'Positivo (1)'])
    plt.title(f'Matriz de Confusión - {model_name.replace("_", " ")}')
    plt.ylabel('Valor Real')
    plt.xlabel('Predicción del Modelo')
    plt.tight_layout()
    plt.savefig(f'{directorio_modelos}/{model_name}_cm.png')
    plt.close()

    #Guardamos la curva ROC
    fpr, tpr, thresholds = roc_curve(y_test_Clas, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title(f'Curva ROC - {model_name.replace("_", " ")}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'{directorio_modelos}/{model_name}_roc.png')
    plt.close()

    #Guardamos modelo(.pkl)
    joblib.dump(pipeline_proc_model, f'{directorio_modelos}/{model_name}.pkl')

#Guardamos métricas globales
joblib.dump(results_data, f'{directorio_modelos}/all_metrics.pkl')

print(X_train.info(verbose=True))