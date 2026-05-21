import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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
azul_graf = '#66C5CC'
amarillo_graf = '#F6CF71'

X_categ_cols = ['Origin_Port', 'Destination_Port', 'Transport_Mode',
            'Product_Category', 'Weather_Condition']
X_cont_cols = ['Distance_km', 'Weight_MT', 'Fuel_Price_Index', 'Geopolitical_Risk_Score',
            'Carrier_Reliability_Score']

X_train, X_test, y_train_Clas, y_test_Clas, y_train_Reg, y_test_Reg = preproc_data(
    data, 
    ['Date', 'Lead_Time_Days', 'Disruption_Occurred','Year','Month','Day'],
    ['Disruption_Occurred'],
    ['Lead_Time_Days']
)

y_train_Reg = y_train_Reg.values.ravel()
y_test_Reg = y_test_Reg.values.ravel()

ct = make_column_transformer(
    (StandardScaler(), X_cont_cols),
    (OneHotEncoder(sparse_output=False), X_categ_cols)
)
ct.set_output(transform='pandas')

ct_tree = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), X_categ_cols),
    remainder='passthrough' 
)


directorio_modelos = 'modelos_regresion'
os.makedirs(directorio_modelos, exist_ok=True)

#Escalados
knn = KNeighborsRegressor(n_neighbors=9)
svm = SVR(kernel='rbf', C=1.0, gamma='scale')
lr = LinearRegression()

#NO escalados: Entrenados con menor profundidad para evitar R2=1 por ser datos sintéticos.
rf = RandomForestRegressor(n_estimators=50, random_state=random_state_value, n_jobs=-1,
                            min_samples_split=5, max_depth=5)
gb = GradientBoostingRegressor(n_estimators=25, random_state=random_state_value)

models_scaled = {
    'K_vecinos_mas_cercanos': knn,
    'Maquinas_de_soporte_vectorial': svm,
    'Regresion_lineal': lr
}

models = {
    'Bosques_aleatorios': rf,
    'Gradient_boosting': gb
}

results_data = {}

for model_name, model in models_scaled.items():
    
    pipeline_proc_model = Pipeline([
        ('Preprocesado', ct),
        ('Modelo', model)
    ])

    pipeline_proc_model.fit(X_train, y_train_Reg)

    y_pred = pipeline_proc_model.predict(X_test)

    mae = mean_absolute_error(y_test_Reg, y_pred)
    mse = mean_squared_error(y_test_Reg, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_Reg, y_pred)

    results_data[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    plt.scatter(y_test_Reg, y_pred, color=azul_graf, alpha=0.5, edgecolors='none')
    plt.plot([y_test_Reg.min(), y_test_Reg.max()], [y_test_Reg.min(), y_test_Reg.max()], 
             color=amarillo_graf, lw=2, linestyle='--')
    plt.title(f'Prediccion vs Realidad - {model_name.replace("_", " ")}')
    plt.ylabel('Prediccion (Dias)')
    plt.xlabel('Valor Real (Dias)')
    plt.tight_layout()
    plt.savefig(f'{directorio_modelos}/{model_name}_dispersion.png')
    plt.close()

    joblib.dump(pipeline_proc_model, f'{directorio_modelos}/{model_name}.pkl')

for model_name, model in models.items():
    
    pipeline_proc_model = Pipeline([
        ('Preprocesado', ct_tree),
        ('Modelo', model)
    ])

    pipeline_proc_model.fit(X_train, y_train_Reg)

    y_pred = pipeline_proc_model.predict(X_test)

    mae = mean_absolute_error(y_test_Reg, y_pred)
    mse = mean_squared_error(y_test_Reg, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_Reg, y_pred)

    results_data[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')
    plt.scatter(y_test_Reg, y_pred, color=azul_graf, alpha=0.5, edgecolors='none')
    plt.plot([y_test_Reg.min(), y_test_Reg.max()], [y_test_Reg.min(), y_test_Reg.max()], 
             color=amarillo_graf, lw=2, linestyle='--')
    plt.title(f'Prediccion vs Realidad - {model_name.replace("_", " ")}')
    plt.ylabel('Prediccion (Dias)')
    plt.xlabel('Valor Real (Dias)')
    plt.tight_layout()
    plt.savefig(f'{directorio_modelos}/{model_name}_dispersion.png')
    plt.close()

    joblib.dump(pipeline_proc_model, f'{directorio_modelos}/{model_name}.pkl')

joblib.dump(results_data, f'{directorio_modelos}/all_metrics.pkl')