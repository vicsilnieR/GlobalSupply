import pandas as pd


from dask.distributed import Client, LocalCluster
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from dask_ml.model_selection import GridSearchCV


if __name__ == "__main__":

    random_state_value = 12

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

    ct = make_column_transformer(
    (StandardScaler(), X_cont_cols),
    (OneHotEncoder(sparse_output=False), X_categ_cols)
    )
    ct.set_output(transform='pandas')

    X_train_proc = ct.fit_transform(X_train)
    X_test_proc = ct.transform(X_test)

    ## Preprocesado Bosques aleatorios y Gradient Boosting (NO escalados)

    ct_tree = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), X_categ_cols),
        remainder='passthrough' 
    )

    X_train_proc_tree = ct_tree.fit_transform(X_train)
    X_test_proc_tree = ct_tree.transform(X_test)

    ### Comenzamos configurando e inicializando el cliente de Dask

    cluster = LocalCluster(
        n_workers=2,                # num procesos
        threads_per_worker=3,       # hilos por proceso (6 núcleos en total)
        memory_limit="950MB",       # la RAM que quedaba libre
        processes=True             
    )

    #Inicialización del cliente de Dask
    client = Client(cluster)

    print("--- Cliente de Dask Inicializado ---")
    print(client)

    #Modelo que pasamos a gridsearch
    knn = KNeighborsClassifier()

    #Parametros que queremos probar
    param_knn = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance']
    }

    #Realizamos gridsearch de dask
    gridsearch_knn = GridSearchCV(
        estimator=knn,
        param_grid=param_knn,
        cv=5,
        return_train_score=False
    )

    #Entrenamiento 
    gridsearch_knn.fit(X_train_proc, y_train_Clas)

    #Guardado mejores parámetros
    knn_mejores_param = gridsearch_knn.best_params_

    print(knn_mejores_param)

    #Modelo que pasamos a gridsearch con atributos fuera de la búsqueda
    svc = SVC(
        kernel='rbf',
        random_state=random_state_value,
        probability=True
    )

    #Parametros que queremos probar
    param_svc = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }

    #Realizamos gridsearch de dask
    gridsearch_svc = GridSearchCV(
        estimator=svc,
        param_grid=param_svc,
        cv=5,
        return_train_score=False
    )

    #Entrenamiento 
    gridsearch_svc.fit(X_train_proc, y_train_Clas)

    #Guardado mejores parámetros
    svc_mejores_param = gridsearch_svc.best_params_

    print(svc_mejores_param)

    #Modelo que pasamos a gridsearch con atributos fuera de la búsqueda
    svc = SVC(
        kernel='rbf',
        random_state=random_state_value,
        probability=True
    )

    #Parametros que queremos probar
    param_svc = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }

    #Realizamos gridsearch de dask
    gridsearch_svc = GridSearchCV(
        estimator=svc,
        param_grid=param_svc,
        cv=5,
        return_train_score=False
    )

    #Entrenamiento 
    gridsearch_svc.fit(X_train_proc, y_train_Clas)

    #Guardado mejores parámetros
    svc_mejores_param = gridsearch_svc.best_params_

    #Modelo que pasamos a gridsearch con atributos fuera de la búsqueda
    rf = RandomForestClassifier(
        random_state=random_state_value,
    )

    #Parametros que queremos probar
    param_rf = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }

    #Realizamos gridsearch de dask
    gridsearch_rf = GridSearchCV(
        estimator=rf,
        param_grid=param_rf,
        cv=5,
        return_train_score=False
    )

    #Entrenamiento 
    gridsearch_rf.fit(X_train_proc_tree, y_train_Clas)

    #Guardado mejores parámetros
    rf_mejores_param = gridsearch_rf.best_params_

    print(rf_mejores_param)

        #Modelo que pasamos a gridsearch con atributos fuera de la búsqueda
    gb = GradientBoostingClassifier(
        random_state=random_state_value,
    )

    #Parametros que queremos probar
    param_gb = {
        'n_estimators': [50, 100, 150]
    }

    #Realizamos gridsearch de dask
    gridsearch_gb = GridSearchCV(
        estimator=gb,
        param_grid=param_gb,
        cv=5,
        return_train_score=False
    )

    #Entrenamiento 
    gridsearch_gb.fit(X_train_proc_tree, y_train_Clas)

    #Guardado mejores parámetros
    gb_mejores_param = gridsearch_gb.best_params_

    print(gb_mejores_param)

    client.close()