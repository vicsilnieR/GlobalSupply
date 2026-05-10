import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dask.distributed import Client
from dask.ml.model_selection import GridSearchCV
import joblib
import dask.array as da
import time
import os, sys
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVC
import warnings

DIR_MODELOS = 'modelos'

if 'data' not in st.session_state:
    st.session_state.data = None

@st.cache_data
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

@st.cache_resource
def load_all_models():
    """
    Busca y carga todos los modelos entrenados desde la carpeta 'modelos/'.
    """
    model_files = {
        'K vecionos más cercanos':'K_vecinos_mas_cercanos.pkl',
        'Máquinas de soporte vectorial':'Máquinas_de_soporte_vectorial.pkl',
        'Regresión Logística': 'Regresión_logistica.pkl',
        'Bosques Aleatorios': 'Bosques_aleatorios.pkl',
        'Gradient Boosting': 'Gradient_boosting.pkl'
    }

    loaded_models = {}
    for name, filename in model_files.items():
        path = os.path.join(DIR_MODELOS, filename)
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
        else:
            st.error(f"No se encontró el archivo en: {path}")    
    return loaded_models

available_models = load_all_models()

tab1, tab2, tab3 = st.tabs(["Carga de datos", "Visualizaciones descriptivas", "Modelos"])

with tab1:
    st.header('Carga de datos y previsualización')

    with st.spinner('Cargando datos de transporte global'):
        st.session_state.data = load_selected_data("categ_data.parquet")
    st.success('Se han cargado los datos correctamente')

    fil1, fil2, fil3 = st.columns(3)
    ports = st.session_state.data['Origin_Port'].unique().to_list()
    transports_list = st.session_state.data['Transport_Mode'].unique().to_list()
    products = st.session_state.data['Product_Category'].unique().to_list()
    weathers_list = st.session_state.data['Weather_Condition'].unique().to_list()


    with fil1:
        origin_port_fil = st.multiselect("Puerto de origen",
                                            st.session_state.data['Origin_Port'].cat.categories,
                                            default=[])
        if origin_port_fil==[]:
            origin_port_fil = ports
        
        dest_port_fil = st.multiselect("Puerto de destino",
                                            st.session_state.data['Destination_Port'].cat.categories,
                                            default=[])
        if dest_port_fil==[]:
            dest_port_fil = ports
        
        transport_mode_fil = st.multiselect("Transportado vía",
                                                st.session_state.data['Transport_Mode'].cat.categories,
                                                default=[])
        if transport_mode_fil==[]:
            transport_mode_fil = transports_list
        
        prod_categ_fil = st.multiselect("Categoría del producto",
                                                st.session_state.data['Product_Category'].cat.categories,
                                                default=[])
        if prod_categ_fil==[]:
            prod_categ_fil = products
    
    with fil2:
        dist_fil = st.slider("Distancia en Km",
                                st.session_state.data['Distance_km'].min(),
                                st.session_state.data['Distance_km'].max(),
                                (st.session_state.data['Distance_km'].min(),
                                st.session_state.data['Distance_km'].max()))
        weight_fil = st.slider("Peso",
                                st.session_state.data['Weight_MT'].min(),
                                st.session_state.data['Weight_MT'].max(),
                                (st.session_state.data['Weight_MT'].min(),
                                st.session_state.data['Weight_MT'].max()))
        fuel_price_fil = st.slider("Índice de precios del combustible",
                                st.session_state.data['Fuel_Price_Index'].min(),
                                st.session_state.data['Fuel_Price_Index'].max(),
                                (st.session_state.data['Fuel_Price_Index'].min(),
                                st.session_state.data['Fuel_Price_Index'].max()))
        geopolitical_fil = st.slider("Puntuación de riesgo geopolítico",
                                st.session_state.data['Geopolitical_Risk_Score'].min(),
                                st.session_state.data['Geopolitical_Risk_Score'].max(),
                                (st.session_state.data['Geopolitical_Risk_Score'].min(),
                                st.session_state.data['Geopolitical_Risk_Score'].max()))        
    
    with fil3:
        weather_fil = st.multiselect("Condición climática",
                                                st.session_state.data['Weather_Condition'].cat.categories,
                                                default=[])
        if weather_fil==[]:
            weather_fil = weathers_list
    
        carrier_fil = st.slider("Confianza del transportista",
                                st.session_state.data['Carrier_Reliability_Score'].min(),
                                st.session_state.data['Carrier_Reliability_Score'].max(),
                                (st.session_state.data['Carrier_Reliability_Score'].min(),
                                st.session_state.data['Carrier_Reliability_Score'].max()))
        lead_time_fil = st.slider("Días en tránsito",
                                st.session_state.data['Lead_Time_Days'].min(),
                                st.session_state.data['Lead_Time_Days'].max(),
                                (st.session_state.data['Lead_Time_Days'].min(),
                                st.session_state.data['Lead_Time_Days'].max()))
        disrr_sel = st.multiselect("Incidencia ocurrida",
                                                ['Sí', 'No'],
                                                default=[])

    disrr_map = {'Sí': 1, 'No':0}
    disrr_fil = [disrr_map[i] for i in disrr_sel]
    if disrr_fil == []:
        disrr_fil = [0, 1]

    query_filter_df = (f'Origin_Port == {origin_port_fil} and '
                       f'Destination_Port == {dest_port_fil} and '
                       f'Transport_Mode == {transport_mode_fil} and '
                       f'Product_Category == {prod_categ_fil} and '
                       f'Distance_km >= {dist_fil[0]} and '
                       f'Distance_km <= {dist_fil[1]} and '
                       f'Weight_MT >= {weight_fil[0]} and '
                       f'Weight_MT <= {weight_fil[1]} and '
                       f'Fuel_Price_Index >= {fuel_price_fil[0]} and '
                       f'Fuel_Price_Index <= {fuel_price_fil[1]} and '
                       f'Geopolitical_Risk_Score >= {geopolitical_fil[0]} and '
                       f'Geopolitical_Risk_Score <= {geopolitical_fil[1]} and '
                       f'Weather_Condition == {weather_fil} and '
                       f'Carrier_Reliability_Score >= {carrier_fil[0]} and '
                       f'Carrier_Reliability_Score <= {carrier_fil[1]} and '
                       f'Lead_Time_Days >= {lead_time_fil[0]} and '
                       f'Lead_Time_Days <= {lead_time_fil[1]} and '
                       f'Disruption_Occurred == {disrr_fil}'
    )

    st.dataframe(st.session_state.data.query(query_filter_df),
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="multi-row",
                    )

with tab2:
    st.header('Visualizaciones descriptivas')

    #Columnas en: Categóricas/Continuas

    categ_cols = ['Origin_Port', 'Destination_Port', 'Transport_Mode',
                 'Product_Category', 'Weather_Condition', 'Disruption_Occurred']
    cont_cols = ['Distance_km', 'Weight_MT', 'Fuel_Price_Index', 'Geopolitical_Risk_Score',
                 'Carrier_Reliability_Score', 'Lead_Time_Days']

    #Listas: categorías en variables categóricas
    transports = st.session_state.data['Transport_Mode'].unique().to_numpy()
    weathers = st.session_state.data['Weather_Condition'].unique().to_numpy()

    columns_type = st.session_state.data.dtypes.apply(lambda x: x.name)

    type_count = columns_type.value_counts().reset_index()
    type_count.columns = ['Tipo', 'Cantidad']

    fig = px.pie(type_count,
                 values='Cantidad',
                 names = 'Tipo',
                 title = 'Tipo de datos',
                 color_discrete_sequence=px.colors.qualitative.Pastel
                 )
    st.plotly_chart(fig)

    #

    st.header('Zona de test')
    
    
    st.subheader('Test selector')
    selected_test = st.selectbox('Seleccionar medio de transporte',
                                      weathers)
    
    test = st.session_state.data.loc[:,['Weather_Condition','Distance_km', 'Geopolitical_Risk_Score', 'Disruption_Occurred']]
    test = test.sort_values(['Weather_Condition', 'Disruption_Occurred'], ascending= [True, False])
    fig = px.scatter(test[test['Weather_Condition'] == selected_test], 
                     x='Distance_km', 
                     y='Geopolitical_Risk_Score',
                     color = 'Disruption_Occurred',
                     marginal_x='histogram',
                     trendline='ols',
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig)

    #VISUALIZACIÓN BÁSICA VARIABLES CATEGÓRICAS

    selected_categ_col = st.selectbox('Selecciona una variable categórica', categ_cols)
    fig = px.histogram(st.session_state.data, 
                   x=selected_categ_col, 
                   color_discrete_sequence=['rgb(246, 207, 113)'],
                   text_auto=True,
                   labels={
                        selected_categ_col: selected_categ_col.replace('_', ' ').title(), 
                    }
                   )
    fig.update_yaxes(title_text='Total')
    st.plotly_chart(fig)

    st.write(px.colors.qualitative.Pastel)
    #VISUALIZACIÓN BÁSICA VARIABLES CONTINUAS

    #CORRELACIÓN VARIABLES CONTINUAS

    selected_transport2 = st.selectbox('Seleccionar medio de transporte',
                                      transports, key='sb3')
    
    df_temp = st.session_state.data.query(f'Transport_Mode == @selected_transport2')
    corr_matrix = df_temp.loc[:,cont_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=True,     
        aspect="auto",      
        color_continuous_scale='RdBu_r', 
        zmin=-1, zmax=1    
    )

    fig.update_layout(title="Matriz de Correlación")
    st.plotly_chart(fig)

    #MEDIO DE TRANSPORTE + RIESGO GEOPOLITICO

    # MEDIO DE TRANSPORTE + CLIMA
    st.subheader('Días de retraso según medio de transporte y clima')
    selected_transport = st.selectbox('Seleccionar medio de transporte',
                                      transports, key='sb1')
    
    df_temp = st.session_state.data.loc[:,['Transport_Mode', 'Weather_Condition', 'Lead_Time_Days']]
    df_temp = df_temp.sort_values(['Transport_Mode', 'Weather_Condition'])
    fig = px.box(df_temp[df_temp['Transport_Mode']==selected_transport], 
                 x='Weather_Condition', 
                 y='Lead_Time_Days',
                 color='Weather_Condition',
                 color_discrete_sequence=px.colors.qualitative.Pastel                 
                 )
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title_text='Días hasta entrega')
    st.plotly_chart(fig)

    ## VASILANDO DE VISUALISASION

    st.subheader('Días de retraso según medio de transporte, clima e incidencia ocurrida')
    selected_transport1 = st.selectbox('Seleccionar medio de transporte',
                                      transports, key='sb2')
    
    df_temp = st.session_state.data.query(f'Transport_Mode == @selected_transport1').loc[:,['Weather_Condition', 'Lead_Time_Days', 'Disruption_Occurred']]
    show_legend = [True,False,False,False,False]
    fig = go.Figure()

    for i, weather in enumerate(np.sort(weathers)):
        df_filtered = df_temp[df_temp['Weather_Condition'] == weather]
        fig.add_trace(go.Violin(x=df_filtered['Weather_Condition'][df_filtered['Disruption_Occurred'] == 0],
                                y=df_filtered['Lead_Time_Days'][df_filtered['Disruption_Occurred'] == 0],
                                legendgroup='Sin Incidencia',
                                scalegroup='Sin Incidencia',
                                name='Sin Incidencia',
                                side='negative',
                                line_color='rgb(102, 197, 204)',
                                showlegend=show_legend[i]
                                )
                    )
        fig.add_trace(go.Violin(x=df_filtered['Weather_Condition'][df_filtered['Disruption_Occurred'] == 1],
                                y=df_filtered['Lead_Time_Days'][df_filtered['Disruption_Occurred'] == 1],
                                legendgroup='Incidencia',
                                scalegroup='Incidencia',
                                name='Incidencia',
                                side='positive',
                                line_color='rgb(246, 207, 113)',
                                showlegend=show_legend[i]
                                )
                    )
    fig.update_traces(meanline_visible=True,
                      jitter=0.05,  
                      scalemode='count'
                      ) 
    fig.update_layout(
        title_text='Título aquí',
        violingap=0.1, violingroupgap=0, violinmode='overlay')
    st.plotly_chart(fig)

    with tab3:
        st.header("Análisis de Parche (Inferencia)")
    
        if not available_models:
            st.error(f"⚠️ No se ha encontrado ningún modelo en la carpeta '{DIR_MODELOS}/'. Ejecuta el script de entrenamiento primero.")
        else:
            st.markdown("### Configuración del Motor Predictivo")
            
            # El usuario elige el modelo
            selected_model_name = st.selectbox(
                "Selecciona la Inteligencia Artificial a usar:", 
                options=list(available_models.keys()),
                index=len(available_models)-1 
            )
            
            active_model = available_models[selected_model_name]
            
            # --- NUEVO: Lógica para el botón de descarga ---
            # Diccionario para saber qué archivo corresponde a cada nombre del selectbox
            model_filenames = {
            'K vecionos más cercanos':'K_vecinos_mas_cercanos.pkl',
            'Máquinas de soporte vectorial':'Máquinas_de_soporte_vectorial.pkl',
            'Regresión Logística': 'Regresión_logistica.pkl',
            'Bosques Aleatorios': 'Bosques_aleatorios.pkl',
            'Gradient Boosting': 'Gradient_boosting.pkl'
        }

            
            # Obtenemos la ruta del archivo seleccionado
            selected_filename = model_filenames[selected_model_name]
            file_path = os.path.join(DIR_MODELOS, selected_filename)
            
            # Leemos el archivo en modo binario y creamos el botón
            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    st.download_button(
                        label=f"📥 Descargar modelo {selected_model_name} (.pkl)",
                        data=file,
                        file_name=selected_filename,
                        mime="application/octet-stream"
                    )
            # -----------------------------------------------
            
            st.divider() # Línea separadora visual
            
            # Dividimos en dos columnas
            col_upload, col_result = st.columns([1, 1])
            
            with col_upload:
                # Contenedor con borde para estética moderna
                with st.container(border=True):
                    st.subheader("1. Cargar Muestra")
                    uploaded_file = st.file_uploader("Sube un parche histológico (JPG/PNG)", type=["jpg", "png", "jpeg"])
                    
                    if uploaded_file is not None:
                        st.write('Aquí irá previsualización del archivo subido')
                        
                    
            with col_result:
                # Contenedor con borde para el diagnóstico
                with st.container(border=True):
                    st.subheader(f"2. Resultado de IA")
                    
                    if uploaded_file is not None:
                        with st.spinner(f"Analizando tejido con {selected_model_name}..."):
                            # Preprocesamiento
                            img_resized = image.resize((50, 50)) 
                            img_array = np.array(img_resized)
                            img_flat = (img_array / 255.0).flatten()
                            
                            # Predicción
                            prediction = active_model.predict_proba([img_flat])[0][1] 
                            
                            threshold = 0.5
                            is_idc_positive = prediction > threshold
                            
                            st.markdown("### Diagnóstico Asistido:")
                            if is_idc_positive:
                                st.error(f"**Positivo para IDC** (Confianza: {prediction:.2%})")
                            else:
                                st.success(f"**Negativo para IDC** (Confianza: {(1 - prediction):.2%})")
                            
                            # Barra de progreso
                            st.progress(float(prediction), text="Probabilidad de Malignidad")
                    else:
                        st.info("Sube una imagen en la tarjeta de la izquierda para comenzar el análisis.")


        @st.cache_data
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

        X_categ_cols = ['Origin_Port', 'Destination_Port', 'Transport_Mode',
                 'Product_Category', 'Weather_Condition']
        X_cont_cols = ['Distance_km', 'Weight_MT', 'Fuel_Price_Index', 'Geopolitical_Risk_Score',
                    'Carrier_Reliability_Score']

        ## Conjunto de entrenamiento y test. Variable objetivo para clasificación/regresión

        X_train, X_test, y_train_Clas, y_test_Clas, y_train_Reg, y_test_Reg = preproc_data(
            st.session_state.data, 
            ['Date', 'Lead_Time_Days', 'Disruption_Occurred','Year','Month','Day'],
            ['Disruption_Occurred'],
            ['Lead_Time_Days']
        )

        ## Preprocesado KNN y SVM: 

        ct = make_column_transformer(
            (StandardScaler(), X_cont_cols),
            (OneHotEncoder(sparse_output=False), X_categ_cols)
        )
        ct.set_output(transform='pandas')

        X_train_proc = ct.fit_transform(X_train)
        X_test_proc = ct.transform(X_test)
        
        ### Clasificación mediante KNN

        knn_Clas = KNeighborsClassifier(n_neighbors=11)
        knn_Clas.fit(X_train_proc, y_train_Clas)

        knn_y_pred_Clas = knn_Clas.predict(X_test_proc)
        #Porcentaje de aciertos
        st.write('Aciertos clas KNN')
        st.write(knn_Clas.score(X_test_proc, y_test_Clas))

        ### Regresión mediante KNN

        knn_Reg = KNeighborsRegressor(n_neighbors=11)
        knn_Reg.fit(X_train_proc, y_train_Reg)

        knn_y_pred_Reg = knn_Reg.predict(X_test_proc)
        #Métricas de rendimiento
        st.write('Aciertos reg KNN')
        mae = mean_absolute_error(y_test_Reg, knn_y_pred_Reg)
        mse = mean_squared_error(y_test_Reg, knn_y_pred_Reg)
        r2 = r2_score(y_test_Reg, knn_y_pred_Reg)
        st.write(mae, mse, r2)

        ### Clasificación mediante SVM
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

        svm_model.fit(X_train_proc, y_train_Clas)

        #Porcentaje de aciertos
        st.write('Aciertos clas SVM')
        st.write(svm_model.score(X_test_proc,y_test_Clas))

        ## Preprocesado Árboles y Naive-Bayes(revisar)
        ct_arboles = make_column_transformer(
            (OneHotEncoder(handle_unknown='ignore'), X_categ_cols),
            remainder='passthrough' # Esto deja las columnas numéricas intactas sin tocarlas
        )

    
        




    



    
    


