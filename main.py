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
import warnings
import rpy2.robjects as ro
from rpy2.robjects import r

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
        'K vecinos más cercanos':'K_vecinos_mas_cercanos.pkl',
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

@st.cache_data
def load_metrics():
    """
    Carga el diccionario con las métricas guardadas durante el entrenamiento.
    """
    path = os.path.join(DIR_MODELOS, 'all_metrics.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None

available_models = load_all_models()
all_metrics_data = load_metrics()

tab1, tab2, tab3, tab4 = st.tabs(["Carga de datos", "Visualizaciones descriptivas", "Predicción con modelos", "Eficacia de los modelos"])

with tab1:
    st.header('Carga de datos y previsualización')

    with st.spinner('Cargando datos de transporte global'):
        st.session_state.data = load_selected_data("https://raw.githubusercontent.com/vicsilnieR/GlobalSupply/main/datos/supplies_data.parquet")
    st.success('Se han cargado los datos correctamente')

    fil1, fil2, fil3 = st.columns(3)
    ports = st.session_state.data['Origin_Port'].unique().tolist() #Correccion aviso terminal: FutureWarning: Categorical.to_list is deprecated and will be removed in a future version. Use obj.tolist() instead
    transports_list = st.session_state.data['Transport_Mode'].unique().tolist()
    products = st.session_state.data['Product_Category'].unique().tolist()
    weathers_list = st.session_state.data['Weather_Condition'].unique().tolist()


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
                    width='stretch', #corrección aviso en terminal: For `use_container_width=True`, use `width='stretch'`
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="multi-row",
                    )

with tab2:
    st.header('Visualizaciones descriptivas')

    azul_graf = '#66C5CC'
    amarillo_graf = '#F6CF71'
    rojo_graf= '#F89C74'
    #Columnas en: Categóricas/Continuas

    categ_cols = ['Origin_Port', 'Destination_Port', 'Transport_Mode',
                 'Product_Category', 'Weather_Condition', 'Disruption_Occurred']
    cont_cols = ['Distance_km', 'Weight_MT', 'Fuel_Price_Index', 'Geopolitical_Risk_Score',
                 'Carrier_Reliability_Score', 'Lead_Time_Days']

    #Listas: categorías en variables categóricas
    transports = st.session_state.data['Transport_Mode'].unique().to_numpy()
    weathers = st.session_state.data['Weather_Condition'].unique().to_numpy()
    ports2 = st.session_state.data['Origin_Port'].unique().to_numpy()

    with st.container(border=True):
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

    #VISUALIZACIÓN BÁSICA VARIABLES CATEGÓRICAS
    with st.container(border=True):
        selected_categ_col = st.selectbox('Selecciona una variable categórica', categ_cols)
        fig = px.histogram(st.session_state.data, 
                    x=selected_categ_col, 
                    color_discrete_sequence=[amarillo_graf],
                    text_auto=True,
                    labels={
                            selected_categ_col: selected_categ_col.replace('_', ' ').title(), 
                        },
                        title=f'Cantidad de observaciones por {selected_categ_col}'
                    )
        fig.update_yaxes(title_text='Total')
    
        st.plotly_chart(fig)

    #VISUALIZACIÓN BÁSICA VARIABLES CONTINUAS

    #CORRELACIÓN VARIABLES CONTINUAS
    with st.container(border=True):
        selected_transport2 = st.selectbox('Seleccionar medio de transporte',
                                        transports, key='sb3')
        
        df_temp = st.session_state.data.query(f'Transport_Mode == @selected_transport2')
        corr_matrix = df_temp.loc[:,cont_cols].corr()

        fig = px.imshow(
            corr_matrix,
            text_auto=True,     
            aspect="auto",      
            color_continuous_scale=[rojo_graf, amarillo_graf, azul_graf, amarillo_graf, rojo_graf], 
            zmin=-1, zmax=1       
        )

        fig.update_layout(title="Matriz de Correlación")
        st.plotly_chart(fig)

    # MEDIO DE TRANSPORTE + CLIMA
    with st.container(border=True):
        selected_transport = st.selectbox('Seleccionar medio de transporte',
                                        transports, key='sb1')
        
        df_temp = st.session_state.data.loc[:,['Transport_Mode', 'Weather_Condition', 'Lead_Time_Days']]
        df_temp = df_temp.sort_values(['Transport_Mode', 'Weather_Condition'])
        fig = px.box(df_temp[df_temp['Transport_Mode']==selected_transport], 
                    x='Weather_Condition', 
                    y='Lead_Time_Days',
                    color='Weather_Condition',
                    color_discrete_sequence=px.colors.qualitative.Pastel   ,
                    title='Días de retraso según medio de transporte y clima'              
                    )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title_text='Días hasta entrega')
        fig.update_xaxes(title_text='Condición climática')
        st.plotly_chart(fig)

    # MEDIO DE TRANSPORTE + CLIMA + INCIDENCIA OCURRIDA
    with st.container(border=True):
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
                                    line_color=azul_graf,
                                    showlegend=show_legend[i]
                                    )
                        )
            fig.add_trace(go.Violin(x=df_filtered['Weather_Condition'][df_filtered['Disruption_Occurred'] == 1],
                                    y=df_filtered['Lead_Time_Days'][df_filtered['Disruption_Occurred'] == 1],
                                    legendgroup='Incidencia',
                                    scalegroup='Incidencia',
                                    name='Incidencia',
                                    side='positive',
                                    line_color=amarillo_graf,
                                    showlegend=show_legend[i]
                                    )
                        )
        fig.update_traces(meanline_visible=True,
                        jitter=0.05,  
                        scalemode='count',
                        spanmode='hard'
                        ) 
        fig.update_layout(
            title_text='Días de retraso según medio de transporte, clima e incidencia ocurrida',
            violingap=0.1, violingroupgap=0, violinmode='overlay')
        fig.update_yaxes(title_text='Días hasta entrega', range=[-5, None])
        fig.update_xaxes(title_text='Condición climática', range=[-0.6, len(weathers) - 0.4])
        st.plotly_chart(fig)

    # RUTA + INCIDENCIAS + RIESGO GEOPOLÍTICO
    with st.container(border=True):

        selected_origin = st.selectbox('Seleccionar puerto de origen',
                                        ports2)

        #Filtrado, extrayendo climas 'dominantes' en incidencias. Por puerto de origen
        df_temp = st.session_state.data.loc[:,['Origin_Port','Destination_Port','Weather_Condition', 'Geopolitical_Risk_Score', 'Disruption_Occurred']]
        
        df_temp = df_temp[~df_temp['Weather_Condition'].isin(['Hurricane', 'Storm'])]
        df_temp = df_temp[df_temp['Origin_Port'] == selected_origin]

        #Asignamos diferentes niveles de riesgo
        limits_political_risk = [0, 3, 6, 10]
        labels_polical_risk = ['Riesgo Bajo (0-3)', 'Riesgo Medio (3-6)', 'Riesgo Alto (6-10)']
        
        df_temp['Rango_Riesgo'] = pd.cut(
            df_temp['Geopolitical_Risk_Score'], 
            bins=limits_political_risk, 
            labels=labels_polical_risk, 
            include_lowest=True
        )

        #Agrupamos para extraer un conteo
        df_group = df_temp.groupby(
            ['Destination_Port','Rango_Riesgo','Disruption_Occurred'],
            observed=True).size().reset_index(name='Cantidad')
        #Aviso en terminal: observed=True FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
        
        df_group = df_group.rename(columns={'0':'Cantidad'})

        fig = px.bar(
            df_group,
            x='Rango_Riesgo',             
            y='Cantidad',               
            color='Disruption_Occurred',   
            facet_col='Destination_Port', 
            facet_col_wrap=3,             
            color_discrete_map={
                0: 'rgb(102, 197, 204)',  
                1: 'rgb(246, 207, 113)'   
            },
            labels={
                'Rango_Riesgo': 'Nivel de Riesgo Geopolítico',
                'Cantidad': 'Total',
                'Disruption_Occurred': 'Incidencia'
            },
            title=f"Incidencias según riesgo geopolítico en rutas desde {selected_origin}"
        )

        st.plotly_chart(fig)

with tab3:
    st.header("Análisis de Parche (Inferencia)")

    if not available_models:
        st.error(f"No se ha encontrado ningún modelo en la carpeta '{DIR_MODELOS}/'. Ejecuta el script de entrenamiento primero.")
    else:
        st.markdown("### Selección del Motor Predictivo")
        
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
        'K vecinos más cercanos':'K_vecinos_mas_cercanos.pkl',
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
                    label=f"Descargar modelo {selected_model_name} (.pkl)",
                    data=file,
                    file_name=selected_filename,
                    mime="application/octet-stream"
                )
        # -----------------------------------------------
        
        st.divider() # Línea separadora visual
        
        # Contenedor con borde para estética moderna
        with st.container(border=True):
            st.subheader("1. Cargar Muestra")
            fil1, fil2, fil3 = st.columns(3)
            ports = st.session_state.data['Origin_Port'].unique().tolist()
            transports_list = st.session_state.data['Transport_Mode'].unique().tolist()
            products = st.session_state.data['Product_Category'].unique().tolist()
            weathers_list = st.session_state.data['Weather_Condition'].unique().tolist()


            with fil1:
                origin_port_for = st.selectbox("Puerto de origen",
                                                    st.session_state.data['Origin_Port'].cat.categories)
                
                dest_port_for = st.selectbox("Puerto de destino",
                                                    st.session_state.data['Destination_Port'].cat.categories)

                transport_mode_for = st.selectbox("Transportado vía",
                                                        st.session_state.data['Transport_Mode'].cat.categories)
                prod_categ_for = st.selectbox("Categoría del producto",
                                                        st.session_state.data['Product_Category'].cat.categories)
            
            with fil2:
                dist_for = st.slider("Distancia en Km",
                                        st.session_state.data['Distance_km'].min(),
                                        st.session_state.data['Distance_km'].max(),
                                        st.session_state.data['Distance_km'].mean())
                weight_for = st.slider("Peso",
                                        st.session_state.data['Weight_MT'].min(),
                                        st.session_state.data['Weight_MT'].max(),
                                        st.session_state.data['Weight_MT'].mean())
                fuel_price_for = st.slider("Índice de precios del combustible",
                                        st.session_state.data['Fuel_Price_Index'].min(),
                                        st.session_state.data['Fuel_Price_Index'].max(),
                                        st.session_state.data['Fuel_Price_Index'].mean())
                geopolitical_for = st.slider("Puntuación de riesgo geopolítico",
                                        st.session_state.data['Geopolitical_Risk_Score'].min(),
                                        st.session_state.data['Geopolitical_Risk_Score'].max(),
                                        st.session_state.data['Geopolitical_Risk_Score'].mean())        
            
            with fil3:
                weather_for = st.selectbox("Condición climática",
                                                        st.session_state.data['Weather_Condition'].cat.categories)
            
                carrier_for = st.slider("Confianza del transportista",
                                        st.session_state.data['Carrier_Reliability_Score'].min(),
                                        st.session_state.data['Carrier_Reliability_Score'].max(),
                                        st.session_state.data['Carrier_Reliability_Score'].mean())

            instance_classif = pd.DataFrame([{
                'Origin_Port': origin_port_for,
                'Destination_Port': dest_port_for,
                'Transport_Mode': transport_mode_for,
                'Product_Category': prod_categ_for,
                'Distance_km': dist_for,
                'Weight_MT': weight_for,
                'Fuel_Price_Index': fuel_price_for,
                'Geopolitical_Risk_Score': geopolitical_for,
                'Weather_Condition': weather_for,
                'Carrier_Reliability_Score': carrier_for
            }])

            X_categ_cols = ['Origin_Port', 'Destination_Port', 'Transport_Mode', 'Product_Category', 'Weather_Condition']

            for col in X_categ_cols:
                instance_classif[col] = instance_classif[col].astype('category')

            st.write(instance_classif)
            
            # Contenedor con borde para el diagnóstico
        with st.container(border=True):
            st.subheader(f"2. Resultado de IA")
            
            if instance_classif is not None:
                with st.spinner(f"Analizando tejido con {selected_model_name}..."):
                    
                    prediction = active_model.predict_proba(instance_classif)[0][1] 
                    
                    threshold = 0.5
                    is_disrr_positive = prediction > threshold
                    
                    st.markdown("### Diagnóstico Asistido:")
                    if is_disrr_positive:
                        st.error(f"**Ocurrirá una incidencia** (Confianza: {prediction:.2%})")
                    else:
                        st.success(f"**No ocurrirá una incidencia** (Confianza: {(1 - prediction):.2%})")
                    
                    # Barra de progreso
                    st.progress(float(prediction), text="Probabilidad de Malignidad")
            else:
                st.info("Sube una imagen en la tarjeta de la izquierda para comenzar el análisis.")

with tab4:
    st.header("Análisis Comparativo y Métricas")
    
    if all_metrics_data is None:
        st.warning(f"No se encontraron datos de métricas en '{DIR_MODELOS}/all_metrics.pkl'. Ejecuta el entrenamiento primero.")
    else:
        st.markdown("""
        A continuación se muestran los resultados obtenidos por cada modelo 
        durante la fase de prueba con datos nunca antes vistos.
        """)
        
        # 1. Selector de modelo para ver su detalle
        modelo_stats = st.selectbox(
            "Selecciona el modelo para ver su detalle técnico:",
            options=list(all_metrics_data.keys())
        )
        
        # 2. Mostrar métricas clave dinámicas
        m = all_metrics_data[modelo_stats]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Precisión Global (Accuracy)", f"{m['Accuracy']:.2%}")
        col2.metric("Sensibilidad (Recall)", f"{m['Recall']:.2%}")
        col3.metric("Precisión Positivos (Precision)", f"{m['Precision']:.2%}")
        col4.metric("F1-Score", f"{m['F1']:.2%}")
        
        
        
        st.warning("""
        🩺 **Nota Clínica sobre la Sensibilidad (Recall):** En la detección de cáncer, la Sensibilidad es la métrica más crítica. Representa la capacidad del modelo para no dejar escapar ningún caso positivo. 
        Un valor bajo (por ejemplo, cercano al 50%) es **clínicamente inaceptable**, ya que significa que el modelo está fallando en detectar casi la mitad de los tejidos malignos reales (Falsos Negativos). En entornos médicos, se busca priorizar esta métrica por encima del 90-95%.
        """)

        st.divider()
        # 3. Mostrar las imágenes correspondientes a ese modelo
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.subheader("Matriz de Confusión")
            img_cm = os.path.join(DIR_MODELOS, f"{modelo_stats}_cm.png")
            if os.path.exists(img_cm):
                st.image(img_cm, width='stretch')
            else:
                st.info("Imagen no encontrada.")
                
        with col_img2:
            st.subheader("Curva ROC (AUC)")
            img_roc = os.path.join(DIR_MODELOS, f"{modelo_stats}_roc.png")
            if os.path.exists(img_roc):
                st.image(img_roc, width='stretch')
            else:
                st.info("Imagen no encontrada.")

        st.divider()
        
        # 4. Tabla resumen comparativa
        st.subheader("Resumen Comparativo General")
        # Convertimos el diccionario a DataFrame para mostrarlo bonito
        df_metrics = pd.DataFrame(all_metrics_data).T
        
        # Formatear el DataFrame a porcentajes antes de mostrarlo
        df_metrics_styled = df_metrics.style.format("{:.2%}").highlight_max(axis=0, color="#86c5ce")
        st.dataframe(df_metrics_styled, width='stretch')


    
        




    



    
    


