import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dask.distributed import Client
import joblib
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import os, sys
import warnings
import rpy2.robjects as ro
from rpy2.robjects import r
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

DIR_MODELOS_CLAS = 'modelos_clasificacion'
DIR_MODELOS_REG = 'modelos_regresion'
URL_DATOS = 'https://raw.githubusercontent.com/vicsilnieR/GlobalSupply/main/datos/supplies_data.parquet'

if 'data' not in st.session_state:
    st.session_state.data = None

@st.cache_data
def load_selected_data(dataset_name):
    data = pd.read_parquet(dataset_name)
    cols_to_category = ['Origin_Port', 'Destination_Port', 'Transport_Mode',
                        'Product_Category', 'Weather_Condition', 'Disruption_Occurred']
    if data['Disruption_Occurred'].dtype != 'object':
        data['Disruption_Occurred'] = data['Disruption_Occurred'].map({1: 'Sí', 0: 'No'}).astype(str)
    else:
        data['Disruption_Occurred'] = data['Disruption_Occurred'].astype(str)
    for col in cols_to_category:
        data[col] = data[col].astype('category')
    data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    return data

@st.cache_resource
def dask_client():
    
    ### Comenzamos configurando e inicializando el cliente de Dask
    cluster = LocalCluster(
        n_workers=2,                # num procesos
        threads_per_worker=3,       # hilos por proceso (6 núcleos en total)
        memory_limit="950MB",       # la RAM que quedaba libre
        processes=False             
    )
    #Inicialización del cliente de Dask
    client = Client(cluster)

if __name__ == '__main__':
    client = dask_client()

@st.cache_resource
def load_data_dask(dataset_name):
    data = dd.read_parquet(dataset_name)
    cols_to_category = ['Origin_Port', 'Destination_Port', 'Transport_Mode',
                        'Product_Category', 'Weather_Condition', 'Disruption_Occurred']
    disruption_map = {1: 'Sí', 0: 'No'}
   # Reemplaza el bloque del mapeo por esto:
    data['Disruption_Occurred'] = data['Disruption_Occurred'].replace({1: 'Sí', 0: 'No'}).astype(str)
    for col in cols_to_category:
        data[col] = data[col].astype('category')
    data['Date'] = dd.to_datetime(data['Date'], format='%Y/%m/%d')
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data = data.drop(columns=['Date'])
    return data

#Carga modelos y métricas: Clasificación
@st.cache_resource
def load_clas_models():
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
        path = os.path.join(DIR_MODELOS_CLAS, filename)
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
        else:
            st.error(f"No se encontró el archivo en: {path}")    
    return loaded_models

@st.cache_data
def load_clas_metrics():
    """
    Carga el diccionario con las métricas guardadas durante el entrenamiento.
    """
    path = os.path.join(DIR_MODELOS_CLAS, 'all_metrics.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None

available_clas_models = load_clas_models()
clas_metrics_data = load_clas_metrics()

#Carga modelos y métricas: Regresión
@st.cache_resource
def load_reg_models():
    """
    Busca y carga todos los modelos entrenados desde la carpeta 'modelos/'.
    """
    model_files = {
        'K vecinos más cercanos':'K_vecinos_mas_cercanos.pkl',
        'Máquinas de soporte vectorial':'Maquinas_de_soporte_vectorial.pkl',
        'Regresión Lineal': 'Regresion_lineal.pkl',
        'Bosques Aleatorios': 'Bosques_aleatorios.pkl',
        'Gradient Boosting': 'Gradient_boosting.pkl'
    }

    loaded_models = {}
    for name, filename in model_files.items():
        path = os.path.join(DIR_MODELOS_REG, filename)
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
        else:
            st.error(f"No se encontró el archivo en: {path}")    
    return loaded_models

@st.cache_data
def load_reg_metrics():
    """
    Carga el diccionario con las métricas guardadas durante el entrenamiento.
    """
    path = os.path.join(DIR_MODELOS_REG, 'all_metrics.pkl')
    if os.path.exists(path):
        return joblib.load(path)
    return None

available_reg_models = load_reg_models()
reg_metrics_data = load_reg_metrics()

#Carga de funciones en script R
@st.cache_resource
def load_R_functions():
    with localconverter(ro.default_converter + pandas2ri.converter):
        r.source('R_functions.R')
        r_functions = {
            "contar_filas": ro.globalenv['contar_filas'],
            "media_envio": ro.globalenv['media_envio']
        }
    
    return r_functions

dict_R = load_R_functions()

@st.cache_resource
def apply_political_risk_range(pandas_partition):
    #Función para mapear el df de dask
    limits_political_risk = [0, 3, 6, 10]
    labels_polical_risk = ['Riesgo Bajo (0-3)', 'Riesgo Medio (3-6)', 'Riesgo Alto (6-10)']
    
    # pd.cut sobre la partición que toque
    pandas_partition['Rango_Riesgo'] = pd.cut(
        pandas_partition['Geopolitical_Risk_Score'], 
        bins=limits_political_risk, 
        labels=labels_polical_risk, 
        include_lowest=True
    )
    return pandas_partition

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Presentación",
                                        "Carga de datos",
                                        "Visualizaciones descriptivas",
                                        "Predicción con modelos", 
                                        "Eficacia de los modelos"])

with tab1:
    st.header('Presentación del problema')

    with st.container(border=False):
        st.markdown(
            "<div style='padding-left: 20px;'>\n\n"
            "#### Conjunto de datos\n\n"
            "Nuestro conjunto de datos recoge envíos de mercancía entre distintos puertos de todo el mundo con múltiples parámetros.\n\n"
            "#### Objetivo\n\n"
            "El objetivo que se presenta será predecir si ocurrirá una incidencia en envíos futuros, además de la cantidad de días que requerirá dicho envío. "
            "Además, se intentará señalar qué parámetros tienen mayor influencia en el resultado.\n\n"
            "#### Detalle de los datos\n\n"
            "* **Puerto de origen**: Lugar desde el que se envían las mercancías.\n"
            "* **Puerto de destino**: Lugar hasta el que llegan las mercancías.\n"
            "* **Medio de transporte**: Puede ser por mar, aire y tierra (tren o carretera).\n"
            "* **Categoría del producto**: Textil, automovilístico, electrónica...\n"
            "* **Distancia recorrida** (Km).\n"
            "* **Peso** (TM): Peso de los productos enviados.\n"
            "* **Índice de precio del combustible**: Multiplicador del coste del combustible normalizado.\n"
            "* **Puntuación de riesgo geopolítico**: Índice de riesgo basado en la estabilidad regional (0-10).\n"
            "* **Condición climática**: Condiciones atmosféricas durante el tránsito (Despejado, lluvia, tormenta...).\n"
            "* **Puntuación de confianza del transportista** (0.5-1).\n"
            "* **Días de tránsito**: Tiempo que ha tomado el envío (Objetivo de la regresión).\n"
            "* **Incidencia ocurrida**: Si ocurrió (1), o no (0), una incidencia durante el tránsito.\n\n"
            "</div>", 
            unsafe_allow_html=True
        )

with tab2:
    st.header('Carga de datos y previsualización')

    with st.spinner('Cargando datos de transporte global'):
        st.session_state.data = load_selected_data(URL_DATOS)
    st.success('Se han cargado los datos correctamente')

    fil1, fil2, fil3 = st.columns(3)
    orig_ports = st.session_state.data['Origin_Port'].unique().tolist()
    dest_ports = st.session_state.data['Destination_Port'].unique().tolist() #Correccion aviso terminal: FutureWarning: Categorical.to_list is deprecated and will be removed in a future version. Use obj.tolist() instead
    transports_list = st.session_state.data['Transport_Mode'].unique().tolist()
    products = st.session_state.data['Product_Category'].unique().tolist()
    weathers_list = st.session_state.data['Weather_Condition'].unique().tolist()


    with fil1:
        origin_port_fil = st.multiselect("Puerto de origen",
                                            st.session_state.data['Origin_Port'].cat.categories,
                                            default=[])
        if origin_port_fil==[]:
            origin_port_fil = orig_ports
        
        dest_port_fil = st.multiselect("Puerto de destino",
                                            st.session_state.data['Destination_Port'].cat.categories,
                                            default=[])
        if dest_port_fil==[]:
            dest_port_fil = dest_ports
        
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

        disrr_fil = disrr_sel
        if disrr_fil == []:
            disrr_fil = ['Sí', 'No']

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
    df_filtered_visualization = st.session_state.data.query(query_filter_df)
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_dataframe = ro.conversion.py2rpy(df_filtered_visualization)     

        num_filas_filtradas_r = dict_R["contar_filas"](r_dataframe)
        num_filas_filtradas_py = ro.conversion.rpy2py(num_filas_filtradas_r)
        num_filas_filtradas = int(num_filas_filtradas_py[0])

        media_envios_filtrados_r = dict_R["media_envio"](r_dataframe)
        media_envios_filtrados_py = ro.conversion.rpy2py(media_envios_filtrados_r)
        media_envios_filtrados = int(media_envios_filtrados_py[0])

    st.markdown('#### Con las características seleccionadas:')
    col1a, col2a = st.columns(2)
    col1a.metric("Observaciones:", num_filas_filtradas)
    col2a.metric("Días de transporte medio:", media_envios_filtrados)

    st.dataframe(df_filtered_visualization,
                    width='stretch', #corrección aviso en terminal: For `use_container_width=True`, use `width='stretch'`
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="multi-row",
                    )
    
with tab3:
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
            fig.add_trace(go.Violin(x=df_filtered['Weather_Condition'][df_filtered['Disruption_Occurred'] == 'No'],
                                    y=df_filtered['Lead_Time_Days'][df_filtered['Disruption_Occurred'] == 'No'],
                                    legendgroup='Sin Incidencia',
                                    scalegroup='Sin Incidencia',
                                    name='Sin Incidencia',
                                    side='negative',
                                    line_color=azul_graf,
                                    showlegend=show_legend[i]
                                    )
                        )
            fig.add_trace(go.Violin(x=df_filtered['Weather_Condition'][df_filtered['Disruption_Occurred'] == 'Sí'],
                                    y=df_filtered['Lead_Time_Days'][df_filtered['Disruption_Occurred'] == 'Sí'],
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
        #Pasamos a df de dask
        df_temp = load_data_dask(URL_DATOS)
        #Filtrado, extrayendo climas 'dominantes' en incidencias. Por puerto de origen
        
        df_temp = df_temp.query("Weather_Condition != 'Hurricane' and Weather_Condition != 'Storm'")
        df_temp = df_temp.query(f"Origin_Port == '{selected_origin}'")

        #Asignamos diferentes niveles de riesgo con map_partitions y la función creada previamente
        df_temp = df_temp.map_partitions(
            apply_political_risk_range,
            meta = df_temp._meta.assign(Rango_Riesgo = 'category'))
            #Indicamos la nueva columna que generará la función 
            #que se mapea.

        #Agrupamos para extraer un conteo
        df_group = df_temp.groupby(
            ['Destination_Port','Rango_Riesgo','Disruption_Occurred'],
            observed=True).size()
        #Aviso en terminal: observed=True FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
        
        df_group = df_group.reset_index()
        df_group = df_group.rename(columns={0: 'Cantidad'})
        
        df_grafico = df_group.compute()

        fig = px.bar(
            df_grafico,
            x='Rango_Riesgo',             
            y='Cantidad',               
            color='Disruption_Occurred',   
            facet_col='Destination_Port', 
            facet_col_wrap=3,             
            color_discrete_map={
                'No': 'rgb(102, 197, 204)',  
                'Sí': 'rgb(246, 207, 113)'   
            },
            labels={
                'Rango_Riesgo': 'Nivel de Riesgo Geopolítico',
                'Cantidad': 'Total',
                'Disruption_Occurred': 'Incidencia'
            },
            title=f"Incidencias según riesgo geopolítico en rutas desde {selected_origin}"
        )

        st.plotly_chart(fig)

with tab4:
    st.header("Análisis de transporte")

    colclas, colreg = st.columns(2)

    with colclas:
        if not available_clas_models:
            st.error(f"No se ha encontrado ningún modelo en la carpeta '{DIR_MODELOS_CLAS}/'. Ejecuta el script de entrenamiento primero.")
        else:
            st.markdown("#### Selección del Motor Predictivo: clasificación")
            st.markdown("##### Modelo recomendado: K vecinos más cercanos")
            st.markdown("Alternativa: Regresión logística")
            
            # El usuario elige el modelo
            selected_clas_model_name = st.selectbox(
                "Selecciona la Inteligencia Artificial a usar:", 
                options=list(available_clas_models.keys()),
                index=len(available_clas_models)-1 
            )
            
            active_clas_model = available_clas_models[selected_clas_model_name]
            
            # --- NUEVO: Lógica para el botón de descarga ---
            # Diccionario para saber qué archivo corresponde a cada nombre del selectbox
            model_filenames_clas = {
            'K vecinos más cercanos':'K_vecinos_mas_cercanos.pkl',
            'Máquinas de soporte vectorial':'Máquinas_de_soporte_vectorial.pkl',
            'Regresión Logística': 'Regresión_logistica.pkl',
            'Bosques Aleatorios': 'Bosques_aleatorios.pkl',
            'Gradient Boosting': 'Gradient_boosting.pkl'
        }

            
            # Obtenemos la ruta del archivo seleccionado
            selected_filename_clas = model_filenames_clas[selected_clas_model_name]
            file_path_clas = os.path.join(DIR_MODELOS_CLAS, selected_filename_clas)
            
            # Leemos el archivo en modo binario y creamos el botón
            if os.path.exists(file_path_clas):
                with open(file_path_clas, "rb") as file:
                    st.download_button(
                        label=f"Descargar modelo clasificación {selected_clas_model_name} (.pkl)",
                        data=file,
                        file_name=selected_filename_clas,
                        mime="application/octet-stream",
                        key='db1'
                    )
    with colreg:
        if not available_reg_models:
            st.error(f"No se ha encontrado ningún modelo en la carpeta '{DIR_MODELOS_REG}/'. Ejecuta el script de entrenamiento primero.")
        else:
            st.markdown("#### Selección del Motor Predictivo: regresión")
            st.markdown("##### Modelo recomendado: Bosques aleatorios")

            # El usuario elige el modelo
            selected_reg_model_name = st.selectbox(
                "Selecciona la Inteligencia Artificial a usar:", 
                options=list(available_reg_models.keys()),
                index=len(available_reg_models)-1 
            )
            
            active_reg_model = available_reg_models[selected_reg_model_name]
            
            # --- NUEVO: Lógica para el botón de descarga ---
            # Diccionario para saber qué archivo corresponde a cada nombre del selectbox
            model_filenames_reg = {
            'K vecinos más cercanos':'K_vecinos_mas_cercanos.pkl',
            'Máquinas de soporte vectorial':'Maquinas_de_soporte_vectorial.pkl',
            'Regresión Lineal': 'Regresion_lineal.pkl',
            'Bosques Aleatorios': 'Bosques_aleatorios.pkl',
            'Gradient Boosting': 'Gradient_boosting.pkl'
        }

            
            # Obtenemos la ruta del archivo seleccionado
            selected_filename_reg = model_filenames_reg[selected_reg_model_name]
            file_path_reg = os.path.join(DIR_MODELOS_REG, selected_filename_reg)
            
            # Leemos el archivo en modo binario y creamos el botón
            if os.path.exists(file_path_reg):
                with open(file_path_reg, "rb") as file:
                    st.download_button(
                        label=f"Descargar modelo regresión {selected_reg_model_name} (.pkl)",
                        data=file,
                        file_name=selected_filename_reg,
                        mime="application/octet-stream",
                        key='db2'
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

        st.markdown("#### Previsualización de la instancia")
        st.write(instance_classif)
        
        # Contenedor con borde para el diagnóstico
    
    #Resultado de la inferencia IA
    with st.container(border=True):
        st.subheader(f"2. Resultado de IA")

        with st.spinner(f"Analizando, incidencia con {selected_clas_model_name} y días de transporte con {selected_reg_model_name}..."):
            
            prediction_clas = active_clas_model.predict_proba(instance_classif)[0][1] 
            prediction_reg = active_reg_model.predict(instance_classif)[0]
            threshold = 0.5
            is_disrr_positive = prediction_clas > threshold
            mm = reg_metrics_data[selected_filename_reg[:-4]]


            st.markdown("#### Diagnóstico Asistido:")
            if is_disrr_positive:
                st.error(f"**Ocurrirá una incidencia** (Confianza: {prediction_clas:.2%})")
            else:
                st.success(f"**No ocurrirá una incidencia** (Confianza: {(1 - prediction_clas):.2%})")
            
            st.info(f"**Tiempo estimado de transporte:** {prediction_reg:.1f} días (± {mm['MAE']:.1f} días)")


with tab5:
    st.header("Análisis Comparativo y Métricas")
    
    if clas_metrics_data is None:
        st.warning(f"No se encontraron datos de métricas en '{DIR_MODELOS_CLAS}/all_metrics.pkl'. Ejecuta el entrenamiento primero.")
    else:
        st.markdown("""
        A continuación se muestran los resultados obtenidos por cada modelo 
        durante la fase de prueba con datos nunca antes vistos.
        """)
        
        # 1. Selector de modelo para ver su detalle
        modelo_stats = st.selectbox(
            "Selecciona el modelo para ver su detalle técnico:",
            options=list(clas_metrics_data.keys())
        )
        
        # 2. Mostrar métricas clave dinámicas
        m = clas_metrics_data[modelo_stats]
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
            img_cm = os.path.join(DIR_MODELOS_CLAS, f"{modelo_stats}_cm.png")
            if os.path.exists(img_cm):
                st.image(img_cm, width='stretch')
            else:
                st.info("Imagen no encontrada.")
                
        with col_img2:
            st.subheader("Curva ROC (AUC)")
            img_roc = os.path.join(DIR_MODELOS_CLAS, f"{modelo_stats}_roc.png")
            if os.path.exists(img_roc):
                st.image(img_roc, width='stretch')
            else:
                st.info("Imagen no encontrada.")

        st.divider()
        
        # 4. Tabla resumen comparativa
        st.subheader("Resumen Comparativo General")
        # Convertimos el diccionario a DataFrame para mostrarlo bonito
        df_metrics = pd.DataFrame(clas_metrics_data).T
        
        # Formatear el DataFrame a porcentajes antes de mostrarlo
        df_metrics_styled = df_metrics.style.format("{:.2%}").highlight_max(axis=0, color="#86c5ce")
        st.dataframe(df_metrics_styled, width='stretch')

#_______________________________________________________________

    if reg_metrics_data is None:
        st.warning(f"No se encontraron datos de métricas en '{DIR_MODELOS_REG}/all_metrics.pkl'. Ejecuta el entrenamiento primero.")
    else:
        st.markdown("""
        A continuación se muestran los resultados obtenidos por cada modelo 
        durante la fase de prueba con datos nunca antes vistos.
        """)
        
        # 1. Selector de modelo para ver su detalle
        modelo_stats = st.selectbox(
            "Selecciona el modelo para ver su detalle técnico:",
            options=list(reg_metrics_data.keys())
        )
        
        # 2. Mostrar métricas clave dinámicas
        m = reg_metrics_data[modelo_stats]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Error Absoluto Medio (MAE)", f"{m['MAE']:.2f} días")
        col2.metric("Error Cuadrático Medio (MSE)", f"{m['MSE']:.2f}")
        col3.metric("Raíz del Error Cuadrático Medio (RMSE)", f"{m['RMSE']:.2f} días")
        col4.metric("Coeficiente de Determinación (R²)", f"{m['R2']:.2%}")
        
        
        
        st.warning("""
        🩺 **Nota Clínica sobre la Sensibilidad (Recall):** En la detección de cáncer, la Sensibilidad es la métrica más crítica. Representa la capacidad del modelo para no dejar escapar ningún caso positivo. 
        Un valor bajo (por ejemplo, cercano al 50%) es **clínicamente inaceptable**, ya que significa que el modelo está fallando en detectar casi la mitad de los tejidos malignos reales (Falsos Negativos). En entornos médicos, se busca priorizar esta métrica por encima del 90-95%.
        """)

        st.divider()
        # 3. Mostrar las imágenes correspondientes a ese modelo
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.subheader("Gráfico CALIDAD REG: PONER NOMBRE")
            img_dis = os.path.join(DIR_MODELOS_REG, f"{modelo_stats}_dispersion.png")
            if os.path.exists(img_dis):
                st.image(img_dis, width='stretch')
            else:
                st.info("Imagen no encontrada.")
                
        st.divider()
        
        # 4. Tabla resumen comparativa
        st.subheader("Resumen Comparativo General")
        # Convertimos el diccionario a DataFrame para mostrarlo bonito
        df_metrics = pd.DataFrame(reg_metrics_data).T
        
        # Formatear el DataFrame a porcentajes antes de mostrarlo
        columnas_error = ['MAE', 'MSE', 'RMSE']
        columna_r2 = ['R2']

        df_metrics_styled = (
            df_metrics.style.format("{:.2f}", subset=columnas_error)
            .format("{:.2%}", subset=columna_r2)
            .highlight_min(axis=0, color="#86c5ce", subset=columnas_error)
            .highlight_max(axis=0, color="#86c5ce", subset=columna_r2)
        )
        st.dataframe(df_metrics_styled, width='stretch')
    
        




    



    
    


