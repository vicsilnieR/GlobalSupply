import pandas as pd
import kagglehub
import os

def load_selected_data(dataset_name):
    data = pd.read_parquet(dataset_name)
    cols_to_category = ['Origin_Port', 'Destination_Port', 'Transport_Mode',
                        'Product_Category', 'Weather_Condition', 'Disruption_Occurred']
    for col in cols_to_category:
        data[col] = data[col].astype('category')
    data['Date'] = pd.to_datetime(data['Date'], format='%Y/%m/%d')
    data = data.drop(columns=['Disruption_Occurred', 'Lead_Time_Days'])
    return data

df = load_selected_data("supplies_data_bis.parquet")
print(df.head())

ultimas_cuatro_fechas = df['Date'].drop_duplicates().sort_values().tail(4)

fechas_simulacion = ['2025-12-28', '2025-12-29', '2025-12-30', '2025-12-31']

for fecha in fechas_simulacion:
    # Filtramos el DataFrame para quedarnos solo con los envíos de ese día
    df_del_dia = df[df['Date'] == fecha]
    
    # Creamos el nombre del archivo (ej: envios_2025-12-28.csv)
    nombre_archivo = f"supplies_{fecha}.csv"
    
    # Lo exportamos a CSV sin el índice de Pandas para que quede limpio
    df_del_dia.to_csv(nombre_archivo, index=False)
    
    print(f"Generado con éxito: {nombre_archivo} ({len(df_del_dia)} envíos)")

