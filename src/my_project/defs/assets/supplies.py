import kagglehub
import dagster as dg
import os

from dagster_duckdb import DuckDBResource
# from my_project.defs.assets import constants
from dagster import asset, definitions, resources
from github import Github, GithubException
from my_project.defs.partitions import daily_partition

@dg.asset(group_name="training")
def rawTrainingGlobalSupply(database: DuckDBResource) -> dg.MaterializeResult:
    
    path = kagglehub.dataset_download("nudratabbas/global-supply-chain-risk-and-logistics-2024-2026")
    csv_file = os.path.join(path, "global_supply_chain_risk_2026.csv")

    dates_saved = "('2025-12-28', '2025-12-29', '2025-12-30', '2025-12-31')"
    #Las hemos guardado para job (schedule o sensor)

    query = f"""
        CREATE OR REPLACE TABLE supplies AS
        SELECT * FROM read_csv_auto('{csv_file}')
        WHERE Date NOT IN {dates_saved}
        ORDER BY Date ASC;
    """
    
    with database.get_connection() as conn:
        conn.execute(query)
        num_rows = conn.execute("SELECT COUNT(*) FROM supplies").fetchone()[0]

    return dg.MaterializeResult(
        metadata={
            'Número de filas': dg.MetadataValue.int(num_rows)
        }
    )

@dg.asset(deps=[rawTrainingGlobalSupply],
          group_name="training")
def suppliesGlobalSupply(database: DuckDBResource) -> None:
    
    os.makedirs("datos_local", exist_ok=True)
    output_path = "datos_local/supplies_data.parquet"
    exclude_column = "Shipment_ID"

    query = f"""
        COPY ( 
            SELECT * EXCLUDE ({exclude_column})
            FROM supplies
        ) TO '{output_path}' (FORMAT PARQUET);
    """

    with database.get_connection() as conn:
        conn.execute(query)

    #variables necesarias para la conexión con github
    token = os.getenv('GITHUB_TOKEN')
    repo_name = "vicsilnieR/GlobalSupply"
    branch = "main"
    github_path = "datos/supplies_data.parquet"
    commit_message = "supplies_data desde dagster"

    if not token:
        raise ValueError('Token de GitHub no disponible')
    
    with open(output_path, 'rb') as file:
        content = file.read()
    
    #Conectar con github
    g_conn = Github(token)
    repo = g_conn.get_repo(repo_name)

    try:
        contents = repo.get_contents(github_path, ref=branch)
        repo.update_file(
            path=github_path,
            message=commit_message,
            content=content,
            sha=contents.sha,
            branch=branch
        )
    except GithubException as e:
        #Si el archivo no existe, debemos crearlo por primera vez

        if e.status == 404 or "Not Found":
            repo.create_file(
                path=github_path,
                message=commit_message,
                content=content,
                branch=branch
            )
        else:
            raise e

#__________________________De aquí parriba no tocar___________________________________________

#Inicio tabla vacía: materializar una vez
@dg.asset()
def init_GlobalSupply_history(database: DuckDBResource) -> None:

    path = kagglehub.dataset_download("nudratabbas/global-supply-chain-risk-and-logistics-2024-2026")
    csv_file = os.path.join(path, "global_supply_chain_risk_2026.csv")

    query = f"""
        CREATE TABLE IF NOT EXISTS supplies_history AS
        SELECT * EXCLUDE (Disruption_Occurred, Lead_Time_Days)
        FROM read_csv_auto('{csv_file}')        
        WHERE 1=0;
    """

    with database.get_connection() as conn:
        conn.execute(query)

#Asset que guarda en mi tabla histórica los envíos de un día (ese día)

simulation_map = { #Necesario para simular que 'hoy' se corresponde en fecha a nuestros datos
    "2026-05-21": "2025-12-28",
    "2026-05-22": "2025-12-29",
    "2026-05-23": "2025-12-30",
    "2026-05-24": "2025-12-31"
}

@dg.asset(
    #deps=["init_GlobalSupply_history"],
    partitions_def=daily_partition,
    group_name="new_supplies")
def add_GlobalSupply_history(
    context: dg.AssetExecutionContext,
    database: DuckDBResource) -> dg.MaterializeResult:
    
    #Obtenemos fecha de hoy 
    date_today = context.partition_key
    #Mapeo para simular que 'hoy' se corresponde en fecha a nuestros datos
    date_sim_today = simulation_map[date_today]

    #Busco los datos en mi carpeta
    csv_file = f"https://raw.githubusercontent.com/vicsilnieR/GlobalSupply/refs/heads/main/datos/simulacion/supplies_{date_sim_today}.csv"

    query_delete = f"""
        DELETE FROM supplies_history 
        WHERE Date = '{date_sim_today}';
    """

    query_add = f"""
        INSERT INTO supplies_history
        SELECT * FROM read_csv_auto('{csv_file}');
    """
    
    with database.get_connection() as conn:
        conn.execute(query_delete)
        conn.execute(query_add)
        num_rows = conn.execute("SELECT COUNT(*) FROM supplies_history").fetchone()[0]

    return dg.MaterializeResult(
        metadata={
            'Número de filas': dg.MetadataValue.int(num_rows)
        }
    )

@dg.asset(
    deps=["add_GlobalSupply_history"],
    partitions_def=daily_partition,
    group_name="new_supplies")
def new_GlobalSupply_prediction(
    context: dg.AssetExecutionContext,
    database: DuckDBResource) -> None:

    date_today = context.partition_key
    date_sim_today = simulation_map[date_today]

    os.makedirs("datos_local/predecir", exist_ok=True)
    output_path = f"datos_local/predecir/supplies_to_predict_{date_sim_today}.parquet"    
  
    query = f"""
        COPY (
            SELECT *
            FROM supplies_history  
            WHERE Date = '{date_sim_today}'
        ) TO '{output_path}' (FORMAT PARQUET);
    """
    with database.get_connection() as conn:
        conn.execute(query)

    #variables necesarias para la conexión con github
    token = os.getenv('GITHUB_TOKEN')
    repo_name = "vicsilnieR/GlobalSupply"
    branch = "main"
    github_path = f"datos/supplies_to_predict_{date_sim_today}.parquet"
    commit_message = "supplies_to_predict desde dagster"

    if not token:
        raise ValueError('Token de GitHub no disponible')
    
    with open(output_path, 'rb') as file:
        content = file.read()
    
    #Conectar con github
    g_conn = Github(token)
    repo = g_conn.get_repo(repo_name)

    try:
        contents = repo.get_contents(github_path, ref=branch)
        repo.update_file(
            path=github_path,
            message=commit_message,
            content=content,
            sha=contents.sha,
            branch=branch
        )
    except GithubException as e:
        #Si el archivo no existe, debemos crearlo por primera vez

        if e.status == 404 or "Not Found":
            repo.create_file(
                path=github_path,
                message=commit_message,
                content=content,
                branch=branch
            )
        else:
            raise e



#___________________________Eliminar cuando estemos listos ____________________________________

@dg.asset
def rawTrainingGlobalSupplyBis(database: DuckDBResource) -> dg.MaterializeResult:
    
    path = kagglehub.dataset_download("nudratabbas/global-supply-chain-risk-and-logistics-2024-2026")
    csv_file = os.path.join(path, "global_supply_chain_risk_2026.csv")

    query = f"""
        CREATE OR REPLACE TABLE supplies_bis AS
        SELECT * FROM read_csv_auto('{csv_file}')
        ORDER BY Date ASC;
    """
    
    with database.get_connection() as conn:
        conn.execute(query)
        num_rows = conn.execute("SELECT COUNT(*) FROM supplies").fetchone()[0]

    return dg.MaterializeResult(
        metadata={
            'Número de filas': dg.MetadataValue.int(num_rows)
        }
    )

@dg.asset(deps=[rawTrainingGlobalSupplyBis])
def suppliesBisGlobalSupply(database: DuckDBResource) -> None:
    output_path = "supplies_data_bis.parquet"

    query = f"""
        COPY ( 
            SELECT *
            FROM supplies_bis
        ) TO '{output_path}' (FORMAT PARQUET);
    """

    with database.get_connection() as conn:
        conn.execute(query)