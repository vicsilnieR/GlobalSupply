import kagglehub
import dagster as dg
import os

from dagster_duckdb import DuckDBResource
# from my_project.defs.assets import constants
from dagster import asset, definitions, resources
from github import Github, GithubException

@dg.asset
def rawGlobalSupply(database: DuckDBResource) -> dg.MaterializeResult:
    
    path = kagglehub.dataset_download("nudratabbas/global-supply-chain-risk-and-logistics-2024-2026")
    csv_file = os.path.join(path, "global_supply_chain_risk_2026.csv")

    query = f"""
        CREATE OR REPLACE TABLE supplies AS
        SELECT * FROM read_csv_auto('{csv_file}')
        ORDER BY Shipment_ID ASC
        LIMIT 4000;
    """
    
    with database.get_connection() as conn:
        conn.execute(query)
        num_rows = conn.execute("SELECT COUNT(*) FROM supplies").fetchone()[0]

    return dg.MaterializeResult(
        metadata={
            'Número de filas': dg.MetadataValue.int(num_rows)
        }
    )

@dg.asset(deps=[rawGlobalSupply])
def suppliesGlobalSupply(database: DuckDBResource) -> None:
    output_path = "supplies_data.parquet"
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

@dg.asset
def new_rawGlobalSupply(database: DuckDBResource) -> dg.MaterializeResult:
    
    path = kagglehub.dataset_download("nudratabbas/global-supply-chain-risk-and-logistics-2024-2026")
    csv_file = os.path.join(path, "global_supply_chain_risk_2026.csv")

    query = f"""
        CREATE OR REPLACE TABLE supplies_predict AS
        SELECT * FROM read_csv_auto('{csv_file}')
        ORDER BY Shipment_ID ASC
        LIMIT 500 
        OFFSET 4000;
    """
    
    with database.get_connection() as conn:
        conn.execute(query)
        num_rows = conn.execute("SELECT COUNT(*) FROM supplies_predict").fetchone()[0]

    return dg.MaterializeResult(
        metadata={
            'Número de filas': dg.MetadataValue.int(num_rows)
        }
    )