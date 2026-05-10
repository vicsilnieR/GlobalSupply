import kagglehub
import dagster as dg
import os

from dagster_duckdb import DuckDBResource
# from my_project.defs.assets import constants
from dagster import asset, definitions, resources

@dg.asset
def rawGlobalSupply(database: DuckDBResource) -> dg.MaterializeResult:
    
    path = kagglehub.dataset_download("nudratabbas/global-supply-chain-risk-and-logistics-2024-2026")
    csv_file = os.path.join(path, "global_supply_chain_risk_2026.csv")

    query = f"""
        CREATE OR REPLACE TABLE supplies AS
        SELECT * FROM read_csv_auto('{csv_file}')
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
def categGlobalSupply(database: DuckDBResource) -> None:
    output_path = "categ_data.parquet"
    exclude_column = "Shipment_ID"

    query = f"""
        COPY ( 
            SELECT * EXCLUDE ({exclude_column})
            FROM supplies
        ) TO '{output_path}' (FORMAT PARQUET);
    """

    with database.get_connection() as conn:
        conn.execute(query)

@dg.asset(deps=[rawGlobalSupply])
def regGlobalSupply(database: DuckDBResource) -> None:
    output_path = "reg_data.parquet"
    exclude_column = "Disruption_Occurred"

    query = f"""
        COPY ( 
            SELECT * EXCLUDE ({exclude_column})
            FROM supplies
        ) TO '{output_path}' (FORMAT PARQUET);
    """

    with database.get_connection() as conn:
        conn.execute(query)

