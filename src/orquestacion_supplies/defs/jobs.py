import dagster as dg
from orquestacion_supplies.defs.partitions import daily_partition

supplies_update_job = dg.define_asset_job(
    name="supplies_update_job",
    partitions_def=daily_partition,
    selection=dg.AssetSelection.groups("new_supplies")
)



