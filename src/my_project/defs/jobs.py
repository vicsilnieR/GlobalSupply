import dagster as dg
from my_project.defs.partitions import daily_partition

supplies_update_job = dg.define_asset_job(
    name="supplies_update_job",
    partitions_def=daily_partition,
    selection=dg.AssetSelection.groups("new_supplies")
)



