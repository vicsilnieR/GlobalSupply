import dagster as dg

supplies_update_job = dg.define_asset_job(
    name="supplies_update_job",
    selection=dg.AssetSelection.all()
)

