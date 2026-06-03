import dagster as dg

start_date = "2026-05-29"

daily_partition = dg.DailyPartitionsDefinition(
    start_date=start_date,
    timezone="Europe/Madrid",
)