import dagster as dg

start_date = "2026-05-21"
end_date = "2026-05-25"

daily_partition = dg.DailyPartitionsDefinition(
    start_date=start_date,
    #hour_offset=5,
    #minute_offset=3,
    timezone="Europe/Madrid",
    #end_date=end_date
)