import dagster as dg
from orquestacion_supplies.defs.jobs import supplies_update_job
import datetime

@dg.schedule(
    job=supplies_update_job,
    cron_schedule="55 6 * * 1-5", # Tu cron original: 02:06 AM de Lunes a Viernes
    execution_timezone="Europe/Madrid"
)
def supplies_update_job_schedule(context):
    # Fecha en la que se está ejecutando el Schedule
    execution_date = context.scheduled_execution_time.date()
    
    #Restamos 1 día para obtener la fecha de los datos que queremos procesar (ayer)
    previous_day = execution_date - datetime.timedelta(days=1)
    partition_date_str = previous_day.strftime("%Y-%m-%d")
    
    # Información partición clara 
    return dg.RunRequest(
        run_key=partition_date_str,       # Una partición por día
        partition_key=partition_date_str, # context.partition_key en el asset
    )

