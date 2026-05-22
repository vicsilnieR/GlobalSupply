import dagster as dg
from my_project.defs.jobs import supplies_update_job

import datetime



@dg.schedule(
    job=supplies_update_job,
    cron_schedule="28 6 * * 1-5", # Tu cron original: 02:06 AM de Lunes a Viernes
    execution_timezone="Europe/Madrid"
)
def supplies_update_job_schedule(context):
    # 1. Obtenemos la fecha en la que se está ejecutando el Schedule
    execution_date = context.scheduled_execution_time.date()
    
    # 2. Restamos 1 día para obtener la fecha de los datos que queremos procesar (ayer)
    previous_day = execution_date - datetime.timedelta(days=1)
    partition_date_str = previous_day.strftime("%Y-%m-%d")
    
    # 3. Le mandamos la partición masticada a Dagster
    return dg.RunRequest(
        run_key=partition_date_str,       # Evita que se duplique la ejecución el mismo día
        partition_key=partition_date_str, # Esto llegará a tu asset como context.partition_key
    )




#supplies_update_schedule = dg.ScheduleDefinition(
 #   job=supplies_update_job,
  #  name="supplies_update_schedule",
   # cron_schedule="52 5 * * 1-5", #Minuto Hora Día(mes) Mes Días(semana)
    #execution_timezone="Europe/Madrid"
#)
