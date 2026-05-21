import dagster as dg
from my_project.defs.jobs import supplies_update_job

supplies_update_schedule = dg.ScheduleDefinition(
    job=supplies_update_job,
    cron_schedule="35 19 * * 1-5" #Minuto Hora Día(mes) Mes Días(semana)
    #-2 Horas por el uso horario
)