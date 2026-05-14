contar_filas <- function(x) {
  nrow(x)[1]
}
media_envio <- function(x) {
  mean(x$Lead_Time_Days)
}
