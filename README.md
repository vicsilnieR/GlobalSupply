# Global Supply

## Iniciando el proyecto

### Instalar dependencias

Crearemos un entorno virtual e instalaremos las dependencias requeridas usando _sync_:

```bash
uv sync
```

Posteriormente, activaremos el entorno virtual:

| OS | Command |
| --- | --- |
| MacOS | ```source .venv/bin/activate``` |
| Windows | ```.venv\Scripts\activate``` |

### Inicializar Dagster

Iniciar UI Dagster web server:

```bash
dg dev
```

Abrimos http://localhost:3000 en el navegador para ver el proyecto

#### Reproducción del proyecto

**Paso 1:**
Iniciamos la UI Dagster para materializar los assets. Esto crea los archivos en la nube que después cargaremos en los distintos scripts.
Puede ser necesario hacer backfills de los datos diarios.

**Paso 2:**
Entrenar los modelos ejecutando los correspondientes scripts en la carpeta *entrenamiento_modelos*.

**Paso 3:**
Iniciar la aplicación en local:

```bash
uv run streamlit run main.py
```

**Paso 4:**
Deploy con streamlit y github para acceso desde cualquier navegador.

#### Objetivo del proyecto

##### Objetivo
El objetivo que se presenta será predecir si ocurrirá una incidencia en envíos futuros, además de la cantidad de días que requerirá dicho envío. 
También, se intentará señalar qué parámetros tienen mayor influencia en el resultado.\n\n"
##### Detalle de los datos
* **Puerto de origen**: Lugar desde el que se envían las mercancías.
* **Puerto de destino**: Lugar hasta el que llegan las mercancías.
* **Medio de transporte**: Puede ser por mar, aire y tierra (tren o carretera).
* **Categoría del producto**: Textil, automovilístico, electrónica...
* **Distancia recorrida** (Km).
* **Peso** (TM): Peso de los productos enviados.
* **Índice de precio del combustible**: Multiplicador del coste del combustible normalizado.
* **Puntuación de riesgo geopolítico**: Índice de riesgo basado en la estabilidad regional (0-10).
* **Condición climática**: Condiciones atmosféricas durante el tránsito (Despejado, lluvia, tormenta...).
* **Puntuación de confianza del transportista** (0.5-1).
* **Días de tránsito**: Tiempo que ha tomado el envío (Objetivo de la regresión).
* **Incidencia ocurrida**: Si ocurrió (1), o no (0), una incidencia durante el tránsito.