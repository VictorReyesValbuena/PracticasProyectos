# ✈️ Práctica Data Warehouse & SQL — Análisis de Vuelos

Práctica del módulo **Data Warehouse & SQL Advanced** de un curso de 24 horas. El objetivo es explorar, limpiar y analizar un dataset real de vuelos usando PostgreSQL, aplicando técnicas progresivas desde consultas básicas hasta funciones de ventana y CTEs encadenadas.

---

## 📁 Estructura del repositorio

```
├── Practica_SQL.sql        # Queries completas de los 11 enunciados
├── flights.csv             # Dataset principal de vuelos
├── airports.csv            # Información de aeropuertos (país, nombre, código IATA)
├── airlines_postgres.sql   # Tabla de aerolíneas
└── README.md
```

---

## 🗃️ Dataset

El dataset `flights.csv` contiene un historial de snapshots de vuelos. Cada vuelo puede aparecer varias veces, registrando la evolución de su estado a lo largo del tiempo.

| Columna | Descripción |
|---|---|
| `unique_identifier` | ID del vuelo (formato: `IB-100-20240124-MAD-JFK`) |
| `departure_airport` | Código IATA del aeropuerto de salida |
| `arrival_airport` | Código IATA del aeropuerto de llegada |
| `airline_code` | Código IATA de la aerolínea |
| `local_departure` | Fecha/hora de salida programada |
| `local_actual_departure` | Fecha/hora de salida real |
| `delay_mins` | Minutos de retraso (negativo = adelanto) |
| `arrival_status` | Estado del vuelo: `OT`, `DY`, `CNL`, `DIV` |
| `created_at` | Cuándo se creó el primer registro del vuelo |
| `updated_at` | Cuándo se actualizó este snapshot |

---

## 📋 Enunciados resueltos

### E1 — Exploración inicial
Análisis del tamaño del dataset: total de registros, vuelos distintos e identificación de duplicados con `COUNT`, `COUNT DISTINCT`, `GROUP BY` y `HAVING`.

### E2 — ¿Por qué hay duplicados?
Selección de vuelos concretos para analizar su evolución temporal. El dataset funciona como un historial de snapshots: cada cambio de estado genera una nueva fila.

### E3 — Calidad del dato
Validación de dos reglas de consistencia:
- `created_at` debe ser único por vuelo
- `updated_at` nunca puede ser anterior a `created_at`

### E4 — Último estado de cada vuelo
Creación de la tabla `flight_mod` con el snapshot más reciente de cada vuelo usando subconsulta con `MAX(updated_at)` e `IN`.

```sql
CREATE TABLE flight_mod AS
SELECT *
FROM flights
WHERE updated_at IN (
    SELECT MAX(updated_at)
    FROM flights
    GROUP BY unique_identifier
);
```

### E5 — Reconstrucción de fechas nulas
Uso de `UPDATE` con `CASE WHEN` para rellenar valores nulos en las columnas de fechas con valores alternativos. Creación de nuevas columnas con `ALTER TABLE`.

### E6 — Análisis de estados de vuelo
Distribución de vuelos por estado (`OT` = On Time, `DY` = Delayed, `CNL` = Cancelled, `DIV` = Diverted) usando `GROUP BY`.

### E7 — País de salida
`INNER JOIN` entre `flight_mod` y `airports` para enriquecer los datos con información geográfica y contar vuelos por país.

### E8 — Delay medio por país
Cálculo del retraso medio con `AVG` y `ROUND`, y distribución de estados por país de salida para identificar aeropuertos con problemas operativos.

### E9 — Estacionalidad
Clasificación de vuelos por estación del año usando `CASE WHEN` con `EXTRACT(MONTH FROM ...)`. Análisis del impacto de la época del año en el delay medio.

```sql
WITH estacion_vuelos AS (
    SELECT *,
        CASE
            WHEN EXTRACT(MONTH FROM created_at) IN (12,1,2) THEN 'Invierno'
            WHEN EXTRACT(MONTH FROM created_at) IN (3,4,5)  THEN 'Primavera'
            WHEN EXTRACT(MONTH FROM created_at) IN (6,7,8)  THEN 'Verano'
            WHEN EXTRACT(MONTH FROM created_at) IN (9,10,11) THEN 'Otoño'
        END AS estacion
    FROM flight_mod
    INNER JOIN airports ON departure_airport = airport_code
)
SELECT country, estacion, ROUND(AVG(delay_mins), 2), COUNT(*)
FROM estacion_vuelos
GROUP BY country, estacion;
```

### E10 — Frecuencia de actualización
Cálculo del intervalo entre actualizaciones consecutivas de cada vuelo usando `LAG() OVER (PARTITION BY ... ORDER BY ...)` y `EXTRACT(EPOCH FROM ...)`. Media de horas entre updates por aeropuerto de salida.

> 💡 El resultado del dataset muestra una frecuencia media de **~6 horas** entre actualizaciones, coherente con las observaciones manuales.

### E11 — Consistencia del unique_identifier
Verificación de que el `unique_identifier` contiene la aerolínea, fecha, aeropuerto de origen y destino correctos. Uso de `TO_CHAR`, `LIKE` con concatenación `||` y CTEs encadenadas para identificar inconsistencias por aerolínea.

---

## 🧠 Competencias SQL adquiridas

| Concepto | Descripción |
|---|---|
| `SELECT`, `WHERE`, `ORDER BY` | Filtrado y ordenación básica |
| `GROUP BY` + `HAVING` | Agrupación y filtrado de grupos |
| `COUNT`, `AVG`, `MAX`, `ROUND` | Funciones de agregación |
| `JOIN` / `INNER JOIN` | Cruce de tablas por clave común |
| `CASE WHEN ... THEN ... END` | Lógica condicional en SQL |
| `COALESCE` / `IS NULL` | Manejo de valores nulos |
| `CTE` (`WITH ... AS`) | Subconsultas reutilizables y legibles |
| `UPDATE` + `ALTER TABLE` | Modificación de datos y estructura |
| `ROW_NUMBER() OVER()` | Funciones de ventana: numeración por grupo |
| `LAG() OVER()` | Funciones de ventana: valor de fila anterior |
| `PARTITION BY` | Segmentación en funciones de ventana |
| `EXTRACT` + `TO_CHAR` | Manipulación de fechas |
| `LIKE` + `\|\|` | Búsqueda de patrones con concatenación dinámica |
| `CREATE TABLE AS` | Materialización de resultados |

---

## 🛠️ Tecnología

- **Base de datos:** PostgreSQL
- **Editor:** DBeaver / pgAdmin

---

## 📌 Notas

- El E4 se resolvió sin `ROW_NUMBER` ni `PARTITION BY`, usando una subconsulta con `MAX` e `IN` para practicar alternativas más básicas.
- Los E10 y E11 requirieron consultar sintaxis específica (`EXTRACT(EPOCH ...)`, `TO_CHAR`, `LIKE` con `||`), aunque la lógica fue desarrollada de forma independiente.
