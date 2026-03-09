
ENUNCIADO 1.
--Explora el fichero flights y analiza:



select *
	from flights
	limit 5;
	--1. Cuántos registros hay en total
	select 
		count(*) 
	from flights;
	
	--2. Cuántos vuelos distintos hay
	select 
		count(distinct unique_identifier) 
	from flights;
	--3. Cuántos vuelos tienen más de un registro
	select
		unique_identifier,
		count(*) as total_registros
	from flights
	group by unique_identifier
	having count(*) > 1;


ENUNCIADO 2.
	Por qué hay registro duplicados para un mismo vuelo. Para ello, selecciona varios vuelos y analiza la evolución temporal de cada vuelo.



	--1. Qué información cambia de un registro a otro
	select 
		arrival_status,
		delay_mins,
		unique_identifier,
		created_at,
		updated_at
	from flights
	where unique_identifier = 'AA-101-20240219-JFK-MAD';


ENUNCIADO 3.
	Evalúa la calidad del dato. La calidad del dato nos indica si la información es consistente, completa, coherente y representa una realidad verosímil. Para ello debemos establecer unos criterios:


	
	--1. La información de created_at debe ser única para cada vuelo aunque tenga más de un registro.
	select
		unique_identifier,
		count(distinct created_at) as created_at_unique
	from flights
	group by unique_identifier
	having count(distinct created_at) > 1;
	--2. La información de updated_at deber ser igual o más que la información de created_at, lo que nos indica coherencia y consistencia
	select
		unique_identifier,
		created_at,
		updated_at
	from flights
	where updated_at < created_at;
	

ENUNCIADO 4.
	El último estado de cada vuelo. Cada vuelo puede aparecer varias veces en el dataset, para avanzar con nuestro análisis necesitamos quedarnos solo con el último registro de cada vuelo.
	Puedes crear una tabla o vista resultante de esta query en tu base de datos local, la utilizaremos en los siguientes enunciados. Si prefieres no guardar la última información, tendrás que hacer uso de esa query como una CTE en los enunciados siguientes.


	
	create table  flight_mod as
	select *
	from flights
	where updated_at in(
		select 
			max(updated_at) as ult_modificacion
		from flights
		group by unique_identifier
	)


ENUNCIADO 5.
	Considerando que los campos local_departure y local_actual_departure son necesarios para el análisis, valida y reconstruye estos valores siguiendo estas reglas:

	--1. Si local_departure es nulo, utiliza created_at.
	--2. Si local_actual_departure es nulo, utiliza local_departure. Si este también es nulo, utiliza created_at.
	--Extra: Realiza las validaciones para los campos local_arrival y local_actual_arrival.



	update flight_mod
	set 
		local_actual_departure = case
        when local_actual_departure is not null then local_actual_departure
       	else created_at
    end,
    	local_departure = case
        when local_departure is not null then local_departure
       	else created_at
    end,
    	local_actual_arrival = case
        when local_actual_arrival is not null then local_actual_arrival
       	else created_at
    end,
    	local_arrival = case
        when local_arrival is not null then local_arrival
       	else created_at
    end;

--Crea dos nuevos campos:
	--● effective_local_departure
	--● effective_local_actual_departure

	alter table flight_mod
	add column effective_local_departure timestamp,
	add column effective_local_actual_departure timestamp;


ENUNCIADO 6.
	Análisis del estado del vuelo. Haciendo uso del resultado del enunciado 4, analiza los estados de los vuelos.
	--1. Qué estados de vuelo existen
	--2. Cuántos vuelos hay por cada estado
--¿Podrías decir qué significa las siglas de cada estado?



	select *
		from flight_mod
		limit 5;
	select
		arrival_status,
		count(*)
	from flight_mod
	group by arrival_status;
	--Puede que OT sea On time / DY sea Delayed y CX
	

ENUNCIADO 7.
	País de salida de cada vuelo. Tienes disponible un csv. con información de aeropuertos airports.csv. Haciendo uso del resultado del enunciado 4, analiza los aeropuertos de salida.
	--1. De qué país despegan los vuelos

	select
		fmod.unique_identifier,
		air.airport_name
	from flight_mod as fmod 
	inner join airports as air 
	on fmod.departure_airport = air.airport_code;
	
	--2. Cuántos vuelos despegan por país

	select
		air.country,
		count(*) as despegues_por_pais
	from flight_mod as fmod 
	inner join airports as air 
	on fmod.departure_airport = air.airport_code
	group by air.country;


ENUNCIADO 8.
	Delay medio y estado de vuelo por país de salida. Haciendo uso del resultado del enunciado 4, analiza el estado y el delay/retraso medio con el objetivo de identificar si existen países que pueden presentar problemas operativos en los aeropuertos de salida.
	--1. Cuál es el delay medio por país


	select
		air.country,
		round(avg(delay_mins), 2) as delay_medio
	from flight_mod as fmod 
	inner join airports as air 
	on fmod.departure_airport = air.airport_code
	where delay_mins is not null
	group by air.country;
	--2. Cuál es la distribución de estados de vuelos por país.


	select
		air.country,
		arrival_status,
		count(*) as num_vuelos
	from flight_mod as fmod 
	inner join airports as air 
	on fmod.departure_airport = air.airport_code 
	group by air.country, arrival_status;
	
--Extra:
--Representa gráficamente la distribución de estados por país. Puedes dibujar un gráfico de barras o representarlo como creas que mejor se visualiza.


ENUNCIADO 9.
--El estado de vuelo por país y por época del año. Dado que no en todas las épocas del año las condiciones climatólogicas son iguales, analiza si la estaciones del año impactan en el delay medio por país. Considera la siguiente clasificación de meses del año por época:
	--● Invierno: diciembre, enero, febrero
	--● Primavera: marzo, abril, mayo
	--● Verano: junio, julio, agosto
	--● Otoño: septiembre, octubre, noviembre



	with estacion_vuelos as(
		select*,
			case
				when extract( month from created_at) in (12,1,2) then 'Invierno'
				when extract( month from created_at) in (3,4,5) then 'Primavera'
				when extract( month from created_at) in (6,7,8) then 'Verano'
				when extract( month from created_at) in (9,10,11) then 'Otoño'
			end as estacion
		from flight_mod as fmod 
		inner join airports as air 
		on fmod.departure_airport = air.airport_code 
	)
	select
		country,
		estacion,
		round(avg(delay_mins), 2) as delay_medio,
		count(*) as num_vuelos
	from estacion_vuelos
	group by country, estacion
	order by country desc;


ENUNCIADO 10.
	Frecuencia de actualización de los vuelos. Volviendo al análisis de la calidad del dataset, explora con qué frecuencia se registran actualizaciones de cada vuelo y calcula la frecuencia media de actualización por aeropuerto de salida.



--PARA ESTE ENUNCIADO, LA CTE LA TUVE QUE MIRAR UN POCO POR IA... la lógica la tenía pero me atasqué un poco con el extract
with intervalos as (
	select
		departure_airport,
        extract(epoch from (updated_at - lag(updated_at) OVER ( --  extract(epoch from para obtener los SEGUNDOS de un intervalo
                partition by unique_identifier	-- Construyo el intervalo como la diferencia entre update_at y el valor anterior agrupado por 
                order by updated_at asc			-- los unique_identifier y ordenado de manera ascendente para asegurar que la resta sea correcta.
            )
        )) / 3600 as horas_entre_updates -- Divido el resultado entre 3600 para sacar las horas.
    from flights
)
select
    departure_airport,
    round(avg(horas_entre_updates), 2) as media_horas_entre_updates -- Hago la media
from intervalos
where horas_entre_updates is not null
group by departure_airport -- Agrupo por departure_airport y horas entre updates
order by media_horas_entre_updates;

-- EL RESULTADO QUE ME DA SON 6 HORAS Y ME CUADRA CON LAS OBSERVACIONES "MANUALES" EN EL DATASET ORIGINAL.


ENUNCIADO 11.
--Consistencia del dato. El campo unique_identifier identifica el vuelo y se construye con: aerolínea, número de vuelo, fecha y aeropuertos. Para cada vuelo (último snapshot), comprueba si la información del unique_identifier es consistente con las columnas del dataset.



-- EN EL 11 TAMBIÉN HE TIRADO UN POCO DE IA, HE INTENTADO HACERLO YO Y, MÁS O MENOS, LA LÓGICA LA TENÍA (O LA ESTRUCTURA), PERO ME ESTANQUÉ EN DOS COSAS:
	--1. Crea un flag is_consistent.
	

with vuelos_fechanueva as (
    select
        unique_identifier,
        airline_code,
        departure_airport,
        arrival_airport,
        to_char(local_departure::timestamp, 'YYYYMMDD') as new_local_departure
        --ESTO LO TERMINÉ CONSULTANDO CON LA IA PERO SI QUE LA "LÓGICA" DE LA QUERY LA TENÍA. CONSULTÉ MÁS BIEN LA SINTAXIS.
    from flight_mod
)
select
    unique_identifier,
    airline_code,
    departure_airport,
    arrival_airport,
    new_local_departure,
    case
        when unique_identifier like '%' || airline_code       || '%'
         and unique_identifier like '%' || departure_airport  || '%'
         and unique_identifier like '%' || arrival_airport    || '%'
         and unique_identifier like '%' || new_local_departure || '%'
         --ESTO TAMBIÉN LO CONSULTÉ. EL UTILIZAR CONCATENAR PARA GENERAR UNA "CADENA" CON EL VALOR DE UNA COLUMNA 
        then true
        else false
    end as dato_consistente
from vuelos_fechanueva;
	
	--2. Calcula cuántos vuelos no son consistentes.
	

with vuelos_fechanueva as (
	select
   		unique_identifier,
        airline_code,
        departure_airport,
        arrival_airport,
        to_char(local_departure::timestamp, 'YYYYMMDD') as new_local_departure
    from flight_mod
), vuelos_consistentes as(
	select
    	unique_identifier,
    	airline_code,
    	departure_airport,
    	arrival_airport,
    	new_local_departure,
    	case
        	when unique_identifier like '%' || airline_code       || '%'
         	and unique_identifier like '%' || departure_airport  || '%'
         	and unique_identifier like '%' || arrival_airport    || '%'
         	and unique_identifier like '%' || new_local_departure || '%'
        	then true
        	else false
    	end as dato_consistente
	from vuelos_fechanueva
)
select
	count(*) as vuelos_incosistentes
from vuelos_consistentes
where dato_consistente = FALSE;

	--3. Usando la tabla airlines, muestra el nombre de la aerolínea y cuántos vuelos no consistentes tiene.
	-- ESTE PUNTO A PESAR DE SER EL MÁS COMPLEJO, FUE EL QUE MEJOR SE ME DIÓ, FUI COMBINANDO CTEs QUITANDO Y PONIENDO Y FINALMENTE CONSEGUÍ LLEGAR AL OBJETIVO.

	

with vuelos_fechanueva as (
	select
   		unique_identifier,
        airline_code,
        departure_airport,
        arrival_airport,
        to_char(local_departure::timestamp, 'YYYYMMDD') as new_local_departure
    from flight_mod
), vuelos_consistentes as(
	select
    	unique_identifier,
    	airline_code,
    	departure_airport,
    	arrival_airport,
    	new_local_departure,
    	case
        	when unique_identifier like '%' || airline_code       || '%'
         	and unique_identifier like '%' || departure_airport  || '%'
         	and unique_identifier like '%' || arrival_airport    || '%'
         	and unique_identifier like '%' || new_local_departure || '%'
        	then TRUE
        	else FALSE
    	end as dato_consistente
	from vuelos_fechanueva
)
select
	air.name,
	count(*) as vuelos_incosistentes
from vuelos_consistentes as vue
inner join airlines as air 
on vue.airline_code = air.airline_code
where vue.dato_consistente = FALSE
group by air.name;