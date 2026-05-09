import pandas as pd
import os
import google.generativeai as genai

# =================================================================
# CONFIGURACIÓN DE SEGURIDAD
# =================================================================
API_KEY = ""
genai.configure(api_key=API_KEY)

# =================================================================
# LIMPIEZA DE LOS DATOS DEL CSV
# =================================================================

def limpiar_valor(valor):
    if pd.isna(valor) or str(valor).strip() in ['', ';']:
        return 0.0
    try:
        s = str(valor).replace(';', '').strip()
        valor_limpio = s.replace('.', '').replace(',', '.')
        return float(valor_limpio)
    except ValueError:
        return 0.0

def obtener_datos(df, patron):
    col_nombre = 'Cuenta de Pérdidas y Ganancias'
    mask = df[col_nombre].str.contains(patron, case=False, na=False)
    sub = df[mask]
    return {
        '2025': sum(limpiar_valor(v) for v in sub['AÑO 2025']),
        '2024': sum(limpiar_valor(v) for v in sub['AÑO 2024'])
    }

# =================================================================
# ESTA ES LA SOLUCIÓN QUE ME DIO GEMS PARA EL PROBLEMA CON LA API KEY
# =================================================================

def buscar_modelo_valido():
    """Busca en tu cuenta qué modelos están realmente disponibles para evitar el 404."""
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                # Priorizamos la familia flash por rapidez y gratuidad
                if 'flash' in m.name:
                    return m.name
        # Si no hay flash, devolvemos el primero que soporte generación
        return genai.list_models().name
    except Exception as e:
        print(f"Error al listar modelos: {e}")
        return None
    
# =================================================================
# DESARROLLO DE LA SOLUCIÓN AL PROBLEMA 
# =================================================================

# Una vez que gems me generó el código, intenté verlo junto con los apuntes del curso
# para intentar vere similitudes y aprender la estructura general.

# ESTO SERÍA COMO LA PARTE DE CHAT_SIMPLE
def generar_informe_ia(datos, nombre_modelo, temperatura=0.4):
    """Redacta el informe usando el modelo detectado automáticamente."""
    try:
        #Le he metido una temperatura de 0.4 para que esté a medio camino entre creatividad y precisión.
        generation_config = genai.GenerationConfig(temperature=temperatura)
        model = genai.GenerativeModel(nombre_modelo, generation_config=generation_config)

# En esta parte, retoqué un poco el prompt para que tocara alguna partida más.
        prompt = f"""
        Actúa como un Asesor Fiscal contable experto. Analiza estos datos contables 2025 vs 2024:
        {datos}

        Redacta el informe con EXACTAMENTE esta estructura, abordando cada punto aunque no haya incidencias:

        [RESULTADO E IMPUESTO]
        Compara ingresos y gastos 2025 vs 2024. Explica por qué varía el impuesto en relación al margen obtenido.
        Si la facturación en menor a un millón de euros, señala que el impuesto es reducido (al 23%) y que si es mayor, el impuesto es del 25%.

        [ALERTA FALSOS AUTÓNOMOS]
        Si la cuenta 6230 es alta y los salarios son cero, alerta del riesgo. Si no aplica, indícalo brevemente.

        [ARRENDAMIENTOS]
        Verifica si son estables. Alerta si hay variación relevante.

        [SUMINISTROS]
        Verifica si son estables. Alerta si hay variación relevante. Puede denotar una imputación de gastos por suministros que no corresponde a la actividad de la empresa, más bien de los socios.

        [AMORTIZACIONES]
        Compara las amortizaciones. Si aumentan, señala que probablemente se adquirió nuevo inmovilizado.

        [GASTOS FINANCIEROS]
        Si aumentan significativamente, indica que puede ser señal de mayor financiación ajena y riesgo.

        [CONCLUSIÓN DEL CIERRE CONTABLE]
        Resume en una frase el motivo principal del aumento o disminución del impuesto.

        [FINAL DEL MAIL]
        Terminar diciendo 
        "Para cualquier consulta, quedamos a su total disposición para concertar una cita presencial o resolver las dudas mediante una llamada telefónica o videollamada.
        Un saludo y gracias."

        El Tono de todo el informe debe ser profesional y directo. No omitas ninguna sección.
        """

        # Aquí según los datos que le metamos (que será lo que está definido en el JSON resumen) llamará a la API
        # y generará el texto correspondiente.
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"\n[ERROR DE GENERACIÓN]: {str(e)}\n"

def main():
    print("--- SISTEMA DE CIERRE FISCAL INTELIGENTE v7.1 ---")
    # AQUÍ ES DONDE ME PIDE EL ARCHIVO EN CUESTIÓN QUE QUIERO ANALIZAR.
    archivo = input("Nombre del CSV (ej. datos.cliente.csv): ").strip()

    if not os.path.exists(archivo):
        print(f"Error: {archivo} no encontrado.")
        return

    # PASO CRÍTICO: Detectar qué modelo quiere tu API KEY
    print("Buscando modelos disponibles para tu clave...")
    modelo_activo = buscar_modelo_valido()
    
    if not modelo_activo:
        print("No se encontraron modelos disponibles. Revisa tu API KEY en Google AI Studio.")
        return
    
    print(f"Modelo detectado y listo: {modelo_activo}")

    try:
        df = pd.read_csv(archivo, sep=';', encoding='utf-8')
        df.columns = [c.strip() for c in df.columns]

        resumen = {
            "Ventas": obtener_datos(df, "1. Importe neto"),
            "Gastos": obtener_datos(df, "7. Otros gastos"),
            "Profesionales (6230)": obtener_datos(df, "6230"),
            "Salarios": obtener_datos(df, "640|642"),
            "Amortizaciones": obtener_datos(df, "8. Amortización de inmovilizado"),
            "Arrendamientos": obtener_datos(df, "621"),
            "Suministros": obtener_datos(df, "628"),
            "Gastos financieros": obtener_datos(df, "15. Gastos financieros"),
            "Impuesto": obtener_datos(df, "Impuesto sobre sociedades"),
            "Resultado": obtener_datos(df, r"D\) RESULTADO DEL")
        }

        print("Redactando informe inteligente...")
        # En redacción, ya ejecutamos nuestra salida (como si fuera el chat_simple), metiendo los datos del diccionario y el modelo activo.
        redaccion = generar_informe_ia(resumen, modelo_activo)

        with open("Informe_Final_Gemini.txt", "w", encoding="utf-8") as f:
            f.write(f"Estimado cliente,\n\n{redaccion}\n\nUn saludo.")
        
        print("\n[ÉXITO] Informe generado en 'Informe_Final_Gemini.txt'")

    except Exception as e:
        print(f"Error procesando el CSV: {e}")

if __name__ == "__main__":
    main()