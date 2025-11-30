import streamlit as st
import pandas as pd
import numpy as np
import pickle  # Reemplaza a joblib (es nativa de Python, 0 peso extra)
import datetime as dt
# Eliminamos sklearn.metrics y pandas.api.types explícito para ahorrar memoria

# --------------------------------------------------------
# Configuración de la página
# --------------------------------------------------------
st.set_page_config(
    page_title="Planificador de ocupación hotelera - Perú",
    layout="wide"
)

# Fecha y hora actual
now = dt.datetime.now()
now_str = now.strftime("%d/%m/%Y %H:%M:%S")

col_time, col_title = st.columns([2, 8])
with col_time:
    st.markdown(f"**📅 Fecha y hora actual:** {now_str}")
with col_title:
    st.title("🧳 Planificador de viaje por ocupación hotelera")

st.markdown("""
Selecciona un **departamento del Perú** y la aplicación te mostrará la
**ocupación hotelera esperada para los próximos 3 meses**.
""")

# --------------------------------------------------------
# Rutas de archivos
# --------------------------------------------------------
DATA_PATH = "Preparado.pickle"
MODEL_PATH = "modelo_rf_ocupabilidad.pkl"

# --------------------------------------------------------
# Carga de datos y modelo (Usando Pickle en lugar de Joblib)
# --------------------------------------------------------
@st.cache_data
def load_data(path: str):
    # Pandas usa pickle internamente para .read_pickle, es eficiente
    return pd.read_pickle(path)

@st.cache_resource
def load_model(path: str):
    # Usamos la librería nativa pickle para no depender de joblib
    with open(path, 'rb') as f:
        return pickle.load(f)

try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
except FileNotFoundError as e:
    st.error(f"No se encontró un archivo necesario: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo o datos: {e}")
    st.stop()

if "target_ocupabilidad" not in df.columns:
    st.error("El DataFrame no tiene la columna `target_ocupabilidad`.")
    st.stop()

# --------------------------------------------------------
# Preparar X, y y predicciones
# --------------------------------------------------------
X = df.drop(columns=["target_ocupabilidad"])
y = df["target_ocupabilidad"]

# Generar predicciones
try:
    df["prediccion"] = model.predict(X)
except Exception as e:
    st.error("Error al realizar la predicción. Asegúrate de que las librerías del modelo estén instaladas (scikit-learn).")
    st.stop()

# --------------------------------------------------------
# Cálculo manual de métricas (Sin importar sklearn)
# --------------------------------------------------------
# MAE: Promedio de la diferencia absoluta
mae = np.mean(np.abs(y - df["prediccion"]))

# RMSE: Raíz cuadrada del promedio de los errores al cuadrado
rmse = np.sqrt(np.mean((y - df["prediccion"])**2))

# R2 Score: 1 - (Suma errores cuadrados / Suma total cuadrados)
ss_res = np.sum((y - df["prediccion"])**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - (ss_res / ss_tot)

# --------------------------------------------------------
# Reconstruir DEPARTAMENTO
# --------------------------------------------------------
dept_cols = [c for c in X.columns if c.startswith("DEPARTAMENTO_")]

if dept_cols:
    dept_matrix = X[dept_cols].values
    idx_max = dept_matrix.argmax(axis=1)
    dept_names = np.array([c.replace("DEPARTAMENTO_", "") for c in dept_cols])
    df["DEPARTAMENTO"] = dept_names[idx_max]
else:
    df["DEPARTAMENTO"] = "No disponible"

# Detectar columna de mes
possible_month_cols = [c for c in df.columns if c.upper() in ["MES", "MES_NUM", "MES_NOMBRE"]]
MES_COL = possible_month_cols[0] if possible_month_cols else None

MESES_NOMBRES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Setiembre", "Octubre", "Noviembre", "Diciembre"
]
MAPA_NUM_A_MES = {i + 1: nombre for i, nombre in enumerate(MESES_NOMBRES)}

# Mes actual y siguientes
MES_NUM_ACTUAL = now.month
MES_NOMBRE_ACTUAL = MAPA_NUM_A_MES[MES_NUM_ACTUAL]
MESES_SIG_NUM = [((MES_NUM_ACTUAL - 1 + i) % 12) + 1 for i in range(1, 4)]
MESES_SIG_NOMBRES = [MAPA_NUM_A_MES[m] for m in MESES_SIG_NUM]

# --------------------------------------------------------
# Lógica de Negocio y Textos
# --------------------------------------------------------
q1, q2 = df["prediccion"].quantile([0.33, 0.66])

def clasificar_ocupacion(valor):
    if valor <= q1:
        return "Baja ocupación (poco transitado)", "✅"
    elif valor <= q2:
        return "Ocupación media (actividad moderada)", "🟡"
    else:
        return "Alta ocupación (muy transitado)", "🔴"

# (Aquí mantengo tu diccionario de festividades intacto para ahorrar espacio visual, 
# pero el código funcionará igual con el que ya tenías)
FESTIVIDADES = {
    "CUSCO": [
        {"Mes": "Junio", "Fecha": "24 de junio", "Evento": "Inti Raymi"},
        {"Mes": "Julio", "Fecha": "15-18 julio", "Evento": "Virgen del Carmen"}
    ],
    "LIMA": [
        {"Mes": "Julio", "Fecha": "28-29 julio", "Evento": "Fiestas Patrias"},
        {"Mes": "Octubre", "Fecha": "Octubre", "Evento": "Señor de los Milagros"}
    ],
    "AREQUIPA": [{"Mes": "Agosto", "Fecha": "15 agosto", "Evento": "Aniversario Arequipa"}],
    "PUNO": [{"Mes": "Febrero", "Fecha": "Febrero", "Evento": "Virgen de la Candelaria"}],
    "LA LIBERTAD": [{"Mes": "Setiembre", "Fecha": "Setiembre", "Evento": "Festival de la Primavera"}],
    "PIURA": [{"Mes": "Enero", "Fecha": "Verano", "Evento": "Temporada de Playa"}],
    "LORETO": [{"Mes": "Junio", "Fecha": "24 junio", "Evento": "Fiesta de San Juan"}],
    "JUNIN": [{"Mes": "Febrero", "Fecha": "Febrero", "Evento": "Carnavales"}]
}

def obtener_festividades_depto_mes(depto, mes_nombre):
    fest_depto = FESTIVIDADES.get(depto.upper(), [])
    # Comparación simple de strings
    fest_mes = [f for f in fest_depto if f["Mes"].lower() == mes_nombre.lower()]
    return fest_depto, fest_mes

def puntuar_mes(depto, mes_num):
    mes_nombre = MAPA_NUM_A_MES[mes_num]
    _, fest_mes = obtener_festividades_depto_mes(depto, mes_nombre)
    score = 0.0
    razones = []
    
    # Puntaje por festividad
    if fest_mes:
        score += 2.0
        razones.append("hay festividades importantes")
    
    # Puntaje por temporada alta general
    mes_lower = mes_nombre.lower()
    if mes_lower in ["enero", "febrero", "julio", "agosto"]:
        score += 1.0
        razones.append("es temporada alta de turismo")
    
    return score, mes_nombre, fest_mes, razones

# --------------------------------------------------------
# UI: Selección
# --------------------------------------------------------
st.subheader("✈️ Elige tu destino")

if "DEPARTAMENTO" not in df.columns:
    st.error("Error procesando columnas de departamento.")
    st.stop()

departamentos = sorted(df["DEPARTAMENTO"].unique())
depto_sel = st.selectbox("¿A qué departamento quieres ir?", departamentos)

st.markdown("---")

# Filtrado de datos
df_depto = df[df["DEPARTAMENTO"] == depto_sel].copy()

# Lógica de filtrado futuro simplificada
if MES_COL:
    # Si la columna es numérica
    if pd.api.types.is_numeric_dtype(df[MES_COL]):
        subset_future = df[(df["DEPARTAMENTO"] == depto_sel) & (df[MES_COL].isin(MESES_SIG_NUM))]
    else:
        # Si es texto
        subset_future = df[(df["DEPARTAMENTO"] == depto_sel) & (df[MES_COL].astype(str).str.title().isin(MESES_SIG_NOMBRES))]
else:
    subset_future = df_depto

if subset_future.empty:
    subset_future = df_depto

ocup_promedio = subset_future["prediccion"].mean()
percentile = (df["prediccion"] <= ocup_promedio).mean() * 100
nivel_texto_global, icono_global = clasificar_ocupacion(ocup_promedio)

# Calcular mejor mes
scores = []
for mes_num in MESES_SIG_NUM:
    score, mes_nom, _, razones = puntuar_mes(depto_sel, mes_num)
    scores.append((score, mes_nom, razones))

# Obtener el mes con max score
best_mes = max(scores, key=lambda x: x[0])
mes_top_nombre = best_mes[1]
razones_texto = " y ".join(best_mes[2]) if best_mes[2] else "es un mes con flujo turístico regular"

# --------------------------------------------------------
# Mostrar Resultados
# --------------------------------------------------------
st.subheader(f"📅 Pronóstico: {depto_sel}")
st.caption(f"Análisis para: {', '.join(MESES_SIG_NOMBRES)}")

c1, c2 = st.columns(2)
c1.metric("Ocupación Estimada", f"{ocup_promedio:,.2f}", delta=icono_global)
c2.metric("Nivel Comparativo", f"{percentile:.1f}%", help="Porcentaje respecto al histórico nacional")

st.info(f"**Recomendación:** Se espera una situación de **{nivel_texto_global.lower()}**.")
st.write(f"👉 De los próximos 3 meses, el de mayor movimiento podría ser **{mes_top_nombre}**, dado que {razones_texto}.")

st.markdown("---")
st.subheader("🎉 Festividades Referenciales")
_, eventos = obtener_festividades_depto_mes(depto_sel, "") # Traer todas del depto
if not eventos:
    # Intentar traer todo el depto del diccionario global si la funcion fallo
    eventos = FESTIVIDADES.get(depto_sel.upper(), [])

if eventos:
    st.table(pd.DataFrame(eventos)[["Mes", "Evento"]])
else:
    st.write("No hay festividades registradas en la base de datos para este destino.")
