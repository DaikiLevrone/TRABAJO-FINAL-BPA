import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas.api.types as ptypes
import datetime as dt

# --------------------------------------------------------
# Configuración de la página
# --------------------------------------------------------
st.set_page_config(
    page_title="Planificador de ocupación hotelera - Perú",
    layout="wide"
)

# Fecha y hora actual (desde la PC/servidor donde corre la app)
now = dt.datetime.now()
now_str = now.strftime("%d/%m/%Y %H:%M:%S")

# Cabecera con fecha/hora a la izquierda y título a la derecha
col_time, col_title = st.columns([2, 8])
with col_time:
    st.markdown(f"**📅 Fecha y hora actual:** {now_str}")
with col_title:
    st.title("🧳 Planificador de viaje por ocupación hotelera")

st.markdown("""
Selecciona un **departamento del Perú** y la aplicación te mostrará la
**ocupación hotelera esperada para los próximos 3 meses** (a partir del mes actual).

Además, indicará **en cuál de esos meses es más probable que haya mayor afluencia de visitantes**
y explicará el motivo (festividades y/o temporada turística).
""")

# --------------------------------------------------------
# Rutas de archivos (misma carpeta que app.py)
# --------------------------------------------------------
DATA_PATH = "Preparado.pickle"
MODEL_PATH = "modelo_rf_ocupabilidad.pkl"

# --------------------------------------------------------
# Carga de datos y modelo
# --------------------------------------------------------
@st.cache_data
def load_data(path: str):
    return pd.read_pickle(path)

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
except FileNotFoundError as e:
    st.error(f"No se encontró un archivo necesario: {e}")
    st.stop()

if "target_ocupabilidad" not in df.columns:
    st.error("El DataFrame no tiene la columna `target_ocupabilidad`. Revisa `Preparado.pickle`.")
    st.write("Columnas disponibles:", list(df.columns))
    st.stop()

# --------------------------------------------------------
# Preparar X, y y predicciones
# --------------------------------------------------------
X = df.drop(columns=["target_ocupabilidad"])
y = df["target_ocupabilidad"]

df["prediccion"] = model.predict(X)

# Métricas globales (por si las usas en el informe)
mae  = mean_absolute_error(y, df["prediccion"])
rmse = np.sqrt(mean_squared_error(y, df["prediccion"]))
r2   = r2_score(y, df["prediccion"])

# --------------------------------------------------------
# Reconstruir DEPARTAMENTO desde one-hot (DEPARTAMENTO_...)
# --------------------------------------------------------
dept_cols = [c for c in X.columns if c.startswith("DEPARTAMENTO_")]

if dept_cols:
    dept_matrix = X[dept_cols].values
    idx_max = dept_matrix.argmax(axis=1)
    dept_names = np.array([c.replace("DEPARTAMENTO_", "") for c in dept_cols])
    df["DEPARTAMENTO"] = dept_names[idx_max]
else:
    df["DEPARTAMENTO"] = "No disponible"

# Detectar posible columna de mes en tu preparado
possible_month_cols = [c for c in df.columns if c.upper() in ["MES", "MES_NUM", "MES_NOMBRE"]]
MES_COL = possible_month_cols[0] if possible_month_cols else None

# Lista de meses (nombres y números)
MESES_NOMBRES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Setiembre", "Octubre", "Noviembre", "Diciembre"
]
MAPA_MES_A_NUM = {nombre: i + 1 for i, nombre in enumerate(MESES_NOMBRES)}
MAPA_NUM_A_MES = {i + 1: nombre for i, nombre in enumerate(MESES_NOMBRES)}

# Mes actual del usuario (según la PC)
MES_NUM_ACTUAL = now.month
MES_NOMBRE_ACTUAL = MAPA_NUM_A_MES[MES_NUM_ACTUAL]

# Próximos 3 meses (por número y nombre)
MESES_SIG_NUM = [((MES_NUM_ACTUAL - 1 + i) % 12) + 1 for i in range(1, 4)]
MESES_SIG_NOMBRES = [MAPA_NUM_A_MES[m] for m in MESES_SIG_NUM]

# --------------------------------------------------------
# Clasificación de ocupación (baja / media / alta, por percentiles)
# --------------------------------------------------------
q1, q2 = df["prediccion"].quantile([0.33, 0.66])

def clasificar_ocupacion(valor):
    if valor <= q1:
        return "Baja ocupación (poco transitado)", "✅"
    elif valor <= q2:
        return "Ocupación media (actividad moderada)", "🟡"
    else:
        return "Alta ocupación (muy transitado)", "🔴"

# --------------------------------------------------------
# Catálogo de festividades por departamento
# --------------------------------------------------------
FESTIVIDADES = {
    "CUSCO": [
        {"Mes": "Junio", "Fecha": "24 de junio", "Evento": "Inti Raymi (Fiesta del Sol)"},
        {"Mes": "Junio", "Fecha": "Mediados de junio (jueves de Corpus Christi)", "Evento": "Corpus Christi en Cusco"},
        {"Mes": "Julio", "Fecha": "15 al 18 de julio", "Evento": "Fiesta de la Virgen del Carmen en Paucartambo"}
    ],
    "LIMA": [
        {"Mes": "Julio", "Fecha": "28 y 29 de julio", "Evento": "Fiestas Patrias y desfiles cívicos"},
        {"Mes": "Octubre", "Fecha": "18, 19 y 28 de octubre", "Evento": "Procesiones del Señor de los Milagros"}
    ],
    "AREQUIPA": [
        {"Mes": "Mayo", "Fecha": "1 de mayo", "Evento": "Peregrinación a la Virgen de Chapi"},
        {"Mes": "Agosto", "Fecha": "15 de agosto", "Evento": "Aniversario de Arequipa"}
    ],
    "PUNO": [
        {"Mes": "Febrero", "Fecha": "Primera quincena de febrero", "Evento": "Festividad de la Virgen de la Candelaria"}
    ],
    "LA LIBERTAD": [
        {"Mes": "Setiembre", "Fecha": "Tercera semana de setiembre", "Evento": "Festival Internacional de la Primavera (Trujillo)"}
    ],
    "PIURA": [
        {"Mes": "Enero", "Fecha": "Verano (enero-febrero)", "Evento": "Alta afluencia a playas como Máncora y Vichayito"},
        {"Mes": "Octubre", "Fecha": "13 al 19 de octubre", "Evento": "Fiesta del Señor Cautivo de Ayabaca"}
    ],
    "LORETO": [
        {"Mes": "Junio", "Fecha": "24 de junio", "Evento": "Fiesta de San Juan en la Amazonía peruana"}
    ],
    "JUNIN": [
        {"Mes": "Febrero", "Fecha": "Segunda quincena de febrero", "Evento": "Carnavales en el Valle del Mantaro"},
        {"Mes": "Marzo", "Fecha": "Inicios de marzo", "Evento": "Fiestas costumbristas y continuaciones de carnaval"}
    ]
}

def obtener_festividades_depto_mes(depto, mes_nombre):
    fest_depto = FESTIVIDADES.get(depto.upper(), [])
    fest_mes = [f for f in fest_depto if f["Mes"].lower() == mes_nombre.lower()]
    return fest_depto, fest_mes

def factores_temporada(mes_nombre):
    mes_lower = mes_nombre.lower()
    razones = []
    score = 0.0

    if mes_lower in ["enero", "febrero", "marzo"]:
        score += 1.5
        razones.append("es plena temporada de verano y vacaciones escolares")
    elif mes_lower in ["julio", "agosto"]:
        score += 1.0
        razones.append("coincide con las vacaciones de medio año y Fiestas Patrias")
    elif mes_lower in ["noviembre", "diciembre"]:
        score += 1.0
        razones.append("se acerca fin de año, cuando suelen aumentar los viajes familiares y de ocio")

    return score, razones

def puntuar_mes(depto, mes_num):
    mes_nombre = MAPA_NUM_A_MES[mes_num]
    _, fest_mes = obtener_festividades_depto_mes(depto, mes_nombre)

    score = 0.0
    razones = []

    if fest_mes:
        score += 2.0
        razones.append(
            "se celebran festividades como " +
            ", ".join(f["Evento"] for f in fest_mes)
        )

    s_temp, razones_temp = factores_temporada(mes_nombre)
    score += s_temp
    razones.extend(razones_temp)

    return score, mes_nombre, fest_mes, razones

def construir_explicacion_global(depto, nivel_texto, porcentaje, meses_nombres):
    lista_meses = ", ".join(meses_nombres)
    base = (
        f"Para **{depto}**, considerando los próximos tres meses ({lista_meses}), "
        f"se estima un nivel de ocupación promedio que se sitúa aproximadamente por encima del "
        f"**{porcentaje:.1f}%** de los niveles históricos observados para el destino, "
        f"lo que se clasifica globalmente como **{nivel_texto.lower()}**."
    )
    return base

# --------------------------------------------------------
# UI: selección de departamento
# --------------------------------------------------------
st.subheader("✈️ Elige tu destino")

if "DEPARTAMENTO" not in df.columns or df["DEPARTAMENTO"].nunique() <= 1:
    st.error("No se pudo reconstruir correctamente la columna DEPARTAMENTO.")
    st.stop()

departamentos = sorted(df["DEPARTAMENTO"].unique())
depto_sel = st.selectbox("¿A qué departamento quieres ir?", departamentos)

st.markdown("---")

# --------------------------------------------------------
# Filtrar datos para el departamento y meses futuros
# --------------------------------------------------------
df_depto = df[df["DEPARTAMENTO"] == depto_sel].copy()

if MES_COL is not None:
    serie_mes = df[MES_COL]
    if ptypes.is_numeric_dtype(serie_mes):
        subset_future = df[(df["DEPARTAMENTO"] == depto_sel) &
                           (serie_mes.isin(MESES_SIG_NUM))].copy()
    else:
        subset_future = df[(df["DEPARTAMENTO"] == depto_sel) &
                           (serie_mes.astype(str).str.title().isin(MESES_SIG_NOMBRES))].copy()
else:
    subset_future = df_depto.copy()

# Si no hay datos específicos para esos meses, usamos todo el histórico del depto
if subset_future.empty:
    subset_future = df_depto.copy()

ocup_promedio = subset_future["prediccion"].mean()

# --------------------------------------------------------
# Porcentaje como percentil histórico
# --------------------------------------------------------
# Ejemplo: 85% significa que esta ocupación está por encima del 85% de los valores históricos
percentile = (df["prediccion"] <= ocup_promedio).mean() * 100
porcentaje = round(percentile, 1)

nivel_texto_global, icono_global = clasificar_ocupacion(ocup_promedio)

# --------------------------------------------------------
# Determinar qué mes de los 3 es más probable que esté más transitado
# (según festividades y temporada)
# --------------------------------------------------------
scores = []
for mes_num in MESES_SIG_NUM:
    score, mes_nombre, fest_mes, razones = puntuar_mes(depto_sel, mes_num)
    scores.append((score, mes_num, mes_nombre, fest_mes, razones))

score_top, mes_top_num, mes_top_nombre, fest_top, razones_top = max(scores, key=lambda x: x[0])

if score_top == 0:
    razon_top = (
        "no se identifican festividades importantes ni temporadas altas muy marcadas, "
        "pero el destino mantiene un flujo turístico relativamente estable."
    )
else:
    razon_top = " y ".join(razones_top)

lista_meses_texto = ", ".join(MESES_SIG_NOMBRES)
explicacion_global = construir_explicacion_global(
    depto_sel, nivel_texto_global, porcentaje, MESES_SIG_NOMBRES
)
explicacion_mes_top = (
    f"Entre estos meses (**{lista_meses_texto}**), "
    f"el mes con mayor probabilidad de estar más transitado es **{mes_top_nombre}**, "
    f"porque {razon_top}."
)

# --------------------------------------------------------
# Mostrar predicción global para los próximos 3 meses
# --------------------------------------------------------
st.subheader(f"📅 Pronóstico de ocupación para los próximos 3 meses en {depto_sel}")
st.caption(f"(A partir del mes actual: **{MES_NOMBRE_ACTUAL}**)")

col_m1, col_m2 = st.columns(2)
with col_m1:
    st.markdown(f"### {icono_global} {nivel_texto_global}")
    st.metric("Ocupación esperada promedio (escala del modelo)", f"{ocup_promedio:,.2f}")
with col_m2:
    st.metric("Nivel promedio vs histórico (percentil)", f"{porcentaje:.1f}%")

st.write(explicacion_global)
st.write(explicacion_mes_top)

# --------------------------------------------------------
# Tabla de festividades del departamento seleccionado
# --------------------------------------------------------
st.markdown("---")
st.subheader(f"🎉 Festividades en {depto_sel}")

fest_depto = FESTIVIDADES.get(depto_sel.upper(), [])

if fest_depto:
    df_fest = pd.DataFrame(fest_depto)
    df_fest = df_fest.rename(columns={"Fecha": "Fecha referencial"})
    st.write("Eventos festivos referenciales de este departamento:")
    st.table(df_fest[["Mes", "Fecha referencial", "Evento"]])
else:
    st.info(
        "Por ahora no se tienen registradas festividades específicas para este departamento en la app. "
        "Puedes complementar esta sección manualmente en el informe."
    )
