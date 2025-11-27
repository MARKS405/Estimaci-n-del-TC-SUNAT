import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from datetime import date, datetime, timedelta

# ------------------------------------------------------------
# Parámetros globales
# ------------------------------------------------------------

# Serie diaria: TC Sistema bancario SBS (S/ por US$) - Venta (BCRP)
SERIE_TC_SBS_VENTA = "PD04640PD"

# Fecha mínima que quieres usar como histórico
FECHA_INICIO_SERIE = date(2021, 1, 1)

# Tabla simulada de feriados en Perú (ejemplos).
# >>> IMPORTANTE: reemplaza/actualiza esta lista manualmente.
FERIADOS_PE = {
    date(2021, 4, 1),   # Jueves Santo
    date(2021, 4, 2),   # Viernes Santo
    date(2021, 5, 1),   # Día de trabajo
    date(2021, 6, 29),   # Día de San Pedro y San Pablo
    date(2021, 7, 28),   # Fiestas Patrias
    date(2021, 7, 29),   # Fiestas Patrias
    date(2021, 8, 30),   # Santa Rosa de Lima
    date(2021, 10, 8),   # Combate de Angamos
    date(2021, 11, 1),   # Día de Todos los Santos
    date(2021, 12, 8),   # Día de la Inmaculada Concepción
    date(2021, 12, 25),   # Navidad    
    date(2022, 1, 1),   # Año Nuevo
    date(2022, 4, 14),   # Jueves Santo
    date(2022, 4, 15),   # Viernes Santo
    date(2022, 4, 17),   # Domingo de Resurrección
    date(2022, 5, 1),   # Día Internacional de los Trabajadores (Día del Trabajo)
    date(2022, 6, 29),   # Día de San Pedro y San Pablo
    date(2022, 7, 28),   # Fiestas Patrias
    date(2022, 7, 29),   # Fiestas Patrias
    date(2022, 8, 6),   # Batalla de Junín
    date(2022, 8, 30),   # Santa Rosa de Lima
    date(2022, 10, 8),   # Combate Naval de Angamos
    date(2022, 11, 1),   # Día de Todos los Santos
    date(2022, 12, 8),   # Día de la Inmaculada Concepción
    date(2022, 12, 9),   # Batalla de Ayacucho
    date(2022, 12, 25),   # Navidad    
    date(2023, 1, 1),   # Año Nuevo
    date(2023, 4, 6),   # Jueves Santo
    date(2023, 4, 7),   # Viernes Santo
    date(2023, 5, 1),   # Día Internacional de los Trabajadores (Día del Trabajo)
    date(2023, 6, 29),   # Día de San Pedro y San Pablo
    date(2023, 7, 23),   # Día de la Fuerza Aérea del Perú
    date(2023, 7, 28),   # Fiestas Patrias
    date(2023, 7, 29),   # Fiestas Patrias
    date(2023, 8, 6),   # Batalla de Junín
    date(2023, 8, 30),   # Santa Rosa de Lima
    date(2023, 10, 8),   # Combate Naval de Angamos
    date(2023, 11, 1),   # Día de Todos los Santos
    date(2023, 12, 8),   # Día de la Inmaculada Concepción
    date(2023, 12, 9),   # Batalla de Ayacucho
    date(2023, 12, 25),   # Navidad    
    date(2024, 1, 1),   # Año Nuevo
    date(2024, 3, 28),   # Jueves Santo
    date(2024, 3, 29),   # Viernes Santo
    date(2024, 5, 1),   # Día Internacional de los Trabajadores (Día del Trabajo)
    date(2024, 6, 7),   # Batalla de Arica y Día de la Bandera
    date(2024, 6, 29),   # Día de San Pedro y San Pablo
    date(2024, 7, 23),   # Día de la Fuerza Aérea del Perú
    date(2024, 7, 28),   # Fiestas Patrias
    date(2024, 7, 29),   # Fiestas Patrias
    date(2024, 8, 6),   # Batalla de Junín
    date(2024, 8, 30),   # Santa Rosa de Lima
    date(2024, 10, 8),   # Combate Naval de Angamos
    date(2024, 11, 1),   # Día de Todos los Santos
    date(2024, 12, 8),   # Día de la Inmaculada Concepción
    date(2024, 12, 9),   # Batalla de Ayacucho
    date(2024, 12, 25),   # Navidad    
    date(2025, 1, 1),   # Año Nuevo
    date(2025, 4, 17),   # Jueves Santo
    date(2025, 4, 18),   # Viernes Santo
    date(2025, 5, 1),   # Día Internacional de los Trabajadores (Día del Trabajo)
    date(2025, 6, 7),   # Batalla de Arica y Día de la Bandera
    date(2025, 6, 29),   # Día de San Pedro y San Pablo
    date(2025, 7, 23),   # Día de la Fuerza Aérea del Perú
    date(2025, 7, 28),   # Fiestas Patrias
    date(2025, 7, 29),   # Fiestas Patrias
    date(2025, 8, 6),   # Batalla de Junín
    date(2025, 8, 30),   # Santa Rosa de Lima
    date(2025, 10, 8),   # Combate Naval de Angamos
    date(2025, 11, 1),   # Día de Todos los Santos
    date(2025, 12, 8),   # Día de la Inmaculada Concepción
    date(2025, 12, 9),   # Batalla de Ayacucho
    date(2025, 12, 25),   # Navidad
    # Añade aquí todos los feriados que quieras considerar...
}

# Diccionario de meses en español
MAPA_MESES_ES = {
    "ENE": 1,
    "FEB": 2,
    "MAR": 3,
    "ABR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AGO": 8,
    "SET": 9,   # el BCRP suele usar 'Set'
    "SEP": 9,   # por si acaso
    "OCT": 10,
    "NOV": 11,
    "DIC": 12,
}

# ------------------------------------------------------------
# Utilidades de calendario
# ------------------------------------------------------------

def es_dia_habil(fecha: date) -> bool:
    """True si es lunes-viernes y no está en la tabla de feriados."""
    return fecha.weekday() < 5 and fecha not in FERIADOS_PE


def contar_dias_habiles(fecha_ini: date, fecha_fin: date) -> int:
    """
    Cuenta días hábiles estrictamente posteriores a fecha_ini y hasta fecha_fin.
    Es decir, si fecha_ini = hoy, cuenta los días hábiles futuros.
    """
    dias = 0
    f = fecha_ini + timedelta(days=1)
    while f <= fecha_fin:
        if es_dia_habil(f):
            dias += 1
        f += timedelta(days=1)
    return dias


def generar_fechas_habiles(fecha_ini: date, n_dias: int):
    """
    Genera una lista de n_dias hábiles posteriores a fecha_ini.
    No incluye fecha_ini.
    """
    fechas = []
    f = fecha_ini
    while len(fechas) < n_dias:
        f += timedelta(days=1)
        if es_dia_habil(f):
            fechas.append(f)
    return fechas

# ------------------------------------------------------------
# Parser de fechas del BCRP (meses en español)
# ------------------------------------------------------------

def parse_periodo_es(s):
    """
    Convierte strings tipo '04.Ene.21' (u otras variantes con separadores)
    a datetime usando meses en español. Si no puede parsear, devuelve NaT.
    """
    if pd.isna(s):
        return pd.NaT

    s = str(s).strip()

    import re
    # día (1-2 dígitos), separador, mes (3 letras), separador, año (2-4 dígitos)
    m = re.search(r"(\d{1,2})[.\-/ ]+([A-Za-zÁÉÍÓÚÑñ]{3})[.\-/ ]+(\d{2,4})", s)
    if not m:
        return pd.NaT

    dia_str, mes_str, anio_str = m.groups()

    mes_key = mes_str[:3].upper()
    mes = MAPA_MESES_ES.get(mes_key)
    if mes is None:
        return pd.NaT

    try:
        dia = int(dia_str)
        anio_val = int(anio_str)
    except ValueError:
        return pd.NaT

    # Año de 2 dígitos -> 20xx (simple)
    if len(anio_str) == 2:
        anio = 2000 + anio_val if anio_val <= 50 else 1900 + anio_val
    else:
        anio = anio_val

    try:
        return datetime(anio, mes, dia)
    except ValueError:
        return pd.NaT

# ------------------------------------------------------------
# Carga de datos desde BCRP (via requests)
# ------------------------------------------------------------

@st.cache_data(ttl=86400)
def obtener_dataframe_bcrp(
    codigo_serie: str = SERIE_TC_SBS_VENTA,
    fecha_ini: date = FECHA_INICIO_SERIE,
    fecha_fin: date | None = None,
) -> pd.DataFrame:
    """
    Descarga una serie del BCRP vía requests y devuelve un DataFrame con:
      - índice datetime (Periodo)
      - una columna 'tc_sbs_venta' (float)
    """
    if fecha_fin is None:
        fecha_fin = date.today()

    url_base = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api/"
    formato_salida = "json"

    periodo_inicial = fecha_ini.strftime("%Y-%m-%d")
    periodo_final = fecha_fin.strftime("%Y-%m-%d")

    url = f"{url_base}{codigo_serie}/{formato_salida}/{periodo_inicial}/{periodo_final}"

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error en la consulta al BCRP: {response.status_code}")

    try:
        consulta = response.json()
    except ValueError:
        raise Exception("Error: La consulta no devolvió un JSON válido")

    # Periodos y valores
    datos = consulta.get("periods", [])
    periodo = [i.get("name") for i in datos]
    valores = [(i.get("values") or [np.nan])[0] for i in datos]

    df = pd.DataFrame({"Periodo": periodo, "valor": valores})

    # Reemplazar 'n.d.' por NaN y convertir a float
    df["valor"] = df["valor"].replace("n.d.", np.nan).astype(float)

    # Convertir 'Periodo' usando el parser de meses en español
    df["Periodo"] = df["Periodo"].apply(parse_periodo_es)
    df = df.dropna(subset=["Periodo"])
    df = df.set_index("Periodo").sort_index()

    # Renombrar columna a nombre estándar
    df = df.rename(columns={"valor": "tc_sbs_venta"})

    return df

from datetime import date

def construir_tc_sunat(df_sbs: pd.DataFrame):
    """
    A partir de la serie de TC SBS (venta) construye la serie de TC SUNAT:

    1. Se reindexa la serie SBS a TODOS los días calendario (incluyendo fines de semana),
       desde la primera fecha disponible hasta 'hoy'.
    2. Se rellena hacia adelante (ffill) cuando falten datos SBS: regla "tomar el TC
       del día inmediato anterior".
    3. Se define TC_SUNAT(t) = TC_SBS_filled(t-1).
    4. Se marcan fines de semana y feriados, y se construye un DataFrame solo con
       días hábiles reales (sin sábados, domingos ni feriados).

    Devuelve:
      - df_full: con todos los días calendario y tc_sunat
      - df_habiles: solo días hábiles reales (lunes-viernes y no feriados)
    """
    if df_sbs.empty:
        df_vacio = df_sbs.copy()
        df_vacio["tc_sunat"] = pd.Series(dtype=float)
        df_vacio["es_fin_de_semana"] = pd.Series(dtype=bool)
        df_vacio["es_feriado"] = pd.Series(dtype=bool)
        df_vacio["es_habil_real"] = pd.Series(dtype=bool)
        return df_vacio, df_vacio

    df = df_sbs.copy()

    # 1) Índice calendario completo: desde el primer dato SBS hasta hoy
    first_date = df.index.min().date()
    today = date.today()
    idx_full = pd.date_range(start=first_date, end=today, freq="D")

    # 2) Reindexar SBS a calendario y rellenar con último valor conocido
    df_full = df.reindex(idx_full)
    df_full.index.name = "Periodo"
    df_full["tc_sbs_venta"] = df_full["tc_sbs_venta"].ffill()

    # 3) SUNAT(t) = SBS_filled(t-1)
    df_full["tc_sunat"] = df_full["tc_sbs_venta"].shift(1)
    df_full["tc_sunat"].fillna(method="bfill", inplace=True)

    # 4) Marcar fines de semana y feriados
    df_full["es_fin_de_semana"] = df_full.index.weekday >= 5
    df_full["es_feriado"] = df_full.index.date.map(lambda d: d in FERIADOS_PE)

    # Día hábil real = no fin de semana, no feriado
    df_full["es_habil_real"] = ~(df_full["es_fin_de_semana"] | df_full["es_feriado"])

    # 5) DataFrame solo con días hábiles reales
    df_habiles = df_full[df_full["es_habil_real"]].copy()

    return df_full, df_habiles


def calcular_retornos_log(df_habiles: pd.DataFrame) -> pd.Series:
    """Retornos logarítmicos diarios del TC SUNAT (solo días hábiles)."""
    r = np.log(df_habiles["tc_sunat"]).diff().dropna()
    return r

# ------------------------------------------------------------
# Modelos de simulación
# ------------------------------------------------------------

def parametros_escenario(mu_hist: float, sigma_hist: float, escenario: str):
    """
    Ajusta media y volatilidad según el escenario.
    Devuelve (mu_esc, sigma_esc, escala_bootstrap).
    """
    if escenario == "Base":
        return mu_hist, sigma_hist, 1.0
    elif escenario == "Poco riesgoso":
        return mu_hist + 0.0002, sigma_hist * 0.8, 0.8
    elif escenario == "Riesgoso":
        return mu_hist + 0.0006, sigma_hist * 1.1, 1.2
    elif escenario == "Muy riesgoso":
        return mu_hist + 0.001, sigma_hist * 1.5, 1.5
    else:
        return mu_hist, sigma_hist, 1.0


def simular_gbm(S0: float, mu: float, sigma: float,
                n_steps: int, n_sims: int, dt: float = 1.0) -> np.ndarray:
    """
    Simula trayectorias bajo un GBM discreto.
    Devuelve un array (n_sims, n_steps+1) con la trayectoria completa.
    """
    S = np.zeros((n_sims, n_steps + 1), dtype=float)
    S[:, 0] = S0

    drift = (mu - 0.5 * sigma**2) * dt

    for t in range(1, n_steps + 1):
        z = np.random.normal(size=n_sims)
        S[:, t] = S[:, t - 1] * np.exp(drift + sigma * np.sqrt(dt) * z)

    return S


def simular_bootstrap(S0: float, retornos_log: np.ndarray,
                      n_steps: int, n_sims: int, escala_vol: float = 1.0) -> np.ndarray:
    """
    Simula trayectorias re-muestreando retornos históricos (bootstrapping).
    retornos_log: array 1D con retornos logarítmicos históricos.
    """
    S = np.zeros((n_sims, n_steps + 1), dtype=float)
    S[:, 0] = S0
    n_hist = len(retornos_log)

    for t in range(1, n_steps + 1):
        idx = np.random.randint(0, n_hist, size=n_sims)
        r = retornos_log[idx] * escala_vol
        S[:, t] = S[:, t - 1] * np.exp(r)

    return S


def resumen_paths(paths: np.ndarray, fecha_inicio: date, fechas_future):
    """
    Calcula media y cuantiles de las trayectorias.
    Indexa el DataFrame con [fecha_inicio] + fechas_future.
    """
    media = paths.mean(axis=0)
    p05 = np.percentile(paths, 5, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    index = [fecha_inicio] + list(fechas_future)
    index = pd.to_datetime(index)

    df = pd.DataFrame(
        {"media": media, "p05": p05, "p50": p50, "p95": p95},
        index=index
    )
    return df

# ------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------

def plot_hist_y_sim(df_hist: pd.DataFrame,
                    fechas_future,
                    paths: np.ndarray,
                    ventana_anos: int = 1):
    """
    Gráfico 1: Histórico TC SUNAT (últimos 'ventana_anos' años) + algunas trayectorias simuladas.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # --- 1) Recortar histórico a la ventana deseada ---
    if len(df_hist) > 0:
        ultima_fecha_hist = df_hist.index[-1]
        cutoff = ultima_fecha_hist - pd.DateOffset(years=ventana_anos)
        df_hist_plot = df_hist[df_hist.index >= cutoff]
    else:
        df_hist_plot = df_hist

    # --- 2) Plot del histórico ---
    ax.plot(
        df_hist_plot.index,
        df_hist_plot["tc_sunat"],
        label="SUNAT histórico (último año)",
        linewidth=2.0,   # un poco más grueso
    )

    # --- 3) Plot de simulaciones (más visibles) ---
    n_sims = paths.shape[0]
    n_mostrar = min(60, n_sims)  # mostramos hasta 60 trayectorias
    idx = np.random.choice(n_sims, n_mostrar, replace=False)

    fechas_all = [df_hist.index[-1]] + list(pd.to_datetime(fechas_future))

    for i in idx:
        ax.plot(
            fechas_all,
            paths[i, :],
            alpha=0.25,     # más opacas
            linewidth=1.2,  # más gruesas
        )

    # --- 4) Estilo del gráfico ---
    ax.set_title(
        "Histórico reciente del TC SUNAT + simulaciones",
        fontsize=13,
        pad=10,
    )
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Tipo de cambio (S/ por US$)", fontsize=11)

    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

    # Quitar bordes superior y derecho
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(frameon=False, fontsize=9, loc="upper left")

    fig.tight_layout()
    st.pyplot(fig)

def plot_media(df_resumen: pd.DataFrame):
    """Gráfico 2: media y banda 5%-95% de las proyecciones."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df_resumen.index, df_resumen["media"], label="Media proyectada")
    ax.fill_between(df_resumen.index,
                    df_resumen["p05"],
                    df_resumen["p95"],
                    alpha=0.2,
                    label="Banda 5%-95%")

    ax.set_title("Media y banda de proyecciones del TC SUNAT")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Tipo de cambio (S/ por US$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)


def plot_backtesting(df_sunat_full: pd.DataFrame,
                     fecha_ini: date,
                     fecha_fin: date,
                     tc_proj: float):
    """
    Gráfico 3: backtesting simple.
    Muestra el histórico del TC SUNAT entre fecha_ini y fecha_fin
    y una línea horizontal con el TC proyectado para la fecha final.
    """
    mask = (df_sunat_full.index.date >= fecha_ini) & (df_sunat_full.index.date <= fecha_fin)
    df_bt = df_sunat_full.loc[mask]

    if df_bt.empty:
        st.info("No hay datos históricos en el rango seleccionado para el backtesting.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df_bt.index, df_bt["tc_sunat"], label="SUNAT histórico")
    ax.axhline(tc_proj, linestyle="--",
               label=f"TC proyectado para {fecha_fin} (media)")

    ax.set_title("Backtesting simple vs proyección")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Tipo de cambio (S/ por US$)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
