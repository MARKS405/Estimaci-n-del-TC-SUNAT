
import streamlit as st
from datetime import date, timedelta

from tc_sunat_model import (
    contar_dias_habiles,
    generar_fechas_habiles,
    obtener_dataframe_bcrp,
    construir_tc_sunat,
    calcular_retornos_log,
    parametros_escenario,
    simular_gbm,
    simular_bootstrap,
    resumen_paths,
    plot_hist_y_sim,
    plot_media,
    plot_backtesting,
)


def main():
    st.set_page_config(
        page_title="Proyección TC SUNAT USD/PEN",
        layout="wide"
    )

    st.title("Proyección del Tipo de Cambio SUNAT (USD/PEN)")
    st.caption("Modelo simplificado basado en datos del BCRP + simulaciones de Monte Carlo.")

    hoy = date.today()

    # Sidebar: inputs
    st.sidebar.header("Configuración de horizonte")

    fecha_inicio = st.sidebar.date_input(
        "Fecha de inicio",
        value=hoy
    )

    opcion_horizonte = st.sidebar.radio(
        "¿Cómo quieres definir el horizonte?",
        ["Plazo en días calendario", "Fecha final"],
        index=0
    )

    if opcion_horizonte == "Plazo en días calendario":
        plazo_dias_cal = st.sidebar.number_input(
            "Plazo (días calendario)",
            min_value=1,
            max_value=365 * 3,
            value=30,
            step=1
        )
        plazo_dias_cal = int(plazo_dias_cal)
        fecha_final_cal = fecha_inicio + timedelta(days=plazo_dias_cal)
    else:
        fecha_final_cal = st.sidebar.date_input(
            "Fecha final",
            value=hoy + timedelta(days=30)
        )
        plazo_dias_cal = (fecha_final_cal - fecha_inicio).days

    plazo_dias_habiles = contar_dias_habiles(fecha_inicio, fecha_final_cal)

    st.sidebar.markdown("---")
    st.sidebar.header("Escenario y modelo")

    escenario = st.sidebar.selectbox(
        "Escenario de riesgo",
        ["Base", "Poco riesgoso", "Riesgoso", "Muy riesgoso"]
    )

    modelo = st.sidebar.selectbox(
        "Modelo de simulación",
        ["GBM (Browniano geométrico)", "Histórico (bootstrapping)"]
    )

    n_sims = st.sidebar.number_input(
        "Número de simulaciones",
        min_value=100,
        max_value=50000,
        value=10000,
        step=100
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Días hábiles (plazo limpio) entre {fecha_inicio} y {fecha_final_cal}: "
        f"**{plazo_dias_habiles}**"
    )
    st.sidebar.caption(
        "Los feriados usados están simulados en el código. "
        "Actualiza la tabla FERIADOS_PE con la lista real."
    )

    if st.button("Simular proyecciones"):
        if plazo_dias_habiles <= 0:
            st.error(
                "El horizonte debe tener al menos 1 día hábil. "
                "Revisa la fecha de inicio y la fecha final/plazo."
            )
            return

        # 1) Cargar datos BCRP y construir TC SUNAT
        df_tc_sbs = obtener_dataframe_bcrp()
        df_sunat_full, df_sunat_habiles = construir_tc_sunat(df_tc_sbs)

        mask_hist = df_sunat_habiles.index.date <= fecha_inicio
        df_hist = df_sunat_habiles.loc[mask_hist]

        if len(df_hist) < 60:
            st.error(
                "No hay suficientes datos históricos antes de la fecha de inicio "
                "(se recomienda al menos ~60 días hábiles)."
            )
            return

        # 2) Estimar parámetros históricos
        retornos_log = calcular_retornos_log(df_hist)
        mu_hist = retornos_log.mean()
        sigma_hist = retornos_log.std()

        mu_esc, sigma_esc, escala_boot = parametros_escenario(mu_hist, sigma_hist, escenario)

        # 3) Generar fechas hábiles futuras (plazo limpio)
        fechas_future = generar_fechas_habiles(fecha_inicio, plazo_dias_habiles)

        # S0: TC SUNAT del último día hábil <= fecha_inicio
        S0 = df_hist["tc_sunat"].iloc[-1]

        # 4) Simular según el modelo escogido
        n_sims_int = int(n_sims)

        if "GBM" in modelo:
            paths = simular_gbm(
                S0=S0,
                mu=mu_esc,
                sigma=sigma_esc,
                n_steps=plazo_dias_habiles,
                n_sims=n_sims_int
            )
        else:
            paths = simular_bootstrap(
                S0=S0,
                retornos_log=retornos_log.values,
                n_steps=plazo_dias_habiles,
                n_sims=n_sims_int,
                escala_vol=escala_boot
            )

        # 5) Resumen de trayectorias
        df_resumen = resumen_paths(paths, fecha_inicio, fechas_future)
        fecha_final_efectiva = fechas_future[-1]

        # Resultados numéricos
        st.subheader("Resumen numérico de la proyección")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "TC SUNAT actual (aprox.)",
                f"{S0:.4f}"
            )
        with col2:
            st.metric(
                "TC proyectado (media)",
                f"{df_resumen['media'].iloc[-1]:.4f}"
            )
        with col3:
            st.metric(
                "Rango 5%–95%",
                f"{df_resumen['p05'].iloc[-1]:.4f} – {df_resumen['p95'].iloc[-1]:.4f}"
            )

        st.write(
            f"- Fecha final **calendario** solicitada: **{fecha_final_cal}**  \n"
            f"- Fecha final **hábil efectiva** (último paso de la simulación): "
            f"**{fecha_final_efectiva}**"
        )

        # Gráfico 1
        st.subheader("Histórico del TC SUNAT + simulaciones hasta la fecha final")
        plot_hist_y_sim(df_hist, fechas_future, paths, ventana_anos=1)

        # Gráfico 2
        st.subheader("Media de las simulaciones proyectadas (con banda 5%–95%)")
        plot_media(df_resumen)

        # Backtesting si aplica
        hoy_actual = date.today()
        if fecha_final_cal < hoy_actual:
            st.subheader("Backtesting simple del modelo")
            tc_proj = df_resumen["media"].iloc[-1]
            plot_backtesting(df_sunat_full, fecha_inicio, fecha_final_cal, tc_proj)

        # Contexto macro (placeholder)
        st.subheader("Contexto macro (extensión opcional)")
        st.info(
            "Aquí puedes extender el proyecto trayendo series adicionales del BCRP "
            "(tasa de referencia, inflación, EMBI, etc.) y mostrar un pequeño resumen "
            "del entorno macro del escenario seleccionado."
        )


if __name__ == "__main__":
    main()
