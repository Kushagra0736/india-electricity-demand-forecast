"""India Demand Intelligence: the custom dark UI rendered inside Streamlit.
Run: streamlit run dashboard_india.py
Needs next to it: dashboard_template.html, site_data.py, india_state_monthly.csv, forecast_next_months.csv,
backtest_long.csv, weather_state_monthly.csv, sensitivity.json"""
import streamlit as st
import streamlit.components.v1 as components
import site_data

st.set_page_config(page_title="India Demand Intelligence", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
#MainMenu,header,footer{visibility:hidden;height:0}.stApp{background:#070b14}
.block-container{padding:0!important;max-width:100%!important}iframe{border:0}
</style>""", unsafe_allow_html=True)


@st.cache_data
def page():
    return site_data.render()


try:
    if hasattr(st, 'iframe'):                       # newer Streamlit
        st.iframe(page(), height=1750)
    else:                                           # older Streamlit
        components.html(page(), height=1750, scrolling=True)
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Keep the pipeline CSVs, sensitivity.json, site_data.py and dashboard_template.html next to this script.")
