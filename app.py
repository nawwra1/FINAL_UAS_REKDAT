import streamlit as st
import pandas as pd
import pickle
import sqlite3
import textwrap
from datetime import datetime
from supabase import create_client

# =========================
# Konfigurasi halaman
# =========================
st.set_page_config(
    page_title="Prediksi Risiko Penyakit",
    page_icon="🩺",
    layout="wide"
)

# =========================
# Load model
# =========================
@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

rf_model = load_model("rf_model.pkl")
xgb_model = load_model("xgb_model.pkl")

label_mapping = {
    0: "Low",
    1: "Moderate",
    2: "High"
}

def get_saran_jantung(risk):
    if risk == "High":
        return [
            "Segera konsultasi dengan dokter atau tenaga kesehatan.",
            "Pantau tekanan darah, kolesterol, dan denyut jantung secara berkala.",
            "Kurangi makanan tinggi lemak jenuh, garam, dan makanan olahan.",
            "Lakukan aktivitas fisik ringan sesuai kemampuan dan anjuran dokter.",
            "Hindari merokok dan batasi konsumsi alkohol."
        ]
    elif risk == "Moderate":
        return [
            "Perbanyak sayur, buah, dan makanan tinggi serat.",
            "Rutin olahraga ringan hingga sedang, misalnya jalan kaki 30 menit.",
            "Pantau tekanan darah dan kolesterol secara berkala.",
            "Kelola stres dan cukup tidur.",
            "Kurangi makanan tinggi garam dan lemak."
        ]
    else:
        return [
            "Pertahankan pola hidup sehat.",
            "Tetap aktif bergerak dan olahraga teratur.",
            "Jaga berat badan ideal.",
            "Lakukan pemeriksaan kesehatan berkala.",
            "Pertahankan kualitas tidur dan manajemen stres."
        ]


def get_saran_diabetes(risk):
    if risk == "High":
        return [
            "Segera konsultasi dengan dokter untuk pemeriksaan gula darah lebih lanjut.",
            "Pantau kadar gula darah, HbA1c, dan pola makan harian.",
            "Kurangi konsumsi gula tambahan, minuman manis, dan karbohidrat sederhana.",
            "Perbanyak makanan tinggi serat seperti sayur, kacang-kacangan, dan biji-bijian.",
            "Lakukan aktivitas fisik secara rutin sesuai kondisi tubuh."
        ]
    elif risk == "Moderate":
        return [
            "Kurangi makanan dan minuman tinggi gula.",
            "Atur porsi karbohidrat seperti nasi, roti, mie, dan makanan manis.",
            "Mulai olahraga rutin untuk membantu menjaga sensitivitas insulin.",
            "Pantau berat badan dan lingkar pinggang.",
            "Lakukan cek gula darah secara berkala."
        ]
    else:
        return [
            "Pertahankan pola makan seimbang.",
            "Tetap batasi gula berlebih meskipun risiko rendah.",
            "Jaga berat badan tetap ideal.",
            "Tetap aktif bergerak setiap hari.",
            "Lakukan pemeriksaan gula darah secara berkala."
        ]
    
# =========================
# Database SQLite
# =========================
DB_NAME = "health_prediction.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        Patient_Name TEXT,
        Checkup_Date TEXT,

        Age INTEGER,
        Gender TEXT,
        BMI REAL,
        Smoking_Status TEXT,
        Alcohol_Consumption TEXT,
        Physical_Activity_Level TEXT,
        Diet_Type TEXT,
        Family_History_CVD INTEGER,
        Family_History_T2D INTEGER,
        Stress_Level REAL,
        Depression_Score INTEGER,
        Anxiety_Score INTEGER,
        Social_Isolation_Index INTEGER,
        Sleep_Hours REAL,
        Sleep_Quality TEXT,
        Waist_Circumference REAL,

        Cholesterol REAL,
        Glucose_Level REAL,
        HbA1c REAL,
        PRS_Cardiometabolic REAL,
        PRS_Type2Diabetes REAL,
        APOE_e4_Carrier INTEGER,
        BRCA_Pathogenic_Variant INTEGER,
        Resting_Heart_Rate INTEGER,
        HRV INTEGER,
        Systolic_BP INTEGER,
        Diastolic_BP INTEGER,
        LDL REAL,
        HDL REAL,
        Triglycerides REAL,
        CRP REAL,
        eGFR REAL,

        Model_Used TEXT,
        Heart_Risk TEXT,
        Diabetes_Risk TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_prediction_to_db(final_data, model_used, heart_risk, diabetes_risk):
    data = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Patient_Name": final_data["Patient_Name"],
        "Checkup_Date": final_data["Checkup_Date"],

        "Age": final_data["Age"],
        "Gender": final_data["Gender"],
        "BMI": final_data["BMI"],
        "Smoking_Status": final_data["Smoking_Status"],
        "Alcohol_Consumption": final_data["Alcohol_Consumption"],
        "Physical_Activity_Level": final_data["Physical_Activity_Level"],
        "Diet_Type": final_data["Diet_Type"],
        "Family_History_CVD": final_data["Family_History_CVD"],
        "Family_History_T2D": final_data["Family_History_T2D"],
        "Stress_Level": final_data["Stress_Level"],
        "Depression_Score": final_data["Depression_Score"],
        "Anxiety_Score": final_data["Anxiety_Score"],
        "Social_Isolation_Index": final_data["Social_Isolation_Index"],
        "Sleep_Hours": final_data["Sleep_Hours"],
        "Sleep_Quality": final_data["Sleep_Quality"],
        "Waist_Circumference": final_data["Waist_Circumference"],

        "Cholesterol": final_data["Cholesterol"],
        "Glucose_Level": final_data["Glucose_Level"],
        "HbA1c": final_data["HbA1c"],
        "PRS_Cardiometabolic": final_data["PRS_Cardiometabolic"],
        "PRS_Type2Diabetes": final_data["PRS_Type2Diabetes"],
        "APOE_e4_Carrier": final_data["APOE_e4_Carrier"],
        "BRCA_Pathogenic_Variant": final_data["BRCA_Pathogenic_Variant"],
        "Resting_Heart_Rate": final_data["Resting_Heart_Rate"],
        "HRV": final_data["HRV"],
        "Systolic_BP": final_data["Systolic_BP"],
        "Diastolic_BP": final_data["Diastolic_BP"],
        "LDL": final_data["LDL"],
        "HDL": final_data["HDL"],
        "Triglycerides": final_data["Triglycerides"],
        "CRP": final_data["CRP"],
        "eGFR": final_data["eGFR"],

        "Model_Used": model_used,
        "Heart_Risk": heart_risk,
        "Diabetes_Risk": diabetes_risk
    }

    try:
        supabase.table("predictions").insert(data).execute()
    except Exception as e:
        st.error("Gagal menyimpan data ke Supabase.")
        st.exception(e)
        st.stop()


def get_all_predictions():
    response = (
        supabase
        .table("predictions")
        .select("*")
        .order("id", desc=True)
        .execute()
    )

    return pd.DataFrame(response.data)

# init_db()

@st.cache_resource
def get_supabase_client():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = get_supabase_client()

@st.dialog("Database Tersimpan")
def show_success_popup():
    st.success("Hasil prediksi berhasil disimpan ke database.")
    st.write("Data pasien dan hasil prediksi sudah masuk ke file SQLite.")
    
    if st.button("OK"):
        st.session_state.show_db_popup = False
        st.rerun()

@st.dialog("Data Basic Tersimpan")
def show_basic_saved_popup():
    st.success("Data basic sudah berhasil disimpan.")
    st.write("Silakan lanjut ke bagian Data Medical Check Up.")

    if st.button("OK"):
        st.session_state.show_basic_popup = False
        st.rerun()

# =========================
# Navbar Atas
# =========================

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

query_params = st.query_params

if "nav" in query_params:
    nav = query_params["nav"]

    if nav == "dashboard":
        st.session_state.page = "Dashboard"
    elif nav == "basic":
        st.session_state.page = "Input Basic User"
    elif nav == "medical":
        st.session_state.page = "Input Medical Check Up"
    elif nav == "history":
        st.session_state.page = "Riwayat Prediksi"

if "theme" in query_params:
    selected_theme = query_params["theme"]

    if selected_theme == "light":
        st.session_state.theme_mode = "light"
    elif selected_theme == "dark":
        st.session_state.theme_mode = "dark"

theme_mode = st.session_state.theme_mode

if theme_mode == "light":
    theme_label = "Light"
else:
    theme_label = "Dark"

active_dashboard = "active" if st.session_state.page == "Dashboard" else ""
active_basic = "active" if st.session_state.page == "Input Basic User" else ""
active_medical = "active" if st.session_state.page == "Input Medical Check Up" else ""
active_history = "active" if st.session_state.page == "Riwayat Prediksi" else ""

# =========================
# Warna Tema Light / Dark
# =========================
if theme_mode == "light":
    app_bg = "#F8FAFC"
    app_text = "#0F172A"
    navbar_bg = "#FFFFFF"
    navbar_border = "#E5E7EB"
    logo_color = "#0F172A"
    menu_color = "#64748B"
    menu_hover = "#0F172A"
    active_bg = "#E2E8F0"
    active_color = "#0F172A"
    dropdown_bg = "#FFFFFF"
    dropdown_border = "#E5E7EB"
    dropdown_hover = "#F1F5F9"
    form_bg = "#FFFFFF"
    form_border = "#E2E8F0"
    input_bg = "#F8FAFC"
    input_text = "#111827"
    label_color = "#1E293B"
else:
    app_bg = "#0E1117"
    app_text = "#FAFAFA"
    navbar_bg = "#0E1117"
    navbar_border = "#2D2D2D"
    logo_color = "#FFFFFF"
    menu_color = "#999999"
    menu_hover = "#FFFFFF"
    active_bg = "#1E1E1E"
    active_color = "#FFFFFF"
    dropdown_bg = "#111827"
    dropdown_border = "#2D2D2D"
    dropdown_hover = "#1F2937"
    form_bg = "#111827"
    form_border = "#2D3748"
    input_bg = "#1F2937"
    input_text = "#F9FAFB"
    label_color = "#E5E7EB"

current_nav = query_params.get("nav", "dashboard")
active_light = "active-theme" if theme_mode == "light" else ""
active_dark = "active-theme" if theme_mode == "dark" else ""

st.markdown(f"""
<style>
[data-testid="stSidebar"] {{ display: none; }}
header[data-testid="stHeader"] {{ display: none !important; }}

.block-container {{
    padding-top: 80px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

.navbar {{
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 40px;
    height: 60px;
    background: {navbar_bg};
    border-bottom: 1px solid {navbar_border};
}}

.navbar-logo {{
    font-size: 1rem;
    font-weight: 600;
    color: {logo_color};
}}

.navbar-menu {{
    display: flex;
    gap: 4px;
    align-items: center;
}}

.navbar-menu a {{
    text-decoration: none;
    font-size: 0.88rem;
    font-weight: 500;
    color: {menu_color};
    padding: 7px 16px;
    border-radius: 7px;
}}

.navbar-menu a:hover,
.navbar-menu a.active {{
    color: {active_color};
    background: {active_bg};
}}

.theme-menu {{
    position: relative;
    display: inline-block;
}}

.theme-dots {{
    color: {menu_color};
    padding: 7px 14px;
    border-radius: 7px;
    font-weight: 700;
    cursor: pointer;
}}

.theme-menu:hover .theme-dots {{
    color: {menu_hover};
    background: {active_bg};
}}

.theme-dropdown {{
    display: none;
    position: absolute;
    top: 36px;
    right: 0;
    min-width: 110px;
    background: {dropdown_bg};
    border: 1px solid {dropdown_border};
    border-radius: 10px;
    padding: 8px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.25);
}}

.theme-menu:hover .theme-dropdown {{
    display: block;
}}

.theme-dropdown a {{
    display: block;
    padding: 8px 12px !important;
    color: {menu_color} !important;
}}

.theme-dropdown a:hover,
.theme-dropdown a.active-theme {{
    background: {active_bg} !important;
    color: {active_color} !important;
}}

/* =========================
   STYLE INPUT BASIC CLEAN ELEGANT
========================= */

.stApp {{
    background: {app_bg} !important;
    color: {app_text} !important;
}}

/* Warning box */
div[data-testid="stAlert"] {{
    background: #fffde7 !important;
    border: 1px solid #facc15 !important;
    border-radius: 13px !important;
}}

div[data-testid="stAlert"] p {{
    color: #7a6000 !important;
    font-weight: 700 !important;
}}

div[data-testid="stAlert"] svg {{
    color: #ca8a04 !important;
    fill: #ca8a04 !important;
}}

/* Label semua input/selectbox di luar dan dalam form */
label,
label p,
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] label p,
div[data-testid="stNumberInput"] label,
div[data-testid="stNumberInput"] label p,
div[data-testid="stTextInput"] label,
div[data-testid="stTextInput"] label p {{
    color: {label_color} !important;
    font-weight: 800 !important;
}}

/* Judul utama halaman */
h1 {{
    color: #0f172a !important;
    font-size: 2.35rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.05em !important;
    margin-bottom: 14px !important;
}}

h1::after {{
    content: "";
    display: block;
    width: 86px;
    height: 5px;
    margin-top: 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, #0f172a, #64748b);
}}

/* Deskripsi halaman */
.stMarkdown p {{
    color: #475569 !important;
    font-size: 1rem !important;
    line-height: 1.7 !important;
}}

/* Form utama */
[data-testid="stForm"] {{
    background: {form_bg} !important;
    border: 1px solid {form_border} !important;
    border-radius: 24px !important;
    padding: 30px !important;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08) !important;
}}

/* Label input */
[data-testid="stForm"] label,
[data-testid="stForm"] label p {{
    color: {label_color} !important;
    font-size: 0.92rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.01em !important;
}}

/* Input text dan angka */
.stTextInput input,
.stNumberInput input {{
    background: {input_bg} !important;
    border: 1px solid {form_border} !important;
    border-radius: 13px !important;
    color: {input_text} !important;
}}

/* Selectbox */
.stSelectbox > div > div {{
    background: {input_bg} !important;
    border: 1px solid {form_border} !important;
    border-radius: 13px !important;
    color: {input_text} !important;
}}

/* Hover input */
.stTextInput input:hover,
.stNumberInput input:hover,
.stSelectbox > div > div:hover {{
    background: {input_bg} !important;
    border: 1px solid {form_border} !important;
    border-radius: 13px !important;
    color: {input_text} !important;
}}

/* Focus input */
.stTextInput input:focus,
.stNumberInput input:focus,
.stSelectbox > div > div:focus-within {{
    border-color: #64748b !important;
    box-shadow: 0 0 0 3px rgba(100, 116, 139, 0.15) !important;
}}

/* Icon panah selectbox */
.stSelectbox svg {{
    color: {input_text} !important;
    fill: {input_text} !important;
}}

/* Balikin tombol plus minus ke normal */
.stNumberInput button {{
    background: {input_bg} !important;
    color: {input_text} !important;
    border: 1px solid {form_border} !important;
    box-shadow: none !important;
    transform: none !important;
}}

.stNumberInput button svg {{
    fill: {input_text} !important;
    color: {input_text} !important;
}}

.stNumberInput button:hover {{
    background: {active_bg} !important;
    color: {active_color} !important;
    border-color: {form_border} !important;
}}

.stNumberInput button:hover svg {{
    fill: {active_color} !important;
    color: {active_color} !important;
}}

/* Tombol Simpan dan Next, tidak hijau */
[data-testid="stFormSubmitButton"] button {{
    height: 46px !important;
    border-radius: 13px !important;
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
    color: #1f2937 !important;
    font-weight: 800 !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}}

/* Hover tombol Simpan dan Next */
[data-testid="stFormSubmitButton"] button:hover {{
    background: #0f172a !important;
    color: #ffffff !important;
    border-color: #0f172a !important;
    transform: translateY(-1px);
}}

/* Spasi antar input */
[data-testid="stForm"] [data-testid="stVerticalBlock"] {{
    gap: 0.7rem !important;
}}

/* === MODEL SELECTOR ONLY - outside form === */
div[data-testid="stSelectbox"]:not([data-testid="stForm"] div[data-testid="stSelectbox"]) > div > div {{
    background: #e8eaf6 !important;
    border: 2px solid #9fa8da !important;
    border-radius: 13px !important;
    color: #1a1a2e !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}}

div[data-testid="stSelectbox"]:not([data-testid="stForm"] div[data-testid="stSelectbox"]) > div > div:hover {{
    background: #c5cae9 !important;
    border-color: #5c6bc0 !important;
}}

div[data-testid="stSelectbox"]:not([data-testid="stForm"] div[data-testid="stSelectbox"]) > div > div:focus-within {{
    background: linear-gradient(135deg, #1a1a2e 0%, #0d0d1f 100%) !important;
    border-color: #6a6aaa !important;
    color: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(106,106,170,0.25) !important;
}}

    /* Tombol Download CSV */
div[data-testid="stDownloadButton"] > button {{
    height: 46px !important;
    border-radius: 13px !important;
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
    color: #1f2937 !important;
    font-weight: 800 !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}}

/* Tombol biasa, termasuk Input Pasien Baru */
div[data-testid="stButton"] > button {{
    height: 46px !important;
    border-radius: 13px !important;
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
    color: #1f2937 !important;
    font-weight: 800 !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}}

/* Hover Download CSV */
div[data-testid="stDownloadButton"] > button:hover {{
    background: #0f172a !important;
    color: #ffffff !important;
    border-color: #0f172a !important;
    transform: translateY(-1px);
}}

/* Hover tombol biasa */
div[data-testid="stButton"] > button:hover {{
    background: #0f172a !important;
    color: #ffffff !important;
    border-color: #0f172a !important;
    transform: translateY(-1px);
}}

/* Saat diklik / focus */
div[data-testid="stDownloadButton"] > button:focus,
div[data-testid="stDownloadButton"] > button:active,
div[data-testid="stButton"] > button:focus,
div[data-testid="stButton"] > button:active {{
    background: #0f172a !important;
    color: #ffffff !important;
    border-color: #0f172a !important;
}}

</style>

<div class="navbar"><div class="navbar-logo">HealthCare</div><div class="navbar-menu"><a href="?nav=dashboard&theme={theme_mode}" target="_self" class="{active_dashboard}">Dashboard</a><a href="?nav=basic&theme={theme_mode}" target="_self" class="{active_basic}">Input Basic</a><a href="?nav=medical&theme={theme_mode}" target="_self" class="{active_medical}">Medical Check Up</a><a href="?nav=history&theme={theme_mode}" target="_self" class="{active_history}">Riwayat</a><div class="theme-menu"><div class="theme-dots">...</div><div class="theme-dropdown"><a href="?nav={current_nav}&theme=light" target="_self" class="{active_light}">Light</a><a href="?nav={current_nav}&theme=dark" target="_self" class="{active_dark}">Dark</a></div></div></div></div>
""", unsafe_allow_html=True)

page = st.session_state.page
    
page_options = [
    "Dashboard",
    "Input Basic User",
    "Input Medical Check Up",
    "Riwayat Prediksi"
]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if st.session_state.page not in page_options:
    st.session_state.page = "Dashboard"

# =========================
# Session State
# =========================
if "basic_data" not in st.session_state:
    st.session_state.basic_data = {}

if "medical_data" not in st.session_state:
    st.session_state.medical_data = {}

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "show_db_popup" not in st.session_state:
    st.session_state.show_db_popup = False

if "show_basic_popup" not in st.session_state:
    st.session_state.show_basic_popup = False

import streamlit.components.v1 as components
if page == "Dashboard":
    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
    *{box-sizing:border-box;margin:0;padding:0;font-family:'DM Sans',sans-serif;}
    body{background:transparent;padding:0;}
    .section-dark{
        background:linear-gradient(135deg,#1a1a2e 0%,#0d0d1f 100%);
        padding:36px 40px;
        margin:0 24px;
    }
    .hero-badge{font-size:.72rem;color:#6a6aaa;letter-spacing:.12em;text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:8px;}
    .hero-badge::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:#6a6aaa;}
    .hero-title{font-size:3rem;font-weight:800;color:#fff;line-height:1.15;margin-bottom:12px;}
    .hero-desc{font-size:.88rem;color:#8888aa;line-height:1.7;}
    .hero-row{display:flex;align-items:center;justify-content:space-between;}
    .hero-icon{font-size:4rem;filter:drop-shadow(0 0 20px rgba(106,106,170,.4));animation:pulse 3s ease-in-out infinite;}
    @keyframes pulse{0%,100%{transform:scale(1);}50%{transform:scale(1.06);}}
    .section-light{
        background:transparent;
        padding:20px 24px;
    }
    .warning-box{background:#fffde7;border:1px solid #f9d900;border-radius:12px;padding:12px 20px;color:#7a6000;font-size:.85rem;display:flex;align-items:center;gap:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .section-dark2{
        background:transparent;
        padding:36px 24px;
    }
    .section-title-light{font-size:.72rem;color:#3a5aaa;letter-spacing:.12em;text-transform:uppercase;margin-bottom:20px;display:flex;align-items:center;gap:8px;}
    .section-title-light::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:#3a5aaa;}
    .stats-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;}
    .stat-card{background:#111827;border:1px solid #1e2d4a;border-radius:16px;padding:26px 20px;text-align:center;cursor:pointer;transition:all .2s;position:relative;overflow:hidden;box-shadow:0 2px 10px rgba(60,90,200,.07);}
    .stat-card::after{content:"";position:absolute;bottom:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#3a5aaa,#6a8aee,#3a5aaa);opacity:0;transition:opacity .2s;}
    .stat-card:hover{background:#f0f4ff;border-color:#8aaae0;transform:translateY(-3px);box-shadow:0 10px 28px rgba(60,100,220,.15);}
    .stat-card:hover::after{opacity:1;}
    .stat-card:active{transform:scale(.97);}
    .stat-number{font-size:2.4rem;font-weight:800;color:#ffffff;margin-bottom:6px;line-height:1;}
    .stat-label{font-size:.75rem;color:#5577aa;font-weight:600;letter-spacing:.04em;}
    .stat-card:hover .stat-number{
        color:#0f172a;
    }
    .stat-card:hover .stat-label{
        color:#1e40af;
    }
    .modal-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.82);z-index:99999;justify-content:center;align-items:center;}
    .modal-overlay.active{display:flex;}
    .modal-box{background:linear-gradient(160deg,#111827,#0a0f1e);border:1px solid #2d3a6a;border-radius:18px;padding:34px;max-width:500px;width:90%;position:relative;animation:popIn .22s cubic-bezier(.34,1.56,.64,1);}
    @keyframes popIn{from{opacity:0;transform:scale(.88);}to{opacity:1;transform:scale(1);}}
    .modal-close{position:absolute;top:14px;right:16px;color:#555577;cursor:pointer;background:#1a1a33;border:1px solid #2d2d4a;padding:4px 10px;border-radius:8px;font-size:.9rem;transition:all .15s;}
    .modal-close:hover{color:#fff;background:#2a2a4a;border-color:#6a6aaa;}
    .modal-title{font-size:1.05rem;font-weight:700;color:#9898cc;margin-bottom:20px;padding-right:28px;}
    .modal-item{display:flex;align-items:flex-start;gap:10px;margin-bottom:13px;}
    .modal-dot{width:6px;height:6px;min-width:6px;background:linear-gradient(135deg,#6a6aaa,#9898cc);border-radius:50%;margin-top:7px;}
    .modal-text{font-size:.83rem;color:#b0b0cc;line-height:1.65;}
    .section-light2{
        background:transparent;
        padding:36px 24px 24px 24px;
    }
    .section-heading{font-size:1.45rem;font-weight:800;color:#3a5aaa;margin-bottom:24px;letter-spacing:-.03em;}
    .tips-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;}
    .tip-card{background:#0d1f3c;border:1px solid #1a3060;border-top:3px solid transparent;border-radius:16px;padding:24px 22px;transition:all .2s;min-height:175px;display:flex;flex-direction:column;}
    .tip-card:hover{transform:translateY(-4px);box-shadow:0 12px 32px rgba(20,60,180,.2);border-color:#2a4a90;border-top-color:#3a6acc;}
    .tip-icon{font-size:1.5rem;margin-bottom:10px;}
    .tip-title{font-size:.87rem;font-weight:700;color:#c8d8f8;margin-bottom:7px;}
    .tip-desc{font-size:.78rem;color:#6a88bb;line-height:1.65;}
    /* CTA + BUTTON: sama lebar dengan tips grid, padding sama persis */
    .section-bottom{
        background:transparent;
        padding:0 24px 50px 24px;
        display:flex;
        flex-direction:column;
        gap:14px;
    }
    .cta-box{
        background:#111827;
        border:1px solid #1e2d4a;
        border-radius:16px;
        padding:26px 32px;
        min-height:86px;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:18px;
        box-shadow:0 2px 10px rgba(60,90,200,.07);
    }
    .cta-button{
        text-decoration:none;
        background:linear-gradient(135deg,#1a3a6a,#2a5aaa);
        border:1px solid #3a6acc;
        color:#ffffff;
        font-size:0.95rem;
        font-weight:800;
        padding:13px 24px;
        border-radius:12px;
        cursor:pointer;
        transition:all .2s;
        white-space:nowrap;
    }
    .cta-button:hover{
        background:linear-gradient(135deg,#2a5aaa,#3a6acc);
        box-shadow:0 8px 24px rgba(30,80,200,.35);
        transform:translateY(-2px);
    }
    .cta-left{display:flex;flex-direction:column;gap:4px;}
    .cta-text{font-size:1.05rem;font-weight:700;color:#ffffff;}
    .cta-sub{font-size:.82rem;color:#6a7aaa;}
    .start-btn{width:100%;background:linear-gradient(135deg,#1a3a6a,#2a5aaa);border:1px solid #3a6acc;color:#fff;font-size:1rem;font-weight:700;padding:15px 0;border-radius:12px;cursor:pointer;transition:all .2s;box-shadow:0 4px 16px rgba(30,80,200,.2);letter-spacing:.01em;}
    .start-btn:hover{background:linear-gradient(135deg,#2a5aaa,#3a6acc);box-shadow:0 8px 24px rgba(30,80,200,.35);transform:translateY(-2px);}
    .start-btn:active{transform:translateY(0);}
    </style>
    </head>
    <body>
    <div class="section-dark">
      <div class="hero-row">
        <div style="flex:1;">
          <div class="hero-badge">HEALTHCARE · MACHINE LEARNING</div>
          <div class="hero-title">Prediksi Risiko Penyakit<br>Jantung &amp; Diabetes</div>
          <div class="hero-desc">Aplikasi ini menggunakan teknologi Machine Learning untuk membantu mendeteksi risiko penyakit jantung dan diabetes berdasarkan data kesehatan dan gaya hidupmu — cepat, mudah, dan akurat.</div>
        </div>
        <div class="hero-icon">🫀</div>
      </div>
    </div>
    <div class="section-light">
      <div class="warning-box">
        ⚠️ Catatan: Hasil prediksi ini hanya untuk edukasi dan bukan diagnosis medis resmi. Selalu konsultasikan kondisi kesehatan Anda kepada tenaga medis profesional.
      </div>
    </div>
    <div class="section-dark2">
      <div class="section-title-light">STATISTIK APLIKASI</div>
      <div class="stats-grid">
        <div class="stat-card" id="card-model"><div class="stat-number">2</div><div class="stat-label">Model ML</div></div>
        <div class="stat-card" id="card-fitur"><div class="stat-number">34</div><div class="stat-label">Fitur Kesehatan</div></div>
        <div class="stat-card" id="card-penyakit"><div class="stat-number">2</div><div class="stat-label">Jenis Penyakit</div></div>
        <div class="stat-card" id="card-risiko"><div class="stat-number">3</div><div class="stat-label">Level Risiko</div></div>
      </div>
    </div>
    <div class="modal-overlay" id="modalOverlay">
      <div class="modal-box">
        <button class="modal-close" id="modalClose">✕</button>
        <div class="modal-title" id="modalTitle"></div>
        <div id="modalContent"></div>
      </div>
    </div>
    <div class="section-light2">
      <div class="section-heading">Tips Menjaga Kesehatan Jantung &amp; Diabetes</div>
      <div class="tips-grid">
        <div class="tip-card">
          <div class="tip-icon">🥗</div>
          <div class="tip-title">Pola Makan Sehat</div>
          <div class="tip-desc">Kurangi makanan tinggi gula, lemak jenuh, dan garam. Perbanyak sayur, buah, dan biji-bijian untuk menjaga kadar gula darah dan tekanan darah tetap normal.</div>
        </div>
        <div class="tip-card">
          <div class="tip-icon">🏃</div>
          <div class="tip-title">Olahraga Rutin</div>
          <div class="tip-desc">Lakukan aktivitas fisik minimal 30 menit sehari, 5 hari seminggu. Jalan kaki, bersepeda, atau berenang sangat baik untuk memperkuat jantung dan mengontrol gula darah.</div>
        </div>
        <div class="tip-card">
          <div class="tip-icon">💧</div>
          <div class="tip-title">Cukup Minum Air</div>
          <div class="tip-desc">Minum 8 gelas air putih per hari membantu ginjal membuang racun, mengatur tekanan darah, dan mencegah dehidrasi yang bisa memicu lonjakan gula darah.</div>
        </div>
        <div class="tip-card">
          <div class="tip-icon">😴</div>
          <div class="tip-title">Tidur Berkualitas</div>
          <div class="tip-desc">Tidur 7–9 jam setiap malam membantu tubuh mengatur hormon insulin dan kortisol. Kurang tidur kronis berkaitan erat dengan risiko diabetes tipe 2 dan penyakit jantung.</div>
        </div>
        <div class="tip-card">
          <div class="tip-icon">🚭</div>
          <div class="tip-title">Hindari Rokok &amp; Alkohol</div>
          <div class="tip-desc">Merokok merusak pembuluh darah dan meningkatkan risiko serangan jantung. Alkohol berlebih meningkatkan tekanan darah dan kadar trigliserida secara signifikan.</div>
        </div>
        <div class="tip-card">
          <div class="tip-icon">🩺</div>
          <div class="tip-title">Cek Kesehatan Berkala</div>
          <div class="tip-desc">Pantau tekanan darah, gula darah puasa, dan kolesterol minimal setahun sekali. Deteksi dini adalah cara terbaik mencegah komplikasi serius di kemudian hari.</div>
        </div>
      </div>
    </div>
    <script>

    var modalData={
      'card-model':{title:'🤖 2 Model Machine Learning',items:['Random Forest — model ensemble berbasis banyak pohon keputusan, unggul dalam akurasi dan tahan terhadap data yang tidak seimbang.','XGBoost — model ensemble berbasis gradient boosting yang membangun pohon keputusan secara bertahap untuk meningkatkan akurasi prediksi.']},
      'card-fitur':{title:'📋 34 Fitur Kesehatan',items:['Data Demografis: nama pasien, tanggal pemeriksaan, usia, jenis kelamin, indeks massa tubuh (BMI).','Riwayat Medis: hipertensi, diabetes sebelumnya, riwayat keluarga penyakit jantung.','Hasil Lab: kadar gula darah puasa, kolesterol total, LDL, HDL, trigliserida.','Pengukuran Fisik: tekanan darah sistolik & diastolik, detak jantung istirahat.','Gaya Hidup: aktivitas fisik, kebiasaan merokok, konsumsi alkohol, pola tidur.','Gejala Klinis: nyeri dada, sesak napas, kelelahan tidak wajar, pembengkakan kaki.']},
      'card-penyakit':{title:'🏥 2 Jenis Penyakit',items:['Penyakit Jantung — mencakup penyakit arteri koroner, gagal jantung, dan aritmia. Diprediksi berdasarkan fitur kardiovaskular dan gaya hidup.','Diabetes Tipe 2 — gangguan metabolisme gula darah kronis akibat resistensi insulin. Diprediksi berdasarkan kadar glukosa, BMI, usia, dan riwayat keluarga.']},
      'card-risiko':{title:'⚠️ 3 Level Risiko',items:['🟢 Risiko Rendah (di bawah 30%) — tetap jaga pola hidup sehat dan lakukan cek rutin tahunan.','🟡 Risiko Sedang (30–60%) — disarankan konsultasi dokter, ubah gaya hidup, dan pemeriksaan lebih lanjut.','🔴 Risiko Tinggi (di atas 60%) — segera konsultasi dokter spesialis untuk evaluasi medis menyeluruh.']}
    };
    function openModal(k){var d=modalData[k];if(!d)return;document.getElementById('modalTitle').innerHTML=d.title;document.getElementById('modalContent').innerHTML=d.items.map(function(i){return'<div class="modal-item"><div class="modal-dot"></div><div class="modal-text">'+i+'</div></div>';}).join('');document.getElementById('modalOverlay').classList.add('active');}
    function closeModal(){document.getElementById('modalOverlay').classList.remove('active');}
    ['card-model','card-fitur','card-penyakit','card-risiko'].forEach(function(id){document.getElementById(id).addEventListener('click',function(){openModal(id);});});
    document.getElementById('modalClose').addEventListener('click',closeModal);
    document.getElementById('modalOverlay').addEventListener('click',function(e){if(e.target===this)closeModal();});
    </script>
    </body>
    </html>
    """, height=1080, scrolling=False)

    cta_html = f"""
    <style>
    .cta-dashboard-box {{
        background: #111827;
        border: 1px solid #1e2d4a;
        border-radius: 16px;
        padding: 26px 32px;
        margin: 0 24px 0 24px;
        min-height: 86px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        box-shadow: 0 2px 10px rgba(60,90,200,.07);
    }}

    .cta-dashboard-left {{
        display: flex;
        flex-direction: column;
        gap: 4px;
    }}

    .cta-dashboard-title {{
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 800;
    }}

    .cta-dashboard-sub {{
        color: #6a7aaa;
        font-size: 0.86rem;
    }}

    .cta-dashboard-button {{
        text-decoration: none !important;
        background: linear-gradient(135deg,#1a3a6a,#2a5aaa);
        border: 1px solid #3a6acc;
        color: #ffffff !important;
        font-size: 0.95rem;
        font-weight: 800;
        padding: 14px 28px;
        border-radius: 12px;
        cursor: pointer;
        transition: all .2s;
        white-space: nowrap;
        min-width: 220px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(30,80,200,.25);
        display: inline-block;
    }}

    .cta-dashboard-button:hover {{
        background: linear-gradient(135deg,#2a5aaa,#3a6acc);
        box-shadow: 0 8px 24px rgba(30,80,200,.35);
        transform: translateY(-2px);
    }}
    </style>
    <div class="cta-dashboard-box"><div class="cta-dashboard-left"><div class="cta-dashboard-title">Siap memulai prediksi risiko kesehatan?</div><div class="cta-dashboard-sub">Isi data basic terlebih dahulu untuk memulai.</div></div><a class="cta-dashboard-button" href="?nav=basic&theme={theme_mode}" target="_self">Mulai Input Data →</a></div>
    """

    st.markdown(cta_html, unsafe_allow_html=True)

    st.markdown("""
    <style>
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #1a3a6a 0%, #2a5aaa 100%) !important;
        color: #ffffff !important;
        border: 1px solid #3a6acc !important;
        border-radius: 16px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 15px 0 !important;
        transition: all .2s !important;
        box-shadow: 0 4px 16px rgba(30,80,200,.25) !important;
        letter-spacing: .01em !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #2a5aaa 0%, #3a6acc 100%) !important;
        box-shadow: 0 8px 24px rgba(30,80,200,.4) !important;
        transform: translateY(-2px) !important;
    }
    div[data-testid="stButton"] > button:active {
        transform: translateY(0) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================
# HALAMAN 1: INPUT BASIC USER
# =========================
elif page == "Input Basic User":
    import streamlit.components.v1 as components
    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'DM Sans', sans-serif; }
    body { background: transparent; padding: 4px 0 16px 0; }
    .hero-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0d0d1f 100%);
        border: 1px solid #2d2d4a;
        border-radius: 16px;
        padding: 32px 40px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-badge {
        font-size: 0.72rem;
        color: #6a6aaa;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.3;
        margin-bottom: 12px;
    }
    .hero-desc {
        font-size: 0.88rem;
        color: #999999;
        line-height: 1.6;
        white-space: nowrap;
    }
    </style>
    </head>
    <body>
    <div class="hero-card">
        <div>
            <div class="hero-badge">INPUT DATA · STEP 1 OF 2</div>
            <div class="hero-title">Input Basic User</div>
            <div class="hero-desc">
                Halaman ini berisi data yang dapat diisi sendiri oleh user tanpa pemeriksaan rumah sakit.
            </div>
        </div>
    </div>
    </body>
    </html>
    """, height=170, scrolling=False)

    model_choice = st.selectbox(
        "Pilih Model Machine Learning",
        ["Random Forest", "XGBoost"],
        key="model_basic"
    )

    if model_choice == "Random Forest":
        model = rf_model
    else:
        model = xgb_model

    with st.form("basic_user_form", enter_to_submit=False):
        identitas_col1, identitas_col2 = st.columns(2)

        with identitas_col1:
            patient_name = st.text_input("Nama Pasien")

        with identitas_col2:
            checkup_date = st.date_input("Tanggal Pemeriksaan")

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Umur",
                min_value=1,
                max_value=120,
                value=30
            )

            jenis_kelamin = st.selectbox(
                "Jenis Kelamin",
                ["Laki-laki", "Perempuan"]
            )

            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                max_value=80.0,
                value=22.0
            )

            smoking_status = st.selectbox(
                "Status Merokok",
                ["Tidak pernah merokok", "Mantan perokok", "Perokok aktif"]
            )

            alcohol_consumption = st.selectbox(
                "Konsumsi Alkohol",
                ["Rendah", "Sedang", "Tinggi"]
            )

            physical_activity_level = st.selectbox(
                "Tingkat Aktivitas Fisik",
                ["Tidak aktif", "Ringan", "Sedang", "Aktif"]
            )

            diet_type = st.selectbox(
                "Tipe Diet",
                ["Seimbang", "Tidak sehat", "Vegetarian", "Tinggi protein"]
            )

            waist_circumference = st.number_input(
                "Lingkar Pinggang",
                min_value=0.0,
                value=80.0
            )

        with col2:
            family_history_cvd = st.selectbox(
                "Riwayat Keluarga Penyakit Jantung",
                [0, 1],
                format_func=lambda x: "Tidak ada" if x == 0 else "Ada"
            )

            family_history_t2d = st.selectbox(
                "Riwayat Keluarga Diabetes",
                [0, 1],
                format_func=lambda x: "Tidak ada" if x == 0 else "Ada"
            )

            stress_level = st.number_input(
                "Tingkat Stres",
                min_value=0.0,
                max_value=10.0,
                value=5.0
            )

            depression_score = st.number_input(
                "Skor Depresi",
                min_value=0,
                value=0
            )

            anxiety_score = st.number_input(
                "Skor Kecemasan",
                min_value=0,
                value=0
            )

            social_isolation_index = st.number_input(
                "Indeks Isolasi Sosial",
                min_value=0,
                value=0
            )

            sleep_hours = st.number_input(
                "Jam Tidur per Hari",
                min_value=0.0,
                max_value=24.0,
                value=7.0
            )

            sleep_quality = st.selectbox(
                "Kualitas Tidur",
                ["Buruk", "Cukup", "Baik", "Sangat Baik"]
            )

        button_col1, button_col2 = st.columns(2)

        with button_col1:
            save_basic = st.form_submit_button(
                "Simpan Data Basic",
                use_container_width=True
            )

        with button_col2:
            next_basic = st.form_submit_button(
                "Next ",
                use_container_width=True
            )

    if save_basic or next_basic:
        missing_fields = []

        if patient_name.strip() == "":
            missing_fields.append("Nama Pasien")

        if not jenis_kelamin:
            missing_fields.append("Jenis Kelamin")

        if not smoking_status:
            missing_fields.append("Status Merokok")

        if not alcohol_consumption:
            missing_fields.append("Konsumsi Alkohol")

        if not physical_activity_level:
            missing_fields.append("Tingkat Aktivitas Fisik")

        if not diet_type:
            missing_fields.append("Tipe Diet")

        if not sleep_quality:
            missing_fields.append("Kualitas Tidur")

        if missing_fields:
            st.error("Data berikut wajib diisi: " + ", ".join(missing_fields))
        else:
            st.session_state.basic_data = {
                'Patient_Name': patient_name,
                'Checkup_Date': str(checkup_date),
                'Age': age,
                'Gender': jenis_kelamin,
                'BMI': bmi,
                'Smoking_Status': smoking_status,
                'Alcohol_Consumption': alcohol_consumption,
                'Physical_Activity_Level': physical_activity_level,
                'Diet_Type': diet_type,
                'Family_History_CVD': family_history_cvd,
                'Family_History_T2D': family_history_t2d,
                'Stress_Level': stress_level,
                'Depression_Score': depression_score,
                'Anxiety_Score': anxiety_score,
                'Social_Isolation_Index': social_isolation_index,
                'Sleep_Hours': sleep_hours,
                'Sleep_Quality': sleep_quality,
                'Waist_Circumference': waist_circumference
            }

            if save_basic:
                st.session_state.show_basic_popup = True

            if next_basic:
                st.session_state.page = "Input Medical Check Up"
                st.query_params["nav"] = "medical"
                st.rerun()
    
    if st.session_state.show_basic_popup:
        show_basic_saved_popup()

    if st.session_state.basic_data:
        data = st.session_state.basic_data
        family_cvd = "Ada" if data["Family_History_CVD"] == 1 else "Tidak ada"
        family_t2d = "Ada" if data["Family_History_T2D"] == 1 else "Tidak ada"

        components.html(f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'DM Sans', sans-serif; }}
        body {{ background: transparent; padding: 0; }}

        .section {{
            background: linear-gradient(135deg, #0f0f2e 0%, #1a1a4e 100%);
            border: 1px solid #2d2d6a;
            border-radius: 18px;
            padding: 28px 32px;
            margin-bottom: 14px;
        }}

        .section-header {{
            margin-bottom: 18px;
            padding-bottom: 14px;
            border-bottom: 1px solid #2d2d6a;
        }}

        .section-title {{
            font-size: 1rem;
            font-weight: 800;
            color: #a5b4fc;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        .section-subtitle {{
            font-size: 0.7rem;
            color: #7c82c8;
            font-weight: 500;
            margin-top: 2px;
        }}

        .identity-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}

        .data-grid-3 {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }}

        .data-card {{
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 18px;
        }}

        .data-label {{
            font-size: 0.65rem;
            font-weight: 700;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 7px;
        }}

        .data-value {{
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.3;
        }}

        </style>
        </head>
        <body>

        <div class="section">
            <div class="section-header">
                <div class="section-title">Identitas Pasien</div>
                <div class="section-subtitle">Data pribadi dan jadwal pemeriksaan</div>
            </div>
            <div class="identity-grid">
                <div class="data-card">
                    <div class="data-label">Nama Pasien</div>
                    <div class="data-value">{data['Patient_Name']}</div>
                </div>
                <div class="data-card">
                    <div class="data-label">Tanggal Pemeriksaan</div>
                    <div class="data-value">{data['Checkup_Date']}</div>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <div class="section-title">Data Fisik &amp; Gaya Hidup</div>
                <div class="section-subtitle">Informasi kesehatan fisik dan kebiasaan sehari-hari</div>
            </div>
            <div class="data-grid-3">
                <div class="data-card"><div class="data-label">Umur</div><div class="data-value">{data['Age']} tahun</div></div>
                <div class="data-card"><div class="data-label">Status Merokok</div><div class="data-value">{data['Smoking_Status']}</div></div>
                <div class="data-card"><div class="data-label">Jam Tidur</div><div class="data-value">{data['Sleep_Hours']} jam</div></div>
                <div class="data-card"><div class="data-label">BMI</div><div class="data-value">{data['BMI']}</div></div>
                <div class="data-card"><div class="data-label">Konsumsi Alkohol</div><div class="data-value">{data['Alcohol_Consumption']}</div></div>
                <div class="data-card"><div class="data-label">Kualitas Tidur</div><div class="data-value">{data['Sleep_Quality']}</div></div>
                <div class="data-card"><div class="data-label">Gender</div><div class="data-value">{data['Gender']}</div></div>
                <div class="data-card"><div class="data-label">Aktivitas Fisik</div><div class="data-value">{data['Physical_Activity_Level']}</div></div>
                <div class="data-card"><div class="data-label">Lingkar Pinggang</div><div class="data-value">{data['Waist_Circumference']} cm</div></div>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <div class="section-title">Riwayat &amp; Kondisi Psikososial</div>
                <div class="section-subtitle">Riwayat keluarga dan kondisi kesehatan mental</div>
            </div>
            <div class="data-grid-3">
                <div class="data-card"><div class="data-label">Riwayat Penyakit Jantung</div><div class="data-value">{family_cvd}</div></div>
                <div class="data-card"><div class="data-label">Tingkat Stres</div><div class="data-value">{data['Stress_Level']} / 10</div></div>
                <div class="data-card"><div class="data-label">Skor Kecemasan</div><div class="data-value">{data['Anxiety_Score']}</div></div>
                <div class="data-card"><div class="data-label">Riwayat Diabetes</div><div class="data-value">{family_t2d}</div></div>
                <div class="data-card"><div class="data-label">Skor Depresi</div><div class="data-value">{data['Depression_Score']}</div></div>
                <div class="data-card"><div class="data-label">Indeks Isolasi Sosial</div><div class="data-value">{data['Social_Isolation_Index']}</div></div>
            </div>
        </div>

        </body>
        </html>
        """, height=1050, scrolling=False)

# =========================
# HALAMAN 2: INPUT MEDICAL CHECK UP
# =========================
elif page == "Input Medical Check Up":
    import streamlit.components.v1 as components
    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'DM Sans', sans-serif; }
    body { background: transparent; padding: 0; }
    .hero-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0d0d1f 100%);
        border: 1px solid #2d2d4a;
        border-radius: 16px;
        padding: 28px 40px;
        min-height: 150px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-badge {
        font-size: 0.72rem;
        color: #6a6aaa;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.3;
        margin-bottom: 12px;
    }
    .hero-desc {
        font-size: 0.88rem;
        color: #999999;
        line-height: 1.6;
        white-space: nowrap;
    }
    </style>
    </head>
    <body>
    <div class="hero-card">
        <div>
            <div class="hero-badge">MEDICAL DATA · STEP 2 OF 2</div>
            <div class="hero-title">Input Medical Check Up</div>
            <div class="hero-desc">
                Halaman ini berisi data yang biasanya diperoleh dari hasil pemeriksaan medis, laboratorium, rumah sakit, atau tenaga kesehatan.
            </div>
        </div>
    </div>
    </body>
    </html>
    """, height=160, scrolling=False)

    model_choice = st.selectbox(
        "Pilih Model Machine Learning",
        ["Random Forest", "XGBoost"],
        key="model_medical"
    )

    if model_choice == "Random Forest":
        model = rf_model
    else:
        model = xgb_model

    if not st.session_state.basic_data:
        st.warning(
            "Anda belum mengisi data basic. Sebaiknya isi Halaman 1 terlebih dahulu."
        )

    with st.form("medical_checkup_form", enter_to_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            cholesterol = st.number_input(
                "Kolesterol",
                min_value=0.0,
                value=200.0
            )

            glucose_level = st.number_input(
                "Kadar Glukosa",
                min_value=0.0,
                value=90.0
            )

            hba1c = st.number_input(
                "HbA1c",
                min_value=0.0,
                max_value=20.0,
                value=5.0
            )

            resting_heart_rate = st.number_input(
                "Denyut Jantung Istirahat",
                min_value=0,
                value=75
            )

            hrv = st.number_input(
                "HRV",
                min_value=0,
                value=50
            )

            systolic_bp = st.number_input(
                "Tekanan Darah Sistolik",
                min_value=0,
                value=120
            )

            diastolic_bp = st.number_input(
                "Tekanan Darah Diastolik",
                min_value=0,
                value=80
            )

            ldl = st.number_input(
                "LDL",
                min_value=0.0,
                value=100.0
            )

        with col2:
            hdl = st.number_input(
                "HDL",
                min_value=0.0,
                value=50.0
            )

            triglycerides = st.number_input(
                "Trigliserida",
                min_value=0.0,
                value=150.0
            )

            crp = st.number_input(
                "CRP",
                min_value=0.0,
                value=1.0
            )

            egfr = st.number_input(
                "eGFR",
                min_value=0.0,
                value=90.0
            )

            prs_cardiometabolic = st.number_input(
                "PRS Cardiometabolic",
                value=0.0
            )

            prs_type2diabetes = st.number_input(
                "PRS Type2Diabetes",
                value=0.0
            )

            apoe_e4_carrier = st.selectbox(
                "APOE e4 Carrier",
                [0, 1],
                format_func=lambda x: "Tidak" if x == 0 else "Ya"
            )

            brca_pathogenic_variant = st.selectbox(
                "BRCA Pathogenic Variant",
                [0, 1],
                format_func=lambda x: "Tidak" if x == 0 else "Ya"
            )

        button_med_col1, button_med_col2 = st.columns(2)

        with button_med_col1:
            back_button = st.form_submit_button(
            "Kembali",
            use_container_width=True
        )

        with button_med_col2:
            predict_button = st.form_submit_button(
            "Prediksi Risiko",
            use_container_width=True
    )   
            
    if back_button:
        st.session_state.page = "Input Basic User"
        st.query_params["nav"] = "basic"
        st.rerun()

    if predict_button:
        if not st.session_state.basic_data:
            st.error("Data basic belum lengkap. Silakan isi Halaman 1 terlebih dahulu.")
        elif systolic_bp <= diastolic_bp:
            st.error("Systolic BP harus lebih besar dari Diastolic BP.")
        elif hdl <= 0 or ldl <= 0:
            st.error("Nilai HDL dan LDL harus lebih dari 0.")
        else:
            st.session_state.medical_data = {
                'Cholesterol': cholesterol,
                'Glucose_Level': glucose_level,
                'HbA1c': hba1c,
                'PRS_Cardiometabolic': prs_cardiometabolic,
                'PRS_Type2Diabetes': prs_type2diabetes,
                'APOE_e4_Carrier': apoe_e4_carrier,
                'BRCA_Pathogenic_Variant': brca_pathogenic_variant,
                'Resting_Heart_Rate': resting_heart_rate,
                'HRV': hrv,
                'Systolic_BP': systolic_bp,
                'Diastolic_BP': diastolic_bp,
                'LDL': ldl,
                'HDL': hdl,
                'Triglycerides': triglycerides,
                'CRP': crp,
                'eGFR': egfr
            }

            # Gabungkan data halaman 1 dan halaman 2
            final_data = {
                **st.session_state.basic_data,
                **st.session_state.medical_data
            }

            # Urutan kolom harus sama seperti saat training model
            input_data = pd.DataFrame([final_data])

            expected_columns = [
                'Age',
                'Gender',
                'BMI',
                'Smoking_Status',
                'Alcohol_Consumption',
                'Physical_Activity_Level',
                'Diet_Type',
                'Cholesterol',
                'Glucose_Level',
                'HbA1c',
                'PRS_Cardiometabolic',
                'PRS_Type2Diabetes',
                'APOE_e4_Carrier',
                'BRCA_Pathogenic_Variant',
                'Family_History_CVD',
                'Family_History_T2D',
                'Stress_Level',
                'Depression_Score',
                'Anxiety_Score',
                'Social_Isolation_Index',
                'Sleep_Hours',
                'Sleep_Quality',
                'Resting_Heart_Rate',
                'HRV',
                'Systolic_BP',
                'Diastolic_BP',
                'LDL',
                'HDL',
                'Triglycerides',
                'CRP',
                'eGFR',
                'Waist_Circumference'
            ]

            input_data = input_data[expected_columns]

            prediction = model.predict(input_data)

            heart_prediction = label_mapping[prediction[0][0]]
            diabetes_prediction = label_mapping[prediction[0][1]]

            st.session_state.prediction_result = {
                "heart_prediction": heart_prediction,
                "diabetes_prediction": diabetes_prediction,
                "input_data": input_data,
                "model_choice": model_choice
            }

            save_prediction_to_db(
                final_data=final_data,
                model_used=model_choice,
                heart_risk=heart_prediction,
                diabetes_risk=diabetes_prediction
            )

            st.session_state.show_db_popup = True

    if st.session_state.show_db_popup:
        show_success_popup()

    if st.session_state.prediction_result is not None:
        heart_prediction = st.session_state.prediction_result["heart_prediction"]
        diabetes_prediction = st.session_state.prediction_result["diabetes_prediction"]
        input_data = st.session_state.prediction_result["input_data"]

        def get_risk_style(risk):
            if risk == "High":
                return {
                    "label": "Tinggi",
                    "emoji": "🔴",
                    "soft_bg": "#fdecec",
                    "soft_border": "#f5b5b5",
                    "top_border": "#ef4444",
                    "title_color": "#b91c1c",
                    "badge_bg": "#fff5f5",
                    "badge_border": "#fca5a5",
                    "dot_color": "#ef4444",
                    "value_color": "#dc2626",
                    "dark_badge_bg": "#2b1212",
                    "dark_badge_border": "#ef4444",
                    "dark_badge_text": "#ef4444"
                }
            elif risk == "Moderate":
                return {
                    "label": "Sedang",
                    "emoji": "🟡",
                    "soft_bg": "#fff8e8",
                    "soft_border": "#f7d88a",
                    "top_border": "#f59e0b",
                    "title_color": "#b45309",
                    "badge_bg": "#fffdf5",
                    "badge_border": "#fcd34d",
                    "dot_color": "#f59e0b",
                    "value_color": "#d97706",
                    "dark_badge_bg": "#2c220f",
                    "dark_badge_border": "#f59e0b",
                    "dark_badge_text": "#f59e0b"
                }
            else:
                return {
                    "label": "Rendah",
                    "emoji": "🟢",
                    "soft_bg": "#ecfdf3",
                    "soft_border": "#9ae6b4",
                    "top_border": "#22c55e",
                    "title_color": "#15803d",
                    "badge_bg": "#f4fff8",
                    "badge_border": "#bbf7d0",
                    "dot_color": "#22c55e",
                    "value_color": "#15803d",
                    "dark_badge_bg": "#0f2e1b",
                    "dark_badge_border": "#22c55e",
                    "dark_badge_text": "#22c55e"
                }

        hs = get_risk_style(heart_prediction)
        ds = get_risk_style(diabetes_prediction)

        def build_saran_items(saran_list):
            html = ""
            for i, saran in enumerate(saran_list, 1):
                html += f"""
                <div class="saran-item">
                    <div class="saran-num">{i}</div>
                    <div class="saran-text">{saran}</div>
                </div>
                """
            return html

        heart_items = build_saran_items(get_saran_jantung(heart_prediction))
        diabetes_items = build_saran_items(get_saran_diabetes(diabetes_prediction))

        components.html(f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            * {{
                box-sizing: border-box;
                font-family: Arial, sans-serif;
            }}

            body {{
                margin: 0;
                padding: 0;
                background: transparent;
            }}

            .wrapper {{
                width: 100%;
                max-width: none;
                margin: 0;
                background: #eef3f8;
                padding: 18px;
                border-radius: 16px;
            }}

            .white-box {{
                background: #f8fafc;
                border: 1px solid #dbe3ec;
                border-radius: 14px;
                padding: 18px;
                margin-bottom: 14px;
            }}

            .section-title {{
                font-size: 14px;
                font-weight: 700;
                color: #111827;
                margin-bottom: 14px;
            }}

            .risk-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }}

            .risk-card {{
                border-radius: 12px;
                padding: 14px;
                min-height: 116px;
                border: 1px solid;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }}

            .risk-header {{
                display: flex;
                align-items: center;
                gap: 6px;
                margin-bottom: 8px;
            }}

            .risk-header span {{
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }}

            .risk-badge {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                border-radius: 10px;
                padding: 8px 12px;
                width: fit-content;
                margin-bottom: 8px;
                border: 1px solid;
            }}

            .risk-dot {{
                width: 14px;
                height: 14px;
                border-radius: 50%;
                box-shadow: inset 0 0 6px rgba(255,255,255,0.4);
            }}

            .risk-value {{
                font-size: 14px;
                font-weight: 700;
            }}

            .risk-desc {{
                font-size: 10px;
                color: #64748b;
                line-height: 1.5;
            }}

            .navy-box {{
                background: #f8fafc;
                border: 1px solid #dbe3ec;
                border-radius: 16px;
                padding: 18px;
                margin-bottom: 14px;
            }}

            .navy-title {{
                font-size: 14px;
                font-weight: 700;
                color: #111827;
                margin-bottom: 14px;
            }}

            .saran-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
            }}

            .saran-card {{
                background: #ffffff;
                border: 1px solid #dbe3ec;
                border-radius: 12px;
                overflow: hidden;
            }}

            .saran-head {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 12px;
                border-bottom: 1px solid #e5e7eb;
                background: #f8fafc;
            }}

            .saran-head-title {{
                font-size: 12px;
                font-weight: 700;
                color: #111827;
            }}

            .saran-badge {{
                font-size: 9px;
                font-weight: 700;
                padding: 3px 8px;
                border-radius: 6px;
                border: 1px solid;
                text-transform: uppercase;
            }}

            .saran-body {{
                padding: 10px 12px;
            }}

            .saran-item {{
                display: flex;
                gap: 8px;
                padding: 8px 0;
                border-bottom: 1px solid #e5e7eb;
            }}

            .saran-item:last-child {{
                border-bottom: none;
            }}

            .saran-num {{
                width: 18px;
                height: 18px;
                min-width: 18px;
                border-radius: 5px;
                background: #eef2ff;
                color: #1e3a8a;
                font-size: 10px;
                font-weight: 700;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-top: 1px;
            }}

            .saran-text {{
                font-size: 10px;
                line-height: 1.6;
                color: #334155;
            }}

            .warning-box {{
                background: #fffbea;
                border: 1px solid #facc15;
                border-radius: 12px;
                padding: 10px 12px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}

            .warning-text {{
                font-size: 10px;
                color: #8a6a00;
                line-height: 1.5;
            }}
        </style>
        </head>
        <body>
            <div class="wrapper">

                <div class="white-box">
                    <div class="section-title">Hasil Prediksi Risiko</div>

                    <div class="risk-grid">
                        <div class="risk-card" style="background:{hs['soft_bg']}; border-color:{hs['soft_border']}; border-top:4px solid {hs['top_border']};">
                            <div>
                                <div class="risk-header">
                                    <div>❤️</div>
                                    <span style="color:{hs['title_color']};">Risiko Penyakit Jantung</span>
                                </div>

                                <div class="risk-badge" style="background:{hs['badge_bg']}; border-color:{hs['badge_border']};">
                                    <div class="risk-dot" style="background:{hs['dot_color']};"></div>
                                    <div class="risk-value" style="color:{hs['value_color']};">{hs['label']}</div>
                                </div>
                            </div>

                            <div class="risk-desc">
                                Estimasi tingkat risiko berdasarkan data yang dimasukkan.
                            </div>
                        </div>

                        <div class="risk-card" style="background:{ds['soft_bg']}; border-color:{ds['soft_border']}; border-top:4px solid {ds['top_border']};">
                            <div>
                                <div class="risk-header">
                                    <div>🩸</div>
                                    <span style="color:{ds['title_color']};">Risiko Diabetes</span>
                                </div>

                                <div class="risk-badge" style="background:{ds['badge_bg']}; border-color:{ds['badge_border']};">
                                    <div class="risk-dot" style="background:{ds['dot_color']};"></div>
                                    <div class="risk-value" style="color:{ds['value_color']};">{ds['label']}</div>
                                </div>
                            </div>

                            <div class="risk-desc">
                                Estimasi tingkat risiko berdasarkan data yang dimasukkan.
                            </div>
                        </div>
                    </div>
                </div>

                <div class="navy-box">
                    <div class="navy-title">Saran Tindakan</div>

                    <div class="saran-grid">
                        <div class="saran-card">
                            <div class="saran-head">
                                <div class="saran-head-title">Panduan Menjaga Kesehatan Jantung</div>
                                <div class="saran-badge"
                                    style="background:{hs['dark_badge_bg']}; border-color:{hs['dark_badge_border']}; color:{hs['dark_badge_text']};">
                                    {hs['label']}
                                </div>
                            </div>
                            <div class="saran-body">
                                {heart_items}
                            </div>
                        </div>

                        <div class="saran-card">
                            <div class="saran-head">
                                <div class="saran-head-title">Panduan Menjaga Kadar Gula Darah</div>
                                <div class="saran-badge"
                                    style="background:{ds['dark_badge_bg']}; border-color:{ds['dark_badge_border']}; color:{ds['dark_badge_text']};">
                                    {ds['label']}
                                </div>
                            </div>
                            <div class="saran-body">
                                {diabetes_items}
                            </div>
                        </div>
                    </div>
                </div>

                <div class="warning-box">
                    <div class="warning-icon">⚠️</div>
                    <div class="warning-text">
                        Catatan: Saran ini bersifat edukatif dan tidak menggantikan diagnosis atau konsultasi medis langsung.
                    </div>
                </div>

            </div>
        </body>
        </html>
        """, height=620, scrolling=False)

        if st.button("Lihat Riwayat Prediksi", use_container_width=True):
            st.session_state.page = "Riwayat Prediksi"
            st.query_params["nav"] = "history"
            st.rerun()

elif page == "Riwayat Prediksi":
    import streamlit.components.v1 as components
    components.html("""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'DM Sans', sans-serif; }
    body { background: transparent; padding: 4px 0 16px 0; }
    .hero-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #0d0d1f 100%);
        border: 1px solid #2d2d4a;
        border-radius: 16px;
        padding: 32px 40px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-badge {
        font-size: 0.72rem;
        color: #6a6aaa;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .hero-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.3;
        margin-bottom: 12px;
    }
    .hero-desc {
        font-size: 0.88rem;
        color: #999999;
        line-height: 1.6;
        white-space: nowrap;
    }
    </style>
    </head>
    <body>
    <div class="hero-card">
        <div>
            <div class="hero-badge">DATABASE · PREDICTION HISTORY</div>
            <div class="hero-title">Riwayat Prediksi</div>
            <div class="hero-desc">Halaman ini menampilkan seluruh data prediksi yang sudah tersimpan di database.</div>
        </div>
    </div>
    </body>
    </html>
    """, height=180, scrolling=False)

    history_df = get_all_predictions()

    if history_df.empty:
        st.info("Belum ada data prediksi yang tersimpan.")
    else:
        history_df["Checkup_Date"] = pd.to_datetime(
            history_df["Checkup_Date"],
            errors="coerce"
        )

        history_df["Checkup_Date_Display"] = history_df["Checkup_Date"].dt.strftime("%d-%m-%Y")

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            selected_year = st.selectbox(
                "Pilih Tahun",
                ["Semua"] + sorted(
                    history_df["Checkup_Date"].dt.year.dropna().unique().astype(int).tolist(),
                    reverse=True
                )
            )

        with filter_col2:
            selected_month = st.selectbox(
                "Pilih Bulan",
                ["Semua","Januari","Februari","Maret","April","Mei","Juni",
                 "Juli","Agustus","September","Oktober","November","Desember"]
            )

        with filter_col3:
            search_name = st.text_input("Cari Nama Pasien")

        filtered_df = history_df.copy()

        if selected_year != "Semua":
            filtered_df = filtered_df[filtered_df["Checkup_Date"].dt.year == selected_year]

        month_map = {
            "Januari":1,"Februari":2,"Maret":3,"April":4,"Mei":5,"Juni":6,
            "Juli":7,"Agustus":8,"September":9,"Oktober":10,"November":11,"Desember":12
        }

        if selected_month != "Semua":
            filtered_df = filtered_df[filtered_df["Checkup_Date"].dt.month == month_map[selected_month]]

        if search_name:
            filtered_df = filtered_df[
                filtered_df["Patient_Name"].str.contains(search_name, case=False, na=False)
            ]

        total_data = len(filtered_df)
        total_high_heart = int((filtered_df["Heart_Risk"] == "High").sum())
        total_high_diabetes = int((filtered_df["Diabetes_Risk"] == "High").sum())

        components.html(f"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'DM Sans', sans-serif; }}
        body {{ background: transparent; padding: 8px 0 16px 0; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
        }}
        .stat-card {{
            background: #111827;
            border: 1px solid #1e2d4a;
            border-radius: 14px;
            padding: 24px 28px;
        }}
        .stat-label {{
            font-size: 0.72rem;
            color: #6a6aaa;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .stat-number {{
            font-size: 2.4rem;
            font-weight: 700;
            color: #ffffff;
            line-height: 1;
        }}
        .stat-sub {{
            font-size: 0.76rem;
            color: #444466;
            margin-top: 6px;
        }}
        </style>
        </head>
        <body>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Data</div>
                <div class="stat-number">{total_data}</div>
                <div class="stat-sub">Seluruh riwayat prediksi</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Risiko Jantung Tinggi</div>
                <div class="stat-number">{total_high_heart}</div>
                <div class="stat-sub">Pasien dengan risiko High</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Risiko Diabetes Tinggi</div>
                <div class="stat-number">{total_high_diabetes}</div>
                <div class="stat-sub">Pasien dengan risiko High</div>
            </div>
        </div>
        </body>
        </html>
        """, height=170, scrolling=False)

        display_df = filtered_df.rename(columns={
            "id": "ID",
            "created_at": "Waktu Tersimpan",
            "Patient_Name": "Nama Pasien",
            "Checkup_Date": "Tanggal Pemeriksaan",
            "Age": "Umur",
            "Gender": "Jenis Kelamin",
            "BMI": "BMI",
            "Smoking_Status": "Status Merokok",
            "Alcohol_Consumption": "Konsumsi Alkohol",
            "Physical_Activity_Level": "Aktivitas Fisik",
            "Diet_Type": "Tipe Diet",
            "Family_History_CVD": "Riwayat Keluarga Jantung",
            "Family_History_T2D": "Riwayat Keluarga Diabetes",
            "Stress_Level": "Tingkat Stres",
            "Depression_Score": "Skor Depresi",
            "Anxiety_Score": "Skor Kecemasan",
            "Social_Isolation_Index": "Indeks Isolasi Sosial",
            "Sleep_Hours": "Jam Tidur",
            "Sleep_Quality": "Kualitas Tidur",
            "Waist_Circumference": "Lingkar Pinggang",
            "Cholesterol": "Kolesterol",
            "Glucose_Level": "Kadar Glukosa",
            "HbA1c": "HbA1c",
            "PRS_Cardiometabolic": "PRS Cardiometabolic",
            "PRS_Type2Diabetes": "PRS Type2Diabetes",
            "APOE_e4_Carrier": "APOE e4 Carrier",
            "BRCA_Pathogenic_Variant": "BRCA Pathogenic Variant",
            "Resting_Heart_Rate": "Denyut Jantung Istirahat",
            "HRV": "HRV",
            "Systolic_BP": "Tekanan Darah Sistolik",
            "Diastolic_BP": "Tekanan Darah Diastolik",
            "LDL": "LDL",
            "HDL": "HDL",
            "Triglycerides": "Trigliserida",
            "CRP": "CRP",
            "eGFR": "eGFR",
            "Model_Used": "Model Digunakan",
            "Heart_Risk": "Risiko Jantung",
            "Diabetes_Risk": "Risiko Diabetes"
        })

        if "Checkup_Date" in display_df.columns:
            display_df = display_df.drop(columns=["Checkup_Date"])

        risk_translation = {"Low": "Rendah", "Moderate": "Sedang", "High": "Tinggi"}
        display_df["Risiko Jantung"] = display_df["Risiko Jantung"].map(risk_translation)
        display_df["Risiko Diabetes"] = display_df["Risiko Diabetes"].map(risk_translation)

        st.dataframe(display_df, use_container_width=True)

        csv = display_df.to_csv(index=False).encode("utf-8")

        button_col1, button_col2, button_col3 = st.columns([1, 1, 5])

        with button_col1:
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="riwayat_prediksi_filtered.csv",
                mime="text/csv",
                use_container_width=True
            )

        with button_col2:
            if st.button("Input Pasien Baru", use_container_width=True):
                st.session_state.basic_data = {}
                st.session_state.medical_data = {}
                st.session_state.prediction_result = None
                st.session_state.show_db_popup = False
                st.session_state.page = "Input Basic User"
                st.query_params["nav"] = "basic"
                st.rerun()
