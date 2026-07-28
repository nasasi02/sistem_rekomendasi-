# Smartwatch Recommendation System - Full Thesis Application (Streamlit Dashboard)
# Hybrid Recommender System (Weighted TF-IDF Content-Based + Preference-Based Scoring)
# Dibuat untuk Skripsi Pemrograman

import streamlit as st
import pandas as pd
import numpy as np
import re
import difflib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Smartwatch Recommendation System",
    layout="wide",
    page_icon="⌚",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS with Glassmorphism and Outfit Typography
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stSidebar"] {
        font-family: 'Outfit', sans-serif;
        background-color: #f8fafc;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Main Dashboard Header Styling */
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.3);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        margin: 10px 0 0 0;
        opacity: 0.9;
        font-size: 1.05rem;
    }

    /* Glassmorphism Product Card Styling */
    .product-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.04), 0 4px 6px -4px rgba(15, 23, 42, 0.04);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .product-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 20px 25px -5px rgba(79, 70, 229, 0.1), 0 8px 10px -6px rgba(79, 70, 229, 0.1);
        border: 1px solid rgba(79, 70, 229, 0.3);
    }
    
    /* Badges & Text details */
    .brand-badge {
        background-color: #e0e7ff;
        color: #4338ca;
        padding: 4px 12px;
        border-radius: 30px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 12px;
    }
    .product-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 10px;
        line-height: 1.4;
    }
    .price-tag {
        font-size: 1.35rem;
        font-weight: 700;
        color: #059669;
        margin: 10px 0;
    }
    .info-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        color: #475569;
        margin-bottom: 8px;
        border-bottom: 1px dashed #f1f5f9;
        padding-bottom: 4px;
    }
    .info-label {
        font-weight: 500;
    }
    .info-value {
        font-weight: 600;
        color: #0f172a;
    }
    
    /* Explainable AI (XAI) Explanation box */
    .explanation-box {
        background-color: #f8fafc;
        border-left: 4px solid #4f46e5;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.82rem;
        color: #334155;
        margin-top: 14px;
        line-height: 1.5;
    }
    
    /* Score display chips */
    .score-container {
        display: flex;
        gap: 8px;
        margin-top: 12px;
        flex-wrap: wrap;
    }
    .score-chip {
        background: #f1f5f9;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 500;
        color: #64748b;
    }
    .score-chip-highlight {
        background: #e0f2fe;
        color: #0369a1;
        font-weight: 600;
    }
    
    /* Styled sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
        border-right: 1px solid #cbd5e1;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2);
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD & PREPROCESS DATA
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / 'dataset_jam_baru.csv', sep=';')
    df.columns = df.columns.str.strip()
    # Hapus baris kosong yang tidak valid
    df = df.dropna(subset=['Nama Produk']).reset_index(drop=True)
    return df

@st.cache_data
def load_survey_weights():
    # Perhitungan Dynamic Weights dari kuisioner.csv (Bab 3 & 4 Skripsi)
    try:
        survey = pd.read_csv(BASE_DIR / "kuisioner.csv")
        survey_col = [c for c in survey.columns if "Fitur apa yang Anda anggap penting" in c]
        if survey_col:
            survey_features = survey[survey_col[0]].fillna("")
            survey_features_split = survey_features.apply(
                lambda x: [item.strip() for item in x.split(",") if item.strip()]
            )
            from sklearn.preprocessing import MultiLabelBinarizer
            mlb = MultiLabelBinarizer()
            encoded_features = pd.DataFrame(
                mlb.fit_transform(survey_features_split),
                columns=mlb.classes_
            )
            feature_scores = encoded_features.sum()
            preference_weights = feature_scores / feature_scores.sum()
            
            # Map survey responses to database columns
            survey_to_dataset_mapping = {
                "GPS": "GPS",
                "Heart rate": "Heart Rate Monitor",
                "Heart rate monitor": "Heart Rate Monitor",
                "Waterproof": "Water Resistant_bin",
                "AMOLED": "AMOLED Display",
                "Bluetooth": "Bluetooth Calling",
                "Sleep tracking": "SpO2 Monitoring"  # Dialihkan ke SpO2 karena tidak ada sleep tracking di dataset baru
            }
            
            dynamic_weights = {}
            for survey_feat, dataset_col in survey_to_dataset_mapping.items():
                if survey_feat in preference_weights:
                    dynamic_weights[dataset_col] = dynamic_weights.get(dataset_col, 0) + preference_weights[survey_feat]
            
            # Normalisasi bobot agar jumlahnya = 1.0
            total_dynamic_weight = sum(dynamic_weights.values())
            if total_dynamic_weight > 0:
                dynamic_weights = {k: v / total_dynamic_weight for k, v in dynamic_weights.items()}
            else:
                raise ValueError
        else:
            raise KeyError
    except Exception:
        # Fallback default weights jika kuisioner.csv gagal dibaca
        dynamic_weights = {
            "GPS": 0.22,
            "Heart Rate Monitor": 0.20,
            "Water Resistant_bin": 0.20,
            "AMOLED Display": 0.18,
            "Bluetooth Calling": 0.20
        }
    return dynamic_weights

raw_products = load_data()
products = raw_products.copy()

# ==========================================
# PREPROCESSING
# ==========================================
# 1. Preprocessing Kolom Biner
binary_cols = ['Bluetooth Calling', 'AMOLED Display', 'SpO2 Monitoring', 'Heart Rate Monitor', 'GPS']
for col in binary_cols:
    if col in products.columns:
        products[col] = products[col].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0, '1': 1, '0': 0})
        products[col] = products[col].fillna(0).astype(int)

# 2. Preprocessing Ketahanan Air (Water Resistant)
if 'Water Resistant' in products.columns:
    products['Water Resistant_bin'] = products['Water Resistant'].apply(
        lambda x: 0 if str(x).strip().lower() in ['no', 'nan', ''] else 1
    )

# 3. Preprocessing Harga (IDR)
if 'Harga (IDR)' in products.columns:
    if products['Harga (IDR)'].dtype == object:
        products['Harga (IDR)'] = (
            products['Harga (IDR)'].astype(str)
            .str.replace('Rp', '', regex=False)
            .str.replace('.', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
    products['Harga (IDR)'] = pd.to_numeric(products['Harga (IDR)'], errors='coerce')
    products['Harga (IDR)'].fillna(products['Harga (IDR)'].median() or 500000.0, inplace=True)

# 4. Preprocessing Rating
if 'Rating (Max 5)' in products.columns:
    products['Rating (Max 5)'] = pd.to_numeric(products['Rating (Max 5)'], errors='coerce')
    products['Rating (Max 5)'].fillna(products['Rating (Max 5)'].median() or 4.5, inplace=True)

# 5. Extract Brand from Nama Produk
products['brand'] = products['Nama Produk'].apply(lambda x: str(x).split(' ')[0].strip().upper() if pd.notna(x) else 'OTHER')
raw_products['brand'] = products['brand']
raw_products['Harga (IDR)'] = products['Harga (IDR)']

# ==========================================
# FEATURE ENGINEERING & TF-IDF
# ==========================================
# Ambil bobot kuesioner dinamis
dynamic_weights = load_survey_weights()

def create_weighted_corpus(row, weights):
    corpus = [
        str(row['brand']),
        str(row['Nama Produk']),
        str(row['Compatibility']),
        str(row['Style'])
    ]
    # Ulangi kata kunci fitur penting berdasarkan bobot kuesioner (Weighted TF-IDF)
    for feat, weight in weights.items():
        if row.get(feat) == 1:
            repeat_count = int(weight * 15) + 1
            corpus.extend([feat] * repeat_count)
    return " ".join(corpus)

# Terapkan korpus fitur terbobot
products["weighted_features"] = products.apply(lambda r: create_weighted_corpus(r, dynamic_weights), axis=1)

# Fit TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(products['weighted_features'])
similarity = cosine_similarity(tfidf_matrix)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def match_product(name):
    names = products['Nama Produk']
    if name in names.values:
        return name
    # Case-insensitive
    lower_map = {n.lower(): n for n in names}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    # Fuzzy match
    match = difflib.get_close_matches(name, names, n=1, cutoff=0.6)
    return match[0] if match else None

def generate_explanation(row, selected_features):
    matched = []
    for f in selected_features:
        col_name = {
            'GPS': 'GPS',
            'Heart Rate Monitor': 'Heart Rate Monitor',
            'AMOLED Display': 'AMOLED Display',
            'Bluetooth Calling': 'Bluetooth Calling',
            'Water Resistant': 'Water Resistant_bin'
        }.get(f, f)
        
        if row.get(col_name) == 1:
            matched.append(f)
            
    if matched:
        return f"Direkomendasikan karena memiliki fitur preferensi Anda: <b>{', '.join(matched)}</b>."
    else:
        return "Direkomendasikan berdasarkan kemiripan spesifikasi umum dengan produk referensi."

# ==========================================
# RECOMMENDATION ENGINE
# ==========================================
def recommend(seed_product_name, brand_filter, selected_features, price_range, alpha=0.5, top_n=5, dynamic_weights=None):
    if dynamic_weights is None:
        dynamic_weights = {
            "GPS": 0.2, "Heart Rate Monitor": 0.2, "Water Resistant_bin": 0.2, "AMOLED Display": 0.2, "Bluetooth Calling": 0.2
        }

    matched_name = match_product(seed_product_name)
    if not matched_name:
        return pd.DataFrame()

    seed_idx = products[products['Nama Produk'] == matched_name].index[0]
    sim_scores = list(enumerate(similarity[seed_idx]))

    results = []
    for idx, sim_score in sim_scores:
        if idx == seed_idx:
            continue

        row = products.iloc[idx]
        
        # 1. Brand Filter
        if brand_filter != "All" and row['brand'] != brand_filter:
            continue

        # 2. Price Filter
        if not (price_range[0] <= row['Harga (IDR)'] <= price_range[1]):
            continue

        # 3. Preference Scoring (0.0 - 1.0)
        pref_score = 0
        total_possible_weight = 0
        for f in selected_features:
            col_name = {
                'GPS': 'GPS',
                'Heart Rate Monitor': 'Heart Rate Monitor',
                'AMOLED Display': 'AMOLED Display',
                'Bluetooth Calling': 'Bluetooth Calling',
                'Water Resistant': 'Water Resistant_bin'
            }.get(f, f)

            weight = dynamic_weights.get(col_name, 0.2)
            total_possible_weight += weight
            if row.get(col_name) == 1:
                pref_score += weight

        if total_possible_weight > 0:
            pref_score = pref_score / total_possible_weight
        else:
            pref_score = 0.0

        # 4. Hybrid Scoring Formula
        final_score = (alpha * sim_score) + ((1 - alpha) * pref_score)
        results.append((idx, final_score, sim_score, pref_score))

    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]
    if not results:
        return pd.DataFrame()

    idxs = [r[0] for r in results]
    output = raw_products.iloc[idxs].copy()
    
    output['final_score'] = [r[1] for r in results]
    output['sim_score'] = [r[2] for r in results]
    output['pref_score'] = [r[3] for r in results]
    output['brand'] = [products.iloc[idx]['brand'] for idx in idxs]
    output['price_formatted'] = output['Harga (IDR)'].apply(lambda x: f"Rp {int(x):,}".replace(',', '.'))
    output['explanation'] = output.apply(lambda r: generate_explanation(products.loc[r.name], selected_features), axis=1)

    return output

# ==========================================
# EVALUATION METRICS ENGINE
# ==========================================
def calculate_dynamic_evaluation(brand_filter, selected_features, price_range, dynamic_weights):
    alpha_vals = np.linspace(0, 1, 6)
    precision_list = []
    recall_list = []
    f1_list = []

    # Produk relevan: produk yang memenuhi range harga, brand, dan memiliki minimal salah satu fitur pilihan
    relevant_indices = []
    for idx, row in products.iterrows():
        if not (price_range[0] <= row['Harga (IDR)'] <= price_range[1]):
            continue
        if brand_filter != "All" and row['brand'] != brand_filter:
            continue
        
        feature_match = True
        if selected_features:
            feature_match = any([
                row.get({
                    'GPS': 'GPS',
                    'Heart Rate Monitor': 'Heart Rate Monitor',
                    'AMOLED Display': 'AMOLED Display',
                    'Bluetooth Calling': 'Bluetooth Calling',
                    'Water Resistant': 'Water Resistant_bin'
                }.get(f, f)) == 1 for f in selected_features
            ])
        
        if feature_match:
            relevant_indices.append(idx)
            
    total_relevant = len(relevant_indices)
    seeds = products['Nama Produk'].head(3).tolist() # Mengambil sampel 3 seed produk
    
    if not seeds or total_relevant == 0:
        return alpha_vals, [0.0]*6, [0.0]*6, [0.0]*6

    for a in alpha_vals:
        precisions = []
        recalls = []
        for seed in seeds:
            res = recommend(seed, brand_filter, selected_features, price_range, alpha=a, top_n=5, dynamic_weights=dynamic_weights)
            if res.empty:
                precisions.append(0.0)
                recalls.append(0.0)
                continue
            
            recommended_indices = res.index.tolist()
            true_positives = len([i for i in recommended_indices if i in relevant_indices])
            
            prec = true_positives / len(recommended_indices)
            rec = true_positives / total_relevant
            precisions.append(prec)
            recalls.append(rec)
            
        avg_prec = np.mean(precisions)
        avg_rec = np.mean(recalls)
        avg_f1 = (2 * avg_prec * avg_rec / (avg_prec + avg_rec)) if (avg_prec + avg_rec) > 0 else 0.0
        
        precision_list.append(round(avg_prec, 3))
        recall_list.append(round(avg_rec, 3))
        f1_list.append(round(avg_f1, 3))

    return alpha_vals, precision_list, recall_list, f1_list


# ==========================================
# SIDEBAR DASHBOARD CONTROL PANEL
# ==========================================
st.sidebar.markdown("### 🔎 CONTROL PANEL")

# 1. Product Reference Selection
st.sidebar.markdown("**1. Produk Referensi**")
selected_seed = st.sidebar.selectbox(
    "Pilih Smartwatch Acuan",
    products['Nama Produk'].unique().tolist(),
    index=0
)

# 2. Brand Selection
st.sidebar.markdown("**2. Filter Brand**")
brand_list = ['All'] + sorted(products['brand'].unique().tolist())
brand = st.sidebar.selectbox("Brand", brand_list)

# 3. Preference Features Multiselect
st.sidebar.markdown("**3. Fitur Preferensi**")
features = st.sidebar.multiselect(
    "Pilih Fitur Kunci",
    ['GPS', 'Heart Rate Monitor', 'AMOLED Display', 'Bluetooth Calling', 'Water Resistant'],
    default=['GPS', 'AMOLED Display']
)

# 4. Price Slider
st.sidebar.markdown("**4. Range Budget (Rp)**")
min_price = int(products['Harga (IDR)'].min())
max_price = int(products['Harga (IDR)'].max())
price_range = st.sidebar.slider(
    "Filter Harga",
    min_value=200000,
    max_value=4000000,
    value=(250000, 3000000),
    step=50000,
    format="Rp %d"
)

# 5. Model Parameters
st.sidebar.markdown("**5. Parameter Model**")
alpha = st.sidebar.slider(
    "Alpha (Bobot Hybrid)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    help="0.0 = Hanya Preferensi Fitur, 1.0 = Hanya Content Similarity (Weighted TF-IDF)"
)
top_n = st.sidebar.slider("Jumlah Rekomendasi (Top-N)", 1, 10, 6)

# Menu Router
st.sidebar.markdown("---")
page = st.sidebar.radio("📋 MENU NAVIGATION", ["Recommendation", "Evaluation", "About"])


# ==========================================
# PAGE 1: RECOMMENDATIONS
# ==========================================
if page == "Recommendation":
    # Header Banner
    st.markdown("""
        <div class="main-header">
            <h1>⌚ Smartwatch Hybrid Recommender</h1>
            <p>Sistem Rekomendasi Skripsi berbasis kecerdasan gabungan (Weighted TF-IDF & Preference-Based Scoring)</p>
        </div>
    """, unsafe_allow_html=True)
    
    # 1. Tampilkan Detail Produk Referensi Pilihan
    matched_seed_name = match_product(selected_seed)
    if matched_seed_name:
        seed_row = raw_products[raw_products['Nama Produk'] == matched_seed_name].iloc[0]
        seed_price = f"Rp {int(seed_row['Harga (IDR)']):,}".replace(',', '.')
        seed_img = seed_row['Gambar Produk'] if pd.notna(seed_row['Gambar Produk']) and str(seed_row['Gambar Produk']).strip() != "" else ""
        
        st.markdown("### 🎯 Produk Referensi (Acuan Utama)")
        seed_html = f"""
        <div style="background: rgba(79, 70, 229, 0.05); border: 2px solid #4f46e5; border-radius: 20px; padding: 24px; margin-bottom: 30px;">
            <div style="display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 100px; max-width: 120px; text-align: center;">
                    <img src="{seed_img}" style="width: 100%; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); object-fit: cover;" 
                    onerror="this.onerror=null; this.src='https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&q=80&w=200';" />
                </div>
                <div style="flex: 3; min-width: 250px;">
                    <span class="brand-badge" style="background-color: #4f46e5; color: white;">ACUAN AKTIF</span>
                    <h4 style="margin: 4px 0 8px 0; color: #1e293b;">{seed_row['Nama Produk']}</h4>
                    <div style="font-size: 1.4rem; font-weight: 700; color: #4f46e5; margin-bottom: 8px;">{seed_price}</div>
                    <div style="display: flex; gap: 15px; font-size: 0.85rem; color: #475569; flex-wrap: wrap;">
                        <span>⭐ Rating: <b>{seed_row['Rating (Max 5)']:.1f}</b></span>
                        <span>🔋 Baterai: <b>{seed_row['Battery Life']}</b></span>
                        <span>📱 Sistem: <b>{seed_row['Compatibility']}</b></span>
                        <span>🎨 Tampilan: <b>{seed_row['Display']} ({seed_row['Style']})</b></span>
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(seed_html, unsafe_allow_html=True)
        
    st.markdown("### 🚀 Hasil Rekomendasi Teratas (Produk Terdekat)")
    
    # Run recommendation
    result = recommend(selected_seed, brand, features, price_range, alpha, top_n, dynamic_weights)
    
    if result.empty:
        st.warning("⚠️ Tidak ada produk smartwatch yang memenuhi kriteria filter Anda saat ini. Silakan perluas jangkauan harga atau filter brand di Panel Kontrol.")
    else:
        st.success(f"Ditemukan {len(result)} produk rekomendasi teratas:")
        
        # Display recommendations in grid
        cols = st.columns(2)
        for i, (idx, row) in enumerate(result.iterrows()):
            with cols[i % 2]:
                image_url = row['Gambar Produk'] if pd.notna(row['Gambar Produk']) and str(row['Gambar Produk']).strip() != "" else ""
                
                # HTML Card Structure
                card_html = f"""
                <div class="product-card">
                    <div style="display: flex; gap: 20px; align-items: start;">
                        <div style="flex: 1; min-width: 120px; max-width: 140px; text-align: center;">
                            <img src="{image_url}" style="width: 100%; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); object-fit: cover;" 
                            onerror="this.onerror=null; this.src='https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&q=80&w=200';" />
                        </div>
                        <div style="flex: 2;">
                            <span class="brand-badge">{row['brand']}</span>
                            <div class="product-title">{row['Nama Produk']}</div>
                            <div class="price-tag">{row['price_formatted']}</div>
                            <div class="info-row">
                                <span class="info-label">Rating Produk:</span>
                                <span class="info-value">⭐ {row['Rating (Max 5)']:.1f}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Daya Tahan Baterai:</span>
                                <span class="info-value">🔋 {row['Battery Life']}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Konektivitas Display:</span>
                                <span class="info-value">📱 {row['Compatibility']} ({row['Display']})</span>
                            </div>
                            <div class="explanation-box">
                                💡 {row['explanation']}
                            </div>
                            <div class="score-container">
                                <div class="score-chip score-chip-highlight">Match Score: {row['final_score']:.2%}</div>
                                <div class="score-chip">Similarity: {row['sim_score']:.2f}</div>
                                <div class="score-chip">Preference: {row['pref_score']:.2f}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)


# ==========================================
# PAGE 2: REAL-TIME DYNAMIC EVALUATION
# ==========================================
elif page == "Evaluation":
    st.markdown("""
        <div class="main-header" style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);">
            <h1>📊 Evaluation & Hyperparameter Tuning</h1>
            <p>Dashboard Analisis Performa Model (Precision, Recall, F1-Score) Secara Dinamis Langsung dari Filter Pengguna</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Analisis Parameter Alpha (Hybrid Balance)")
    st.write("Hasil di bawah ini dihitung secara dinamis berdasarkan filter aktif pada Panel Kontrol sidebar Anda:")
    
    alpha_vals, precision, recall, f1_score = calculate_dynamic_evaluation(brand, features, price_range, dynamic_weights)
    
    df_eval = pd.DataFrame({
        "Alpha": alpha_vals,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
    })
    
    col_table, col_chart = st.columns([2, 3])
    
    with col_table:
        st.subheader("Tabel Hasil Evaluasi")
        st.dataframe(df_eval, use_container_width=True)
        
        st.markdown("""
        **Keterangan Parameter:**
        - **Alpha = 0.0**: Sistem berbasis preferensi murni (*Preference-Based Only*).
        - **Alpha = 1.0**: Sistem berbasis kemiripan konten murni (*Content-Based Only*).
        - **Alpha 0.1 - 0.9**: Kombinasi hybrid (Keseimbangan bobot).
        """)
        
    with col_chart:
        st.subheader("Grafik Performa Akurasi")
        # Line chart visualization
        st.line_chart(df_eval.set_index("Alpha"))
        
    st.markdown("---")
    st.markdown("### 🧬 Bobot Preferensi Dinamis dari Kuesioner")
    st.write("Berdasarkan pengolahan data survei kuisioner (`kuisioner.csv`) menggunakan metode pembobotan frekuensi:")
    
    # Visualisasi bobot preferensi dari survei kuesioner
    df_weights = pd.DataFrame({
        "Fitur Smartwatch": list(dynamic_weights.keys()),
        "Bobot Relatif (Normalized)": list(dynamic_weights.values())
    }).sort_values(by="Bobot Relatif (Normalized)", ascending=False)
    
    st.dataframe(df_weights, use_container_width=True)
    st.info("💡 **Catatan Skripsi:** Bobot di atas dihitung dari frekuensi pemilihan fitur oleh responden dalam data kuesioner Anda untuk memberikan representasi keputusan pembelian nyata.")


# ==========================================
# PAGE 3: ABOUT
# ==========================================
elif page == "About":
    st.markdown("""
        <div class="main-header" style="background: linear-gradient(135deg, #475569 0%, #334155 100%);">
            <h1>ℹ️ About the Recommender System</h1>
            <p>Metodologi dan Informasi Teknis Sistem Rekomendasi Skripsi</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Metodologi Model Hybrid Recommendation
    
    Sistem rekomendasi ini menggabungkan dua pendekatan (*Hybrid Recommender System*) untuk menghasilkan rekomendasi yang akurat bagi pengguna:
    
    1. **Content-Based Filtering (Weighted TF-IDF & Cosine Similarity)**
       - Setiap data smartwatch digabungkan menjadi korpus teks (metadata).
       - Untuk meningkatkan pengaruh fitur yang dinilai penting oleh responden survei kuesioner, sistem menerapkan **Weighted TF-IDF** dengan merepetisi istilah fitur kunci sebanding dengan bobot dinamis dari `kuisioner.csv`.
       - Matriks TF-IDF kemudian dihitung kemiripannya menggunakan rumus *Cosine Similarity*:
         $$Cosine Similarity(A, B) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$$
         
    2. **Preference-Based Scoring**
       - Pengguna memasukkan preferensi fitur biner (GPS, AMOLED, dll.) di UI panel filter.
       - Sistem menghitung persentase kecocokan antara spesifikasi smartwatch dengan preferensi pengguna secara terbobot berdasarkan data kuesioner.
       
    3. **Hybrid Combination Formula**
       - Skor akhir rekomendasi dihitung dengan menggabungkan Content Similarity dan Preference Score menggunakan bobot parameter $\\alpha$ (Alpha):
         $$\\text{Final Score} = (\\alpha \\times \\text{Similarity Score}) + ((1 - \\alpha) \\times \\text{Preference Score})$$
         
    ### Struktur Dataset
    - **Dataset Utama**: `dataset_jam_baru.csv` berisi 110 produk smartwatch dari retail dengan pemisah titik koma (`;`).
    - **Dataset Evaluasi**: `kuisioner.csv` digunakan untuk pembobotan dinamis dan simulasi validasi Precision & Recall di Bab 4.
    """)
    st.success("🎓 Aplikasi Streamlit siap digunakan untuk keperluan demo dan sidang skripsi pemrograman Anda!")
