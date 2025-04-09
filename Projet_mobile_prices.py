import pandas as pd
import joblib
import streamlit as st
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Lire les données
df = pd.read_csv("mobile_prices.csv")

# Séparer les features et la target
X = df.drop("price_range", axis=1)
y = df["price_range"]

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Normalisation
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entraînement du modèle
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
joblib.dump(model,"model.pkl")
joblib.dump(scaler,"scaler.pkl")



# Interface Streamlit
st.title("Prédiction du prix d'un téléphone")

with st.form("formulaire_caractéristiques"):
    battery_power = st.number_input("Battery Power", min_value= , max_value= )
    blue = st.selectbox("Bluetooth", [0, 1])
    clock_speed = st.number_input("Clock Speed", step=0.1)
    dual_sim = st.selectbox("Dual SIM", [0, 1])
    fc = st.number_input("Front Camera (MP)", min_value=, max_value=)
    four_g = st.selectbox("4G", [0, 1])
    int_memory = st.number_input("Mémoire Interne (Go)", min_value=, max_value=)
    m_dep = st.number_input("Mobile Depth (cm)", step=0.1)
    mobile_wt = st.number_input("Poids du téléphone (g)", min_value=, max_value=)
    n_cores = st.number_input("Nombre de cœurs CPU", min_value=, max_value=)
    pc = st.number_input("Camera Principale (MP)", min_value=, max_value=)
    px_height = st.number_input("Hauteur écran (px)", min_value=, max_value=)
    px_width = st.number_input("Largeur écran (px)", min_value=, max_value=)
    ram = st.number_input("RAM (Mo)", min_value=, max_value=)
    sc_h = st.number_input("Hauteur écran (cm)", min_value=, max_value=)
    sc_w = st.number_input("Largeur écran (cm)", min_value=, max_value=)
    talk_time = st.number_input("Autonomie en communication (h)", min_value=, max_value=)
    three_g = st.selectbox("3G", [0, 1])
    touch_screen = st.selectbox("Écran tactile", [0, 1])
    wifi = st.selectbox("WiFi", [0, 1])

    submit = st.form_submit_button("Prédire")

# Traitement de la prédiction
if submit:
    user_input = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g,
                            int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width,
                            ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi]])
    
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    
    st.success(f" Le modèle prédit que le téléphone appartient à la classe de prix est : *{prediction}*")

    # Optionnel : affichage des performances du modèle
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    st.write(f"*Précision du modèle sur les données de test* : {accuracy * 100:.2f}%")
