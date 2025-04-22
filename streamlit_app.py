import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("xgbmodel.pkl")
encoders = joblib.load("label_encoding.pkl")
scaler = joblib.load("scaler.pkl")

def main():
    st.title("Loan Acceptance Prediction Model App")

    # Form input user
    def user_input_features():
        person_age = st.slider("Usia", min_value=15, max_value=100, value=15)
        person_gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        person_education = st.selectbox("Tingkat Pendidikan Tertinggi", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
        person_income = st.number_input("Pendapatan (dalam US$)", min_value=1000, max_value=6000000, value=1000, step=1)
        person_emp_exp = st.slider("Lama Bekerja (tahun)", min_value=0, max_value=100, value=0)
        person_home_ownership = st.selectbox("Status Tempat Tinggal", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        loan_amnt = st.slider("Jumlah Pinjaman", min_value=0, max_value=35000, value=0, step=500)
        loan_intent = st.selectbox("Tujuan Pinjaman", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        loan_int_rate = st.slider("Bunga Pinjaman (%)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
        loan_percent_income = st.slider("Persentase Pinjaman dari Pendapatan Tahunan", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        cb_person_cred_hist_length = st.slider("Lama Riwayat Kredit (tahun)", min_value=0, max_value=30, value=0)
        credit_score = st.slider("Skor Kredit", min_value=400, max_value=800, value=400, step=10)
        previous_loan_defaults_on_file = st.selectbox("Memiliki Tunggakan Pinjaman Sebelumnya?", ["Yes", "No"])


        data = {
            "person_age": person_age,
            "person_gender": person_gender,
            "person_education": person_education,
            "person_income": person_income,
            "person_emp_exp": person_emp_exp,
            "person_home_ownership": person_home_ownership,
            "loan_amnt": loan_amnt,
            "loan_intent": loan_intent,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "credit_score": credit_score,
            "previous_loan_defaults_on_file": previous_loan_defaults_on_file
        }

        return pd.DataFrame([data])

    # Ambil input user
    input_df = user_input_features()

    # Proses data
    if st.button("Predict"):
        numerical = ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]
        categorical = ["person_home_ownership", "loan_intent", "person_education", "previous_loan_defaults_on_file", "person_gender"]

        encoded_cat = []
        for col in categorical:
            le = encoders[col]
            encoded = le.transform(input_df[col])
            encoded_cat.append(encoded.reshape(-1, 1))

        x_cat = np.hstack(encoded_cat)
        x_num = scaler.transform(input_df[numerical])

        # Gabung
        x_final = np.hstack([x_num, x_cat])

        # Prediksi
        prediction = model.predict(x_final)[0]
        pred_label = "Diterima" if prediction == 1 else "Ditolak"

        st.subheader("Hasil Prediksi:")
        st.success(f"Status Pengajuan Pinjaman: **{pred_label}**")

    
if __name__ == '__main__':
    main()
