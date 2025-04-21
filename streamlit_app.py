import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("xgbmodel.pkl")
encoders = joblib.load("label_encoding.pkl")
scaler = joblib.load("scaler.pkl")

def main():
    st.title("Loan Acceptance Prediction App 🏦")

    # Form input user
    def user_input_features():
        person_age = st.number_input("Usia", min_value=18, max_value=100, value=30)
        person_income = st.number_input("Pendapatan", min_value=0, value=5000)
        person_home_ownership = st.selectbox("Status Tempat Tinggal", ["Rent", "Own", "Mortgage", "Other"])
        person_emp_length = st.number_input("Lama Bekerja (tahun)", min_value=0, value=5)
        loan_intent = st.selectbox("Tujuan Pinjaman", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        loan_grade = st.selectbox("Grade Pinjaman", ["A", "B", "C", "D", "E", "F", "G"])
        loan_amnt = st.number_input("Jumlah Pinjaman", min_value=500, value=1000)
        loan_int_rate = st.number_input("Bunga Pinjaman (%)", min_value=0.0, value=10.0)
        cb_person_default_on_file = st.selectbox("Pernah Default?", ["Y", "N"])
        cb_person_cred_hist_length = st.number_input("Lama Riwayat Kredit (tahun)", min_value=0, value=5)
        person_gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])

        data = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "person_gender": person_gender
        }

        return pd.DataFrame([data])

    # Ambil input user
    input_df = user_input_features()

    # Proses data
    if st.button("Predict"):
        numerical = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "cb_person_cred_hist_length"]
        categorical = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file", "person_gender"]

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