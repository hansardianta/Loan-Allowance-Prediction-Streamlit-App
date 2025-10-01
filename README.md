---

# Loan Acceptance Prediction

Proyek ini membangun model machine learning untuk memprediksi **kelayakan pengajuan pinjaman (loan acceptance)** berdasarkan data demografis, finansial, dan riwayat kredit seseorang.
Model yang digunakan adalah **XGBoost Classifier** dengan preprocessing (encoding & scaling) dan aplikasi deployment menggunakan **Streamlit**.

---

## 📂 Struktur Repository

```
.
├── 2702249663_model.ipynb         # Notebook eksplorasi & eksperimen awal
├── 2702249663_OOPtrainingmodel.py # Script OOP untuk training model
├── Dataset_A_loan.csv             # Dataset pinjaman
├── xgbmodel.pkl                   # Model terlatih (XGBoost)
├── label_encoding.pkl              # Encoder untuk fitur kategorikal
├── scaler.pkl                      # Scaler untuk fitur numerikal
├── streamlit_app.py                # Aplikasi deployment dengan Streamlit
├── requirements.txt                # Daftar dependencies
```

---

## ⚙️ Setup & Instalasi

1. **Clone repository**

   ```bash
   git clone https://github.com/hansardianta/Loan-Allowance-Prediction-Streamlit-App.git
   cd Loan-Allowance-Prediction-Streamlit-App
   ```

2. **Buat virtual environment (opsional)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Training Model

Model dilatih menggunakan file **`2702249663_OOPtrainingmodel.py`**.
Langkah utama:

* Preprocessing data (membersihkan umur, normalisasi gender, filtering umur kerja).
* Encoding kolom kategorikal menggunakan `LabelEncoder`.
* Scaling fitur numerikal menggunakan `RobustScaler`.
* Training menggunakan **XGBoost Classifier** dengan hyperparameter hasil tuning.
* Model diekspor ke file `.pkl` untuk digunakan di aplikasi.

Jalankan script training:

```bash
python 2702249663_OOPtrainingmodel.py
```

---

## 🚀 Menjalankan Aplikasi

Aplikasi berbasis **Streamlit** untuk melakukan prediksi status pengajuan pinjaman.

Jalankan:

```bash
streamlit run streamlit_app.py
```

Aplikasi akan berjalan di browser pada alamat:

```
http://localhost:8501
```

---

## 🖥️ Cara Menggunakan Aplikasi

1. Isi form input (usia, gender, pendidikan, penghasilan, pengalaman kerja, dll).
2. Klik tombol **Predict**.
3. Aplikasi akan menampilkan hasil prediksi:

   * **Diterima** ✅
   * **Ditolak** ❌

---

## 📌 Catatan

* Dataset asli: **`Dataset_A_loan.csv`**
* Model: **`xgbmodel.pkl`**
* Encoder & scaler harus sesuai dengan hasil training agar prediksi akurat.

---
