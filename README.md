============================================================
AI ADMISSION & ACADEMIC PREDICTION PORTAL
============================================================

Proyek ini adalah sistem berbasis Flask yang menggunakan 
Machine Learning (MLP & Forecasting) untuk memproses 
penerimaan siswa dan memprediksi tren nilai masa depan.

---
1. PRASYARAT
---
- Pastikan kamu telah menginstal Python 3.10.x.
  (Cek dengan perintah: python --version)

---
2. SETUP LINGKUNGAN (VIRTUAL ENVIRONMENT)
---
Sebelum menjalankan aplikasi, pastikan Virtual Environment (venv) aktif:

- Windows:
    .\venv\Scripts\activate

- macOS / Linux:
    source venv/bin/activate

(Jika belum membuat venv, jalankan: python -m venv venv)

---
3. INSTALASI DEPENDENSI
---
Instal semua pustaka yang dibutuhkan dengan menjalankan:

    pip install -r requirements.txt

*Catatan: Jika kamu belum memiliki file requirements.txt, 
jalankan: pip freeze > requirements.txt*

---
4. MENJALANKAN APLIKASI
---
Setelah environment aktif dan dependencies terinstal serta sudah masuk ke path project, jalankan command berikut di terminal:

    python generate_data.py
    python train_model.py
    python train_forecaster.py
    python app.py

Aplikasi akan berjalan di: http://127.0.0.1:5000

---
5. PENTING (MODEL ML)
---
Pastikan file model berikut berada di direktori yang sama 
dengan app.py agar aplikasi tidak error:
- admission_mlp_model.pkl
- admission_scaler.pkl
- grade_forecaster.pkl
- forecaster_scaler.pkl

============================================================
