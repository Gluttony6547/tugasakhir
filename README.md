# Tugas Akhir — Prediksi Sinyal

Project ini berisi aplikasi Flask pada `Prog1`.

## Menjalankan lokal

```powershell
cd Prog1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Buka `http://127.0.0.1:5000`.

## Deploy ke Vercel

Root repository sudah memiliki `vercel.json` dan entrypoint `api/index.py`.
Entrypoint tersebut memakai implementasi sigmoid ringan dengan bobot Dense yang
sama seperti model demo, karena TensorFlow dan file `.h5` terlalu besar untuk
Python Function Vercel. Aplikasi lokal tetap menggunakan `Prog1/model.h5`.

Deploy dari root repository:

```powershell
npm install -g vercel
vercel login
vercel
vercel --prod
```

Untuk deployment berbasis GitHub, import repository `Gluttony6547/tugasakhir`
di Vercel dan gunakan root directory repository.
