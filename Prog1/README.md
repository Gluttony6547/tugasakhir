# H5 Lab — Model Playground

Demo web untuk menjalankan model Keras `.h5` secara lokal. User mengisi tiga fitur numerik, Flask memanggil `model.h5`, lalu UI menampilkan prediksi, confidence, dan riwayat inference. Prediksi yang confidence-nya di bawah `0.65` ditandai `uncertain` dan tidak dipaksa menjadi label 0/1.

## Jalankan

```powershell
cd C:\SMT_7\PRATAAA\Prog1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python create_model.py
python app.py
```

Buka `http://127.0.0.1:5000`.

## Isi model

`create_model.py` melakukan fine-tuning classifier Keras dengan input shape `(3,)` menggunakan data demo reproducible. Script lalu mengevaluasi 4.000 sampel holdout yang tidak dipakai saat training dan menyimpan metriknya di `model_metrics.json`. Label 1 merepresentasikan sinyal `3*f1 - 2*f2 + 1.2*f3 - 0.35 >= 0`. Ini adalah aturan demo, bukan bukti akurasi untuk data produksi. Untuk penggunaan nyata, ganti generator data dengan dataset berlabel yang representatif dan evaluasi metriknya pada data yang tidak dipakai saat training.

Endpoint `POST /api/predict` mengembalikan keputusan bermakna (`Sinyal positif`, `Sinyal negatif`, atau `Belum pasti`), probabilitas kelas 1, confidence, margin keputusan, dan kontribusi setiap fitur. Nilai `NaN`, infinity, serta fitur di luar rentang training `[-2, 2]` ditolak agar model tidak melakukan ekstrapolasi tanpa dasar.

Catatan: file `.h5` adalah kontainer HDF5 untuk menyimpan struktur dan bobot model; website tidak “menjalankan HDF5” secara langsung, melainkan memuat model Keras yang tersimpan di dalamnya.
