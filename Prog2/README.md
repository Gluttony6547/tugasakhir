# Hybrid Signal Engine — Prog2

V2 adalah dashboard riset dark-mode untuk membaca hasil pipeline multimodal saham IDX LQ45. UI menggunakan snapshot yang dibangun dari salinan/hasil baca data di `C:\SMT_7\PRATAAA\CODE`, tanpa mengubah file sumber.

## Jalankan

```powershell
cd C:\SMT_7\PRATAAA\Prog2
pip install -r requirements.txt
python app.py
```

Buka `http://127.0.0.1:5002`.

Untuk membangun ulang snapshot setelah data di `CODE` diperbarui:

```powershell
python prepare_data.py
```

Snapshot mencakup 10 saham, OHLCV + 19 indikator/fusi sentimen, metrik ensemble target 50 hari, hasil ML/DL, berita berlabel, dan history prediksi. Model di `models/` adalah salinan read-only dari `CODE\Price Prediction Model\*\LSTM_*_Target_50.h5`.

Dashboard ini merupakan research UI dan bukan nasihat investasi. Nilai historis pada data sumber berhenti di September 2023; indikator "Market Open" pada mockup hanya elemen visual referensi, bukan feed real-time.
