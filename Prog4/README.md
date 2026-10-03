# Prog4: Forecaster Tervalidasi Walk-Forward

API prediksi harga saham IDX yang memilih kandidat peramal lewat validasi
walk-forward pada jendela latih, dan melaporkan skor kandidat serta peringatan
out-of-distribution di setiap respons. Stateless: tanpa database, tanpa model
H5, inferensi sklearn dalam milidetik sehingga cocok untuk serverless.

## Metode

- Kandidat: `persistence` (harga terakhir), `drift` (regresi linear lokal di
  jendela input, identik dengan baseline drift Prog3), dan `ridge` (StandarScaler
  + Ridge pada fitur trailing: return 10 hari, volatilitas, momentum, slope
  relatif horizon, jarak ke SMA20; target log-return horizon).
- Pemilihan kandidat per (saham, horizon) memakai validasi walk-forward pada
  30% akhir jendela latih. Ridge hanya di-fit pada prefix sebelum validasi.
  Di dalam selisih MAPE 0,05 poin persentase, persistence menang karena tanpa
  parameter.
- Jendela latih berakhir 25 Sep 2023. Titik evaluasi out-of-sample, blokir
  corporate action (lompatan harian >30%), formula MAPE/akurasi sinyal/akurasi
  arah semuanya meniru `Prog3/backtest_models.py` persis, jadi angka kedua
  sistem dibandingkan pada kisi yang sama.
- Ambang sinyal dan aturan batas identik dengan Prog3; paritasnya diuji di
  `test_forecast.py`.
- Harga terkini diambil dari Yahoo Finance; bila gagal, memakai ujung dataset
  penelitian dengan `data_source: research_dataset` dan peringatan basi, tidak
  pernah diam-diam.

## Hasil benchmark out-of-sample

Jendela evaluasi 1 Des 2023 hingga Okt 2026, rerata 10 saham per horizon,
dari `results/evaluation.json` (dibuat `python evaluate.py`):

| Horizon | MAPE Prog4 | MAPE Prog3 LSTM | MAPE persistensi | Akurasi sinyal Prog4 | Akurasi sinyal Prog3 |
| --- | --- | --- | --- | --- | --- |
| 1 hari | 1,79% | 2,44% | 1,79% | 57,8% | 44,4% |
| 5 hari | 3,99% | 5,15% | 3,99% | 51,5% | 46,4% |
| 10 hari | 5,68% | 8,91% | 5,68% | 64,8% | 54,6% |
| 20 hari | 7,99% | 12,38% | 7,92% | 68,6% | 54,8% |
| 50 hari | 12,94% | 18,35% | 12,85% | 54,1% | 47,5% |

Verdict tersimpan di JSON: Prog4 mengalahkan Prog3 pada MAPE dan akurasi sinyal
di semua horizon. Klaim "mengalahkan persistensi" tercatat `false` pada horizon
20 dan 50 karena pilihan ridge yang kalah tipis di sana. Ini memang batas
jujur dari pendekatan berbasis harga saja: persistensi tetap baseline yang
harus dikalahkan, bukan lawan yang otomatis kalah.

Catatan: persistence terpilih di mayoritas pasangan (saham, horizon); ridge
menang validasi di antaranya ANTM T5, BMRI T10/T20, BNGA T20, INCO T20/T50.
Hasilnya bukan saran investasi.

## Menjalankan lokal

```powershell
# dependensi: pandas, scikit-learn, fastapi, yfinance (lihat requirements.txt)
python -m pip install -r requirements.txt
python -m uvicorn app:app --reload          # buka http://127.0.0.1:8000
python -m pytest -q                          # 8 tes, offline
python evaluate.py                           # regenerasi benchmark (butuh Yahoo + CSV baseline Prog3)
```

## Endpoint

- `GET /api/v1/health`: daftar horizon dan ticker.
- `GET /api/v1/forecast/{symbol}?horizon_days=1|5|10|20|50`: harga terkini,
  harga prediksi, sinyal, skor CV ketiga kandidat, z-score OOD, sumber data,
  dan daftar peringatan.
- `GET /api/v1/benchmark`: tabel hasil benchmark out-of-sample.

## Deploy ke Vercel

```powershell
cd Prog4
npx vercel login       # sekali saja
npx vercel --prod
```

`requirements.txt` berisi `fastapi` sehingga Vercel mendeteksi framework dan
merutekan semua request ke `app.py`; `vercel.json` memasang `maxDuration` 30
detik. Tidak butuh `DATABASE_URL`; butuh jaringan Yahoo Finance untuk harga
terkini (fallback dataset penelitian tetap bekerja tanpa jaringan).
