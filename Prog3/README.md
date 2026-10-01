# Dashboard Prediksi Saham

Dashboard FastAPI untuk melihat riwayat harga IDX dan menjalankan artefak LSTM target 50 hari yang sudah tersedia di direktori penelitian. Hasil yang ditampilkan adalah keluaran model regresi harga beserta sinyal berbasis ambang, bukan probabilitas keyakinan dan bukan rekomendasi investasi.

## Pemilihan ERD dan PDM

`Prompt.md` menjadi dasar PDM karena skemanya menelusuri alur penelitian dari saham, data OHLCV, indikator, berita, sentimen, data gabungan, versi model sampai hasil prediksi. Itu membuat asal fitur dan keluaran model lebih mudah dijelaskan dalam skripsi daripada PDM `Prompt_Mermaid.md` yang berfokus pada akun, watchlist, permintaan pengguna, dan hasil prediksi generik.

Skema di aplikasi mempertahankan entitas penelitian tersebut, lalu menambahkan `prediction_requests` untuk mencatat keberhasilan atau kegagalan inferensi dan `model_versions` untuk menautkan prediksi dengan artefak yang dipakai. Tabel berita, sentimen, dan data gabungan disiapkan untuk pipeline penelitian, namun aplikasi tidak mengarang berita atau sentimen. ERD pengguna, admin, dan watchlist dari prompt Mermaid dapat ditambahkan ketika autentikasi dan pengelolaan banyak pengguna benar-benar menjadi kebutuhan.

Relasi utama PDM: `stocks` memiliki banyak `stock_prices`, `technical_indicators`, `news_articles`, `news_sentiment`, `fused_market_data`, dan `prediction_requests`; satu permintaan menghasilkan paling banyak satu `predictions`; satu `model_versions` dapat menghasilkan banyak `predictions`. Kunci unik ticker-tanggal mencegah duplikasi harga, indikator, sentimen, dan fitur gabungan.

Rencana CI/CD pada kedua prompt masih berupa fase atau saran, bukan workflow yang dapat dijalankan. Aplikasi ini menambahkan GitHub Actions untuk memasang dependensi, menjalankan tes, dan membangun image Docker. Job deployment opsional memanggil Render Deploy Hook setelah tes di branch `main`. `RENDER_DEPLOY_HOOK` harus disimpan sebagai GitHub Actions secret dan layanan Render perlu dikonfigurasi agar menggunakan Dockerfile di folder `Prog3`. Untuk deployment, gunakan PostgreSQL atau penyimpanan persisten, bukan SQLite pada filesystem sementara.

Vercel cocok untuk frontend statis, tetapi API ini membawa model H5 dan runtime inferensi Python serta membutuhkan database persisten. Karena itu image Docker lebih sesuai untuk backend ini. Tautan Vercel yang diberikan tetap berguna sebagai referensi eksplorasi, bukan target deployment langsung aplikasi FastAPI.

## Batas model yang tersedia

Artefak yang ditemukan di `CODE` mencakup model LSTM regresi harga `Target_50` untuk sepuluh ticker serta dataset hasil penelitian. Model H5 menerima 50 harga penutupan sebagai satu fitur. Dataset latih yang tersedia berakhir pada 25 September 2023, sehingga performa model untuk data yang lebih baru belum tervalidasi. Skaler input dan target dibangun kembali dari dataset penelitian dengan pembagian acak `random_state=0`, mengikuti notebook asal karena artefak skaler tidak disimpan terpisah.

Kode `training_engine.py` memuat rancangan tiga jalur, tetapi tidak ditemukan artefak tersimpan untuk voter machine learning dan deep learning yang bisa dipakai oleh layanan ini. Oleh sebab itu, dashboard hanya menjalankan LSTM regresi yang tersedia. Indikator teknikal dihitung dan ditampilkan untuk konteks, tetapi tidak diklaim sebagai masukan model. Angka confidence dan sinyal ensemble tidak ditampilkan.

## Arah desain

Reading this as: dashboard penelitian saham untuk mahasiswa dan pembaca data pasar, dengan bahasa visual editorial yang tenang, dial ENERGY 1 / RHYTHM 2 / MOTION 1. Latar terang menjadi default untuk membaca angka dalam sesi panjang, sedangkan mode gelap tersedia sebagai pilihan. Tipografi sistem menghindari unduhan font eksternal dan angka memakai tabular figures. Komposisi memprioritaskan harga serta hasil inferensi; aksen hijau menandai data grafik, sementara warna sinyal hanya dipakai untuk makna BELI, JUAL, atau TAHAN. Jarak antarpanel memisahkan kelompok analisis, dan batas tipis mengelompokkan informasi tanpa membuatnya terlihat mengambang. Ticker, tanggal data, dan sumber harga menjadi motif identitas berulang karena keterlacakan data penting dalam penelitian. Tidak ada logo atau ilustrasi buatan.

## Menjalankan lokal

Gunakan Python 3.12 atau lebih baru dan pasang dependensi:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m uvicorn backend_api:app --reload
```

Buka `http://127.0.0.1:8000`. Jika Yahoo Finance tidak dapat dijangkau, aplikasi mencoba cache lokal lalu dataset penelitian. UI menandai dataset penelitian sebagai historis dan bukan harga terkini.

## Endpoint

- `GET /api/v1/health`: status database dan jumlah artefak model.
- `GET /api/v1/stocks`: ticker yang memiliki artefak penelitian.
- `GET /api/v1/market-data/{symbol}?period=1y`: riwayat harga, indikator, sumber data, dan tanggal data.
- `POST /api/v1/predictions/{symbol}`: jalankan LSTM target 50 hari dan simpan hasil.
- `GET /api/v1/predictions/latest/{symbol}`: hasil terbaru, atau `404` bila belum ada.
- `GET /api/v1/predictions/history/{symbol}`: riwayat hasil inferensi.

## CI/CD

Workflow berada di `.github/workflows/stock-signal.yml`. Pull request dan push menjalankan tes serta build image. Untuk mengaktifkan deploy otomatis ke Render, buat layanan Docker yang mengambil root proyek dari `Prog3`, nonaktifkan auto-deploy bawaan, lalu tambahkan secret repo `RENDER_DEPLOY_HOOK`. Tanpa secret itu, tes dan build tetap berjalan dan job deployment dilewati.
