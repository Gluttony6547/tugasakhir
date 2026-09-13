# PENGEMBANGAN APLIKASI WEB UNTUK SISTEM PENDUKUNG KEPUTUSAN PREDIKSI SINYAL BUY, HOLD, DAN SELL SAHAM SECARA NEAR REAL-TIME DENGAN FITUR TRAINING MODEL

**Development of a Web Application for a Near Real-Time Decision Support System for Buy, Hold, and Sell Stock Signal Prediction with Model Training Capability**

---

## ABSTRAK

Pergerakan harga saham memiliki karakteristik deret waktu, nonlinier, dan volatil sehingga proses pengambilan keputusan investasi memerlukan analisis terhadap berbagai informasi yang tersedia. Perkembangan Machine Learning dan Deep Learning memungkinkan data historis, indikator teknikal, serta sentimen berita digunakan untuk menghasilkan prediksi sinyal transaksi saham. Penelitian Wijaya et al. mengembangkan pendekatan multimodal yang menggabungkan indikator teknikal dan sentimen berita ke dalam arsitektur hybrid berbasis Recurrent Neural Network dan ensemble learning untuk menghasilkan sinyal Buy, Hold, dan Sell. Model tersebut menggunakan tiga jalur prediksi, yaitu jalur prediksi harga berbasis RNN, Machine Learning Voting, dan Deep Learning Voting, yang kemudian menghasilkan keputusan akhir melalui mekanisme voting. Penelitian tersebut memperoleh accuracy 89,34% dan F1-score 89,42% pada konfigurasi terbaik serta rata-rata accuracy 90,53% pada pengujian terhadap sepuluh saham.

Meskipun model prediksi telah dikembangkan dan dievaluasi, model tersebut perlu dioperasionalkan melalui sebuah sistem yang dapat digunakan secara lebih praktis. Oleh karena itu, penelitian ini berfokus pada pengembangan aplikasi web yang mengintegrasikan aplikasi Python sebagai prediction engine dan training engine dengan sumber data pasar terbaru. Sistem menyediakan fitur pengambilan data saham, preprocessing, prediksi sinyal Buy, Hold, dan Sell, training atau retraining model, pengelolaan versi model, penyimpanan riwayat prediksi, serta visualisasi hasil pada dashboard. Pengembangan sistem tidak berfokus pada penciptaan model machine learning baru, melainkan pada integrasi model yang telah tersedia ke dalam sistem pendukung keputusan berbasis web.

Metode pengembangan sistem meliputi analisis kebutuhan, analisis model Python, perancangan arsitektur sistem, perancangan basis data dan API, implementasi aplikasi web dan layanan machine learning, integrasi sumber data pasar, implementasi training dan prediction pipeline, serta pengujian fungsional, integrasi, kinerja, dan evaluasi model. Sistem yang dikembangkan diharapkan dapat menjembatani model prediksi saham berbasis Python dengan antarmuka web yang dapat digunakan untuk membantu pengguna memperoleh informasi sinyal saham berdasarkan data pasar terbaru.

**Kata kunci:** Sistem Pendukung Keputusan, Near Real-Time, Aplikasi Web, Python, Machine Learning, Prediksi Saham, Buy, Hold, Sell, Ensemble Learning.

---

# BAB I  
# PENDAHULUAN

## 1.1 Latar Belakang

Pasar saham merupakan salah satu instrumen keuangan yang memiliki tingkat ketidakpastian dan volatilitas tinggi. Perubahan harga saham dapat dipengaruhi oleh berbagai faktor, antara lain kondisi pasar, karakteristik perusahaan, perkembangan ekonomi, serta respons investor terhadap informasi dan berita. Karakteristik data saham yang bersifat time-series dan nonlinier menjadikan prediksi pergerakan saham sebagai permasalahan yang kompleks. Perkembangan teknologi Data Mining, Machine Learning, dan Deep Learning kemudian memberikan berbagai pendekatan untuk melakukan analisis serta prediksi terhadap data saham. Model dalam keluarga Recurrent Neural Network seperti Long Short-Term Memory (LSTM) dan Gated Recurrent Unit (GRU) banyak digunakan karena mampu menangani ketergantungan temporal pada data sekuensial.

Pendekatan prediksi saham tidak hanya dapat dilakukan berdasarkan pergerakan harga dan indikator teknikal. Informasi eksternal berupa berita dapat mencerminkan sentimen pasar yang tidak selalu dapat direpresentasikan melalui data harga. Penelitian terkait menunjukkan adanya kecenderungan untuk menggabungkan technical analysis dengan news sentiment agar model memperoleh representasi kondisi pasar yang lebih komprehensif. Sentimen berita dapat diproses melalui tahapan text mining untuk menghasilkan kelas atau nilai sentimen yang selanjutnya digunakan bersama data teknikal sebagai fitur prediksi.

Wijaya et al. mengembangkan suatu pendekatan multimodal untuk menghasilkan sinyal transaksi Buy, Sell, dan Hold melalui integrasi dua jenis informasi, yaitu data technical analysis yang berasal dari data transaksi saham dan news sentiment yang diperoleh melalui web crawling serta pseudolabelling. Penelitian tersebut menggunakan data harian selama lima tahun pada sepuluh saham blue-chip yang termasuk dalam kelompok LQ45. Dataset transaksi mencakup 1.219 hari perdagangan, yaitu periode 25 September 2018 sampai 25 September 2023. Data Open, High, Low, Close, dan Volume kemudian digunakan sebagai basis ekstraksi indikator teknikal.

Indikator teknikal yang digunakan meliputi Simple Moving Average (SMA), Exponential Moving Average (EMA), Relative Strength Index (RSI), Moving Average Convergence/Divergence (MACD), dan Bollinger Bands. SMA digunakan untuk mengidentifikasi kecenderungan tren melalui perataan harga, EMA memberikan bobot lebih besar terhadap harga terbaru, MACD merepresentasikan hubungan antara dua EMA, Bollinger Bands digunakan untuk merepresentasikan volatilitas, sedangkan RSI digunakan sebagai indikator momentum. Indikator-indikator tersebut kemudian digunakan sebagai bagian dari predictor vector.

Selain informasi teknikal, penelitian tersebut mengolah berita saham yang diperoleh dari empat portal berita keuangan di Indonesia, yaitu bisnis.com, CNBC Indonesia, investor.id, dan investasi.kontan.co.id. Judul berita diproses menggunakan tahapan character cleaning, case folding, tokenization, stopword removal, dan stemming. Untuk memperluas data berlabel, digunakan pendekatan SVM-based pseudolabelling dengan proses iteratif berdasarkan confidence threshold dari 95% hingga 80%. Validasi manual terhadap 150 data hasil machine labelling menghasilkan tingkat kesesuaian sebesar 94,6%.

Data teknikal dan sentimen kemudian digabungkan berdasarkan atribut tanggal. Penelitian memperhatikan aspek temporal untuk mencegah lookahead bias. Berita yang diterbitkan setelah waktu penutupan pasar pukul 16.00 dialokasikan ke hari perdagangan berikutnya sehingga informasi yang digunakan model sesuai dengan informasi yang secara realistis telah tersedia bagi investor. Hasil penggabungan tersebut kemudian digunakan dalam proses pembentukan fitur dan prediksi.

Tahap prediksi pada penelitian tersebut menggunakan arsitektur hybrid tiga jalur. Jalur pertama menggunakan model RNN untuk melakukan prediksi harga penutupan pada target day tertentu yang selanjutnya dikonversi menjadi sinyal Buy, Hold, atau Sell berdasarkan threshold perubahan harga. Jalur kedua menggunakan Machine Learning Voting Classifier yang terdiri atas Random Forest, AdaBoost, XGBoost, Support Vector Machine, dan K-Nearest Neighbors. Jalur ketiga menggunakan Deep Learning Voting dengan arsitektur LSTM, Bidirectional LSTM, GRU, dan Bidirectional GRU. Ketiga jalur tersebut menghasilkan prediksi yang kemudian digunakan dalam mekanisme voting untuk menentukan sinyal akhir.

Penentuan sinyal pada jalur prediksi harga didasarkan pada perubahan harga prediksi terhadap harga aktual dengan suatu threshold \(k\). Prediksi dikategorikan sebagai Buy apabila perubahan relatif melebihi \(k\), Sell apabila perubahan relatif lebih kecil dari \(-k\), dan Hold apabila perubahan berada di antara kedua threshold tersebut.

Dalam eksperimen, dataset dibagi secara kronologis dengan rasio 80% data training dan 20% data testing tanpa random shuffling. Preprocessing seperti scaling dan imputation hanya difit pada training data, sedangkan sliding window dibangun secara berurutan berdasarkan data masa lalu untuk mencegah leakage dan lookahead bias. Evaluasi menggunakan Accuracy, Precision, Recall, dan F1-score.

Pada target 50 hari, model ensemble memperoleh accuracy 89,34%, precision 89,67%, recall 89,34%, dan F1-score 89,42%. Performa tersebut lebih tinggi dibandingkan single LSTM dengan accuracy 86,48%, Random Forest sebesar 80,74%, dan SVM sebesar 73,77%. Model ensemble juga dibandingkan dengan pendekatan tradisional seperti SMA Crossover dan Buy-and-Hold.

Pengujian lebih lanjut pada sepuluh saham menunjukkan rata-rata accuracy sebesar 90,53%, precision sebesar 90,83%, dan F1-score sebesar 90,53%. Selain itu, walk-forward validation menghasilkan rata-rata out-of-sample accuracy sebesar 87,25% ± 2,15%, yang digunakan untuk menguji ketahanan model terhadap perubahan kondisi temporal dan market concept drift.

Hasil tersebut menunjukkan bahwa model yang tersedia memiliki potensi untuk digunakan sebagai komponen prediksi dalam sebuah sistem pendukung keputusan. Namun, model machine learning yang tersedia dalam bentuk aplikasi atau pipeline Python belum secara otomatis menjadi sebuah sistem yang mudah digunakan oleh pengguna. Pengguna tetap membutuhkan mekanisme untuk memilih saham, memperoleh data terbaru, menjalankan prediction pipeline, melakukan training atau retraining model, melihat hasil evaluasi model, serta memahami hasil prediksi melalui antarmuka yang terintegrasi.

Permasalahan tersebut menjadi semakin relevan ketika model digunakan untuk data pasar terbaru. Sistem perlu memiliki kemampuan memperoleh data baru secara berkala, melakukan preprocessing yang konsisten dengan model, menjalankan inferensi menggunakan model yang sesuai, kemudian menampilkan sinyal hasil prediksi melalui antarmuka web. Selain itu, apabila model dapat dilatih kembali menggunakan dataset baru, sistem memerlukan mekanisme untuk mengelola proses training, hasil evaluasi, dan versi model sehingga model yang digunakan untuk prediksi dapat diketahui secara jelas.

Berdasarkan permasalahan tersebut, penelitian ini mengembangkan aplikasi web sebagai Sistem Pendukung Keputusan yang mengintegrasikan model machine learning berbasis Python, sumber data pasar terbaru, dan antarmuka pengguna berbasis web. Sistem menyediakan fitur pengambilan data pasar, training atau retraining model, prediksi sinyal Buy, Hold, dan Sell, penyimpanan hasil prediksi, serta visualisasi informasi pada dashboard.

Dengan demikian, kontribusi penelitian tidak diarahkan pada pengembangan algoritma prediksi baru, melainkan pada **implementasi dan integrasi model prediksi yang telah tersedia menjadi sistem berbasis web yang operasional**. Sistem diharapkan dapat menjadi lapisan aplikasi yang menjembatani model machine learning dengan pengguna sehingga proses training dan prediksi dapat dilakukan melalui antarmuka yang lebih terstruktur.

---

## 1.2 Identifikasi Masalah

Berdasarkan latar belakang tersebut, permasalahan yang diidentifikasi dalam penelitian ini adalah:

1. Model prediksi saham berbasis Python belum terintegrasi secara optimal ke dalam aplikasi web yang dapat digunakan melalui antarmuka pengguna.
2. Proses memperoleh data pasar, melakukan preprocessing, dan menjalankan prediksi masih memerlukan pipeline Python.
3. Penggunaan data pasar yang terus berubah membutuhkan mekanisme pengambilan data terbaru dan pengelolaan data secara terstruktur.
4. Belum tersedia antarmuka berbasis web yang menyediakan proses training atau retraining model.
5. Hasil prediksi perlu disajikan dalam bentuk dashboard yang dapat memberikan informasi sinyal Buy, Hold, atau Sell secara jelas.
6. Pengelolaan versi model, hasil training, dan riwayat prediksi diperlukan agar proses penggunaan model dapat ditelusuri.
7. Integrasi antara aplikasi web, backend, layanan Python, basis data, dan sumber data pasar membutuhkan arsitektur yang terstruktur.

---

## 1.3 Rumusan Masalah

Rumusan masalah dalam penelitian ini adalah:

1. Bagaimana merancang dan mengembangkan aplikasi web yang dapat mengintegrasikan model prediksi saham berbasis Python sebagai komponen Sistem Pendukung Keputusan?
2. Bagaimana mengintegrasikan sumber data pasar terbaru dengan preprocessing dan model prediksi sehingga sistem dapat menghasilkan sinyal Buy, Hold, dan Sell?
3. Bagaimana mengimplementasikan fitur training atau retraining model melalui aplikasi web?
4. Bagaimana merancang mekanisme pengelolaan model dan hasil training sehingga model yang digunakan untuk prediksi dapat diidentifikasi dan ditelusuri?
5. Bagaimana merancang dashboard untuk menyajikan informasi harga, indikator, hasil prediksi, dan sinyal Buy, Hold, atau Sell?
6. Bagaimana mengevaluasi fungsionalitas, integrasi, performa, dan kemampuan sistem dalam menjalankan proses training dan prediction?

---

## 1.4 Batasan Masalah

Batasan penelitian ditetapkan agar penelitian terfokus pada pengembangan sistem:

1. Model machine learning yang digunakan berasal dari implementasi Python yang telah tersedia dan/atau mengacu pada model penelitian Wijaya et al. (2026).
2. Penelitian tidak berfokus pada penciptaan arsitektur machine learning baru.
3. Sistem menghasilkan tiga kelas sinyal, yaitu **Buy, Hold, dan Sell**.
4. Sistem menggunakan data pasar terbaru yang tersedia melalui sumber data atau API yang dipilih.
5. Data dan preprocessing yang digunakan harus kompatibel dengan pipeline model Python.
6. Sistem mendukung proses training/retraining melalui antarmuka web dengan konfigurasi yang dibatasi sesuai kemampuan model yang tersedia.
7. Model yang berhasil dilatih dapat disimpan sebagai versi model untuk digunakan dalam proses prediksi.
8. Sistem tidak melakukan eksekusi jual atau beli saham secara otomatis.
9. Hasil prediksi diperlakukan sebagai informasi pendukung keputusan, bukan jaminan keuntungan investasi.
10. Istilah **near real-time** digunakan untuk menunjukkan bahwa data dan prediksi diperbarui berdasarkan ketersediaan data terbaru dari sumber data, bukan menjamin pemrosesan tick-by-tick.
11. Evaluasi model tetap mengikuti karakteristik model yang tersedia. Data pasar dan konfigurasi baru tidak diasumsikan menghasilkan performa yang identik dengan hasil eksperimen pada paper.
12. Jumlah dan format fitur final akan mengikuti implementasi Python yang telah tersedia setelah dilakukan tahap analisis dan validasi pipeline.

---

## 1.5 Tujuan Penelitian

### 1.5.1 Tujuan Umum

Mengembangkan aplikasi web Sistem Pendukung Keputusan yang mengintegrasikan model prediksi saham berbasis Python dengan sumber data pasar terbaru untuk menghasilkan sinyal Buy, Hold, dan Sell serta menyediakan fitur training atau retraining model.

### 1.5.2 Tujuan Khusus

1. Mengidentifikasi kebutuhan dan interface dari model Python yang telah tersedia.
2. Merancang arsitektur integrasi antara frontend, backend, layanan Python, basis data, dan sumber data pasar.
3. Mengimplementasikan pengambilan data pasar terbaru untuk kebutuhan prediksi.
4. Mengimplementasikan pipeline preprocessing yang kompatibel dengan model.
5. Mengimplementasikan fitur prediction menggunakan model Python.
6. Mengimplementasikan fitur training atau retraining model melalui aplikasi web.
7. Mengimplementasikan model versioning dan penyimpanan metrik hasil training.
8. Mengembangkan dashboard yang menampilkan data harga dan sinyal Buy, Hold, dan Sell.
9. Mengevaluasi sistem dari aspek fungsionalitas, integrasi, performa, dan konsistensi output prediksi.

---

## 1.6 Manfaat Penelitian

### 1.6.1 Manfaat Akademis

Penelitian ini dapat menjadi referensi implementasi integrasi model Machine Learning dan Deep Learning ke dalam aplikasi web serta penerapan Sistem Pendukung Keputusan berbasis data pasar.

### 1.6.2 Manfaat Teknis

Penelitian menghasilkan arsitektur yang menghubungkan aplikasi web dengan aplikasi Python sehingga proses training dan inference dapat dijalankan melalui suatu sistem terintegrasi.

### 1.6.3 Manfaat Pengguna

Pengguna memperoleh antarmuka untuk melihat informasi saham, data pasar terbaru, hasil prediksi, dan sinyal Buy, Hold, atau Sell tanpa harus berinteraksi langsung dengan kode Python.

### 1.6.4 Manfaat Pengembangan Selanjutnya

Sistem dapat menjadi dasar pengembangan fitur lanjutan seperti model versioning yang lebih kompleks, walk-forward retraining, notifikasi prediksi, monitoring model, atau integrasi additional data source.

---

## 1.7 Kontribusi Penelitian

Kontribusi utama penelitian ini adalah:

1. **Web-based Model Integration**  
   Mengintegrasikan model prediksi saham berbasis Python ke dalam aplikasi web.

2. **Market Data Integration**  
   Mengintegrasikan sumber data pasar terbaru sebagai input bagi prediction pipeline.

3. **Training Capability**  
   Menyediakan fasilitas untuk menjalankan training/retraining model melalui web application.

4. **Model Management**  
   Menyediakan pengelolaan informasi versi model, waktu training, konfigurasi, status, dan metrik evaluasi.

5. **Decision Support Dashboard**  
   Menyajikan hasil prediksi dalam bentuk dashboard dengan sinyal Buy, Hold, dan Sell.

6. **System Evaluation**  
   Melakukan evaluasi terhadap fungsionalitas, integrasi, dan performa aplikasi serta kompatibilitas output model.

Kontribusi tersebut membedakan penelitian ini dari paper utama yang berfokus pada pengembangan dan evaluasi arsitektur prediksi. Paper berfokus pada integrasi multimodal feature dan hybrid ensemble untuk memperoleh prediksi yang lebih stabil, sedangkan penelitian ini berfokus pada **operasionalisasi model tersebut sebagai aplikasi web**.

---

## 1.8 Metodologi Penelitian

Metodologi penelitian terdiri dari beberapa tahapan berikut.

### 1.8.1 Studi Literatur

Studi literatur dilakukan untuk mempelajari:

- Sistem Pendukung Keputusan;
- prediksi saham;
- time-series forecasting;
- technical analysis;
- sentiment analysis;
- Machine Learning;
- Deep Learning;
- RNN, LSTM, dan GRU;
- Ensemble Learning;
- aplikasi web;
- REST API;
- integrasi Python dengan web application;
- model training dan model serving.

Paper utama menjadi dasar untuk memahami model prediksi, dataset, feature engineering, labeling, ensemble voting, dan metode evaluasinya.

### 1.8.2 Analisis Model Python

Tahap ini dilakukan untuk memahami aplikasi Python yang telah tersedia.

Analisis meliputi:

- struktur source code;
- input dataset;
- preprocessing;
- feature engineering;
- format model;
- parameter model;
- mekanisme training;
- mekanisme inference;
- output prediksi;
- dependensi Python;
- waktu eksekusi;
- format model hasil training.

Tahap ini penting karena implementasi aplikasi web harus mengikuti pipeline Python aktual, bukan hanya deskripsi pada paper.

### 1.8.3 Analisis Kebutuhan Sistem

Kebutuhan sistem dibagi menjadi:

**Functional Requirements**

- pengguna dapat melihat daftar saham;
- pengguna dapat melihat harga terbaru;
- pengguna dapat menjalankan prediksi;
- pengguna dapat melihat sinyal;
- pengguna dapat melihat histori prediksi;
- pengguna dapat melakukan training;
- pengguna dapat melihat status training;
- pengguna dapat melihat metrik hasil training;
- pengguna dapat memilih model yang tersedia untuk prediksi.

**Non-Functional Requirements**

- respons aplikasi;
- availability;
- scalability;
- maintainability;
- security;
- usability;
- reliability.

### 1.8.4 Perancangan Sistem

Sistem dirancang menggunakan arsitektur berlapis:

```text
+---------------------------+
|       Web Frontend        |
| Dashboard / Training UI  |
+-------------+-------------+
              |
              | REST API
              v
+---------------------------+
|       Web Backend         |
| API / Auth / Orchestration|
+------+--------------+-----+
       |              |
       |              |
       v              v
+-------------+   +----------------+
| Market Data |   | Python ML      |
| Service     |   | Service        |
+-------------+   |                |
                  | Training       |
                  | Preprocessing  |
                  | Inference      |
                  +-------+--------+
                          |
                          v
                  +---------------+
                  | Model Storage |
                  +---------------+

              |
              v
       +---------------+
       |   Database    |
       +---------------+
```

### 1.8.5 Implementasi

Implementasi meliputi:

- frontend;
- backend;
- REST API;
- Python service;
- market data integration;
- database;
- training service;
- prediction service;
- model storage;
- dashboard.

### 1.8.6 Pengujian

Pengujian meliputi:

1. Functional Testing.
2. Integration Testing.
3. API Testing.
4. Performance Testing.
5. Training Pipeline Testing.
6. Prediction Pipeline Testing.
7. Model Evaluation.

---

## 1.9 Sistematika Penulisan

### BAB I PENDAHULUAN

Berisi latar belakang, identifikasi masalah, rumusan masalah, batasan masalah, tujuan, manfaat, kontribusi, metodologi penelitian, dan sistematika penulisan.

### BAB II TINJAUAN PUSTAKA

Berisi teori dan penelitian terkait mengenai Sistem Pendukung Keputusan, saham, technical analysis, sentiment analysis, Machine Learning, Deep Learning, RNN, LSTM, GRU, ensemble learning, web application, API, dan model serving.

### BAB III ANALISIS DAN PERANCANGAN

Berisi analisis model Python, analisis kebutuhan sistem, arsitektur sistem, use case, activity diagram, sequence diagram, rancangan database, API, prediction pipeline, training pipeline, dan rancangan antarmuka.

### BAB IV IMPLEMENTASI DAN PENGUJIAN

Berisi implementasi aplikasi dan pengujian sistem, termasuk integrasi model Python, market data, training, prediction, dashboard, dan evaluasi performa.

### BAB V KESIMPULAN DAN SARAN

Berisi kesimpulan hasil penelitian dan saran untuk pengembangan berikutnya.

---

# BAB II  
# TINJAUAN PUSTAKA

## 2.1 Sistem Pendukung Keputusan

Sistem Pendukung Keputusan merupakan sistem yang menyediakan informasi dan analisis untuk membantu pengguna dalam mengambil keputusan. Dalam penelitian ini, sistem digunakan sebagai media untuk menyajikan hasil analisis model machine learning dalam bentuk sinyal Buy, Hold, dan Sell.

Sistem tidak mengambil keputusan transaksi secara otomatis. Sistem memberikan informasi hasil prediksi yang dapat digunakan pengguna sebagai salah satu masukan dalam proses pengambilan keputusan.

---

## 2.2 Prediksi Saham

Prediksi saham merupakan proses memperkirakan pergerakan atau kondisi harga saham pada periode mendatang berdasarkan informasi historis dan variabel yang relevan. Data saham bersifat time-series sehingga hubungan temporal menjadi aspek penting dalam pemodelannya.

Paper utama menekankan bahwa karakteristik time-series dan non-linear pada data saham menjadikan prediksi sebagai permasalahan kompleks. RNN, LSTM, dan GRU digunakan karena kemampuan dalam menangani data sekuensial.

---

## 2.3 Technical Analysis

Technical analysis menggunakan informasi historis harga dan volume untuk mengidentifikasi pola serta kecenderungan pasar.

Pada model dasar, data transaksi mencakup:

- Open;
- High;
- Low;
- Close;
- Volume.

Data tersebut digunakan untuk membentuk berbagai indikator seperti SMA, EMA, RSI, MACD, dan Bollinger Bands.

---

## 2.4 Simple Moving Average

SMA digunakan untuk menghitung rata-rata harga dalam periode tertentu:

\[
SMA_t =
\frac{1}{n}\sum_{i=0}^{n-1}P_{t-i}
\]

dengan \(P_t\) sebagai harga penutupan dan \(n\) sebagai periode pengamatan. Paper menggunakan beberapa periode seperti 5, 10, 20, dan 50 hari.

---

## 2.5 Exponential Moving Average

EMA memberikan bobot yang lebih besar terhadap data terbaru:

\[
EMA_t=P_t\alpha+EMA_{t-1}(1-\alpha)
\]

dengan:

\[
\alpha=\frac{2}{n+1}
\]

Paper menggunakan EMA pada beberapa periode seperti 5, 10, 20, dan 50 hari.

---

## 2.6 MACD

MACD digunakan untuk melihat hubungan antara dua EMA:

\[
MACD_t=EMA_{fast,t}-EMA_{slow,t}
\]

Dalam implementasi paper, MACD menggunakan periode 12 dan 26 hari serta MACD Signal 9 hari.

---

## 2.7 Bollinger Bands

Bollinger Bands digunakan untuk menggambarkan volatilitas:

\[
BB_n=MA_n\pm2\sigma_n
\]

yang menghasilkan upper band, middle band, dan lower band.

---

## 2.8 Relative Strength Index

RSI merupakan momentum oscillator yang dihitung berdasarkan perbandingan rata-rata kenaikan dan penurunan:

\[
RS=\frac{Average(Gain)}{Average(Loss)}
\]

\[
RSI=100-\frac{100}{1+RS}
\]

Paper menggunakan periode RSI 14 hari.

---

## 2.9 News Sentiment Analysis

Sentimen berita digunakan untuk menangkap informasi eksternal yang tidak secara langsung tercermin dalam indikator teknikal. Paper memproses judul berita melalui preprocessing, pseudolabelling berbasis SVM, dan Count Vectorizer. Hasil sentimen direpresentasikan sebagai informasi negatif, positif, netral, serta pengukuran agregat sentimen.

---

## 2.10 Recurrent Neural Network

RNN merupakan model neural network yang dirancang untuk memproses data sekuensial. LSTM dan GRU merupakan pengembangan dari RNN yang dirancang untuk menangani permasalahan ketergantungan jangka panjang pada data.

Paper menggunakan RNN, LSTM, GRU, Bidirectional LSTM, dan Bidirectional GRU sebagai bagian dari arsitektur prediksi.

---

## 2.11 Ensemble Learning

Ensemble learning menggabungkan beberapa model untuk menghasilkan keputusan yang lebih stabil dibandingkan satu model. Paper menggunakan beberapa algoritma Machine Learning serta beberapa arsitektur Deep Learning dalam voting mechanism.

---

## 2.12 Buy, Hold, dan Sell Signal

Paper menggunakan tiga label:

| Label | Sinyal |
|---:|---|
| 0 | Sell |
| 1 | Hold |
| 2 | Buy |



Untuk jalur prediksi harga, hasil prediksi harga dibandingkan dengan harga saat ini menggunakan threshold \(k\):

\[
\frac{\hat{P}_{t+n}-P_t}{P_t}>k
\Rightarrow Buy
\]

\[
\frac{\hat{P}_{t+n}-P_t}{P_t}<-k
\Rightarrow Sell
\]

\[
-k\leq
\frac{\hat{P}_{t+n}-P_t}{P_t}
\leq k
\Rightarrow Hold
\]



---

## 2.13 Sliding Window

Model menggunakan pendekatan sliding window. Ukuran window disesuaikan dengan target day. Contohnya, target 50 hari menggunakan data 50 hari sebelumnya sebagai input sequence.

---

## 2.14 Data Leakage dan Lookahead Bias

Pada data time-series, data masa depan tidak boleh digunakan dalam pembentukan input untuk memprediksi masa lalu. Oleh karena itu, paper melakukan pembagian data berdasarkan urutan waktu dan tidak melakukan random shuffling. Scaling dan imputation juga hanya difit menggunakan training set.

Prinsip tersebut juga harus diterapkan pada implementasi aplikasi web.

---

## 2.15 Aplikasi Web

Aplikasi web digunakan sebagai antarmuka untuk mengakses fungsi sistem dari browser. Dalam penelitian ini aplikasi web berfungsi sebagai orchestrator yang menghubungkan pengguna dengan backend, market data service, database, serta Python machine learning service.

---

## 2.16 REST API

REST API digunakan sebagai mekanisme komunikasi antara frontend dan backend serta antara backend dan layanan Python.

Contoh endpoint:

```text
GET  /api/stocks
GET  /api/stocks/{symbol}
GET  /api/stocks/{symbol}/market-data

POST /api/predictions
GET  /api/predictions/{symbol}

POST /api/training
GET  /api/training/{job_id}

GET  /api/models
GET  /api/models/{model_id}
```

---

## 2.17 Model Training dan Model Versioning

Training model pada penelitian ini diperlakukan sebagai proses terpisah dari inference. Model yang telah selesai dilatih dan memenuhi kriteria evaluasi dapat disimpan dengan informasi:

- model ID;
- model version;
- stock;
- training date;
- dataset;
- parameter;
- accuracy;
- precision;
- recall;
- F1-score;
- status.

Hal ini memungkinkan sistem mengetahui model yang digunakan pada sebuah hasil prediksi.

---

# BAB III  
# ANALISIS DAN PERANCANGAN SISTEM

## 3.1 Gambaran Umum Sistem

Sistem yang dikembangkan terdiri atas empat komponen utama:

1. Web Frontend.
2. Web Backend.
3. Python Machine Learning Service.
4. Market Data Service.

Komponen tersebut terhubung dengan database dan penyimpanan model.

### Alur Utama

```text
User
 ↓
Web Dashboard
 ↓
Backend API
 ├──────────────→ Market Data
 │
 ├──────────────→ Python Prediction
 │
 └──────────────→ Database
                         ↑
                   Training Result
                         ↑
                   Python Training
```

---

## 3.2 Aktor Sistem

### User

User dapat:

- melihat saham;
- melihat data pasar;
- menjalankan prediksi;
- melihat hasil prediksi;
- melihat histori prediksi;
- menjalankan training;
- melihat status training;
- melihat metrik model.

### Administrator

Apabila dibutuhkan, administrator dapat:

- mengelola user;
- mengelola model;
- mengelola konfigurasi;
- memantau training job;
- mengelola sumber data.

---

## 3.3 Modul Dashboard

Dashboard merupakan halaman utama sistem.

Informasi yang ditampilkan:

```text
Stock Symbol
Latest Price
Price Change
Volume
Prediction Signal
Prediction Time
Model Version
Confidence / Score*
```

\* Hanya ditampilkan apabila model Python memang menghasilkan nilai confidence/probability yang valid.

Dashboard juga menampilkan:

- price chart;
- indikator teknikal;
- informasi sentimen;
- histori sinyal;
- informasi model.

---

## 3.4 Modul Prediction

Flow:

```text
User memilih saham
        ↓
Request prediction
        ↓
Backend mengambil data terbaru
        ↓
Validasi data
        ↓
Feature generation
        ↓
Preprocessing
        ↓
Python Model
        ↓
Ensemble Voting
        ↓
Buy / Hold / Sell
        ↓
Simpan Prediction Result
        ↓
Dashboard
```

---

## 3.5 Modul Training

Flow:

```text
User membuka Training
        ↓
Memilih saham
        ↓
Memilih dataset/periode
        ↓
Mengatur konfigurasi
        ↓
Submit Training Job
        ↓
Backend
        ↓
Python Training Service
        ↓
Preprocessing
        ↓
Model Training
        ↓
Evaluation
        ↓
Model Version
        ↓
Save Model
        ↓
Training Result
```

---

## 3.6 Model Training Configuration

Sebagai konfigurasi awal, implementasi dapat mengikuti parameter yang dilaporkan paper, sepanjang kompatibel dengan kode Python:

- RNN/LSTM/GRU: 3 layer;
- 96 units per layer;
- dropout 20%;
- maksimum 50 epochs;
- batch size 32;
- validation split 10%;
- early stopping patience 10;
- random seed 42.

Parameter tersebut berasal dari konfigurasi eksperimental paper.

Untuk model klasik, paper melaporkan antara lain Random Forest dengan 100 estimators, SVM dengan \(C=1.0\), KNN dengan 5 neighbors, AdaBoost dengan 50 estimators dan learning rate 1.0, serta XGBoost dengan 100 estimators, learning rate 0.1, dan max depth 6.

Parameter tersebut dapat dijadikan **default configuration**, bukan parameter wajib yang tidak boleh diubah.

---

## 3.7 Prediction Engine

Python service bertanggung jawab terhadap:

```text
Input
 ↓
Data Validation
 ↓
Feature Engineering
 ↓
Scaling / Imputation
 ↓
Sliding Window
 ↓
RNN Path
 ↓
ML Voting Path
 ↓
DL Voting Path
 ↓
Majority Voting
 ↓
Signal
```

Dalam paper, ensemble akhir menggabungkan tiga jalur utama: price prediction, Machine Learning Voting, dan Deep Learning Voting. Kelas dengan jumlah voting terbanyak menjadi sinyal akhir.

---

## 3.8 Perancangan Database

### Tabel `users`

| Field | Tipe |
|---|---|
| id | UUID |
| name | VARCHAR |
| email | VARCHAR |
| password_hash | VARCHAR |
| role | VARCHAR |
| created_at | TIMESTAMP |

### Tabel `stocks`

| Field | Tipe |
|---|---|
| id | UUID |
| symbol | VARCHAR |
| company_name | VARCHAR |
| exchange | VARCHAR |
| created_at | TIMESTAMP |

### Tabel `market_data`

| Field | Tipe |
|---|---|
| id | BIGINT |
| stock_id | UUID |
| timestamp | TIMESTAMP |
| open | DECIMAL |
| high | DECIMAL |
| low | DECIMAL |
| close | DECIMAL |
| volume | BIGINT |

### Tabel `models`

| Field | Tipe |
|---|---|
| id | UUID |
| name | VARCHAR |
| version | VARCHAR |
| stock | VARCHAR |
| model_path | TEXT |
| status | VARCHAR |
| accuracy | DECIMAL |
| precision | DECIMAL |
| recall | DECIMAL |
| f1_score | DECIMAL |
| trained_at | TIMESTAMP |

### Tabel `training_jobs`

| Field | Tipe |
|---|---|
| id | UUID |
| model_id | UUID |
| stock | VARCHAR |
| dataset | TEXT |
| status | VARCHAR |
| started_at | TIMESTAMP |
| completed_at | TIMESTAMP |
| metrics | JSON |

### Tabel `predictions`

| Field | Tipe |
|---|---|
| id | UUID |
| stock_id | UUID |
| model_id | UUID |
| timestamp | TIMESTAMP |
| signal | VARCHAR |
| predicted_price | DECIMAL |
| confidence | DECIMAL |
| input_timestamp | TIMESTAMP |

---

## 3.9 Model Versioning

Contoh:

```text
Model
 ├── ADRO
 │    ├── v1.0
 │    ├── v1.1
 │    └── v1.2
 │
 ├── ANTM
 │    ├── v1.0
 │    └── v1.1
 │
 └── TLKM
      └── v1.0
```

Sistem harus dapat mengetahui:

```text
Prediction
       ↓
Model ID
       ↓
Model Version
       ↓
Training Dataset
       ↓
Training Metrics
```

---

## 3.10 Perancangan API

### Get Stocks

```http
GET /api/stocks
```

### Get Market Data

```http
GET /api/stocks/BBCA/market-data
```

### Run Prediction

```http
POST /api/predictions
```

Request:

```json
{
  "symbol": "BBCA",
  "model_id": "model-001"
}
```

Response:

```json
{
  "symbol": "BBCA",
  "signal": "BUY",
  "model_version": "v1.2",
  "timestamp": "2026-09-09T14:00:00+07:00"
}
```

### Start Training

```http
POST /api/training
```

Request:

```json
{
  "symbol": "BBCA",
  "dataset": "latest",
  "target_days": 50
}
```

Response:

```json
{
  "job_id": "job-001",
  "status": "QUEUED"
}
```

### Training Status

```http
GET /api/training/job-001
```

---

## 3.11 Mekanisme Near Real-Time

Near real-time prediction dirancang dengan mekanisme:

```text
Market Data Provider
        ↓
Periodic Update / Manual Refresh
        ↓
Backend
        ↓
Data Validation
        ↓
Feature Calculation
        ↓
Prediction
        ↓
Dashboard Update
```

Interval pembaruan bergantung pada kemampuan sumber data yang dipilih.

Sistem tidak menganggap data tersedia secara kontinu apabila provider tidak menyediakan streaming data.

---

## 3.12 Pencegahan Lookahead Bias

Pipeline aplikasi harus mempertahankan prinsip temporal sebagaimana penelitian dasar:

```text
Past Data ───────► Training
                      │
                      ▼
                 Trained Model
                      │
Latest Available ────► Prediction
```

Data masa depan tidak boleh masuk ke feature engineering atau preprocessing untuk periode sebelumnya.

Paper secara eksplisit menerapkan chronological split, training-only preprocessing fitting, sequential sliding window, serta pengaturan berita setelah pukul 16.00 ke trading day berikutnya untuk menghindari lookahead bias.

---

# BAB IV  
# RENCANA IMPLEMENTASI DAN PENGUJIAN

## 4.1 Implementasi Frontend

Frontend bertanggung jawab untuk:

- dashboard;
- stock selection;
- prediction page;
- training page;
- model management;
- training history;
- prediction history.

---

## 4.2 Implementasi Backend

Backend bertanggung jawab untuk:

- authentication;
- authorization;
- REST API;
- request validation;
- data orchestration;
- prediction orchestration;
- training orchestration;
- database interaction;
- model management.

---

## 4.3 Implementasi Python Machine Learning Service

Python service bertanggung jawab atas:

```text
Data Input
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
Model Loading
    ↓
Prediction
```

dan:

```text
Dataset
    ↓
Preprocessing
    ↓
Training
    ↓
Evaluation
    ↓
Model Serialization
```

---

## 4.4 Implementasi Market Data Service

Komponen ini menangani:

- pengambilan data harga;
- validasi data;
- timestamp;
- penyimpanan;
- update data;
- error handling.

---

## 4.5 Pengujian Fungsional

Skenario minimal:

| No | Fitur | Pengujian |
|---:|---|---|
| 1 | Dashboard | Dashboard dapat dibuka |
| 2 | Stock Selection | User dapat memilih saham |
| 3 | Market Data | Data terbaru berhasil diperoleh |
| 4 | Prediction | Prediksi berhasil dijalankan |
| 5 | Signal | Buy/Hold/Sell ditampilkan |
| 6 | Training | Training dapat dimulai |
| 7 | Training Status | Status job dapat dilihat |
| 8 | Model | Model hasil training tersimpan |
| 9 | History | Riwayat prediksi tersedia |
| 10 | Model Metrics | Metrik training dapat ditampilkan |

---

## 4.6 Integration Testing

Pengujian integrasi mencakup:

```text
Frontend ↔ Backend
Backend ↔ Database
Backend ↔ Market Data
Backend ↔ Python
Python ↔ Model
Python ↔ Model Storage
```

---

## 4.7 Performance Testing

Parameter yang diukur:

### Prediction Latency

\[
T_{prediction}
=
T_{response}-T_{request}
\]

### Training Duration

\[
T_{training}
=
T_{completed}-T_{started}
\]

Selain itu dapat diukur:

- API response time;
- data retrieval time;
- feature processing time;
- inference time;
- memory usage;
- CPU usage;
- training execution time.

---

## 4.8 Evaluasi Model

Evaluasi model menggunakan metrik yang juga digunakan penelitian dasar:

### Accuracy

\[
Accuracy=
\frac{TP+TN}{TP+TN+FP+FN}
\]

### Precision

\[
Precision=
\frac{TP}{TP+FP}
\]

### Recall

\[
Recall=
\frac{TP}{TP+FN}
\]

### F1-score

\[
F1=
2\times
\frac{Precision\times Recall}
{Precision+Recall}
\]



Karena penelitian menghasilkan tiga kelas, implementasi evaluasi sebaiknya juga mempertimbangkan **macro-averaged metric** agar performa masing-masing kelas Buy, Hold, dan Sell dapat terlihat.

Paper sendiri menggunakan evaluasi lintas horizon dan menunjukkan bahwa target 50 hari memberikan performa terbaik pada eksperimennya, sementara walk-forward validation menghasilkan rata-rata 87,25% ± 2,15%.

---

# BAB V  
# RENCANA HASIL YANG DIHARAPKAN

Hasil akhir penelitian diharapkan berupa sebuah aplikasi web yang memiliki:

### 1. Dashboard Saham

```text
┌─────────────────────────────────────────┐
│ Stock: BBCA                             │
│                                         │
│ Latest Price       Rp xxxx              │
│ Change             +x.xx%               │
│ Volume             xxxxx                │
│                                         │
│            ┌──────────────┐             │
│            │     BUY      │             │
│            └──────────────┘             │
│                                         │
│ Model: Ensemble v1.2                    │
│ Updated: 14:30                          │
└─────────────────────────────────────────┘
```

### 2. Training Interface

```text
Stock         : BBCA
Dataset       : Latest
Target Day    : 50
Epoch         : 50
Batch Size    : 32

              [ START TRAINING ]

Status: TRAINING

Accuracy : --
F1 Score : --
```

Setelah selesai:

```text
Status      : COMPLETED
Model       : BBCA-v1.2
Accuracy    : 89.xx%
Precision   : 90.xx%
Recall      : 89.xx%
F1 Score    : 89.xx%
```

### 3. Prediction History

| Time | Stock | Signal | Model |
|---|---|---|---|
| 10:30 | BBCA | Buy | v1.2 |
| 11:30 | BBCA | Hold | v1.2 |
| 13:30 | BBCA | Buy | v1.2 |

### 4. Model Management

| Model | Version | Accuracy | F1 | Status |
|---|---|---:|---:|---|
| BBCA Ensemble | v1.0 | xx% | xx% | Archived |
| BBCA Ensemble | v1.1 | xx% | xx% | Archived |
| BBCA Ensemble | v1.2 | xx% | xx% | Active |

---

# BAB VI  
# KERANGKA PENELITIAN

Kerangka penelitian dapat digambarkan sebagai berikut:

```text
                 STUDI LITERATUR
                       │
                       ▼
            ANALISIS PAPER DAN
              MODEL PYTHON
                       │
                       ▼
             ANALISIS KEBUTUHAN
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
   Kebutuhan Web              Kebutuhan ML
          │                         │
          └────────────┬────────────┘
                       ▼
              PERANCANGAN SISTEM
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
    Frontend        Backend        Python ML
        │              │              │
        └──────────────┼──────────────┘
                       ▼
               MARKET DATA API
                       │
                       ▼
             TRAINING / PREDICTION
                       │
                       ▼
                DATABASE + MODEL
                       │
                       ▼
                  DASHBOARD
                       │
                       ▼
                    TESTING
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
       Functional  Integration Performance
                       │
                       ▼
               MODEL EVALUATION
                       │
                       ▼
                   KESIMPULAN
```

---

# 7. State of the Art dan Research Gap

Penelitian Wijaya et al. berfokus pada peningkatan kualitas prediksi dengan menggabungkan technical indicators dan news sentiment serta menggunakan hybrid ensemble architecture. Pendekatan tersebut menggabungkan tiga prediction paths dan menghasilkan performa yang lebih tinggi dibandingkan beberapa baseline individual.

Penelitian sebelumnya secara umum masih menghadapi keterbatasan penggunaan satu model atau satu jenis fitur. Paper utama secara eksplisit mengidentifikasi dua permasalahan tersebut, yaitu keterbatasan single-feature dan single-model, kemudian mengatasinya melalui multimodal feature fusion dan ensemble method.

Namun, fokus paper tersebut adalah **pengembangan serta eksperimen terhadap model prediksi**, bukan pembangunan aplikasi web yang mengelola siklus penggunaan model dari pengambilan data, training, penyimpanan model, inference, hingga penyajian hasil melalui dashboard.

Oleh karena itu, research gap penelitian ini terletak pada **lapisan implementasi sistem**:

| Penelitian Dasar | Penelitian Ini |
|---|---|
| Mengembangkan model prediksi | Mengintegrasikan model yang tersedia |
| Eksperimen ML/DL | Aplikasi web + ML |
| Dataset penelitian | Data pasar terbaru + dataset training |
| Offline prediction pipeline | Web-accessible prediction |
| Training melalui pipeline penelitian | Training melalui UI |
| Evaluasi model | Evaluasi model + sistem |
| Fokus model | Fokus system engineering |

Dengan demikian, penelitian ini bukan duplikasi terhadap pengembangan model pada paper, melainkan pengembangan sistem yang menjadikan model tersebut dapat digunakan sebagai komponen Sistem Pendukung Keputusan.

---

# 8. Posisi Model Python dalam Sistem

Model Python diperlakukan sebagai **Machine Learning Engine**.

```text
                    WEB APPLICATION
                          │
                          ▼
                     BACKEND API
                          │
                          ▼
                 PYTHON ML SERVICE
                    /           \
                   /             \
            TRAINING           PREDICTION
               │                    │
               ▼                    ▼
         MODEL STORAGE         PREDICTION
               │                    │
               └────────┬───────────┘
                        ▼
                    DATABASE
                        │
                        ▼
                     WEB UI
```

Dengan pendekatan ini, model Python dapat dikembangkan secara relatif independen dari frontend.

---

# 9. Rencana Teknologi

Teknologi yang diusulkan:

| Komponen | Teknologi |
|---|---|
| Frontend | React / Next.js |
| Backend | FastAPI |
| ML Engine | Python |
| ML Library | Scikit-learn / TensorFlow sesuai implementasi existing |
| Database | PostgreSQL |
| API | REST |
| Model Storage | File/Object Storage |
| Container | Docker |
| Version Control | Git |
| Deployment | Server/Cloud |

Pemilihan teknologi final dapat berubah berdasarkan struktur aplikasi Python yang telah tersedia dan kebutuhan lingkungan deployment.

---

# 10. Risiko Penelitian

### Risiko 1 — Model Python sulit diintegrasikan

Mitigasi:

Model dianalisis terlebih dahulu dan dibuat wrapper/service interface sehingga frontend tidak berinteraksi langsung dengan source code internal model.

### Risiko 2 — Data provider tidak menyediakan data yang sesuai

Mitigasi:

Sistem menggunakan abstraction layer pada market data service sehingga provider dapat diganti tanpa mengubah prediction engine.

### Risiko 3 — Training terlalu berat untuk request HTTP biasa

Mitigasi:

Training diperlakukan sebagai **asynchronous job**, bukan proses synchronous yang membuat browser menunggu hingga training selesai.

### Risiko 4 — Model baru menghasilkan performa berbeda

Mitigasi:

Metrik model setiap training disimpan sehingga pengguna dapat membandingkan model dan menentukan model aktif berdasarkan hasil evaluasi.

### Risiko 5 — Data leakage

Mitigasi:

Pipeline mempertahankan chronological split, training-only preprocessing, sequential sliding window, serta temporal alignment untuk data berita sesuai prinsip paper.

---

# 11. Jadwal Penelitian

| Tahap | Bln 1 | Bln 2 | Bln 3 | Bln 4 | Bln 5 | Bln 6 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Studi Literatur | ● | ● | | | | |
| Analisis Python Model | ● | ● | | | | |
| Analisis Kebutuhan | | ● | | | | |
| Perancangan Sistem | | ● | ● | | | |
| Implementasi Backend | | | ● | ● | | |
| Implementasi Frontend | | | ● | ● | | |
| Integrasi Python | | | | ● | ● | |
| Market Data Integration | | | | ● | ● | |
| Testing | | | | | ● | |
| Evaluasi | | | | | ● | ● |
| Dokumentasi | | | | | ● | ● |

---

# 12. Luaran Penelitian

Luaran yang diharapkan:

1. Aplikasi web Sistem Pendukung Keputusan prediksi saham.
2. Python prediction/training service yang terintegrasi.
3. Database market data, prediction history, dan model metadata.
4. Dashboard sinyal Buy, Hold, dan Sell.
5. Fitur training/retraining model.
6. Dokumentasi API.
7. Dokumentasi pengujian sistem.
8. Laporan tugas akhir.

---

# DAFTAR PUSTAKA

Daftar pustaka berikut diprioritaskan berdasarkan referensi yang digunakan dalam paper utama dan paling relevan terhadap penelitian.

[1] A. Y. Wijaya, C. Fatichah, A. Saikhu, and S. Arifin, “Stock Buy and Sell Prediction Based on Technical and News Sentiment Analysis Using Recurrent Neural Network Architecture and Ensemble Methods,” *International Journal of Intelligent Engineering and Systems*, vol. 19, no. 5, pp. 715–729, 2026, doi: 10.22266/ijies2026.0531.41.

[2] R. M. Dhokane and S. Agarwal, “Enhancing Stock Price Prediction with MACD and EMA Features Using LSTM Algorithm,” *Proc. 2024 International Conference on Emerging Smart Computing and Informatics (ESCI)*, pp. 1–6, 2024.

[3] A. Ojha and V. Saxena, “Understanding Stock Market Trends Using Simple Moving Average (SMA) and Exponential Moving Average (EMA) Indicators,” *Proc. 2023 6th International Conference on Contemporary Computing and Informatics (IC3I)*, pp. 1931–1935, 2023.

[4] S. Liu, G. Liao, and Y. Ding, “Stock transaction prediction modeling and analysis based on LSTM,” *Proc. 2018 13th IEEE Conference on Industrial Electronics and Applications (ICIEA)*, pp. 2787–2790, 2018.

[5] Z. Lanbouri and S. Achchab, “Stock Market prediction on High frequency data using Long-Short Term Memory,” *Procedia Computer Science*, vol. 175, pp. 603–608, 2020.

[6] H. N. Bhandari, B. Rimal, N. R. Pokhrel, R. Rimal, K. R. Dahal, and R. K. C. Khatri, “Predicting stock market index using LSTM,” *Machine Learning with Applications*, vol. 9, p. 100320, 2022.

[7] R. Akita, A. Yoshihara, T. Matsubara, and K. Uehara, “Deep learning for stock prediction using numerical and textual information,” *Proc. 2016 IEEE/ACIS 15th International Conference on Computer and Information Science (ICIS)*, pp. 1–6, 2016.

[8] G. M. Chatziloizos, D. Gunopulos, and K. Konstantinou, “Deep Learning for Stock Market Prediction Using Sentiment and Technical Analysis,” *SN Computer Science*, vol. 5, no. 5, p. 446, 2024.

[9] A. F. Kamara, E. Chen, and Z. Pan, “An ensemble of a boosted hybrid of deep learning models and technical analysis for forecasting stock prices,” *Information Sciences*, vol. 594, pp. 1–19, 2022.

[10] S. Ravikumar and P. Saraf, “Prediction of Stock Prices using Machine Learning (Regression, Classification) Algorithms,” *Proc. 2020 International Conference for Emerging Technology (INCET)*, pp. 1–5, 2020.

[11] T. Sanboon, K. Keatruangkamala, and S. Jaiyen, “A Deep Learning Model for Predicting Buy and Sell Recommendations in Stock Exchange of Thailand using Long Short-Term Memory,” *Proc. 2019 IEEE 4th International Conference on Computer and Communication Systems (ICCCS)*, pp. 757–760, 2019.

[12] A. T. Haryono, R. Sarno, and K. R. Sungkono, “Transformer-Gated Recurrent Unit Method for Predicting Stock Price Based on News Sentiments and Technical Indicators,” *IEEE Access*, vol. 11, pp. 77132–77146, 2023.

[13] J. Han, J. Pei, and H. Tong, *Data Mining: Concepts and Techniques*, 4th ed. Cambridge, MA, USA: Morgan Kaufmann, 2023.

[14] N. Jing, Z. Wu, and H. Wang, “A hybrid model integrating deep learning with investor sentiment analysis for stock price prediction,” *Expert Systems with Applications*, vol. 178, p. 115019, 2021, doi: 10.1016/j.eswa.2021.115019.

[15] H. Lee, J. H. Kim, and H. S. Jung, “Deep-learning-based stock market prediction incorporating ESG sentiment and technical indicators,” *Scientific Reports*, vol. 14, no. 1, p. 10262, 2024, doi: 10.1038/s41598-024-61106-2.

[16] T. Chong, W. K. Ng, and V. Liew, “Revisiting the Performance of MACD and RSI Oscillators,” *Journal of Risk and Financial Management*, vol. 7, no. 1, pp. 1–12, 2014, doi: 10.3390/jrfm7010001.

[17] R. Ren, D. D. Wu, and T. Liu, “Forecasting Stock Market Movement Direction Using Sentiment Analysis and Support Vector Machine,” *IEEE Systems Journal*, vol. 13, no. 1, pp. 760–770, 2019.

[18] J. D. Kothapalli et al., “Predicting Buy and Sell Signals for Stocks using Bollinger Bands and MACD with the Help of Machine Learning,” *Proc. 2023 International Conference on Sustainable Computing and Smart Systems (ICSCSS)*, pp. 333–340, 2023.

[19] W. Ding and S. Wu, “A cross-entropy based stacking method in ensemble learning,” *Journal of Intelligent & Fuzzy Systems*, vol. 39, no. 3, pp. 4677–4688, 2020.

[20] M. Sui, C. Zhang, L. Zhou, S. Liao, and C. Wei, “An Ensemble Approach to Stock Price Prediction Using Deep Learning and Time Series Models,” *Proc. 2024 IEEE 6th International Conference on Power, Intelligent Computing and Systems (ICPICS)*, pp. 793–797, 2024.

[21] I. D. Mienye and Y. Sun, “A Survey of Ensemble Learning: Concepts, Algorithms, Applications, and Prospects,” *IEEE Access*, vol. 10, pp. 99129–99149, 2022.

[22] K. Valiant, Y. Lukito, and R. G. Santosa, “Sistem Prediksi Harga Saham LQ45 Dengan Random Forest Classifier,” *JUTEI*, vol. 3, no. 2, pp. 127–136, 2021.

[23] Y. Ni, M. Y. Day, P. Huang, and S. R. Yu, “The profitability of Bollinger Bands: Evidence from the constituent stocks of Taiwan 50,” *Physica A: Statistical Mechanics and its Applications*, vol. 551, p. 124144, 2020.

---

# PENUTUP

Proposal ini menempatkan penelitian secara jelas sebagai **software/system engineering research yang memanfaatkan model machine learning existing**, bukan sebagai penelitian yang menciptakan ulang algoritma pada paper. Model dasar tetap menggunakan pendekatan multimodal dan hybrid ensemble yang telah diuji oleh Wijaya et al., sedangkan kontribusi tugas akhir diarahkan pada pembangunan **web application, market-data integration, training capability, model management, prediction service, dan decision-support dashboard**. Model dasar sendiri menggunakan kombinasi RNN-based price regression, Machine Learning Voting, dan Deep Learning Voting dengan final majority voting.

Hal yang paling penting untuk tahap implementasi berikutnya adalah menjadikan **kode Python yang sudah tersedia sebagai source of truth** untuk input feature, preprocessing, model file, output, threshold, dan mekanisme training. Dengan cara tersebut, proposal tetap konsisten dengan paper sekaligus realistis terhadap implementasi sistem yang benar-benar akan dibangun.