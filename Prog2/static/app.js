let dashboard;
let active = 'ADRO';
const labels = ['SELL', 'HOLD', 'BUY'];
const $ = (id) => document.getElementById(id);
const money = (v) => `Rp ${Math.round(v).toLocaleString('id-ID')}`;
const pct = (v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`;

function signalClass(value) { return value === 2 ? 'buy' : value === 0 ? 'sell' : 'hold'; }

function renderTicker() {
  $('ticker-bar').innerHTML = Object.entries(dashboard.stocks).slice(0, 5).map(([key, stock], index) => {
    const change = ((stock.close - stock.previous_close) / stock.previous_close) * 100;
    return `<button class="ticker ${index === 0 ? 'active' : ''}" data-symbol="${key}"><strong>${key}.JK</strong><span class="${change >= 0 ? 'up' : 'down'}">${pct(change)}</span></button>`;
  }).join('') + '<span class="ticker-spacer">IDX LQ45 &nbsp;·&nbsp; <i></i> Market Open &nbsp; 15:58 WIB</span>';
  document.querySelectorAll('.ticker').forEach((el) => el.addEventListener('click', () => selectStock(el.dataset.symbol)));
}

function renderChart(stock) {
  const canvas = $('chart'); const ctx = canvas.getContext('2d'); const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth; const height = 300; canvas.width = width * ratio; canvas.height = height * ratio; ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height); const data = stock.chart; const all = data.flatMap((d) => [d.close, d.sma20, d.sma50]); const min = Math.min(...all) * .96; const max = Math.max(...all) * 1.04;
  const x = (i) => 5 + (i / (data.length - 1)) * (width - 12); const y = (v) => height - 12 - ((v - min) / (max - min)) * (height - 30);
  ctx.strokeStyle = '#202024'; ctx.lineWidth = 1; for (let i = 0; i < 5; i++) { const gy = 12 + i * 57; ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(width, gy); ctx.stroke(); }
  const line = (key, color, lineWidth = 1.4) => { ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.beginPath(); data.forEach((d, i) => i ? ctx.lineTo(x(i), y(d[key])) : ctx.moveTo(x(i), y(d[key]))); ctx.stroke(); };
  ctx.strokeStyle = '#10b981'; data.forEach((d, i) => { const px = x(i); const py = y(d.close); ctx.beginPath(); ctx.moveTo(px, py - 4); ctx.lineTo(px, py + 4); ctx.stroke(); ctx.fillStyle = d.close >= (data[i - 1]?.close || d.close) ? '#10b981' : '#f43f5e'; ctx.fillRect(px - 2.5, py - 2.5, 5, 5); });
  line('sma20', '#60a5fa'); line('sma50', '#f5b843');
  const axis = data.filter((_, i) => i % 14 === 0 || i === data.length - 1).map((d) => `<span>${d.date}</span>`).join(''); $('chart-axis').innerHTML = axis;
}

function renderPaths(stock) {
  const pathNames = ['Path 1 · LSTM price regression', 'Path 2 · ML voter', 'Path 3 · DL voter'];
  $('paths').innerHTML = pathNames.map((name, i) => `<article class="path"><div class="path-top"><h4>${name}</h4><span class="signal-chip">${labels[stock.paths[i] ?? stock.signal]}</span></div><small>${i === 0 ? 'Prediksi harga t+50 dari sequence model' : i === 1 ? 'Random Forest · XGBoost · SVM · KNN · AdaBoost' : 'LSTM · Bi-LSTM · GRU · Bi-GRU'}</small><div class="path-meta"><span>Model F1</span><b>${(stock.metrics[i === 0 ? 'Price Prediction' : i === 1 ? 'Machine Learning' : 'Deep Learning']?.f1 * 100 || 0).toFixed(2)}%</b></div></article>`).join('');
}

function renderStock(key) {
  active = key; const stock = dashboard.stocks[key]; const change = ((stock.close - stock.previous_close) / stock.previous_close) * 100; const signal = labels[stock.signal];
  $('symbol').textContent = stock.symbol; $('company').textContent = stock.name; $('signal').textContent = signal; $('signal').className = signalClass(stock.signal); $('consensus').textContent = stock.consensus; $('confidence').textContent = `${Math.round((stock.metrics['Weighted Ensemble']?.accuracy || .8) * 100)}%`; $('close').textContent = money(stock.close); $('change').textContent = `${money(stock.close - stock.previous_close)} (${pct(change)})`; $('target').textContent = money(stock.target_price); $('upside').textContent = `${pct(((stock.target_price - stock.close) / stock.close) * 100)} implied`; $('rsi').textContent = stock.rsi.toFixed(1); $('macd').textContent = stock.macd >= 0 ? 'Bullish' : 'Bearish'; $('macd').className = stock.macd >= 0 ? 'positive' : 'negative'; $('sentiment').textContent = stock.sentiment === 2 ? 'Positive' : stock.sentiment === -1 ? 'Negative' : 'Neutral'; $('chart-meta').textContent = `Daily close · ${stock.chart.length} observations · as of ${stock.date}`; $('majority-signal').textContent = signal; $('majority-signal').className = signalClass(stock.signal); $('majority-score').textContent = `${stock.consensus} paths`; $('consensus-badge').textContent = `${stock.consensus} ${signal}`;
  renderChart(stock); renderPaths(stock); $('news').innerHTML = stock.news.map((item) => `<div class="news-row"><span class="news-time">${String(item.date).slice(-8, -3) || '—'}</span><span class="news-source">${item.publisher}</span><span class="news-title" title="${item.title}">${item.title}</span><span class="sentiment ${item.sentiment === 2 ? 'sent-positive' : item.sentiment < 0 ? 'sent-negative' : 'sent-neutral'}">${item.sentiment === 2 ? 'Positive' : item.sentiment < 0 ? 'Negative' : 'Neutral'}</span></div>`).join('');
  $('history').innerHTML = stock.history.map((row) => `<tr><td>${row.date}</td><td>${row.close.toLocaleString('id-ID')}</td><td><span class="sig ${signalClass(row.truth)}">${labels[row.truth] || '—'}</span></td><td><span class="sig ${signalClass(row.prediction)}">${labels[row.prediction] || '—'}</span></td><td class="${row.correct ? 'correct' : 'wrong'}">${row.correct ? '✓ Correct' : '× Review'}</td></tr>`).join(''); $('history-count').textContent = `Showing latest ${stock.history.length} research rows · ${stock.model_date} model snapshot`;
}

function selectStock(key) { document.querySelectorAll('.ticker').forEach((el) => el.classList.toggle('active', el.dataset.symbol === key)); renderStock(key); }

fetch('/api/dashboard').then((r) => r.json()).then((data) => { dashboard = data; renderTicker(); renderStock(active); });
window.addEventListener('resize', () => dashboard && renderChart(dashboard.stocks[active]));
$('search').addEventListener('keydown', (e) => { if (e.key === 'Enter') { const key = e.target.value.toUpperCase().replace('.JK', ''); if (dashboard.stocks[key]) selectStock(key); } });
$('export').addEventListener('click', () => { const rows = dashboard.stocks[active].history; const csv = ['date,close,ground_truth,prediction,correct', ...rows.map((r) => [r.date, r.close, labels[r.truth], labels[r.prediction], r.correct].join(','))].join('\n'); const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'})); a.download = `${active}_prediction_history.csv`; a.click(); URL.revokeObjectURL(a.href); });
