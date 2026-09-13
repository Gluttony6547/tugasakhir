const form = document.querySelector('#prediction-form');
const message = document.querySelector('#form-message');
const runsBody = document.querySelector('#runs-body');
const confidenceBar = document.querySelector('#confidence-bar');
const trend = document.querySelector('#trend');

function renderRuns(runs) {
  if (!runs.length) return;
  runsBody.innerHTML = runs.map((run, index) => `<tr><td>${runs.length - index}</td><td>${run.features[0].toFixed(2)}</td><td>${run.features[1].toFixed(2)}</td><td>${run.features[2].toFixed(2)}</td><td class="output-cell">${run.class_name}</td><td>${run.confidence.toFixed(2)}</td><td>${run.time}</td></tr>`).join('');
  [...trend.children].forEach((dot, index) => {
    const item = runs.slice(0, 3).reverse()[index];
    dot.style.setProperty('--index', index);
    dot.style.setProperty('--position', item ? `${Math.max(10, item.confidence * 88)}%` : '50%');
    dot.classList.toggle('has-value', Boolean(item));
  });
}

async function refreshHealth() {
  const response = await fetch('/api/health');
  const health = await response.json();
  document.querySelector('#health-status').textContent = health.status === 'ready' ? 'Ready' : 'Unavailable';
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = form.querySelector('button');
  button.disabled = true;
  message.textContent = 'Running inference…';
  const data = Object.fromEntries(new FormData(form));
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    document.querySelector('#prediction-value').textContent = result.class_name;
    document.querySelector('#probability-label').textContent = `P(kelas 1): ${result.class_1_probability.toFixed(2)} · margin: ${result.decision_margin.toFixed(2)}`;
    document.querySelector('#decision-summary').textContent = result.explanation;
    document.querySelector('#confidence-value').textContent = result.confidence.toFixed(2);
    confidenceBar.style.width = `${result.confidence * 100}%`;
    renderRuns(await (await fetch('/api/runs')).json());
    const impacts = Array.isArray(result.feature_contributions)
      ? result.feature_contributions.map((value, index) => `F${index + 1} ${value >= 0 ? '+' : ''}${value.toFixed(2)}`).join(', ')
      : 'tersedia setelah response model terbaru';
    message.textContent = result.status === 'uncertain'
      ? 'Model belum cukup yakin; hasil tidak dipaksakan menjadi 0 atau 1.'
      : `Keputusan selesai. Dampak fitur: ${impacts}.`;
  } catch (error) { message.textContent = error.message; }
  finally { button.disabled = false; }
});

document.querySelector('#reload-button').addEventListener('click', refreshHealth);
refreshHealth();
