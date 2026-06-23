from flask import Flask, request, jsonify, render_template_string
from Pro.analyzer import analyze_from_bytes
import traceback
import json

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB máx

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DeepFace — Análisis Facial</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #0d1117;
      --surface:  #161b22;
      --surface2: #21262d;
      --accent:   #3fb950;
      --accent2:  #58a6ff;
      --text:     #e6edf3;
      --muted:    #8b949e;
      --border:   #30363d;
      --danger:   #f85149;
      --radius:   12px;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, 'Segoe UI', sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    /* ===== HEADER ===== */
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 16px 28px;
      display: flex;
      align-items: center;
      gap: 14px;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .hdr-icon {
      width: 40px; height: 40px;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      border-radius: 10px;
      display: grid; place-items: center;
      font-size: 18px; flex-shrink: 0;
    }
    header h1 { font-size: 1.15rem; font-weight: 700; }
    header p  { font-size: 0.76rem; color: var(--muted); margin-top: 1px; }
    .badge {
      margin-left: auto;
      border: 1px solid var(--accent);
      color: var(--accent);
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 0.7rem;
      font-weight: 600;
    }

    /* ===== MAIN ===== */
    main {
      flex: 1;
      max-width: 960px;
      width: 100%;
      margin: 0 auto;
      padding: 28px 20px 40px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    /* ===== SECCIÓN DE CARGA ===== */
    .upload-section {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: start;
    }

    .upload-card {
      background: var(--surface);
      border: 2px dashed var(--border);
      border-radius: var(--radius);
      padding: 28px 20px;
      text-align: center;
      cursor: pointer;
      transition: border-color .2s, background .2s;
      position: relative;
    }
    .upload-card:hover, .upload-card.drag {
      border-color: var(--accent);
      background: rgba(63,185,80,.05);
    }
    .upload-card input[type="file"] {
      position: absolute; inset: 0;
      opacity: 0; cursor: pointer; width: 100%; height: 100%;
    }
    .upload-icon { font-size: 2rem; margin-bottom: 8px; }
    .upload-card h2 { font-size: 0.95rem; font-weight: 600; margin-bottom: 4px; }
    .upload-card p  { font-size: 0.76rem; color: var(--muted); }
    #file-name {
      margin-top: 8px; font-size: 0.76rem;
      color: var(--accent); font-weight: 600;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }

    #analyze-btn {
      padding: 0 28px;
      height: 100%;
      min-height: 56px;
      min-width: 160px;
      background: var(--accent);
      color: #0d1117;
      font-weight: 700; font-size: 0.9rem;
      border: none; border-radius: var(--radius);
      cursor: pointer;
      white-space: nowrap;
      transition: opacity .2s, transform .15s;
      align-self: stretch;
    }
    #analyze-btn:disabled { opacity: .3; cursor: not-allowed; }
    #analyze-btn:not(:disabled):hover { opacity: .88; transform: translateY(-1px); }

    /* ===== SPINNER ===== */
    .spinner-wrap {
      display: none; flex-direction: column;
      align-items: center; gap: 10px;
      padding: 32px; color: var(--muted); font-size: 0.84rem;
    }
    .spinner {
      width: 34px; height: 34px;
      border: 3px solid var(--border); border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin .8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ===== RESULTADO LAYOUT ===== */
    .result-wrap {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      align-items: start;
    }

    /* Imagen anotada */
    .result-img-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
    }
    .result-img-box img {
      width: 100%; display: block;
    }
    .result-img-label {
      padding: 10px 14px;
      font-size: 0.72rem; font-weight: 700;
      letter-spacing: .6px; text-transform: uppercase;
      color: var(--muted);
      border-top: 1px solid var(--border);
    }

    /* Panel de datos */
    .data-panel {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    /* Tarjeta por rostro */
    .face-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      overflow: hidden;
      animation: fadeUp .3s ease both;
    }
    @keyframes fadeUp {
      from { opacity:0; transform:translateY(8px); }
      to   { opacity:1; transform:translateY(0); }
    }
    .face-card:nth-child(2) { animation-delay: .07s; }
    .face-card:nth-child(3) { animation-delay: .14s; }

    .face-card-header {
      background: var(--surface2);
      padding: 10px 16px;
      display: flex; align-items: center; gap: 8px;
      border-bottom: 1px solid var(--border);
    }
    .face-card-header .num {
      font-size: 0.68rem; font-weight: 700;
      letter-spacing: .8px; text-transform: uppercase;
      color: var(--accent);
    }
    .face-card-header .confidence-chip {
      margin-left: auto;
      background: rgba(63,185,80,.12);
      border: 1px solid rgba(63,185,80,.3);
      color: var(--accent);
      font-size: 0.68rem; font-weight: 700;
      padding: 2px 8px; border-radius: 12px;
    }

    /* Grid de stats principales: 2 columnas */
    .stats-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
    }
    .stat-cell {
      padding: 12px 16px;
      border-right: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
    }
    .stat-cell:nth-child(2n) { border-right: none; }
    .stat-cell:nth-last-child(-n+2) { border-bottom: none; }

    .stat-cell .sc-label {
      font-size: 0.68rem; color: var(--muted);
      text-transform: uppercase; letter-spacing: .5px;
      margin-bottom: 4px;
      display: flex; align-items: center; gap: 5px;
    }
    .stat-cell .sc-value {
      font-size: 0.95rem; font-weight: 700;
    }
    .stat-cell .sc-sub {
      font-size: 0.72rem; color: var(--accent);
      margin-top: 1px;
    }

    /* Sección emociones */
    .emo-section {
      padding: 14px 16px;
      border-top: 1px solid var(--border);
    }
    .emo-section-title {
      font-size: 0.68rem; font-weight: 700;
      letter-spacing: .6px; text-transform: uppercase;
      color: var(--muted); margin-bottom: 10px;
    }
    .emo-row { margin-bottom: 8px; }
    .emo-lbl {
      display: flex; justify-content: space-between;
      font-size: 0.74rem; margin-bottom: 4px;
    }
    .emo-lbl .emo-name { font-weight: 500; }
    .emo-lbl .emo-pct  { color: var(--muted); }
    .emo-bg {
      height: 6px; background: var(--surface2);
      border-radius: 4px; overflow: hidden;
    }
    .emo-fill {
      height: 100%; border-radius: 4px;
      background: linear-gradient(90deg, var(--accent2), var(--accent));
      transition: width .5s ease;
    }
    .emo-fill.dominant {
      background: linear-gradient(90deg, var(--accent), #7ee787);
    }

    /* ===== ERROR ===== */
    .error-box {
      background: rgba(248,81,73,.1);
      border: 1px solid rgba(248,81,73,.3);
      border-radius: var(--radius);
      padding: 14px 18px;
      color: var(--danger);
      font-size: 0.84rem;
      display: flex; gap: 10px;
    }

    /* ===== FOOTER ===== */
    footer {
      text-align: center; padding: 16px;
      font-size: 0.74rem; color: var(--muted);
      border-top: 1px solid var(--border);
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 680px) {
      .upload-section { grid-template-columns: 1fr; }
      #analyze-btn { height: auto; min-height: 48px; width: 100%; }
      .result-wrap { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>

<header>
  <div class="hdr-icon">🧠</div>
  <div>
    <h1>DeepFace — Análisis Facial</h1>
    <p>Género · Etnia · Edad · Emoción</p>
  </div>
  <span class="badge">Básica</span>
</header>

<main>

  <!-- SECCIÓN DE CARGA -->
  <div class="upload-section">
    <div class="upload-card" id="drop-zone">
      <input type="file" id="file-input" accept="image/*">
      <div class="upload-icon">📷</div>
      <h2>Sube una imagen para analizar</h2>
      <p>Haz clic aquí o arrastra una foto · JPG, PNG, WEBP</p>
      <div id="file-name"></div>
    </div>
    <button id="analyze-btn" disabled>🔍 Analizar</button>
  </div>

  <!-- SPINNER -->
  <div class="spinner-wrap" id="spinner">
    <div class="spinner"></div>
    <span>Analizando con DeepFace…</span>
  </div>

  <!-- RESULTADOS -->
  <div id="results"></div>

</main>

<footer>Proyecto educativo · DeepFace + Flask · Versión Básica</footer>

<script>
  const fileInput  = document.getElementById('file-input');
  const analyzeBtn = document.getElementById('analyze-btn');
  const resultsDiv = document.getElementById('results');
  const spinner    = document.getElementById('spinner');
  const fileNameEl = document.getElementById('file-name');
  const dropZone   = document.getElementById('drop-zone');
  let selectedFile = null;

  fileInput.addEventListener('change', e => {
    selectedFile = e.target.files[0];
    if (selectedFile) {
      analyzeBtn.disabled = false;
      fileNameEl.textContent = '✓ ' + selectedFile.name;
    }
  });

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('image/')) {
      selectedFile = f;
      fileNameEl.textContent = '✓ ' + f.name;
      analyzeBtn.disabled = false;
    }
  });

  analyzeBtn.addEventListener('click', async () => {
    resultsDiv.innerHTML = '';
    spinner.style.display = 'flex';
    analyzeBtn.disabled = true;
    try {
      const fd = new FormData();
      fd.append('image', selectedFile);
      const resp = await fetch('/analyze', { method: 'POST', body: fd });
      const data = await resp.json();
      spinner.style.display = 'none';
      analyzeBtn.disabled = false;
      if (data.error) {
        resultsDiv.innerHTML = `<div class="error-box">⚠️ <span>${data.error}</span></div>`;
        return;
      }
      renderResults(data);
    } catch (err) {
      spinner.style.display = 'none';
      analyzeBtn.disabled = false;
      resultsDiv.innerHTML = `<div class="error-box">⚠️ <span>Error de conexión: ${err.message}</span></div>`;
    }
  });

  function renderResults(data) {
    const { faces, annotated_image } = data;

    let imgHtml = '';
    if (annotated_image) {
      imgHtml = `
        <div class="result-img-box">
          <img src="data:image/jpeg;base64,${annotated_image}" alt="Imagen analizada">
          <div class="result-img-label">📸 Imagen analizada — ${faces.length} rostro(s) detectado(s)</div>
        </div>`;
    }

    let cardsHtml = '<div class="data-panel">';
    faces.forEach((face, i) => {
      const emociones = [...(face.emociones_detalle || [])].sort((a,b) => b.porcentaje - a.porcentaje);
      const topEmocion = emociones[0];

      cardsHtml += `
      <div class="face-card">
        <div class="face-card-header">
          <span class="num">◆ Rostro ${i + 1}</span>
          <span class="confidence-chip">Confianza ${face.genero_confianza}%</span>
        </div>

        <div class="stats-grid">
          <div class="stat-cell">
            <div class="sc-label">👤 Género</div>
            <div class="sc-value">${face.genero}</div>
            <div class="sc-sub">${face.genero_confianza}% confianza</div>
          </div>
          <div class="stat-cell">
            <div class="sc-label">🌍 Etnia</div>
            <div class="sc-value">${face.raza_dominante}</div>
          </div>
          <div class="stat-cell">
            <div class="sc-label">🎂 Edad estimada</div>
            <div class="sc-value">${face.edad_estimada} años</div>
          </div>
          <div class="stat-cell">
            <div class="sc-label">😊 Emoción</div>
            <div class="sc-value">${face.emocion}</div>
            ${topEmocion ? `<div class="sc-sub">${topEmocion.porcentaje}%</div>` : ''}
          </div>
        </div>

        <div class="emo-section">
          <div class="emo-section-title">Detalle de emociones</div>
          ${emociones.map((e, idx) => `
            <div class="emo-row">
              <div class="emo-lbl">
                <span class="emo-name">${e.nombre}</span>
                <span class="emo-pct">${e.porcentaje}%</span>
              </div>
              <div class="emo-bg">
                <div class="emo-fill ${idx === 0 ? 'dominant' : ''}"
                     style="width:${Math.min(e.porcentaje, 100)}%"></div>
              </div>
            </div>
          `).join('')}
        </div>
      </div>`;
    });
    cardsHtml += '</div>';

    resultsDiv.innerHTML = `
      <div class="result-wrap">
        ${imgHtml}
        ${cardsHtml}
      </div>`;
  }
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400
    file = request.files["image"]
    try:
        faces, img_b64 = analyze_from_bytes(file.read())
        return jsonify({"faces": faces, "annotated_image": img_b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    print("App corriendo en http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
