// MULTIStream Frontend Logic

const chatMessages = document.getElementById('chat-messages');
const searchForm = document.getElementById('search-form');
const queryInput = document.getElementById('query-input');
const healthBadge = document.getElementById('health-badge');
const healthText = document.getElementById('health-text');
const vramText = document.getElementById('vram-text');

// Polygon Modal Elements
const polyModal = document.getElementById('poly-modal');
const btnOpenPoly = document.getElementById('btn-open-poly');
const btnClosePoly = document.getElementById('btn-close-poly');
const polyCamSelect = document.getElementById('poly-camera-select');
const polyAliasName = document.getElementById('poly-alias-name');
const btnResetPoly = document.getElementById('btn-reset-poly');
const btnSavePoly = document.getElementById('btn-save-poly');
const polyCanvas = document.getElementById('poly-canvas');
const ctx = polyCanvas.getContext('2d');

// Mobile Cam Modal Elements
const mobileModal = document.getElementById('mobile-modal');
const btnOpenMobile = document.getElementById('btn-open-mobile');
const btnCloseMobile = document.getElementById('btn-close-mobile');
const streamUrlInput = document.getElementById('stream-url-input');
const streamCameraName = document.getElementById('stream-camera-name');
const btnStartStream = document.getElementById('btn-start-stream');
const btnStopStream = document.getElementById('btn-stop-stream');
const streamStatusText = document.getElementById('stream-status-text');
const streamDetailText = document.getElementById('stream-detail-text');

let knownCameras = [];
let currentPolyPoints = [];
let currentBgImage = new Image();
let streamPollTimer = null;

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
  fetchHealth();
  fetchCameras();
  setupEventListeners();
  checkStreamStatus();
  streamPollTimer = setInterval(checkStreamStatus, 2000);
});

function setupEventListeners() {
  searchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = queryInput.value.trim();
    if (q) {
      handleSearch(q);
    }
  });

  // Mobile modal events
  if (btnOpenMobile) {
    btnOpenMobile.addEventListener('click', () => {
      mobileModal.classList.remove('hidden');
      checkStreamStatus();
    });
  }

  if (btnCloseMobile) {
    btnCloseMobile.addEventListener('click', () => {
      mobileModal.classList.add('hidden');
    });
  }

  if (btnStartStream) {
    btnStartStream.addEventListener('click', handleStartStream);
  }

  if (btnStopStream) {
    btnStopStream.addEventListener('click', handleStopStream);
  }

  // Polygon modal events
  btnOpenPoly.addEventListener('click', () => {
    polyModal.classList.remove('hidden');
    loadCameraCanvas();
  });

  btnClosePoly.addEventListener('click', () => {
    polyModal.classList.add('hidden');
  });

  polyCamSelect.addEventListener('change', () => {
    loadCameraCanvas();
  });

  polyCanvas.addEventListener('click', (e) => {
    const rect = polyCanvas.getBoundingClientRect();
    const scaleX = polyCanvas.width / rect.width;
    const scaleY = polyCanvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    currentPolyPoints.push([x, y]);
    drawPolygon();
  });

  btnResetPoly.addEventListener('click', () => {
    currentPolyPoints = [];
    drawPolygon();
  });

  btnSavePoly.addEventListener('click', handleSavePolygonAlias);
}

// --- Health & Cameras ---
async function fetchHealth() {
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      healthBadge.querySelector('.status-dot').classList.add('active');
      healthText.textContent = `Online (${data.models.device})`;
      vramText.textContent = `GPU VRAM: ${data.vram.used_mib} / ${data.vram.total_mib} MB`;
    }
  } catch (err) {
    healthText.textContent = 'Offline';
    console.error('Health check failed:', err);
  }
}

async function fetchCameras() {
  try {
    const res = await fetch('/cameras');
    if (res.ok) {
      knownCameras = await res.json();
      polyCamSelect.innerHTML = knownCameras.map(c => 
        `<option value="${c.camera}">${c.camera} (${c.width}x${c.height})</option>`
      ).join('');
    }
  } catch (err) {
    console.error('Failed to load cameras:', err);
  }
}

// --- Search Flow ---
function setQuery(text) {
  queryInput.value = text;
  handleSearch(text);
}

async function handleSearch(queryText) {
  appendUserMessage(queryText);
  queryInput.value = '';

  const loadingMsgId = appendAssistantLoading();

  try {
    const t0 = performance.now();
    const res = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: queryText, top_k: 5 })
    });
    const data = await res.json();
    const elapsed = Math.round(performance.now() - t0);

    removeMessage(loadingMsgId);

    if (data.status === 'clarify') {
      appendClarificationCard(queryText, data.referent, data.options);
    } else if (data.status === 'success') {
      appendResultsCard(queryText, data.parsed, data.results, elapsed);
    }
  } catch (err) {
    removeMessage(loadingMsgId);
    appendAssistantMessage(`Error executing search: ${err.message}`);
  }
}

// --- UI Message Helpers ---
function appendUserMessage(text) {
  const msgEl = document.createElement('div');
  msgEl.className = 'message user-msg';
  msgEl.innerHTML = `
    <div class="msg-avatar">YOU</div>
    <div class="msg-body">${escapeHtml(text)}</div>
  `;
  chatMessages.appendChild(msgEl);
  scrollToBottom();
}

function appendAssistantMessage(htmlContent) {
  const msgEl = document.createElement('div');
  msgEl.className = 'message assistant-msg';
  msgEl.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-body">${htmlContent}</div>
  `;
  chatMessages.appendChild(msgEl);
  scrollToBottom();
}

function appendAssistantLoading() {
  const id = 'loading-' + Date.now();
  const msgEl = document.createElement('div');
  msgEl.className = 'message assistant-msg';
  msgEl.id = id;
  msgEl.innerHTML = `
    <div class="msg-avatar">AI</div>
    <div class="msg-body">
      <span style="color: var(--text-muted);">Analyzing video embeddings and spatio-temporal index...</span>
    </div>
  `;
  chatMessages.appendChild(msgEl);
  scrollToBottom();
  return id;
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendClarificationCard(originalQuery, referent, options) {
  const optionsHtml = options.map(opt => `
    <button class="cam-option-btn" onclick="resolveClarification('${escapeHtml(referent)}', '${escapeHtml(opt)}', '${escapeHtml(originalQuery)}')">
      ${escapeHtml(opt)}
    </button>
  `).join('');

  const html = `
    <div class="clarify-card">
      <div class="clarify-title">
        <span>&#9888; Clarification Required</span>
      </div>
      <p>The location referent <strong>"${escapeHtml(referent)}"</strong> is unknown and not mapped to any camera.</p>
      <p style="margin-top: 8px; font-size: 13px; color: var(--text-muted);">
        Which camera covers this location? Click to resolve and save permanently:
      </p>
      <div class="clarify-options">
        ${optionsHtml}
      </div>
    </div>
  `;
  appendAssistantMessage(html);
}

async function resolveClarification(referent, cameraName, originalQuery) {
  try {
    const res = await fetch('/alias', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: referent, camera: cameraName })
    });
    if (res.ok) {
      appendAssistantMessage(`&#10004; Saved alias: <strong>${escapeHtml(referent)}</strong> &rarr; <strong>${escapeHtml(cameraName)}</strong>. Resuming query...`);
      // Re-run original query
      handleSearch(originalQuery);
    }
  } catch (err) {
    alert('Failed to save alias: ' + err.message);
  }
}

function appendResultsCard(query, parsed, results, elapsedMs) {
  if (!results || results.length === 0) {
    appendAssistantMessage(`
      <p>No matching objects found for <em>"${escapeHtml(query)}"</em>.</p>
      <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px;">
        Object prompt: <code>${escapeHtml(parsed.object_prompt)}</code> | Provider: ${parsed.provider} | Latency: ${elapsedMs}ms
      </div>
    `);
    return;
  }

  const resultsHtml = results.map((r, i) => `
    <div class="result-card">
      <div class="result-preview" id="preview-${r.id}">
        <img src="${r.snapshot_url}" alt="Snapshot of ${r.label}" onerror="this.src='/static/placeholder.jpg'">
      </div>
      <div class="result-info">
        <div>
          <div class="result-meta">
            <span class="meta-badge badge-score">Match: ${(r.score * 100).toFixed(1)}%</span>
            <span class="meta-badge badge-cam">${escapeHtml(r.camera)}</span>
            <span class="meta-badge">${escapeHtml(r.label)}</span>
            ${r.color && r.color !== 'unknown' ? `<span class="meta-badge" style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); font-weight: 600;">Color: ${escapeHtml(r.color)}</span>` : ''}
            <span class="meta-badge">${formatTimestamp(r.timestamp)}</span>
            <span class="meta-badge">Offset: ${r.offset_seconds.toFixed(2)}s</span>
          </div>
          <div style="font-size: 13px; color: var(--text-muted);">
            Result ID: <code>${escapeHtml(r.id)}</code> (${r.type})
          </div>
        </div>
        <button class="btn-play" onclick="playClip('${r.id}', '${r.clip_url}')">Play Video Clip</button>
      </div>
    </div>
  `).join('');

  const fullHtml = `
    <div>
      <div style="font-size: 13px; color: var(--text-muted); margin-bottom: 8px;">
        Found ${results.length} results for <strong>"${escapeHtml(query)}"</strong> (${elapsedMs} ms)
      </div>
      <div class="results-container">
        ${resultsHtml}
      </div>
    </div>
  `;
  appendAssistantMessage(fullHtml);
}

function playClip(resultId, clipUrl) {
  const container = document.getElementById(`preview-${resultId}`);
  if (!container) return;

  container.innerHTML = `
    <video controls autoplay width="100%" height="100%">
      <source src="${clipUrl}" type="video/mp4">
      Your browser does not support HTML5 video.
    </video>
  `;
}

// --- Canvas & Polygon Drawing ---
function loadCameraCanvas() {
  const selectedCam = polyCamSelect.value;
  if (!selectedCam) return;

  // Find camera snapshot or draw dark placeholder
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, polyCanvas.width, polyCanvas.height);
  
  // Try loading first snapshot of that camera
  fetch('/cameras').then(res => res.json()).then(cams => {
    const camInfo = cams.find(c => c.camera === selectedCam);
    if (camInfo) {
      currentBgImage.onload = () => {
        ctx.drawImage(currentBgImage, 0, 0, polyCanvas.width, polyCanvas.height);
        drawPolygon();
      };
      // Load sample snapshot
      currentBgImage.src = `/snapshot/${selectedCam}`;
    }
  });

  currentPolyPoints = [];
  drawPolygon();
}

function drawPolygon() {
  ctx.clearRect(0, 0, polyCanvas.width, polyCanvas.height);
  if (currentBgImage.complete && currentBgImage.naturalWidth > 0) {
    ctx.drawImage(currentBgImage, 0, 0, polyCanvas.width, polyCanvas.height);
  } else {
    ctx.fillStyle = '#161b22';
    ctx.fillRect(0, 0, polyCanvas.width, polyCanvas.height);
  }

  if (currentPolyPoints.length === 0) return;

  // Draw points & path
  ctx.strokeStyle = '#58a6ff';
  ctx.fillStyle = 'rgba(88, 166, 255, 0.25)';
  ctx.lineWidth = 2;

  ctx.beginPath();
  ctx.moveTo(currentPolyPoints[0][0], currentPolyPoints[0][1]);
  for (let i = 1; i < currentPolyPoints.length; i++) {
    ctx.lineTo(currentPolyPoints[i][0], currentPolyPoints[i][1]);
  }
  if (currentPolyPoints.length > 2) {
    ctx.closePath();
    ctx.fill();
  }
  ctx.stroke();

  // Draw vertices
  ctx.fillStyle = '#388bfd';
  for (const pt of currentPolyPoints) {
    ctx.beginPath();
    ctx.arc(pt[0], pt[1], 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

async function handleSavePolygonAlias() {
  const name = polyAliasName.value.trim();
  const cam = polyCamSelect.value;
  if (!name || !cam) {
    alert('Please enter an alias name and select a camera.');
    return;
  }
  if (currentPolyPoints.length < 3) {
    alert('Please click at least 3 points on the canvas to define a valid polygon.');
    return;
  }

  // Normalize points [x/W, y/H] in [0, 1]
  const normPoly = currentPolyPoints.map(p => [
    parseFloat((p[0] / polyCanvas.width).toFixed(4)),
    parseFloat((p[1] / polyCanvas.height).toFixed(4))
  ]);

  try {
    const res = await fetch('/alias', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: name,
        camera: cam,
        polygon: normPoly
      })
    });
    if (res.ok) {
      alert(`Alias '${name}' saved successfully with ${normPoly.length}-point spatial polygon!`);
      polyModal.classList.add('hidden');
      polyAliasName.value = '';
      currentPolyPoints = [];
    }
  } catch (err) {
    alert('Failed to save alias: ' + err.message);
  }
}

// --- Utilities ---
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(str) {
  if (!str) return '';
  return str.toString()
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatTimestamp(isoStr) {
  if (!isoStr) return '';
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return isoStr;
  }
}

// --- Live Mobile Stream Handlers ---
async function checkStreamStatus() {
  if (!streamStatusText || !streamDetailText) return;
  try {
    const res = await fetch('/stream/status');
    if (res.ok) {
      const data = await res.json();
      if (data.status === 'running') {
        streamStatusText.textContent = `Streaming (${data.camera})`;
        streamStatusText.style.color = '#10b981';
        let detail = `FPS: ${data.fps} | Frames: ${data.frames_read} | Tracks: ${data.tracks_indexed}`;
        if (data.latest_track) {
          detail += ` | Last: ${data.latest_track.color || ''} ${data.latest_track.label}`;
        }
        streamDetailText.textContent = detail;
        if (btnStartStream) btnStartStream.style.display = 'none';
        if (btnStopStream) btnStopStream.style.display = 'inline-block';
      } else if (data.status === 'finished') {
        streamStatusText.textContent = 'Finished';
        streamStatusText.style.color = '#f59e0b';
        streamDetailText.textContent = `Processed ${data.frames_read || 0} frames | Indexed ${data.tracks_indexed || 0} tracks`;
        if (btnStartStream) btnStartStream.style.display = 'inline-block';
        if (btnStopStream) btnStopStream.style.display = 'none';
      } else {
        streamStatusText.textContent = data.status || 'Idle';
        streamStatusText.style.color = '#38bdf8';
        streamDetailText.textContent = data.status === 'idle' ? 'No active stream' : `Status: ${data.status}`;
        if (btnStartStream) btnStartStream.style.display = 'inline-block';
        if (btnStopStream) btnStopStream.style.display = 'none';
      }
    }
  } catch (err) {
    console.error('Failed to get stream status:', err);
  }
}

async function handleStartStream() {
  const url = streamUrlInput.value.trim();
  const cam = streamCameraName.value.trim() || 'mobile_cam01';
  if (!url) {
    alert('Please enter a stream URL (e.g. http://192.168.1.105:8080/video or footage/traffic_video.mp4)');
    return;
  }
  btnStartStream.disabled = true;
  btnStartStream.textContent = 'Connecting...';
  try {
    const res = await fetch('/stream/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stream_url: url, camera: cam })
    });
    if (res.ok) {
      const data = await res.json();
      appendAssistantMessage(`&#128249; <strong>Live Mobile Stream Started</strong>: Ingesting from <code>${escapeHtml(url)}</code> under camera <strong>${escapeHtml(cam)}</strong>. Detections, tracks, and dominant colors are stored live into the database! Try asking: <em>"car in ${escapeHtml(cam)}"</em> or <em>"bus in ${escapeHtml(cam)}"</em>.`);
      fetchCameras();
    } else {
      const err = await res.json();
      alert('Error starting stream: ' + (err.detail || res.statusText));
    }
  } catch (err) {
    alert('Failed to connect to stream: ' + err.message);
  } finally {
    btnStartStream.disabled = false;
    btnStartStream.textContent = 'Connect & Start Live Ingest';
    checkStreamStatus();
  }
}

async function handleStopStream() {
  btnStopStream.disabled = true;
  try {
    const res = await fetch('/stream/stop', { method: 'POST' });
    if (res.ok) {
      appendAssistantMessage(`&#9209; <strong>Live Stream Stopped</strong>. Captured tracks are stored in the database.`);
      fetchCameras();
    }
  } catch (err) {
    alert('Failed to stop stream: ' + err.message);
  } finally {
    btnStopStream.disabled = false;
    checkStreamStatus();
  }
}
