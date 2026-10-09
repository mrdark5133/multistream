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
let indexingPollTimer = null;
let previousIndexingState = 'idle';

// Indexing Status Elements
const indexingBadge = document.getElementById('indexing-badge');
const indexingDot = document.getElementById('indexing-dot');
const indexingText = document.getElementById('indexing-text');

// Voice Push-to-Talk Elements
const btnMic = document.getElementById('btn-mic');
const micText = document.getElementById('mic-text');
const voiceStatusBar = document.getElementById('voice-status-bar');
const voiceStatusText = document.getElementById('voice-status-text');
const voicePlayer = document.getElementById('voice-player');

let audioContext = null;
let mediaStream = null;
let scriptProcessor = null;
let audioChunks = [];
let isRecording = false;
let recordingStartTime = 0;

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
  fetchHealth();
  fetchCameras();
  setupEventListeners();
  checkStreamStatus();
  checkIndexingStatus();
  streamPollTimer = setInterval(checkStreamStatus, 2000);
  indexingPollTimer = setInterval(checkIndexingStatus, 2500);
});

function setupEventListeners() {
  searchForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = queryInput.value.trim();
    if (q) {
      handleSearch(q);
    }
  });

  // Push-to-talk microphone event listeners
  if (btnMic) {
    btnMic.addEventListener('mousedown', (e) => {
      e.preventDefault();
      startVoiceRecording();
    });
    btnMic.addEventListener('mouseup', (e) => {
      e.preventDefault();
      stopVoiceRecording(true);
    });
    btnMic.addEventListener('mouseleave', () => {
      if (isRecording) {
        stopVoiceRecording(true);
      }
    });

    // Mobile / touch support
    btnMic.addEventListener('touchstart', (e) => {
      e.preventDefault();
      startVoiceRecording();
    });
    btnMic.addEventListener('touchend', (e) => {
      e.preventDefault();
      stopVoiceRecording(true);
    });
  }

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

  const btnFillIpwebcam = document.getElementById('btn-fill-ipwebcam');
  if (btnFillIpwebcam) {
    btnFillIpwebcam.addEventListener('click', () => {
      streamUrlInput.value = 'http://192.168.1.105:8080/video';
      streamUrlInput.focus();
    });
  }

  const btnFillDemo = document.getElementById('btn-fill-demo');
  if (btnFillDemo) {
    btnFillDemo.addEventListener('click', () => {
      streamUrlInput.value = 'footage/traffic_video.mp4';
      streamUrlInput.focus();
    });
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

// --- Voice Push-To-Talk Logic ---
async function initVoiceAudio() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('Microphone audio capture is not supported in this browser.');
  }
  if (!mediaStream) {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });
  }
  if (!audioContext || audioContext.state === 'closed') {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    audioContext = new AudioCtx();
  }
  if (audioContext.state === 'suspended') {
    await audioContext.resume();
  }
}

async function startVoiceRecording() {
  try {
    await initVoiceAudio();
    audioChunks = [];
    isRecording = true;
    recordingStartTime = Date.now();

    const source = audioContext.createMediaStreamSource(mediaStream);
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    scriptProcessor.onaudioprocess = (e) => {
      if (!isRecording) return;
      const channelData = e.inputBuffer.getChannelData(0);
      audioChunks.push(new Float32Array(channelData));
    };

    source.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination);

    if (btnMic) {
      btnMic.classList.add('recording');
      if (micText) micText.textContent = 'Recording...';
    }
    if (voiceStatusBar) {
      voiceStatusBar.style.display = 'flex';
      if (voiceStatusText) voiceStatusText.textContent = 'Listening... Speak your query (release to send)';
    }
  } catch (err) {
    console.error('Error starting audio recording:', err);
    alert('Microphone access failed: ' + err.message);
    stopVoiceRecording(false);
  }
}

async function stopVoiceRecording(shouldSend = true) {
  if (!isRecording) return;
  isRecording = false;

  if (btnMic) {
    btnMic.classList.remove('recording');
    if (micText) micText.textContent = 'Hold to Talk';
  }

  if (scriptProcessor) {
    try {
      scriptProcessor.disconnect();
    } catch (e) {
      console.warn('ScriptProcessor disconnect warning:', e);
    }
    scriptProcessor = null;
  }

  const duration = (Date.now() - recordingStartTime) / 1000;
  if (!shouldSend || duration < 0.3 || audioChunks.length === 0) {
    if (voiceStatusBar) voiceStatusBar.style.display = 'none';
    return;
  }

  if (voiceStatusText) {
    voiceStatusText.textContent = 'Transcribing with Whisper & searching...';
  }

  // Concatenate Float32Array chunks
  let totalLength = 0;
  for (const chunk of audioChunks) totalLength += chunk.length;
  const mergedSamples = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of audioChunks) {
    mergedSamples.set(chunk, offset);
    offset += chunk.length;
  }

  // Downsample to 16 kHz
  const sampleRate = audioContext.sampleRate;
  const downsampled = downsampleBuffer(mergedSamples, sampleRate, 16000);
  const wavBlob = encodeWAV(downsampled, 16000);

  await sendVoiceQuery(wavBlob);
}

function downsampleBuffer(buffer, inputSampleRate, outputSampleRate = 16000) {
  if (inputSampleRate === outputSampleRate) {
    return buffer;
  }
  const sampleRateRatio = inputSampleRate / outputSampleRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
      accum += buffer[i];
      count++;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult++;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

async function sendVoiceQuery(wavBlob) {
  const formData = new FormData();
  formData.append('file', wavBlob, 'voice_query.wav');

  try {
    const res = await fetch('/voice', {
      method: 'POST',
      body: formData
    });

    if (voiceStatusBar) voiceStatusBar.style.display = 'none';

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      appendAssistantMessage(`⚠️ Voice error: ${escapeHtml(err.detail || res.statusText)}`);
      return;
    }

    const data = await res.json();
    handleVoiceResponse(data);
  } catch (err) {
    if (voiceStatusBar) voiceStatusBar.style.display = 'none';
    console.error('Failed to send voice query:', err);
    appendAssistantMessage(`⚠️ Voice request failed: ${escapeHtml(err.message)}`);
  }
}

function handleVoiceResponse(data) {
  const transcript = data.transcript || '';
  queryInput.value = transcript;

  appendUserMessage(`🎙️ "${transcript}"`);

  // Play audio response if provided
  if (data.audio_url && voicePlayer) {
    voicePlayer.src = data.audio_url;
    voicePlayer.play().catch(e => console.warn('Audio auto-play blocked:', e));
  }

  const askRes = data.ask_response;
  const timings = data.timings_ms || {};
  const timingsBadge = `
    <div class="voice-timings">
      ASR: ${timings.asr || 0}ms | Search: ${timings.search || 0}ms | TTS: ${timings.tts || 0}ms | Total: ${timings.total || 0}ms
    </div>
  `;

  if (!askRes) {
    appendAssistantMessage(`
      <p>${escapeHtml(data.spoken_text || '')}</p>
      ${timingsBadge}
    `);
    return;
  }

  if (askRes.status === 'clarify') {
    appendVoiceClarificationCard(transcript, askRes.referent, askRes.options, data.spoken_text, timingsBadge);
  } else if (askRes.status === 'success') {
    appendVoiceResultsCard(transcript, askRes.parsed, askRes.results, data.spoken_text, timingsBadge);
  }
}

function appendVoiceClarificationCard(query, referent, options, spokenText, timingsBadge) {
  const optionsHtml = (options || []).map(opt => `
    <button class="cam-option-btn" onclick="resolveClarification('${escapeHtml(referent)}', '${escapeHtml(opt)}', '${escapeHtml(query)}')">
      ${escapeHtml(opt)}
    </button>
  `).join('');

  const html = `
    <div class="clarify-card">
      <div class="clarify-title">
        <span>⚠️ Clarification Required</span>
      </div>
      <p>🔊 <em>"${escapeHtml(spokenText)}"</em></p>
      <p style="margin-top: 8px; font-size: 13px; color: var(--text-muted);">
        Which camera covers this location? Click a button or speak the camera name:
      </p>
      <div class="clarify-options">
        ${optionsHtml}
      </div>
      ${timingsBadge}
    </div>
  `;
  appendAssistantMessage(html);
}

function appendVoiceResultsCard(query, parsed, results, spokenText, timingsBadge) {
  if (!results || results.length === 0) {
    appendAssistantMessage(`
      <p>🔊 <em>"${escapeHtml(spokenText)}"</em></p>
      <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px;">
        Object prompt: <code>${escapeHtml(parsed?.object_prompt || '')}</code> | Provider: ${parsed?.provider || 'fast'}
      </div>
      ${timingsBadge}
    `);
    return;
  }

  const resultsHtml = results.map((r) => `
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
      <div style="font-size: 14px; margin-bottom: 8px; color: var(--accent); font-weight: 500;">
        🔊 <em>"${escapeHtml(spokenText)}"</em>
      </div>
      <div class="results-container">
        ${resultsHtml}
      </div>
      ${timingsBadge}
    </div>
  `;
  appendAssistantMessage(fullHtml);
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
      const isStreaming = (data.status === 'streaming' || data.status === 'running');
      const isEnded = (data.status === 'finished' || data.status === 'stopped');

      const previewWrapper = document.getElementById('stream-preview-wrapper');
      const streamLiveImg = document.getElementById('stream-live-img');
      const previewCamBadge = document.getElementById('preview-camera-badge');
      const previewFileBadge = document.getElementById('preview-file-badge');

      if (isStreaming) {
        streamStatusText.textContent = `● STREAMING (${data.camera || 'mobile_cam01'})`;
        streamStatusText.style.background = 'rgba(16, 185, 129, 0.2)';
        streamStatusText.style.color = '#34d399';
        let detail = `FPS: ${data.fps || 0} | Frames: ${data.frames_read || 0} | Tracks: ${data.tracks_indexed || 0}`;
        if (data.latest_track) {
          detail += `\nLatest object: ${data.latest_track.color || ''} ${data.latest_track.label}`;
        }
        streamDetailText.innerText = detail;
        if (btnStartStream) btnStartStream.style.display = 'none';
        if (btnStopStream) btnStopStream.style.display = 'inline-block';

        if (previewWrapper) previewWrapper.style.display = 'block';
        if (previewCamBadge) previewCamBadge.textContent = data.camera || 'mobile_cam01';
        if (previewFileBadge && data.recorded_file) {
          previewFileBadge.textContent = `Recording: ${data.recorded_file}`;
        }
        if (streamLiveImg && (!streamLiveImg.src || streamLiveImg.src.indexOf('/stream/feed') === -1)) {
          streamLiveImg.src = '/stream/feed?t=' + Date.now();
        }
      } else if (isEnded) {
        streamStatusText.textContent = 'STOPPED';
        streamStatusText.style.background = 'rgba(245, 158, 11, 0.2)';
        streamStatusText.style.color = '#fbbf24';
        streamDetailText.innerText = `Stream ended. Processed ${data.frames_read || 0} frames | Indexed ${data.tracks_indexed || 0} tracks.\nRecorded video saved: ${data.recorded_file || 'footage/recorded/'}`;
        if (btnStartStream) btnStartStream.style.display = 'inline-block';
        if (btnStopStream) btnStopStream.style.display = 'none';
        if (streamLiveImg) streamLiveImg.src = '';
        if (previewWrapper) previewWrapper.style.display = 'none';
      } else {
        streamStatusText.textContent = (data.status || 'IDLE').toUpperCase();
        streamStatusText.style.background = 'rgba(56, 189, 248, 0.15)';
        streamStatusText.style.color = '#38bdf8';
        streamDetailText.innerText = data.status === 'idle' ? 'Enter your IP Webcam URL above and click Connect.' : `Status: ${data.status}`;
        if (btnStartStream) btnStartStream.style.display = 'inline-block';
        if (btnStopStream) btnStopStream.style.display = 'none';
        if (streamLiveImg) streamLiveImg.src = '';
        if (previewWrapper) previewWrapper.style.display = 'none';
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
      appendAssistantMessage(`&#9209; <strong>Live Mobile Capture Stopped</strong>.<br>The video has been saved to disk. <strong>Taking time now to run Grounding DINO</strong> to extract fine-grained objects (juice box, power bank, extension board, charger, speaker, laptop, monitor) into persistent storage.<br><br><span style="color: #fbbf24;">⏳ Look at the top header status: <strong>Grounding DINO Indexing</strong> will pulse while processing. Once it turns green, ask in chat to fetch your objects!</span>`);
      fetchCameras();
      checkIndexingStatus();
    }
  } catch (err) {
    alert('Failed to stop stream: ' + err.message);
  } finally {
    btnStopStream.disabled = false;
    checkStreamStatus();
  }
}

// --- Grounding-DINO Indexing Pipeline Status ---
async function checkIndexingStatus() {
  if (!indexingBadge || !indexingText || !indexingDot) return;
  try {
    const res = await fetch('/indexing/status');
    if (res.ok) {
      const data = await res.json();
      if (data.status === 'running') {
        indexingBadge.className = 'status-badge indexing-badge-running';
        indexingDot.className = 'status-dot pulsing';
        indexingDot.style.backgroundColor = '#f59e0b';
        indexingText.textContent = `⏳ Grounding DINO: Processing ${data.video || 'footage'}...`;
        previousIndexingState = 'running';
      } else if (data.status === 'completed') {
        indexingBadge.className = 'status-badge indexing-badge-completed';
        indexingDot.className = 'status-dot active';
        indexingDot.style.backgroundColor = '#10b981';
        indexingText.textContent = `✓ Grounding DINO: Ready to Fetch (${data.tracks_indexed || 0} objects)`;
        if (previousIndexingState === 'running') {
          appendAssistantMessage(`&#9989; <strong>Grounding DINO Indexing Complete!</strong><br>Indexed ${data.tracks_indexed || 0} desk objects into storage with high-precision bounding boxes. You can now fetch them anytime (e.g., <em>"where is the juice box?"</em> or <em>"where is the extension board?"</em>).`);
          previousIndexingState = 'completed';
        }
      } else if (data.status === 'error') {
        indexingBadge.className = 'status-badge indexing-badge-running';
        indexingDot.className = 'status-dot';
        indexingDot.style.backgroundColor = '#ef4444';
        indexingText.textContent = `❌ Indexing Error`;
      } else {
        indexingBadge.className = 'status-badge indexing-badge-idle';
        indexingDot.className = 'status-dot active';
        indexingDot.style.backgroundColor = '#38bdf8';
        indexingText.textContent = `Grounding DINO: Ready (${data.total_database_tracks || 0} tracks)`;
      }
    }
  } catch (err) {
    console.error('Failed to get indexing status:', err);
  }
}
