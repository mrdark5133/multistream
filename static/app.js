// MULTIStream Frontend Logic - Clean Minimalist Boxy Design
// No emojis, no AI fluff, strict geometric layout.

const chatMessages = document.getElementById('chat-messages');
const searchForm = document.getElementById('search-form');
const queryInput = document.getElementById('query-input');
const healthBadge = document.getElementById('health-badge');
const healthIndicator = document.getElementById('health-indicator');
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

let knownCameras = [];
let currentPolyPoints = [];
let currentBgImage = new Image();
let streamPollTimer = null;
let indexingPollTimer = null;
let previousIndexingState = 'idle';

let audioContext = null;
let mediaStream = null;
let mediaRecorder = null;
let scriptProcessor = null;
let audioSourceNode = null;
let recordedChunks = [];
let fallbackAudioChunks = [];
let isRecording = false;
let isInitializing = false;
let pendingStop = false;
let isToggleMode = false;
let pressStartTime = 0;
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
    if (isRecording) {
      stopVoiceRecording(false);
    }
    const q = queryInput.value.trim();
    if (q) {
      handleSearch(q);
    }
  });

  // Dual-mode Microphone Events (Click-to-Toggle and Push-to-Talk)
  if (btnMic) {
    const handlePressStart = (e) => {
      e.preventDefault();
      if (isRecording && isToggleMode) {
        // Second click when in toggle mode stops and submits
        stopVoiceRecording(true);
        return;
      }
      if (isRecording || isInitializing) return;
      pressStartTime = Date.now();
      isToggleMode = false;
      startVoiceRecording();
    };

    const handlePressEnd = (e) => {
      e.preventDefault();
      if (!isRecording && !isInitializing) return;
      const pressDuration = Date.now() - pressStartTime;
      if (pressDuration < 350) {
        // Short click: remain in toggle recording mode
        isToggleMode = true;
        if (btnMic) {
          btnMic.classList.add('recording');
          if (micText) micText.textContent = 'Stop';
        }
        if (voiceStatusBar) {
          voiceStatusBar.style.display = 'flex';
          if (voiceStatusText) voiceStatusText.textContent = 'Recording audio — click Stop when done';
        }
      } else {
        // Long press: push-to-talk release
        isToggleMode = false;
        stopVoiceRecording(true);
      }
    };

    btnMic.addEventListener('mousedown', handlePressStart);
    btnMic.addEventListener('mouseup', handlePressEnd);
    btnMic.addEventListener('mouseleave', () => {
      if (isRecording && !isToggleMode) {
        stopVoiceRecording(true);
      }
    });

    btnMic.addEventListener('touchstart', handlePressStart, { passive: false });
    btnMic.addEventListener('touchend', handlePressEnd, { passive: false });
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
  if (btnOpenPoly) {
    btnOpenPoly.addEventListener('click', () => {
      polyModal.classList.remove('hidden');
      loadCameraCanvas();
    });
  }

  if (btnClosePoly) {
    btnClosePoly.addEventListener('click', () => {
      polyModal.classList.add('hidden');
    });
  }

  if (polyCamSelect) {
    polyCamSelect.addEventListener('change', () => {
      loadCameraCanvas();
    });
  }

  if (polyCanvas) {
    polyCanvas.addEventListener('click', (e) => {
      const rect = polyCanvas.getBoundingClientRect();
      const scaleX = polyCanvas.width / rect.width;
      const scaleY = polyCanvas.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;
      currentPolyPoints.push([x, y]);
      drawPolygon();
    });
  }

  if (btnResetPoly) {
    btnResetPoly.addEventListener('click', () => {
      currentPolyPoints = [];
      drawPolygon();
    });
  }

  if (btnSavePoly) {
    btnSavePoly.addEventListener('click', handleSavePolygonAlias);
  }
}

// --- Health & Cameras ---
async function fetchHealth() {
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      if (healthIndicator) healthIndicator.classList.add('active');
      if (healthText) healthText.textContent = `Online (${data.models.device.toUpperCase()})`;
      if (vramText) vramText.textContent = `VRAM: ${data.vram.used_mib} / ${data.vram.total_mib} MB`;
    }
  } catch (err) {
    if (healthText) healthText.textContent = 'Offline';
    console.error('Health check failed:', err);
  }
}

async function fetchCameras() {
  try {
    const res = await fetch('/cameras');
    if (res.ok) {
      knownCameras = await res.json();
      if (polyCamSelect) {
        polyCamSelect.innerHTML = knownCameras.map(c => 
          `<option value="${c.camera}">${c.camera} (${c.width}x${c.height})</option>`
        ).join('');
      }
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
    if (!res.ok) {
      const errText = await res.text();
      let msg = errText;
      try {
        const parsed = JSON.parse(errText);
        msg = parsed.detail || parsed.message || errText;
      } catch (e) {}
      throw new Error(`Server ${res.status}: ${msg}`);
    }
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
    appendAssistantMessage(`Error executing query: ${err.message}`);
  }
}

// --- Voice Push-To-Talk Logic ---
async function initVoiceAudio() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('Microphone audio capture is not supported in this browser.');
  }
  if (!mediaStream || !mediaStream.active) {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });
  }
}

async function startVoiceRecording() {
  if (isRecording || isInitializing) return;
  isInitializing = true;
  pendingStop = false;

  if (btnMic) {
    btnMic.classList.add('recording');
    if (micText) micText.textContent = isToggleMode ? 'Stop' : 'Listening';
  }
  if (voiceStatusBar) {
    voiceStatusBar.style.display = 'flex';
    if (voiceStatusText) {
      voiceStatusText.textContent = isToggleMode
        ? 'Recording audio — click Stop when done'
        : 'Listening... Speak query';
    }
  }

  try {
    await initVoiceAudio();
    recordingStartTime = Date.now();
    isRecording = true;
    isInitializing = false;

    if (window.MediaRecorder && mediaStream) {
      let mimeType = '';
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
        mimeType = 'audio/webm;codecs=opus';
      } else if (MediaRecorder.isTypeSupported('audio/webm')) {
        mimeType = 'audio/webm';
      } else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) {
        mimeType = 'audio/ogg;codecs=opus';
      }

      recordedChunks = [];
      mediaRecorder = mimeType ? new MediaRecorder(mediaStream, { mimeType }) : new MediaRecorder(mediaStream);
      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          recordedChunks.push(e.data);
        }
      };
      mediaRecorder.start(100);
    } else {
      // AudioContext fallback
      fallbackAudioChunks = [];
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!audioContext || audioContext.state === 'closed') {
        audioContext = new AudioCtx();
      }
      if (audioContext.state === 'suspended') {
        await audioContext.resume();
      }
      audioSourceNode = audioContext.createMediaStreamSource(mediaStream);
      scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
      scriptProcessor.onaudioprocess = (e) => {
        if (!isRecording) return;
        const channelData = e.inputBuffer.getChannelData(0);
        fallbackAudioChunks.push(new Float32Array(channelData));
      };
      audioSourceNode.connect(scriptProcessor);
      const zeroGain = audioContext.createGain();
      zeroGain.gain.value = 0;
      scriptProcessor.connect(zeroGain);
      zeroGain.connect(audioContext.destination);
    }

    if (pendingStop) {
      pendingStop = false;
      stopVoiceRecording(true);
    }
  } catch (err) {
    isInitializing = false;
    isRecording = false;
    isToggleMode = false;
    if (btnMic) {
      btnMic.classList.remove('recording');
      if (micText) micText.textContent = 'Talk';
    }
    if (voiceStatusBar) voiceStatusBar.style.display = 'none';
    console.error('Error starting audio recording:', err);
    appendAssistantMessage(`Microphone access failed: ${escapeHtml(err.message)}`);
  }
}

async function stopVoiceRecording(shouldSend = true) {
  if (isInitializing) {
    pendingStop = true;
    return;
  }
  if (!isRecording) return;

  isRecording = false;
  isToggleMode = false;
  const duration = (Date.now() - recordingStartTime) / 1000;

  if (btnMic) {
    btnMic.classList.remove('recording');
    if (micText) micText.textContent = 'Talk';
  }

  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    return new Promise((resolve) => {
      mediaRecorder.onstop = async () => {
        if (!shouldSend || duration < 0.25 || recordedChunks.length === 0) {
          if (voiceStatusBar) voiceStatusBar.style.display = 'none';
          resolve();
          return;
        }
        if (voiceStatusText) {
          voiceStatusText.textContent = 'Processing: Whisper ASR & retrieval...';
        }
        const mime = mediaRecorder.mimeType || 'audio/webm';
        const blob = new Blob(recordedChunks, { type: mime });
        const ext = mime.includes('ogg') ? 'voice.ogg' : 'voice.webm';
        await sendVoiceQuery(blob, ext);
        resolve();
      };
      try {
        mediaRecorder.stop();
      } catch (e) {
        console.warn('Error stopping mediaRecorder:', e);
        resolve();
      }
    });
  } else if (scriptProcessor) {
    try {
      scriptProcessor.disconnect();
    } catch (e) {}
    scriptProcessor = null;

    if (!shouldSend || duration < 0.25 || fallbackAudioChunks.length === 0) {
      if (voiceStatusBar) voiceStatusBar.style.display = 'none';
      return;
    }
    if (voiceStatusText) {
      voiceStatusText.textContent = 'Processing: Whisper ASR & retrieval...';
    }

    let totalLength = 0;
    for (const chunk of fallbackAudioChunks) totalLength += chunk.length;
    const mergedSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of fallbackAudioChunks) {
      mergedSamples.set(chunk, offset);
      offset += chunk.length;
    }
    const sampleRate = audioContext ? audioContext.sampleRate : 16000;
    const downsampled = downsampleBuffer(mergedSamples, sampleRate, 16000);
    const wavBlob = encodeWAV(downsampled, 16000);
    await sendVoiceQuery(wavBlob, 'voice.wav');
  }
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

async function sendVoiceQuery(audioBlob, filename = 'voice.webm') {
  const formData = new FormData();
  formData.append('file', audioBlob, filename);

  try {
    const res = await fetch('/voice', {
      method: 'POST',
      body: formData
    });

    if (voiceStatusBar) voiceStatusBar.style.display = 'none';

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      appendAssistantMessage(`Voice error: ${escapeHtml(err.detail || res.statusText)}`);
      return;
    }

    const data = await res.json();
    handleVoiceResponse(data);
  } catch (err) {
    if (voiceStatusBar) voiceStatusBar.style.display = 'none';
    console.error('Failed to send voice query:', err);
    appendAssistantMessage(`Voice request failed: ${escapeHtml(err.message)}`);
  }
}

function handleVoiceResponse(data) {
  const transcript = data.transcript || '';
  queryInput.value = transcript;

  appendUserMessage(`Audio query: "${transcript}"`);

  // Play audio response if provided
  if (data.audio_url && voicePlayer) {
    voicePlayer.src = data.audio_url;
    voicePlayer.play().catch(e => console.warn('Audio auto-play blocked:', e));
  }

  const askRes = data.ask_response;
  const timings = data.timings_ms || {};
  const timingsBadge = `
    <div class="voice-timings">
      ASR: ${timings.asr || 0}ms | SEARCH: ${timings.search || 0}ms | TTS: ${timings.tts || 0}ms | TOTAL: ${timings.total || 0}ms
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
      <div class="clarify-title">CLARIFICATION REQUIRED: REFERENT UNMAPPED</div>
      <p>Spoken prompt: <em>"${escapeHtml(spokenText)}"</em></p>
      <p style="margin-top: 8px; font-size: 12px; color: var(--text-muted);">
        Select camera to bind alias:
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
      <p>Spoken response: <em>"${escapeHtml(spokenText)}"</em></p>
      <div style="font-size: 11px; font-family: var(--font-mono); color: var(--text-muted); margin-top: 6px;">
        PROMPT: ${escapeHtml(parsed?.object_prompt || '')} | PROVIDER: ${parsed?.provider || 'fast'}
      </div>
      ${timingsBadge}
    `);
    return;
  }

  const resultsHtml = results.map((r) => `
    <div class="result-card">
      <div class="result-preview" id="preview-${r.id}">
        <img src="${r.snapshot_url}" alt="Snapshot ${r.label}" onerror="this.src='/static/placeholder.jpg'">
      </div>
      <div class="result-info">
        <div>
          <div class="meta-grid">
            <div class="meta-box">
              <span class="meta-key">Match</span>
              <span class="meta-val">${(r.score * 100).toFixed(1)}%</span>
            </div>
            <div class="meta-box">
              <span class="meta-key">Camera</span>
              <span class="meta-val">${escapeHtml(r.camera)}</span>
            </div>
            <div class="meta-box">
              <span class="meta-key">Object</span>
              <span class="meta-val">${escapeHtml(r.label)}</span>
            </div>
            ${r.color && r.color !== 'unknown' ? `
            <div class="meta-box">
              <span class="meta-key">Color</span>
              <span class="meta-val">${escapeHtml(r.color)}</span>
            </div>` : ''}
            <div class="meta-box">
              <span class="meta-key">Timestamp</span>
              <span class="meta-val">${formatTimestamp(r.timestamp)}</span>
            </div>
            <div class="meta-box">
              <span class="meta-key">Offset</span>
              <span class="meta-val">${r.offset_seconds.toFixed(2)}s</span>
            </div>
          </div>
          <div class="result-id-line">
            ID: ${escapeHtml(r.id)} (${r.type})
          </div>
        </div>
        <button class="btn-play" onclick="playClip('${r.id}', '${r.clip_url}')">Play Clip</button>
      </div>
    </div>
  `).join('');

  const fullHtml = `
    <div>
      <div style="font-family: var(--font-mono); font-size: 12px; margin-bottom: 8px; color: var(--text-main); font-weight: 600;">
        SPOKEN ANSWER: "${escapeHtml(spokenText)}"
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
    <div class="msg-header-tag">USER</div>
    <div class="msg-body">${escapeHtml(text)}</div>
  `;
  chatMessages.appendChild(msgEl);
  scrollToBottom();
}

function appendAssistantMessage(htmlContent) {
  const msgEl = document.createElement('div');
  msgEl.className = 'message assistant-msg';
  msgEl.innerHTML = `
    <div class="msg-header-tag">SYSTEM</div>
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
    <div class="msg-header-tag">SYSTEM</div>
    <div class="msg-body">
      <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">
        PROCESSING: Executing spatio-temporal index search...
      </span>
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
      <div class="clarify-title">CLARIFICATION REQUIRED: REFERENT UNMAPPED</div>
      <p>Location referent <strong>"${escapeHtml(referent)}"</strong> is not bound to a camera.</p>
      <p style="margin-top: 8px; font-size: 12px; color: var(--text-muted);">
        Select camera to bind alias:
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
      appendAssistantMessage(`Alias registered: <strong>${escapeHtml(referent)}</strong> &rarr; <strong>${escapeHtml(cameraName)}</strong>. Resuming query...`);
      handleSearch(originalQuery);
    }
  } catch (err) {
    alert('Failed to register alias: ' + err.message);
  }
}

function appendResultsCard(query, parsed, results, elapsedMs) {
  if (!results || results.length === 0) {
    appendAssistantMessage(`
      <p>No matching objects found for <em>"${escapeHtml(query)}"</em>.</p>
      <div style="font-size: 11px; font-family: var(--font-mono); color: var(--text-muted); margin-top: 6px;">
        OBJECT: ${escapeHtml(parsed.object_prompt)} | PROVIDER: ${parsed.provider} | LATENCY: ${elapsedMs}ms
      </div>
    `);
    return;
  }

  const resultsHtml = results.map((r) => `
    <div class="result-card">
      <div class="result-preview" id="preview-${r.id}">
        <img src="${r.snapshot_url}" alt="Snapshot ${r.label}" onerror="this.src='/static/placeholder.jpg'">
      </div>
      <div class="result-info">
        <div>
          <div class="meta-grid">
            <div class="meta-box">
              <span class="meta-key">Match</span>
              <span class="meta-val">${(r.score * 100).toFixed(1)}%</span>
            </div>
            <div class="meta-box">
              <span class="meta-key">Camera</span>
              <span class="meta-val">${escapeHtml(r.camera)}</span>
            </div>
            <div class="meta-box">
              <span class="meta-key">Object</span>
              <span class="meta-val">${escapeHtml(r.label)}</span>
            </div>
            ${r.color && r.color !== 'unknown' ? `
            <div class="meta-box">
              <span class="meta-key">Color</span>
              <span class="meta-val">${escapeHtml(r.color)}</span>
            </div>` : ''}
            <div class="meta-box">
              <span class="meta-key">Timestamp</span>
              <span class="meta-val">${formatTimestamp(r.timestamp)}</span>
            </div>
            <div class="meta-box">
              <span class="meta-key">Offset</span>
              <span class="meta-val">${r.offset_seconds.toFixed(2)}s</span>
            </div>
          </div>
          <div class="result-id-line">
            ID: ${escapeHtml(r.id)} (${r.type})
          </div>
        </div>
        <button class="btn-play" onclick="playClip('${r.id}', '${r.clip_url}')">Play Clip</button>
      </div>
    </div>
  `).join('');

  const fullHtml = `
    <div>
      <div class="results-summary-bar">
        <span>Results: ${results.length} matched</span>
        <span>Latency: ${elapsedMs} ms</span>
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

  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, polyCanvas.width, polyCanvas.height);
  
  fetch('/cameras').then(res => res.json()).then(cams => {
    const camInfo = cams.find(c => c.camera === selectedCam);
    if (camInfo) {
      currentBgImage.onload = () => {
        ctx.drawImage(currentBgImage, 0, 0, polyCanvas.width, polyCanvas.height);
        drawPolygon();
      };
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
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, polyCanvas.width, polyCanvas.height);
  }

  if (currentPolyPoints.length === 0) return;

  ctx.strokeStyle = '#ffffff';
  ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
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

  // Draw square vertices (boxy style)
  ctx.fillStyle = '#ffffff';
  for (const pt of currentPolyPoints) {
    ctx.fillRect(pt[0] - 3, pt[1] - 3, 6, 6);
  }
}

async function handleSavePolygonAlias() {
  const name = polyAliasName.value.trim();
  const cam = polyCamSelect.value;
  if (!name || !cam) {
    alert('Enter alias name and select camera.');
    return;
  }
  if (currentPolyPoints.length < 3) {
    alert('Mark at least 3 points on canvas to define boundary.');
    return;
  }

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
      alert(`Alias '${name}' registered with ${normPoly.length}-point boundary.`);
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

// --- Live Multi-Camera Stream Handlers ---
let selectedPreviewCam = null;

window.selectPreviewCamera = function(cameraName) {
  selectedPreviewCam = cameraName;
  const previewCamBadge = document.getElementById('preview-camera-badge');
  const streamLiveImg = document.getElementById('stream-live-img');
  const previewWrapper = document.getElementById('stream-preview-wrapper');
  if (previewCamBadge) previewCamBadge.textContent = cameraName;
  if (streamLiveImg) {
    streamLiveImg.src = `/stream/feed?camera=${encodeURIComponent(cameraName)}&t=${Date.now()}`;
  }
  if (previewWrapper) previewWrapper.style.display = 'block';
};

window.handleStopCamera = async function(cameraName) {
  try {
    const res = await fetch('/stream/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera: cameraName })
    });
    if (res.ok) {
      appendAssistantMessage(`STREAM STOPPED: Camera ${escapeHtml(cameraName)} stopped. Ingest complete.`);
      fetchCameras();
      checkIndexingStatus();
    }
  } catch (err) {
    alert(`Failed to stop stream for ${cameraName}: ${err.message}`);
  } finally {
    checkStreamStatus();
  }
};

async function checkStreamStatus() {
  if (!streamStatusText || !streamDetailText) return;
  try {
    const res = await fetch('/stream/status');
    if (res.ok) {
      const data = await res.json();
      const cameras = data.cameras || {};
      const camKeys = Object.keys(cameras);
      const totalActive = data.total_active || 0;

      const activeBadge = document.getElementById('stream-total-active-badge');
      if (activeBadge) activeBadge.textContent = `${totalActive} active`;

      const listContainer = document.getElementById('stream-cameras-list');
      if (listContainer) {
        if (camKeys.length === 0) {
          listContainer.innerHTML = `
            <div style="padding: 12px; font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); text-align: center;">
              No active streams. Add a camera above to start ingest.
            </div>
          `;
        } else {
          listContainer.innerHTML = camKeys.map(camName => {
            const cam = cameras[camName];
            const isRunning = cam.running;
            return `
              <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 10px; border-bottom: 1px solid var(--border-subtle); background: #ffffff;">
                <div style="display: flex; flex-direction: column; gap: 2px;">
                  <div style="display: flex; align-items: center; gap: 6px;">
                    <span style="font-family: var(--font-mono); font-weight: 700; font-size: 11px;">${escapeHtml(camName)}</span>
                    <span style="font-family: var(--font-mono); font-size: 9px; padding: 1px 4px; background: ${isRunning ? '#18181b' : '#71717a'}; color: #fff;">${isRunning ? 'RUNNING' : 'STOPPED'}</span>
                  </div>
                  <span style="font-family: var(--font-mono); font-size: 10px; color: var(--text-muted); word-break: break-all; max-width: 200px;">${escapeHtml(cam.stream_url || '')}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                  <div style="text-align: right; font-family: var(--font-mono); font-size: 10px;">
                    <div>FPS: <b>${cam.effective_fps || 0}</b> | LAG: <b>${cam.lag || 0}s</b></div>
                    <div style="color: var(--text-muted);">REC: ${cam.reconnect_count || 0} | ERR: ${cam.errors ? cam.errors.length : 0}</div>
                  </div>
                  <div style="display: flex; gap: 4px;">
                    <button onclick="selectPreviewCamera('${escapeHtml(camName)}')" class="box-pill-btn" style="padding: 2px 6px; font-size: 10px;">View</button>
                    ${isRunning ? `<button onclick="handleStopCamera('${escapeHtml(camName)}')" class="btn-secondary" style="height: 24px; padding: 0 6px; font-size: 10px;">Stop</button>` : ''}
                  </div>
                </div>
              </div>
            `;
          }).join('');
        }
      }

      // Preview view handling
      const previewWrapper = document.getElementById('stream-preview-wrapper');
      const streamLiveImg = document.getElementById('stream-live-img');
      const previewCamBadge = document.getElementById('preview-camera-badge');

      if (totalActive > 0) {
        streamStatusText.textContent = `INGESTING (${totalActive} CAMERAS)`;
        streamDetailText.innerText = `Active streams: ${camKeys.filter(k => cameras[k].running).join(', ')}`;
        
        if (!selectedPreviewCam || !cameras[selectedPreviewCam] || !cameras[selectedPreviewCam].running) {
          const firstRunning = camKeys.find(k => cameras[k].running);
          if (firstRunning) {
            selectedPreviewCam = firstRunning;
          }
        }

        if (selectedPreviewCam && cameras[selectedPreviewCam] && cameras[selectedPreviewCam].running) {
          if (previewWrapper) previewWrapper.style.display = 'block';
          if (previewCamBadge) previewCamBadge.textContent = selectedPreviewCam;
          if (streamLiveImg && (!streamLiveImg.src || streamLiveImg.src.indexOf('/stream/feed') === -1)) {
            streamLiveImg.src = `/stream/feed?camera=${encodeURIComponent(selectedPreviewCam)}&t=${Date.now()}`;
          }
        }
      } else {
        streamStatusText.textContent = 'IDLE';
        streamDetailText.innerText = 'Ready for multi-camera connections.';
        if (previewWrapper) previewWrapper.style.display = 'none';
        if (streamLiveImg) streamLiveImg.src = '';
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
    alert('Enter stream URL');
    return;
  }
  btnStartStream.disabled = true;
  btnStartStream.textContent = 'Connecting...';
  try {
    const res = await fetch('/stream/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera: cam, url: url })
    });
    if (res.ok) {
      appendAssistantMessage(`STREAM STARTED: Ingesting from ${escapeHtml(url)} under camera ${escapeHtml(cam)}. Live detections and isolated tracking running.`);
      fetchCameras();
      selectedPreviewCam = cam;
      // Clear URL input and advance camera name for next phone
      streamUrlInput.value = '';
      const match = cam.match(/(\d+)$/);
      if (match) {
        const nextNum = parseInt(match[1], 10) + 1;
        const prefix = cam.slice(0, match.index);
        streamCameraName.value = `${prefix}${String(nextNum).padStart(match[1].length, '0')}`;
      } else {
        streamCameraName.value = `${cam}_02`;
      }
    } else {
      const err = await res.json();
      alert('Error starting stream: ' + (err.detail || res.statusText));
    }
  } catch (err) {
    alert('Failed to connect to stream: ' + err.message);
  } finally {
    btnStartStream.disabled = false;
    btnStartStream.textContent = '+ Add / Start Stream';
    checkStreamStatus();
  }
}

async function handleStopStream() {
  if (selectedPreviewCam) {
    await handleStopCamera(selectedPreviewCam);
  } else {
    try {
      await fetch('/stream/stop', { method: 'POST' });
    } catch (e) {}
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
        if (indexingDot) indexingDot.className = 'status-indicator pulsing';
        indexingText.textContent = `DINO: INDEXING ${data.video || 'FOOTAGE'}...`;
        previousIndexingState = 'running';
      } else if (data.status === 'completed') {
        if (indexingDot) indexingDot.className = 'status-indicator active';
        indexingText.textContent = `DINO: READY (${data.tracks_indexed || 0} OBJECTS)`;
        if (previousIndexingState === 'running') {
          appendAssistantMessage(`INDEXING COMPLETE: Indexed ${data.tracks_indexed || 0} fine-grained objects into database.`);
          previousIndexingState = 'completed';
        }
      } else if (data.status === 'error') {
        if (indexingDot) indexingDot.className = 'status-indicator';
        indexingText.textContent = `DINO: ERROR`;
      } else {
        if (indexingDot) indexingDot.className = 'status-indicator active';
        indexingText.textContent = `DINO: READY (${data.total_database_tracks || 0} TRACKS)`;
      }
    }
  } catch (err) {
    console.error('Failed to get indexing status:', err);
  }
}
