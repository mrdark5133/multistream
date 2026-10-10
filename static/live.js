/**
 * MULTIStream - Multi-Camera Live Screen Monitor Controller
 * Clean minimalistic boxy design, zero emojis, zero pill shapes.
 */

// Application State
let activeCameras = {};
let currentGridMode = 'grid-2x2';
let focusedCamera = null;
let pollInterval = null;
let healthInterval = null;

// DOM Elements
const liveGrid = document.getElementById('live-grid');
const emptyStateCard = document.getElementById('empty-state-card');
const focusTabBar = document.getElementById('focus-tab-bar');
const connectDrawer = document.getElementById('connect-drawer');
const btnToggleDrawer = document.getElementById('btn-toggle-drawer');
const drawerCamName = document.getElementById('drawer-cam-name');
const drawerCamUrl = document.getElementById('drawer-cam-url');
const btnSubmitStream = document.getElementById('btn-submit-stream');
const btnStopAll = document.getElementById('btn-stop-all');
const btnLaunchQuad = document.getElementById('btn-launch-quad');
const btnEmptyLaunchQuad = document.getElementById('btn-empty-launch-quad');

// Stats Elements
const statActiveSummary = document.getElementById('stat-active-summary');
const statGlobalFps = document.getElementById('stat-global-fps');
const statAvgLag = document.getElementById('stat-avg-lag');
const streamsCountText = document.getElementById('streams-count-text');
const streamsIndicator = document.getElementById('streams-indicator');
const vramText = document.getElementById('vram-text');

// Init
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  loadSavedPreferences();
  refreshStreams(true);
  fetchHealth();

  // Polling intervals
  pollInterval = setInterval(() => refreshStreams(false), 1500);
  healthInterval = setInterval(fetchHealth, 8000);
});

function setupEventListeners() {
  btnToggleDrawer.addEventListener('click', () => toggleDrawer());

  btnSubmitStream.addEventListener('click', async () => {
    const cam = drawerCamName.value.trim() || 'mobile_cam01';
    const url = drawerCamUrl.value.trim();
    if (!url) {
      alert('Please enter a stream source URL (e.g. http://192.168.1.105:8080/video)');
      return;
    }
    await connectStream(cam, url);
  });

  const btnNavSearch = document.getElementById('btn-nav-search');
  if (btnNavSearch) {
    btnNavSearch.addEventListener('click', (e) => {
      e.preventDefault();
      // Immediately sever all MJPEG stream connections before navigating
      document.querySelectorAll('.camera-live-feed').forEach(img => {
        img.src = '';
      });
      if (pollInterval) clearInterval(pollInterval);
      if (healthInterval) clearInterval(healthInterval);
      window.location.href = '/';
    });
  }

  btnStopAll.addEventListener('click', async () => {
    const activeNames = Object.keys(activeCameras).filter(k => activeCameras[k].running);
    if (activeNames.length === 0) return;
    if (!confirm(`Stop all ${activeNames.length} active camera streams?`)) return;

    btnStopAll.disabled = true;
    // Disconnect all image stream feeds immediately
    document.querySelectorAll('.camera-live-feed').forEach(img => {
      img.src = '';
      img.style.display = 'none';
    });
    document.querySelectorAll('.camera-status-pill').forEach(pill => {
      pill.textContent = 'STOPPED';
      pill.className = 'camera-status-pill stopped';
    });
    document.querySelectorAll('.rec-indicator').forEach(rec => {
      rec.style.display = 'none';
    });

    for (const name of activeNames) {
      if (activeCameras[name]) activeCameras[name].running = false;
      await stopCamera(name, false);
    }
    btnStopAll.disabled = false;
    refreshStreams(true);
  });

  if (btnLaunchQuad) {
    btnLaunchQuad.addEventListener('click', () => launchQuadDemo());
  }
  if (btnEmptyLaunchQuad) {
    btnEmptyLaunchQuad.addEventListener('click', () => launchQuadDemo());
  }
}

function loadSavedPreferences() {
  const savedMode = localStorage.getItem('multistream_live_grid_mode');
  if (savedMode && ['grid-auto', 'grid-2x2', 'grid-3x3', 'grid-1x1'].includes(savedMode)) {
    setGridMode(savedMode);
  }
}

function toggleDrawer(forceOpen = null) {
  if (forceOpen === true) {
    connectDrawer.classList.remove('hidden');
    btnToggleDrawer.textContent = 'Hide Drawer';
  } else if (forceOpen === false) {
    connectDrawer.classList.add('hidden');
    btnToggleDrawer.textContent = '+ Connect Camera';
  } else {
    const isHidden = connectDrawer.classList.toggle('hidden');
    btnToggleDrawer.textContent = isHidden ? '+ Connect Camera' : 'Hide Drawer';
  }
}

function fillPreset(camName, url) {
  drawerCamName.value = camName;
  drawerCamUrl.value = url;
  toggleDrawer(true);
}

// Quick 4-phone quad ingest launcher
async function launchQuadDemo() {
  const quadConfigs = [
    { cam: 'mobile_cam01', url: 'http://127.0.0.1:8081' },
    { cam: 'mobile_cam02', url: 'http://127.0.0.1:8082' },
    { cam: 'mobile_cam03', url: 'http://127.0.0.1:8083' },
    { cam: 'mobile_cam04', url: 'http://127.0.0.1:8084' }
  ];

  if (btnLaunchQuad) btnLaunchQuad.disabled = true;
  if (btnEmptyLaunchQuad) btnEmptyLaunchQuad.disabled = true;

  for (const cfg of quadConfigs) {
    await connectStream(cfg.cam, cfg.url, false);
  }

  if (btnLaunchQuad) btnLaunchQuad.disabled = false;
  if (btnEmptyLaunchQuad) btnEmptyLaunchQuad.disabled = false;

  setGridMode('grid-2x2');
  refreshStreams(true);
}

// Connect stream API
async function connectStream(cam, url, advanceInputs = true) {
  btnSubmitStream.disabled = true;
  btnSubmitStream.textContent = 'Connecting...';
  try {
    const res = await fetch('/stream/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera: cam, url: url })
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      alert(`Error starting ${cam}: ${err.detail || res.statusText}`);
      return;
    }

    if (advanceInputs) {
      drawerCamUrl.value = '';
      const match = cam.match(/(\d+)$/);
      if (match) {
        const nextNum = parseInt(match[1], 10) + 1;
        const prefix = cam.slice(0, match.index);
        drawerCamName.value = `${prefix}${String(nextNum).padStart(match[1].length, '0')}`;
      } else {
        drawerCamName.value = `${cam}_02`;
      }
    }
    refreshStreams(true);
  } catch (err) {
    alert(`Failed to start stream: ${err.message}`);
  } finally {
    btnSubmitStream.disabled = false;
    btnSubmitStream.textContent = '+ Connect Stream';
  }
}

// Stop camera stream API
async function stopCamera(camName, confirmPrompt = true) {
  if (confirmPrompt && !confirm(`Stop camera stream '${camName}'?`)) {
    return;
  }
  // Immediately update local state and sever feed image connection
  if (activeCameras[camName]) {
    activeCameras[camName].running = false;
  }
  const img = document.getElementById(`feed-img-${camName}`);
  if (img) {
    img.src = '';
    img.style.display = 'none';
  }
  const pill = document.getElementById(`status-pill-${camName}`);
  if (pill) {
    pill.textContent = 'STOPPED';
    pill.className = 'camera-status-pill stopped';
  }
  const dot = document.getElementById(`dot-${camName}`);
  if (dot) dot.className = 'status-indicator';
  const recHud = document.getElementById(`rec-hud-${camName}`);
  if (recHud) recHud.style.display = 'none';

  try {
    const res = await fetch('/stream/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera: camName, auto_ingest: true })
    });
    if (!res.ok) {
      console.error(`Stop failed for ${camName}:`, res.statusText);
    }
  } catch (err) {
    console.error(`Stop failed for ${camName}:`, err);
  } finally {
    refreshStreams(false);
  }
}

// Grid layout selector
function setGridMode(mode) {
  currentGridMode = mode;
  localStorage.setItem('multistream_live_grid_mode', mode);

  ['btn-mode-auto', 'btn-mode-2x2', 'btn-mode-3x3', 'btn-mode-1x1'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.classList.remove('active');
  });

  const activeBtn = document.getElementById(`btn-mode-${mode.replace('grid-', '')}`);
  if (activeBtn) activeBtn.classList.add('active');

  liveGrid.className = `live-grid ${mode}`;

  if (mode === 'grid-1x1') {
    focusTabBar.style.display = 'flex';
    renderFocusTabs();
  } else {
    focusTabBar.style.display = 'none';
  }

  renderCameraCards();
}

function focusCamera(camName) {
  focusedCamera = camName;
  setGridMode('grid-1x1');
}

// Synchronize streams from /stream/status
async function refreshStreams(forceRender = false) {
  try {
    const res = await fetch('/stream/status');
    if (!res.ok) return;

    const data = await res.json();
    const cameras = data.cameras || {};
    const camKeys = Object.keys(cameras);
    const totalActive = data.total_active || 0;

    activeCameras = cameras;

    // Update global aggregate metrics
    let totalFps = 0;
    let sumLag = 0;
    let lagCount = 0;

    camKeys.forEach(k => {
      const c = cameras[k];
      if (c.running) {
        totalFps += (c.effective_fps || 0);
        sumLag += (c.lag || 0);
        lagCount++;
      }
    });

    const avgLag = lagCount > 0 ? (sumLag / lagCount) : 0;

    statActiveSummary.innerHTML = `Streams: <b>${totalActive}</b> Active`;
    statGlobalFps.innerHTML = `Total FPS: <b>${totalFps.toFixed(1)}</b>`;
    statAvgLag.innerHTML = `Avg Lag: <b>${avgLag.toFixed(2)}s</b>`;
    streamsCountText.textContent = `${totalActive} Streams Active`;

    if (totalActive > 0) {
      streamsIndicator.className = 'status-indicator active';
    } else {
      streamsIndicator.className = 'status-indicator';
    }

    // Default focused camera if none selected
    if (!focusedCamera || !cameras[focusedCamera]) {
      const firstRunning = camKeys.find(k => cameras[k].running);
      focusedCamera = firstRunning || camKeys[0] || null;
    }

    // Update or re-render camera cards
    updateCameraCardsDOM(camKeys, forceRender);
  } catch (err) {
    console.error('Failed to sync streams:', err);
  }
}

// Reconcile and update camera cards without flickering video feeds
function updateCameraCardsDOM(camKeys, forceRender) {
  if (camKeys.length === 0) {
    if (emptyStateCard) emptyStateCard.style.display = 'flex';
    focusTabBar.style.display = 'none';
    const cards = liveGrid.querySelectorAll('.camera-card');
    cards.forEach(c => c.remove());
    return;
  }

  if (emptyStateCard) emptyStateCard.style.display = 'none';

  if (currentGridMode === 'grid-1x1') {
    focusTabBar.style.display = 'flex';
    renderFocusTabs();
  }

  // If cards count differs or forceRender requested, full render
  const existingCardMap = {};
  liveGrid.querySelectorAll('.camera-card').forEach(card => {
    const id = card.getAttribute('data-camera');
    if (id) existingCardMap[id] = card;
  });

  const visibleKeys = currentGridMode === 'grid-1x1' 
    ? (focusedCamera ? [focusedCamera] : [camKeys[0]])
    : camKeys;

  // Remove cards not in visibleKeys
  Object.keys(existingCardMap).forEach(id => {
    if (!visibleKeys.includes(id)) {
      existingCardMap[id].remove();
      delete existingCardMap[id];
    }
  });

  // Create or update visible cards
  visibleKeys.forEach(camName => {
    const camData = activeCameras[camName] || {};
    let card = existingCardMap[camName];

    if (!card || forceRender) {
      if (card) card.remove();
      card = createCameraCardElement(camName, camData);
      liveGrid.appendChild(card);
    } else {
      updateCardTelemetry(card, camData);
    }
  });
}

function renderCameraCards() {
  const camKeys = Object.keys(activeCameras);
  updateCameraCardsDOM(camKeys, true);
}

function renderFocusTabs() {
  const camKeys = Object.keys(activeCameras);
  focusTabBar.innerHTML = camKeys.map(k => {
    const isAct = k === focusedCamera;
    const isRunning = activeCameras[k] && activeCameras[k].running;
    return `
      <button 
        class="focus-tab-btn ${isAct ? 'active' : ''}" 
        onclick="switchFocusCamera('${escapeHtml(k)}')"
      >
        ${escapeHtml(k)} (${isRunning ? 'ONLINE' : 'STOPPED'})
      </button>
    `;
  }).join('');
}

function switchFocusCamera(camName) {
  focusedCamera = camName;
  renderFocusTabs();
  renderCameraCards();
}

function createCameraCardElement(camName, camData) {
  const isRunning = camData.running;
  const card = document.createElement('div');
  card.className = 'camera-card';
  card.setAttribute('data-camera', camName);

  const fps = (camData.effective_fps || 0).toFixed(1);
  const lag = (camData.lag || 0).toFixed(2);
  const framesRecv = camData.frames_received || 0;
  const tracksIndexed = camData.tracks_indexed || 0;
  const latestTrack = formatLatestTrack(camData.latest_track);
  const recCount = camData.reconnect_count || 0;
  const recFile = camData.recorded_file || 'active_capture.mp4';
  const streamUrl = camData.stream_url || camData.url || '';

  const feedUrl = isRunning 
    ? `/stream/feed?camera=${encodeURIComponent(camName)}&t=${Date.now()}`
    : '';

  card.innerHTML = `
    <div class="camera-card-header">
      <div class="camera-title-wrap">
        <span class="status-indicator ${isRunning ? 'active' : ''}" id="dot-${camName}"></span>
        <span class="camera-id-title">${escapeHtml(camName)}</span>
        <span class="camera-status-pill ${isRunning ? '' : 'stopped'}" id="status-pill-${camName}">
          ${isRunning ? 'STREAMING' : 'STOPPED'}
        </span>
      </div>
      <div class="camera-header-stats">
        <span class="stat-item">FPS: <b id="fps-${camName}">${fps}</b></span>
        <span class="stat-item">LAG: <b id="lag-${camName}">${lag}s</b></span>
      </div>
    </div>

    <div class="camera-screen-wrapper">
      ${isRunning ? `
        <img 
          class="camera-live-feed" 
          id="feed-img-${camName}" 
          src="${feedUrl}" 
          alt="${escapeHtml(camName)} Feed"
        >
      ` : `
        <div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); text-transform: uppercase;">
          Camera Stream Offline
        </div>
      `}

      <div class="camera-screen-overlay-top">
        <div class="rec-indicator" id="rec-hud-${camName}" style="${isRunning ? '' : 'display: none;'}">
          <span class="rec-dot"></span>
          <span>REC</span>
        </div>
        <div class="screen-timestamp" id="hud-cam-${camName}">${escapeHtml(camName)}</div>
      </div>

      <div class="camera-screen-actions">
        <button class="screen-action-btn" onclick="focusCamera('${escapeHtml(camName)}')">Focus</button>
        <button class="screen-action-btn" onclick="captureSnapshot('${escapeHtml(camName)}')">Snapshot</button>
        ${isRunning ? `
          <button class="screen-action-btn" onclick="stopCamera('${escapeHtml(camName)}')">Stop</button>
        ` : ''}
      </div>
    </div>

    <div class="camera-card-telemetry">
      <div class="telemetry-cell">
        <span class="telemetry-label">Frames Ingested</span>
        <span class="telemetry-value" id="frames-${camName}">${framesRecv.toLocaleString()}</span>
      </div>
      <div class="telemetry-cell">
        <span class="telemetry-label">Indexed Objects</span>
        <span class="telemetry-value" id="tracks-${camName}">${tracksIndexed} tracks</span>
      </div>
      <div class="telemetry-cell">
        <span class="telemetry-label">Latest Detections</span>
        <span class="telemetry-value" id="latest-${camName}">${escapeHtml(String(latestTrack))}</span>
      </div>
    </div>

    <div class="camera-card-footer">
      <span class="footer-source-url" title="${escapeHtml(streamUrl)}">
        Source: ${escapeHtml(streamUrl || 'local')}
      </span>
      <div class="footer-btn-group">
        <button class="box-pill-btn" onclick="focusCamera('${escapeHtml(camName)}')">Focus</button>
        <button class="box-pill-btn" onclick="captureSnapshot('${escapeHtml(camName)}')">Snapshot</button>
        ${isRunning ? `
          <button class="btn-secondary" style="height: 22px; padding: 0 8px; font-size: 10px;" onclick="stopCamera('${escapeHtml(camName)}')">Stop</button>
        ` : `
          <button class="btn-primary" style="height: 22px; padding: 0 8px; font-size: 10px;" onclick="reconnectCamera('${escapeHtml(camName)}', '${escapeHtml(streamUrl)}')">Restart</button>
        `}
      </div>
    </div>
  `;

  return card;
}

function formatLatestTrack(track) {
  if (!track) return 'none';
  if (typeof track === 'string') return track;
  if (typeof track === 'object') {
    const label = track.label || track.class || 'object';
    const id = track.id ? ` [${track.id}]` : '';
    return `${label}${id}`;
  }
  return String(track);
}

function updateCardTelemetry(card, camData) {
  const camName = card.getAttribute('data-camera');
  if (!camName) return;

  const isRunning = camData.running;
  const fps = (camData.effective_fps || 0).toFixed(1);
  const lag = (camData.lag || 0).toFixed(2);
  const framesRecv = camData.frames_received || 0;
  const tracksIndexed = camData.tracks_indexed || 0;
  const latestTrack = formatLatestTrack(camData.latest_track);

  const dot = document.getElementById(`dot-${camName}`);
  if (dot) dot.className = `status-indicator ${isRunning ? 'active' : ''}`;

  const pill = document.getElementById(`status-pill-${camName}`);
  if (pill) {
    pill.textContent = isRunning ? 'STREAMING' : 'STOPPED';
    pill.className = `camera-status-pill ${isRunning ? '' : 'stopped'}`;
  }

  const fpsEl = document.getElementById(`fps-${camName}`);
  if (fpsEl) fpsEl.textContent = fps;

  const lagEl = document.getElementById(`lag-${camName}`);
  if (lagEl) lagEl.textContent = `${lag}s`;

  const framesEl = document.getElementById(`frames-${camName}`);
  if (framesEl) framesEl.textContent = framesRecv.toLocaleString();

  const tracksEl = document.getElementById(`tracks-${camName}`);
  if (tracksEl) tracksEl.textContent = `${tracksIndexed} tracks`;

  const latestEl = document.getElementById(`latest-${camName}`);
  if (latestEl) latestEl.textContent = escapeHtml(latestTrack);

  const recHud = document.getElementById(`rec-hud-${camName}`);
  if (recHud) recHud.style.display = isRunning ? 'flex' : 'none';

  // Manage image stream socket connection vs offline placeholder
  const img = document.getElementById(`feed-img-${camName}`);
  const wrapper = card.querySelector('.camera-screen-wrapper');
  let offlineNotice = card.querySelector(`.offline-notice-${camName}`);

  if (!isRunning) {
    if (img) {
      img.src = '';
      img.style.display = 'none';
    }
    if (!offlineNotice && wrapper) {
      offlineNotice = document.createElement('div');
      offlineNotice.className = `offline-notice-${camName}`;
      offlineNotice.style.cssText = 'font-family: var(--font-mono); font-size: 11px; color: var(--text-faint); text-transform: uppercase; text-align: center;';
      offlineNotice.textContent = 'Camera Stream Offline';
      wrapper.appendChild(offlineNotice);
    } else if (offlineNotice) {
      offlineNotice.style.display = 'block';
    }
  } else {
    if (offlineNotice) offlineNotice.style.display = 'none';
    if (img) {
      img.style.display = 'block';
      if (!img.src || img.src === '' || img.src.indexOf('/stream/feed') === -1) {
        img.src = `/stream/feed?camera=${encodeURIComponent(camName)}&t=${Date.now()}`;
      }
    }
  }
}

function reconnectCamera(camName, streamUrl) {
  if (!streamUrl) {
    fillPreset(camName, '');
    return;
  }
  connectStream(camName, streamUrl, false);
}

function captureSnapshot(camName) {
  const img = document.getElementById(`feed-img-${camName}`);
  if (img && img.src) {
    window.open(`/stream/feed?camera=${encodeURIComponent(camName)}`, '_blank');
  } else {
    alert(`No active frame available for camera '${camName}'.`);
  }
}

// Fetch system health & VRAM
async function fetchHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) return;
    const data = await res.json();
    if (data.vram && vramText) {
      vramText.textContent = `VRAM: ${data.vram.used_mib} / ${data.vram.total_mib} MB`;
    }
  } catch (e) {
    // ignore
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
