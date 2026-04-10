// ============================================================
// EvasionEye — Evidence Store (localStorage)
// Shared across Dashboard, Detections, and History pages
// ============================================================

const EVIDENCE_KEY = 'evasioneye_evidence';
const MAX_EVIDENCE_ITEMS = 20;

/**
 * Get all stored evidence entries.
 * @returns {Array} Array of evidence objects
 */
function getEvidence() {
  try {
    const raw = localStorage.getItem(EVIDENCE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.warn('Evidence store: failed to read', e);
    return [];
  }
}

/**
 * Get evidence for a specific plate.
 * @param {string} plate — detected plate string
 * @returns {Object|null} Evidence entry or null
 */
function getEvidenceByPlate(plate) {
  const all = getEvidence();
  return all.find(e => e.plate === plate) || null;
}

/**
 * Save a new evidence entry.
 * Keeps only the latest MAX_EVIDENCE_ITEMS.
 * @param {Object} detection — detection result from API
 * @param {string} dataUrl — captured frame as data URL
 */
function saveEvidence(detection, dataUrl) {
  const entry = {
    plate: detection.detected_plate,
    corrected: detection.correct_plate,
    violation: detection.violation,
    confidence: detection.confidence,
    frame: detection.frame || null,
    image: dataUrl,
    timestamp: new Date().toISOString()
  };

  const all = getEvidence();

  // Replace existing entry for same plate, or add new
  const existingIdx = all.findIndex(e => e.plate === entry.plate);
  if (existingIdx !== -1) {
    all[existingIdx] = entry;
  } else {
    all.push(entry);
  }

  // Keep only the latest MAX_EVIDENCE_ITEMS
  while (all.length > MAX_EVIDENCE_ITEMS) {
    all.shift(); // remove oldest
  }

  try {
    localStorage.setItem(EVIDENCE_KEY, JSON.stringify(all));
  } catch (e) {
    console.warn('Evidence store: storage full, clearing oldest items', e);
    // If storage is full, remove half the entries and retry
    all.splice(0, Math.floor(all.length / 2));
    try {
      localStorage.setItem(EVIDENCE_KEY, JSON.stringify(all));
    } catch (e2) {
      console.error('Evidence store: unable to save', e2);
    }
  }
}

/**
 * Clear all evidence from storage.
 */
function clearEvidence() {
  localStorage.removeItem(EVIDENCE_KEY);
}

/**
 * Capture a video frame with bounding box overlay as JPEG data URL.
 * @param {HTMLVideoElement} video — the video element
 * @param {Object} detection — detection with bbox [x%, y%, w%, h%]
 * @returns {string} JPEG data URL at 0.7 quality
 */
function captureEvidenceFrame(video, detection) {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || video.clientWidth;
  canvas.height = video.videoHeight || video.clientHeight;
  const ctx = canvas.getContext('2d');

  // Draw the video frame
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Draw bounding box if available
  if (detection.bbox && detection.bbox.length >= 4) {
    const [xPct, yPct, wPct, hPct] = detection.bbox;
    const x = (xPct / 100) * canvas.width;
    const y = (yPct / 100) * canvas.height;
    const w = (wPct / 100) * canvas.width;
    const h = (hPct / 100) * canvas.height;

    // Red box for violations
    ctx.strokeStyle = '#EF4444';
    ctx.lineWidth = 3;
    ctx.strokeRect(x, y, w, h);

    // Semi-transparent fill
    ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
    ctx.fillRect(x, y, w, h);

    // Label background
    const label = detection.detected_plate || 'Unknown';
    ctx.font = 'bold 14px Inter, sans-serif';
    const textMetrics = ctx.measureText(label);
    const labelW = textMetrics.width + 16;
    const labelH = 24;

    ctx.fillStyle = '#EF4444';
    ctx.fillRect(x, y - labelH, labelW, labelH);

    // Label text
    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x + 8, y - labelH / 2);
  }

  // Compress as JPEG at 0.7 quality
  return canvas.toDataURL('image/jpeg', 0.7);
}
