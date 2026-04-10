// EvasionEye API Client
// Handles all API calls to the backend with camera-based authentication

// Get API base URL from config
const API_BASE_URL = window.EvasionEyeConfig ? window.EvasionEyeConfig.API_BASE_URL : 'http://localhost:8000';

// Get current camera ID from localStorage
function getCurrentCameraId() {
  const cameraData = localStorage.getItem('evasioneye_camera');
  if (cameraData) {
    try {
      const camera = JSON.parse(cameraData);
      return camera.id || 'CAM-001';
    } catch (e) {
      console.error('Error parsing camera data:', e);
    }
  }
  return 'CAM-001'; // Default camera
}

// Camera API
const CameraAPI = {
  // Get all cameras
  async getAll() {
    const response = await fetch(`${API_BASE_URL}/api/cameras`);
    const data = await response.json();
    return data.cameras;
  },

  // Get camera info
  async get(cameraId) {
    const response = await fetch(`${API_BASE_URL}/api/cameras/${cameraId}`);
    const data = await response.json();
    return data.camera;
  },

  // Update camera info
  async update(cameraId, cameraData) {
    const response = await fetch(`${API_BASE_URL}/api/cameras/${cameraId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cameraData)
    });
    return await response.json();
  }
};

// Detection API
const DetectionAPI = {
  // Get detections for current camera
  async getAll(violationsOnly = false, limit = 100) {
    const cameraId = getCurrentCameraId();
    const response = await fetch(
      `${API_BASE_URL}/api/detections/${cameraId}?violations_only=${violationsOnly}&limit=${limit}`
    );
    const data = await response.json();
    return data.detections;
  },

  // Add detection
  async add(detection) {
    const cameraId = getCurrentCameraId();
    const response = await fetch(`${API_BASE_URL}/api/detections/${cameraId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(detection)
    });
    return await response.json();
  },

  // Clear all detections
  async clearAll() {
    const cameraId = getCurrentCameraId();
    const response = await fetch(`${API_BASE_URL}/api/detections/${cameraId}`, {
      method: 'DELETE'
    });
    return await response.json();
  },

  // Get stats
  async getStats() {
    const cameraId = getCurrentCameraId();
    const response = await fetch(`${API_BASE_URL}/api/stats/${cameraId}`);
    const data = await response.json();
    return data.stats;
  }
};

// Process frame (live monitoring)
async function processFrame(imageBlob) {
  const formData = new FormData();
  formData.append('file', imageBlob, 'frame.jpg');

  const response = await fetch(`${API_BASE_URL}/api/process-frame`, {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  return data.detections;
}

// Analyze video
async function analyzeVideo(videoFile) {
  const formData = new FormData();
  formData.append('video', videoFile);

  const response = await fetch(`${API_BASE_URL}/analyze-video`, {
    method: 'POST',
    body: formData
  });

  const data = await response.json();
  return data.detections;
}

// Export API
window.EvasionEyeAPI = {
  Camera: CameraAPI,
  Detection: DetectionAPI,
  processFrame,
  analyzeVideo,
  getCurrentCameraId
};
