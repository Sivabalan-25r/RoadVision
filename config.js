/**
 * RoadVision Configuration
 * 
 * Centralized configuration for API endpoints and application settings.
 * Update API_BASE_URL for production deployment.
 */

const RoadVisionConfig = {
    // API Configuration
    // For local development: 'http://localhost:8000'
    // For production: update to your deployed backend URL (e.g., 'https://api.roadvision.com')
    API_BASE_URL: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
        ? 'http://localhost:8000'
        : window.location.origin,

    // API Endpoints
    ENDPOINTS: {
        HEALTH: '/health',
        ANALYZE_VIDEO: '/analyze-video',
        PROCESS_FRAME: '/api/process-frame',
        LIVE_DETECTIONS: '/api/live-detections'
    },

    // Video Processing Settings
    MAX_VIDEO_DURATION: 60, // seconds
    FRAME_INTERVAL: 5, // process every Nth frame

    // UI Settings
    DETECTION_CONFIDENCE_THRESHOLD: 0.6,
    ITEMS_PER_PAGE: 10
};

// Helper function to get full API URL
RoadVisionConfig.getApiUrl = function (endpoint) {
    return this.API_BASE_URL + (this.ENDPOINTS[endpoint] || endpoint);
};

// Make config globally available
window.RoadVisionConfig = RoadVisionConfig;
