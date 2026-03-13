# RoadVision Frontend — Walkthrough

## What Was Built

Refactored the entire RoadVision frontend into a unified design system with shared components and 4 consistent pages.

## File Structure

```
RoadVision/
├── index.html              ← Dashboard
├── monitoring.html         ← Live Monitoring
├── detections.html         ← Detections
├── history.html            ← History
├── styles.css              ← Shared design system (all tokens & components)
├── components/
│   ├── sidebar.html        ← Reusable sidebar (fetched via JS)
│   ├── header.html         ← Reusable header (fetched via JS)
│   └── loader.js           ← Auto-loads sidebar/header & highlights active page
└── assets/
    ├── images/
    │   └── traffic-frame.jpg
    └── icons/
        ├── vehicle.svg
        ├── warning.svg
        ├── shield-check.svg
        └── chart-bar.svg
```

## How It Works

### Shared Components
- Each page contains `<div id="sidebar-slot">` and `<div id="header-slot">` placeholders
- [components/loader.js](file:///c:/Desktop/RoadVision/components/loader.js) runs on DOM load, fetches the HTML fragments, and injects them
- Active sidebar link is auto-detected from `window.location` and highlighted via `.active` class
- Page title in the header is set dynamically from a `pageTitles` map

### Design System ([styles.css](file:///c:/Desktop/RoadVision/styles.css))
All pages share a single CSS file with:
- **Color tokens**: `#0B1220` (background), `#111827` (cards), `#2563EB` (primary), `#22C55E` (success), `#EF4444` (alert)
- **Font**: Inter with system fallbacks
- **Component styles**: stat cards, data tables, badges, bounding boxes, detection cards, pagination
- **Micro-animations**: pulse for live badges, hover lift on cards, smooth transitions

### Pages

| Page | Content |
|------|---------|
| **Dashboard** | 4 stat cards + traffic frame with AI bounding box overlays |
| **Live Monitoring** | Large camera feed viewer + bounding boxes + stats row + detection log table |
| **Detections** | Card grid of 6 suspicious vehicles with registration & violation details |
| **History** | Full data table with vehicle images, registrations, violations, timestamps + pagination |

## Important Notes
- Pages must be served via a local server (e.g. `npx serve` or Live Server) for the `fetch()` component loading to work — opening raw HTML files via `file://` protocol will block fetch requests due to CORS
- Sidebar navigation uses `data-page` attributes matched against the filename to determine the active page
