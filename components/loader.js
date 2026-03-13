/**
 * RoadVision — Component Loader
 * Fetches sidebar.html and header.html, injects them into the page,
 * and highlights the active sidebar link based on the current URL.
 */
document.addEventListener('DOMContentLoaded', async () => {
  // Determine current page key from filename
  const path = window.location.pathname;
  const filename = path.substring(path.lastIndexOf('/') + 1) || 'index.html';
  const pageKey = filename.replace('.html', '') || 'index';

  // Page titles map
  const pageTitles = {
    index: 'Dashboard',
    monitoring: 'Live Monitoring',
    detections: 'Detections',
    history: 'History'
  };

  // Load sidebar
  const sidebarSlot = document.getElementById('sidebar-slot');
  if (sidebarSlot) {
    try {
      const res = await fetch('components/sidebar.html');
      const html = await res.text();
      sidebarSlot.innerHTML = html;

      // Highlight active link
      const links = sidebarSlot.querySelectorAll('.sidebar-nav a');
      links.forEach(link => {
        if (link.getAttribute('data-page') === pageKey) {
          link.classList.add('active');
        }
      });
    } catch (e) {
      console.error('Failed to load sidebar:', e);
    }
  }

  // Load header
  const headerSlot = document.getElementById('header-slot');
  if (headerSlot) {
    try {
      const res = await fetch('components/header.html');
      const html = await res.text();
      headerSlot.innerHTML = html;

      // Set page title
      const titleEl = headerSlot.querySelector('#page-title');
      if (titleEl && pageTitles[pageKey]) {
        titleEl.textContent = pageTitles[pageKey];
      }
    } catch (e) {
      console.error('Failed to load header:', e);
    }
  }

  // Load Background (Dark Veil)
  try {
    const res = await fetch('components/dark-veil.html');
    const html = await res.text();
    // Inject at the beginning of body
    document.body.insertAdjacentHTML('afterbegin', html);

    // Initialize WebGL Animation
    const { initDarkVeil } = await import('./dark-veil.js');
    initDarkVeil();
  } catch (e) {
    console.error('Failed to load background:', e);
  }
});
