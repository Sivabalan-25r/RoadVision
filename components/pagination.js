/**
 * EvasionEye Pagination Component
 * Handles client-side pagination for data tables.
 */

function initPagination() {
  const tableBody = document.querySelector('.data-table tbody');
  const paginationControls = document.getElementById('paginationControls');
  const paginationInfo = document.querySelector('.pagination .info');
  
  if (!tableBody || !paginationControls || !paginationInfo) return;

  const rowsPerPage = 10; // Default rows per page
  let currentPage = 1;

  function updateView() {
    const rows = Array.from(tableBody.querySelectorAll('tr'));
    const totalPages = Math.ceil(rows.length / rowsPerPage);
    
    // Boundary check for current page
    if (currentPage > totalPages && totalPages > 0) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    const start = (currentPage - 1) * rowsPerPage;
    const end = start + rowsPerPage;

    // Show/Hide rows
    rows.forEach((row, index) => {
      row.style.display = (index >= start && index < end) ? '' : 'none';
    });

    // Update info text
    const showingStart = rows.length === 0 ? 0 : start + 1;
    const showingEnd = Math.min(end, rows.length);
    paginationInfo.textContent = `Showing ${showingStart} to ${showingEnd} of ${rows.length} entries`;

    // Render controls
    paginationControls.innerHTML = '';
    if (totalPages <= 1) return;

    // Prev
    const prevBtn = document.createElement('button');
    prevBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size:18px">chevron_left</span>';
    prevBtn.disabled = currentPage === 1;
    prevBtn.onclick = () => { currentPage--; updateView(); };
    paginationControls.appendChild(prevBtn);

    // Numbers
    for (let i = 1; i <= totalPages; i++) {
      const btn = document.createElement('button');
      btn.textContent = i;
      if (i === currentPage) btn.classList.add('active');
      btn.onclick = () => { currentPage = i; updateView(); };
      paginationControls.appendChild(btn);
    }

    // Next
    const nextBtn = document.createElement('button');
    nextBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size:18px">chevron_right</span>';
    nextBtn.disabled = currentPage === totalPages;
    nextBtn.onclick = () => { currentPage++; updateView(); };
    paginationControls.appendChild(nextBtn);
  }

  // Initial update
  updateView();

  // Expose update function globally for dynamic content
  window.refreshPagination = updateView;
}

document.addEventListener("DOMContentLoaded", initPagination);
