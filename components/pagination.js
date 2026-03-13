document.addEventListener("DOMContentLoaded", () => {
  const tableBody = document.querySelector('.data-table tbody');
  const paginationControls = document.getElementById('paginationControls');
  const paginationInfo = document.querySelector('.pagination .info');
  
  if (!tableBody || !paginationControls || !paginationInfo) return;

  const rows = Array.from(tableBody.querySelectorAll('tr'));
  const rowsPerPage = 5; // Change this to show more/less rows per page
  const totalPages = Math.ceil(rows.length / rowsPerPage);
  let currentPage = 1;

  function displayRows(page) {
    const start = (page - 1) * rowsPerPage;
    const end = start + rowsPerPage;

    // Hide all rows, then show the ones for current page
    rows.forEach((row, index) => {
      if (index >= start && index < end) {
        row.style.display = ''; // Show
      } else {
        row.style.display = 'none'; // Hide
      }
    });

    // Update info text
    const showingStart = rows.length === 0 ? 0 : start + 1;
    const showingEnd = Math.min(end, rows.length);
    paginationInfo.textContent = `Showing ${showingStart} to ${showingEnd} of ${rows.length} entries`;
  }

  function renderPagination() {
    paginationControls.innerHTML = ''; // Clear existing buttons

    // If there's 1 or fewer pages, we don't need pagination buttons at all
    if (totalPages <= 1) {
      return; 
    }

    // Prev Button
    const prevBtn = document.createElement('button');
    prevBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size:18px">chevron_left</span>';
    prevBtn.disabled = currentPage === 1;
    prevBtn.addEventListener('click', () => {
      if (currentPage > 1) {
        currentPage--;
        updateView();
      }
    });
    paginationControls.appendChild(prevBtn);

    // Page number buttons
    for (let i = 1; i <= totalPages; i++) {
      const pageBtn = document.createElement('button');
      pageBtn.textContent = i;
      if (i === currentPage) {
        pageBtn.classList.add('active');
      }
      
      pageBtn.addEventListener('click', () => {
        currentPage = i;
        updateView();
      });
      paginationControls.appendChild(pageBtn);
    }

    // Next Button
    const nextBtn = document.createElement('button');
    nextBtn.innerHTML = '<span class="material-symbols-outlined" style="font-size:18px">chevron_right</span>';
    nextBtn.disabled = currentPage === totalPages;
    nextBtn.addEventListener('click', () => {
      if (currentPage < totalPages) {
        currentPage++;
        updateView();
      }
    });
    paginationControls.appendChild(nextBtn);
  }

  function updateView() {
    displayRows(currentPage);
    renderPagination();
  }

  // Initialize
  updateView();
});
