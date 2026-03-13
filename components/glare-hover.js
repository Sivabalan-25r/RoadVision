document.addEventListener("DOMContentLoaded", () => {
  // Select all interactive containers we want the effect on
  const selectors = [
    '.stat-card', 
    '.detection-card', 
    '.card', 
    '.frame-viewer', 
    '.upload-zone'
  ];
  
  const allPossibleCards = document.querySelectorAll(selectors.join(','));
  
  // Filter out those marked with data-glare="false"
  const cards = Array.from(allPossibleCards).filter(card => {
    return card.getAttribute('data-glare') !== 'false';
  });

  cards.forEach(card => {
    // Ensure relative positioning for glare container
    const computedStyle = window.getComputedStyle(card);
    if (computedStyle.position === 'static') {
      card.style.position = 'relative';
    }

    // Add glare wrapper and element
    const glareContainer = document.createElement('div');
    glareContainer.className = 'glare-container';
    
    const glare = document.createElement('div');
    glare.className = 'glare';
    
    glareContainer.appendChild(glare);
    card.appendChild(glareContainer);
    
    // Interactive 3D tilt and glare logic
    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left; // x position within the element
      const y = e.clientY - rect.top;  // y position within the element
      
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      
      // Calculate rotation (max tilt: 10 degrees)
      const rotateX = ((y - centerY) / centerY) * -10; 
      const rotateY = ((x - centerX) / centerX) * 10;
      
      // Apply transform with smooth tracking
      card.style.transition = 'transform 0.1s ease-out, box-shadow 0.1s ease-out';
      card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
      
      // Calculate glare position
      const glareX = (x / rect.width) * 100;
      const glareY = (y / rect.height) * 100;
      
      glare.style.transition = 'opacity 0.2s ease';
      glare.style.opacity = '1';
      glare.style.background = `radial-gradient(circle at ${glareX}% ${glareY}%, rgba(255, 255, 255, 0.15) 0%, transparent 60%)`;
      
      // Dynamic box-shadow for depth
      card.style.boxShadow = `
        ${-rotateY}px ${rotateX}px 30px rgba(0, 0, 0, 0.5),
        0 15px 40px rgba(0,0,0,0.6)
      `;
      card.style.zIndex = '50';
    });

    card.addEventListener('mouseleave', () => {
      // Animate back to resting state
      card.style.transition = 'transform 0.5s ease, box-shadow 0.5s ease';
      card.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)`;
      card.style.boxShadow = ''; // Reset to CSS default
      card.style.zIndex = '';
      
      glare.style.transition = 'opacity 0.5s ease';
      glare.style.opacity = '0';
    });
  });
});
