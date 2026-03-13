document.addEventListener("DOMContentLoaded", () => {
  // Select all cards we want the effect on
  const cards = document.querySelectorAll('.stat-card, .detection-card');

  cards.forEach(card => {
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
      
      // Apply transform without transition for instantaneous tracking
      card.style.transition = 'transform 0.1s ease-out';
      card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02, 1.02, 1.02)`;
      
      // Calculate glare position
      const glareX = (x / rect.width) * 100;
      const glareY = (y / rect.height) * 100;
      
      glare.style.transition = 'opacity 0.2s ease';
      glare.style.opacity = '1';
      glare.style.background = `radial-gradient(circle at ${glareX}% ${glareY}%, rgba(255, 255, 255, 0.15) 0%, transparent 60%)`;
      
      // Optional: dynamic box-shadow for depth
      card.style.boxShadow = `
        ${-rotateY}px ${rotateX}px 20px rgba(0, 0, 0, 0.4),
        0 10px 30px rgba(0,0,0,0.5)
      `;
    });

    card.addEventListener('mouseleave', () => {
      // Animate back to resting state
      card.style.transition = 'transform 0.5s ease, box-shadow 0.5s ease';
      card.style.transform = `perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)`;
      card.style.boxShadow = ''; // Reset to CSS default
      
      glare.style.transition = 'opacity 0.5s ease';
      glare.style.opacity = '0';
    });
  });
});
