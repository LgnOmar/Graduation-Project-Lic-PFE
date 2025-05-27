/* Optional JS for JibJob static site (e.g., mobile nav, smooth scroll) */
// Hamburger menu toggle (if needed)
// Smooth scroll for anchor links

document.addEventListener('DOMContentLoaded', function() {
  const hamburger = document.querySelector('.hamburger');
  const nav = document.querySelector('header nav');
  if (hamburger && nav) {
    hamburger.addEventListener('click', function() {
      nav.classList.toggle('mobile-nav-active');
    });
    // Optional: close nav when clicking a link (for better UX)
    nav.querySelectorAll('a').forEach(function(link) {
      link.addEventListener('click', function() {
        nav.classList.remove('mobile-nav-active');
      });
    });
  }
});
