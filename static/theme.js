// Theme toggle: light / dark / auto (system)
(function() {
    const saved = localStorage.getItem('theme');
    if (saved === 'dark') document.documentElement.setAttribute('data-theme', 'dark');
    else if (saved === 'light') document.documentElement.setAttribute('data-theme', 'light');
})();

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const isDarkSystem = window.matchMedia('(prefers-color-scheme: dark)').matches;

    let next;
    if (!current) {
        // Currently auto — go to opposite of system
        next = isDarkSystem ? 'light' : 'dark';
    } else if (current === 'dark') {
        next = 'light';
    } else {
        // Remove override, go back to auto
        next = null;
    }

    if (next) {
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
    } else {
        document.documentElement.removeAttribute('data-theme');
        localStorage.removeItem('theme');
    }
    updateThemeIcon();
}

function updateThemeIcon() {
    const btn = document.getElementById('theme-btn');
    if (!btn) return;
    const theme = document.documentElement.getAttribute('data-theme');
    const isDarkSystem = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDark = theme === 'dark' || (!theme && isDarkSystem);
    btn.textContent = isDark ? '\u2600' : '\u263E';  // sun / moon
    btn.title = isDark ? 'Switch to light mode' : 'Switch to dark mode';
}

document.addEventListener('DOMContentLoaded', updateThemeIcon);
