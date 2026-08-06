// ===============================
// GLOBAL LIGHT / DARK THEME
// ===============================

// Elements
const body = document.body;
const themeBtn = document.getElementById("theme-btn");

// -------------------------------
// Icons (inline SVG — never depends
// on Font Awesome loading, so it
// can't fall back to a wrong glyph
// like a gear icon)
// -------------------------------

const MOON_ICON =
    '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true">' +
    '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>' +
    '</svg>';

const SUN_ICON =
    '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<circle cx="12" cy="12" r="5"/>' +
    '<line x1="12" y1="1" x2="12" y2="3"/>' +
    '<line x1="12" y1="21" x2="12" y2="23"/>' +
    '<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>' +
    '<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>' +
    '<line x1="1" y1="12" x2="3" y2="12"/>' +
    '<line x1="21" y1="12" x2="23" y2="12"/>' +
    '<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>' +
    '<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>' +
    '</svg>';

// -------------------------------
// Apply Theme
// -------------------------------

function setTheme(mode) {

    if (mode === "dark") {

        body.classList.add("dark-mode");

        localStorage.setItem("theme", "dark");

        if (themeBtn) {
            themeBtn.innerHTML = SUN_ICON;
        }

    } else {

        body.classList.remove("dark-mode");

        localStorage.setItem("theme", "light");

        if (themeBtn) {
            themeBtn.innerHTML = MOON_ICON;
        }

    }

}

// -------------------------------
// Load Saved Theme
// -------------------------------

const savedTheme = localStorage.getItem("theme");

if (savedTheme) {

    setTheme(savedTheme);

} else {

    setTheme("light");

}

// -------------------------------
// Toggle Theme
// -------------------------------

if (themeBtn) {

    themeBtn.addEventListener("click", () => {

        if (body.classList.contains("dark-mode")) {

            setTheme("light");

        } else {

            setTheme("dark");

        }

    });

}