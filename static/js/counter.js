/* ===========================
   STAT NUMBER COUNT-UP EFFECT
   Auto-detects numbers inside .stat-card h2
   (works with "100+", "500+", "95%", "10K+" as-is,
   no HTML changes required)
=========================== */

document.addEventListener('DOMContentLoaded', () => {

    const statCards = document.querySelectorAll('.stat-card');

    statCards.forEach((card) => {
        const heading = card.querySelector('h2');
        if (!heading) return;

        const rawText = heading.textContent.trim();

        // Split into number + suffix, e.g. "10K+" -> 10, "K+"
        const match = rawText.match(/^([\d.]+)(.*)$/);
        if (!match) return;

        const targetNumber = parseFloat(match[1]);
        const suffix = match[2] || '';

        // Wrap the numeric part so we can animate just the digits
        heading.innerHTML = `<span class="num">0</span>${suffix}`;
        const numEl = heading.querySelector('.num');

        card.dataset.target = targetNumber;
        card.dataset.suffix = suffix;
        card._numEl = numEl;
        card._counted = false;
    });

    const animateCount = (card) => {
        if (card._counted) return;
        card._counted = true;

        const numEl = card._numEl;
        const target = parseFloat(card.dataset.target);
        const duration = 1600; // ms
        const startTime = performance.now();
        const isDecimal = card.dataset.target.includes('.');

        const step = (now) => {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // ease-out cubic for a smooth, natural finish
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = target * eased;

            numEl.textContent = isDecimal
                ? current.toFixed(1)
                : Math.round(current).toLocaleString();

            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                numEl.textContent = isDecimal
                    ? target.toFixed(1)
                    : target.toLocaleString();
            }
        };

        requestAnimationFrame(step);
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('in-view');
                animateCount(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.4 });

    statCards.forEach((card) => observer.observe(card));

});
