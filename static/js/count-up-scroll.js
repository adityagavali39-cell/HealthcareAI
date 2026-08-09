/* ==================================================
   COUNT-UP ON SCROLL — for .stats .stat-card h2
   ==================================================
   HTML madhe pratyek .stat-card la data-target ani
   (optional) data-suffix add kar. Udaharan:

   <div class="stat-card" data-target="100" data-suffix="+">
       <i class="fa-solid fa-user-doctor"></i>
       <h2>0</h2>
       <p>Doctors</p>
   </div>

   <div class="stat-card" data-target="500" data-suffix="+">
       ...
       <h2>0</h2>
       <p>Hospitals</p>
   </div>

   <div class="stat-card" data-target="95" data-suffix="%">
       ...
       <h2>0</h2>
       <p>Prediction Accuracy</p>
   </div>

   <div class="stat-card" data-target="10000" data-suffix="+" data-compact="true">
       ...
       <h2>0</h2>
       <p>Users</p>
   </div>

   Ha file </body> chya aadhi <script src="count-up-scroll.js"></script>
   asa link kar.
================================================== */

function animateStatCount(card) {
  const target = parseInt(card.dataset.target, 10);
  const suffix = card.dataset.suffix || "";
  const compact = card.dataset.compact === "true";
  const numEl = card.querySelector("h2");
  const duration = 1200;
  const start = performance.now();

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // ease-out
    const value = Math.round(target * eased);

    let display = value;
    if (compact && value >= 1000) {
      display = (value / 1000).toFixed(value % 1000 === 0 ? 0 : 1) + "K";
    }

    numEl.textContent = display + suffix;

    if (progress < 1) {
      requestAnimationFrame(tick);
    }
  }

  requestAnimationFrame(tick);
}

document.addEventListener("DOMContentLoaded", function () {
  const statCards = document.querySelectorAll(".stat-card[data-target]");

  const observer = new IntersectionObserver(
    (entries, obs) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateStatCount(entry.target);
          obs.unobserve(entry.target); // ek vela count zala ki thamb
        }
      });
    },
    { threshold: 0.4 }
  );

  statCards.forEach((card) => observer.observe(card));
});