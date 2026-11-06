// Utilidades comuns das páginas: estrelinhas de giz e formatação de números.

const STAR_PATH = "M12 2 L14.6 9.2 L22 9.4 L16 14 L18.4 21.4 L12 16.9 L5.6 21.6 L8 14.1 L2 9.5 L9.4 9.1 Z";

function starSvg() {
  return `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${STAR_PATH}" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" filter="url(#wobble)"/></svg>`;
}

// Espalha estrelinhas pelo céu, evitando o miolo da página onde fica o texto.
function scatterStars(count) {
  const sky = document.querySelector(".sky");
  if (!sky) return;
  const colors = ["var(--chalk)", "var(--chalk)", "var(--chalk-yellow)", "var(--chalk-pink)", "var(--chalk-blue)"];
  for (let i = 0; i < count; i++) {
    const star = document.createElement("span");
    star.className = "star";
    const size = 10 + Math.random() * 22;
    const left = Math.random() < 0.5 ? Math.random() * 14 : 86 + Math.random() * 12;
    Object.assign(star.style, {
      width: size + "px", height: size + "px",
      left: left + "%", top: 12 + Math.random() * 84 + "%", // abaixo da faixa do menu
      color: colors[i % colors.length],
      animationDelay: (Math.random() * 3).toFixed(2) + "s",
    });
    star.style.setProperty("--r", Math.round(Math.random() * 40 - 20) + "deg");
    star.innerHTML = starSvg();
    sky.appendChild(star);
  }
}

const fmt = (v, d = 1) =>
  v == null ? "–" : Number(v).toLocaleString("pt-BR", { maximumFractionDigits: d, minimumFractionDigits: d });

async function getJson(url) {
  const r = await fetch(url);
  return r.ok ? r.json() : null;
}

document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;
  document.querySelectorAll(".nav a").forEach((a) => {
    if (a.dataset.page === page) a.setAttribute("aria-current", "page");
  });
  scatterStars(window.innerWidth < 600 ? 8 : 16);
});
