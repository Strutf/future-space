// Utilidades comuns das páginas: tradução, idioma, estrelinhas de giz e formatação de números.

const LANG_KEY = "fs-lang";

function storedLang() {
  try { return localStorage.getItem(LANG_KEY); } catch { return null; }
}
function storeLang(lang) {
  try { localStorage.setItem(LANG_KEY, lang); } catch { /* navegação privada: só não lembra */ }
}

// Primeira visita: usa o primeiro idioma do navegador que o site tem (senão, inglês).
// Depois que a pessoa escolhe um idioma no menu, a escolha dela sempre vale.
(function chooseLanguage() {
  const root = document.documentElement;
  let urls = {};
  try { urls = JSON.parse(root.dataset.urls || "{}"); } catch { return; }
  const browser = (navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language || ""])
    .map((l) => l.toLowerCase().slice(0, 2));
  const preferred = storedLang() || browser.find((l) => l in urls) || "en";
  if (preferred !== root.dataset.lang && urls[preferred]) {
    location.replace(urls[preferred] + location.hash);
  }
})();

// Texto traduzido, com {variáveis}: t("winner_few", { n: 3 })
function t(key, vars = {}) {
  const text = (window.T && window.T[key]) ?? key;
  return text.replace(/\{(\w+)\}/g, (_, name) => (vars[name] ?? ""));
}

const fmt = (v, d = 1) =>
  v == null ? "–" : Number(v).toLocaleString(window.T.locale, { maximumFractionDigits: d, minimumFractionDigits: d });

// "07/11" em português, "11/07" em inglês.
const dayLabel = (iso) =>
  new Date(iso).toLocaleDateString(window.T.locale, { day: "2-digit", month: "2-digit", timeZone: "UTC" });

const hour = (iso) => iso.slice(11, 13) + window.T.hourSuffix;

async function getJson(url) {
  const r = await fetch(url);
  return r.ok ? r.json() : null;
}

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

document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;
  document.querySelectorAll(".nav a[data-page]").forEach((a) => {
    if (a.dataset.page === page) a.setAttribute("aria-current", "page");
  });
  const menu = document.querySelector(".lang-menu");
  if (menu) {
    menu.querySelectorAll("a[data-lang]").forEach((a) => {
      a.addEventListener("click", (e) => {
        storeLang(a.dataset.lang);
        // mantém a âncora (#formulas, #storm) ao trocar de idioma
        if (location.hash) { e.preventDefault(); location.href = a.href + location.hash; }
      });
    });
    // fecha ao clicar fora ou apertar Esc
    document.addEventListener("click", (e) => { if (!menu.contains(e.target)) menu.open = false; });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && menu.open) { menu.open = false; menu.querySelector("summary").focus(); } });
  }
  scatterStars(window.innerWidth < 600 ? 8 : 16);
});
