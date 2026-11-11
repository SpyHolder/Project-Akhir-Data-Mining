const D = window.DATA || null;
const R = D ? D.results : {};
const P = D ? D.prep : {};
const E = D ? D.eda : {};

const MODELS = ["Baseline", "Linear Regression", "Random Forest", "Gradient Boosting"];
const $ = id => document.getElementById(id);
const fmt = (v, n = 2) => (v == null || Number.isNaN(Number(v))) ? "-" : Number(v).toLocaleString("id-ID", { maximumFractionDigits: n });

const best = D ? MODELS.reduce((a, b) => R[a].MAE <= R[b].MAE ? a : b) : "Baseline";
const bestModel = D ? MODELS.slice(1).reduce((a, b) => R[a].MAE <= R[b].MAE ? a : b) : "Gradient Boosting";

function getTheme() {
  try {
    return localStorage.getItem("telemetry_theme") || "dark";
  } catch (_) {
    return "dark";
  }
}

function getSeriesColors(theme = getTheme()) {
  if (theme === "light") {
    return {
      "Baseline": "#475569",
      "Linear Regression": "#b45309",
      "Random Forest": "#047857",
      "Gradient Boosting": "#6d28d9",
      "Aktual": "#0369a1",
      "Riwayat": "#334155"
    };
  }
  return {
    "Baseline": "#94a3b8",
    "Linear Regression": "#fbbf24",
    "Random Forest": "#34d399",
    "Gradient Boosting": "#a78bfa",
    "Aktual": "#38bdf8",
    "Riwayat": "#e2e8f0"
  };
}

let COL = getSeriesColors();

function syncChartDefaults(theme = getTheme()) {
  COL = getSeriesColors(theme);
  if (!window.Chart) return;
  const isLight = theme === "light";
  Chart.defaults.font.family = "'Plus Jakarta Sans', -apple-system, sans-serif";
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.color = isLight ? "#475569" : "#94a3b8";
  Chart.defaults.borderColor = isLight ? "rgba(71, 85, 105, 0.16)" : "rgba(148, 163, 184, 0.14)";
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.tooltip.backgroundColor = isLight ? "#0f172a" : "#090d16";
  Chart.defaults.plugins.tooltip.titleColor = "#f1f5f9";
  Chart.defaults.plugins.tooltip.bodyColor = "#e2e8f0";
  Chart.defaults.plugins.tooltip.borderColor = isLight ? "#0369a1" : "#38bdf8";
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.elements.bar.borderRadius = 4;
  Chart.defaults.elements.line.borderWidth = 2;
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem("telemetry_theme", theme);
  } catch (_) {}
  syncChartDefaults(theme);
  const btn = $("theme-toggle");
  if (btn) {
    btn.textContent = theme === "dark" ? "Mode Terang" : "Mode Gelap";
    btn.setAttribute("aria-pressed", String(theme === "light"));
  }
  if (window.Chart && Chart.instances) {
    Object.values(Chart.instances).forEach(ch => {
      if (ch.options && ch.options.scales) {
        Object.values(ch.options.scales).forEach(s => {
          if (s.ticks) s.ticks.color = Chart.defaults.color;
          if (s.title) s.title.color = Chart.defaults.color;
          if (s.grid) s.grid.color = Chart.defaults.borderColor;
        });
      }
      ch.update("none");
    });
  }
  if (typeof window.onThemeChange === "function") {
    window.onThemeChange(theme);
  }
}

syncChartDefaults(getTheme());
document.documentElement.setAttribute("data-theme", getTheme());

const mk = (id, cfg) => {
  const el = $(id);
  if (!el) return null;
  return new Chart(el, cfg);
};

const bar = (id, labels, datasets, opt = {}) => mk(id, { type: "bar", data: { labels, datasets }, options: opt });

const table = (id, head, rows, cls = () => "") => {
  const el = $(id);
  if (!el) return;
  if (!rows || !rows.length) {
    el.innerHTML = `<tbody><tr><td colspan="${head.length}" class="text-muted">Tidak ada baris untuk ditampilkan pada filter ini.</td></tr></tbody>`;
    return;
  }
  el.innerHTML =
    `<thead><tr>${head.map(h => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r, i) => `<tr class="${cls(r, i)}">${r.map(c => `<td class="num">${c}</td>`).join("")}</tr>`).join("") +
    "</tbody>";
};

const PAGES = [
  ["index.html", "Ringkasan"],
  ["eda.html", "Eksplorasi Data"],
  ["data.html", "Kualitas Data"],
  ["model.html", "Model & Fitur"],
  ["evaluasi.html", "Evaluasi"],
  ["prediksi.html", "Prediksi & Uji Akurasi"]
];

function renderState(container, title, detail, actionLabel, onAction) {
  if (!container) return;
  container.innerHTML = `<div class="state-box" role="status" aria-live="polite">
    <h3>${title}</h3>
    <p class="text-muted">${detail}</p>
    ${actionLabel ? `<button type="button" class="btn-control state-action">${actionLabel}</button>` : ""}
  </div>`;
  if (actionLabel && typeof onAction === "function") {
    const btn = container.querySelector(".state-action");
    if (btn) btn.onclick = onAction;
  }
}

function layout(active, title, sub) {
  const currentTheme = getTheme();
  $("top").innerHTML = `
  <a href="#main" class="skip-link">Lewati ke konten utama</a>
  <nav class="site-nav" aria-label="Navigasi utama">
    <div class="shell">
      <div class="nav-bar">
        <a href="index.html" class="brand">
          <span class="brand-mark" aria-hidden="true"></span>
          <span>TELEMETRI DROPPING</span>
        </a>
        <div class="nav-controls">
          <button type="button" id="theme-toggle" class="btn-control" aria-pressed="${currentTheme === "light"}">
            ${currentTheme === "dark" ? "Mode Terang" : "Mode Gelap"}
          </button>
          <button type="button" id="nav-toggle" class="btn-control nav-toggle" aria-expanded="false" aria-controls="nav-menu">
            Menu Halaman
          </button>
        </div>
      </div>
      <div id="nav-menu" class="nav-links">
        ${PAGES.map(([h, l]) => `<a href="${h}" class="nav-link ${h === active ? "active" : ""}" ${h === active ? 'aria-current="page"' : ""}>${l}</a>`).join("")}
      </div>
    </div>
  </nav>
  <header class="page-header">
    <div class="shell page-header-grid">
      <div>
        <h1>${title}</h1>
        <p class="page-sub">${sub}</p>
      </div>
      <div class="meta-tag">Historis: Jun-Agu 2026 | Prediksi: Sep-Okt 2026 (18 Outlet Sampel)</div>
    </div>
  </header>`;

  $("foot").innerHTML = `
  <footer class="site-footer">
    <div class="shell footer-inner">
      <span>Sistem Peramalan Penjualan Dropping Barang (Baseline, Linear Regression, Random Forest, Gradient Boosting)</span>
      <span class="font-mono">Riwayat Aktual: Jun-Agu 2026 | Horizon Prediksi: Sep-Okt 2026</span>
    </div>
  </footer>`;

  const navBtn = $("nav-toggle");
  const navMenu = $("nav-menu");
  if (navBtn && navMenu) {
    navBtn.onclick = () => {
      const open = navMenu.classList.toggle("open");
      navBtn.setAttribute("aria-expanded", String(open));
      navBtn.textContent = open ? "Tutup Menu" : "Menu Halaman";
    };
    document.addEventListener("keydown", e => {
      if (e.key === "Escape" && navMenu.classList.contains("open")) {
        navMenu.classList.remove("open");
        navBtn.setAttribute("aria-expanded", "false");
        navBtn.textContent = "Menu Halaman";
        navBtn.focus();
      }
    });
  }

  const themeBtn = $("theme-toggle");
  if (themeBtn) {
    themeBtn.onclick = () => {
      const next = getTheme() === "dark" ? "light" : "dark";
      applyTheme(next);
    };
  }

  if (!D) {
    renderState(
      $("main"),
      "Data belum termuat",
      "Berkas site/data.js tidak ditemukan atau gagal dimuat. Jalankan perintah python pipeline.py untuk membangkitkan ulang data model.",
      "Muat Ulang Halaman",
      () => window.location.reload()
    );
  }
}

const kpiHTML = (items, focalIdx = 0) =>
  items.map(([label, val, sub], i) => `
    <div class="kpi ${i === focalIdx ? "kpi-focal" : ""}">
      <div class="kpi-label">${label}</div>
      <div class="kpi-val">${val}</div>
      ${sub ? `<div class="kpi-sub">${sub}</div>` : ""}
    </div>`).join("");
