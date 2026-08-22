const API = "";

async function api(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return res.json();
  return res;
}

function navActive() {
  const page = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll("nav a").forEach((a) => {
    const href = a.getAttribute("href") || "";
    a.classList.toggle("active", href.endsWith(page) || (page === "" && href.endsWith("index.html")));
  });
}

function fmtState(s) {
  return s || "—";
}

async function loadHealth() {
  const el = document.getElementById("health-status");
  if (!el) return;
  try {
    const h = await api("/healthz");
    el.textContent = `GPU: ${h.gpu?.device || "n/a"} | Jobs: ${h.jobs} | Queue: ${JSON.stringify(h.queue)}`;
  } catch (e) {
    el.textContent = "Health check failed: " + e.message;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  navActive();
  loadHealth();
});

export { api, fmtState };
