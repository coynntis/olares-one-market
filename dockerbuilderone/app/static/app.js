const STORAGE_KEY = "dockerbuilderone_active_job";
const LOG_PAGE = 500;

const uploadForm = document.getElementById("upload-form");
const buildForm = document.getElementById("build-form");
const projectList = document.getElementById("project-list");
const buildProject = document.getElementById("build-project");
const buildHistory = document.getElementById("build-history");
const queuePanel = document.getElementById("queue-panel");
const uploadStatus = document.getElementById("upload-status");
const buildStatus = document.getElementById("build-status");
const buildLog = document.getElementById("build-log");
const logLoadHint = document.getElementById("log-load-hint");
const refreshBtn = document.getElementById("refresh-projects");
const refreshBuildsBtn = document.getElementById("refresh-builds");
const cancelBuildBtn = document.getElementById("cancel-build");
const kanikoVerbose = document.getElementById("kaniko-verbose");

let activeStream = null;
let pollTimer = null;
let queuePollTimer = null;
let watchedJobId = null;

/** Paginated build log viewer state. */
const logView = {
  jobId: null,
  lines: [],
  startLine: 0,
  endLine: 0,
  totalLines: 0,
  hasOlder: false,
  loadingOlder: false,
};

function setStatus(el, text, kind = "") {
  el.textContent = text;
  el.className = `status ${kind}`.trim();
}

function stateClass(state) {
  if (state === "success") return "ok";
  if (state === "failed" || state === "cancelled") return "err";
  if (state === "running" || state === "queued") return "running";
  return "";
}

/** SSE done payload: JSON {state, error} or legacy plain state string. */
function parseDonePayload(raw) {
  if (!raw) return { state: "failed", error: "" };
  try {
    const o = JSON.parse(raw);
    if (o && typeof o.state === "string") {
      return { state: o.state, error: o.error || "" };
    }
  } catch {
    /* legacy */
  }
  return { state: raw, error: "" };
}

function buildFailureStatus(jobId, state, errorFromEvent, meta) {
  const err = errorFromEvent || meta?.error || "";
  if (state === "success") {
    return { msg: "Build and push succeeded.", kind: "ok" };
  }
  if (state === "cancelled") {
    return {
      msg: err || "Build cancelled.",
      kind: "err",
    };
  }
  const short = err || "Build failed.";
  const lines = err ? err.split("\n") : [];
  const msg =
    lines.length > 1
      ? `${jobId} — failed\n${err}`
      : `${jobId} — failed — ${short}`;
  return { msg, kind: "err" };
}

function stopWatching() {
  if (activeStream) {
    activeStream.close();
    activeStream = null;
  }
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function rememberJob(jobId) {
  if (jobId) {
    localStorage.setItem(STORAGE_KEY, jobId);
  } else {
    localStorage.removeItem(STORAGE_KEY);
  }
}

function canCancel(state) {
  return state === "running" || state === "queued";
}

function updateCancelButton(jobId, state) {
  if (!cancelBuildBtn) return;
  if (jobId && canCancel(state)) {
    cancelBuildBtn.hidden = false;
    cancelBuildBtn.dataset.job = jobId;
  } else {
    cancelBuildBtn.hidden = true;
    cancelBuildBtn.dataset.job = "";
  }
}

function updateLogHint() {
  if (!logLoadHint) return;
  if (!logView.jobId) {
    logLoadHint.textContent = "";
    return;
  }
  const hidden =
    logView.totalLines > logView.lines.length
      ? logView.totalLines - logView.lines.length
      : 0;
  let hint = `Showing lines ${logView.startLine + 1}–${logView.endLine} of ${logView.totalLines}`;
  if (hidden > 0) {
    hint += ` — scroll up to load older (${hidden} hidden)`;
  }
  if (logView.loadingOlder) {
    hint += " — loading…";
  }
  logLoadHint.textContent = hint;
}

function renderLogView({ stickToBottom = true } = {}) {
  const prevScroll = buildLog.scrollTop;
  const prevHeight = buildLog.scrollHeight;
  const atBottom = prevScroll + buildLog.clientHeight >= prevHeight - 40;

  buildLog.textContent = logView.lines.join("\n") + (logView.lines.length ? "\n" : "");

  if (stickToBottom && atBottom) {
    buildLog.scrollTop = buildLog.scrollHeight;
  } else if (!stickToBottom && prevHeight > 0) {
    buildLog.scrollTop = buildLog.scrollHeight - prevHeight + prevScroll;
  }
  updateLogHint();
}

function resetLogView() {
  logView.jobId = null;
  logView.lines = [];
  logView.startLine = 0;
  logView.endLine = 0;
  logView.totalLines = 0;
  logView.hasOlder = false;
  logView.loadingOlder = false;
  buildLog.textContent = "";
  updateLogHint();
}

async function fetchBuildMeta(jobId) {
  const res = await fetch(`/api/builds/${jobId}`);
  if (!res.ok) return null;
  return res.json();
}

async function fetchLogTail(jobId) {
  const res = await fetch(`/api/builds/${jobId}/logs?limit=${LOG_PAGE}`);
  if (!res.ok) return null;
  return res.json();
}

async function fetchLogOlder(jobId, before) {
  const res = await fetch(
    `/api/builds/${jobId}/logs?before=${before}&limit=${LOG_PAGE}`
  );
  if (!res.ok) return null;
  return res.json();
}

async function loadLogTail(jobId, { stickToBottom = true } = {}) {
  const data = await fetchLogTail(jobId);
  if (!data) return null;
  logView.jobId = jobId;
  logView.lines = data.lines || [];
  logView.startLine = data.start_line ?? 0;
  logView.endLine = data.end_line ?? 0;
  logView.totalLines = data.total_lines ?? 0;
  logView.hasOlder = !!data.has_older;
  renderLogView({ stickToBottom });
  return data;
}

async function loadOlderLogs() {
  const id = logView.jobId || watchedJobId;
  if (
    !id ||
    logView.loadingOlder ||
    !logView.hasOlder ||
    logView.startLine <= 0
  ) {
    return;
  }
  logView.loadingOlder = true;
  updateLogHint();
  try {
    const data = await fetchLogOlder(id, logView.startLine);
    if (!data || !data.lines?.length) {
      logView.hasOlder = false;
      return;
    }
    logView.lines = [...data.lines, ...logView.lines];
    logView.startLine = data.start_line ?? 0;
    logView.hasOlder = !!data.has_older;
    renderLogView({ stickToBottom: false });
  } finally {
    logView.loadingOlder = false;
    updateLogHint();
  }
}

buildLog.addEventListener("scroll", () => {
  if (buildLog.scrollTop < 64) {
    loadOlderLogs();
  }
});

function appendStreamLine(line) {
  logView.lines.push(line);
  logView.endLine += 1;
  logView.totalLines = Math.max(logView.totalLines, logView.endLine);
  const atBottom =
    buildLog.scrollTop + buildLog.clientHeight >= buildLog.scrollHeight - 40;
  renderLogView({ stickToBottom: atBottom });
}

async function loadSettings() {
  if (!kanikoVerbose) return;
  try {
    const res = await fetch("/api/settings");
    if (!res.ok) return;
    const data = await res.json();
    kanikoVerbose.checked = !!data.kaniko_verbose;
  } catch {
    /* ignore */
  }
}

async function saveKanikoVerbose(enabled) {
  try {
    const res = await fetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kaniko_verbose: enabled }),
    });
    if (!res.ok) throw new Error("Failed to save settings");
  } catch (err) {
    setStatus(buildStatus, String(err.message || err), "err");
  }
}

async function loadProjects() {
  const res = await fetch("/api/projects");
  const data = await res.json();
  projectList.innerHTML = "";
  buildProject.innerHTML = "";

  if (!data.projects.length) {
    projectList.innerHTML = "<li>No projects yet.</li>";
    return;
  }

  for (const p of data.projects) {
    const li = document.createElement("li");
    const df = p.has_dockerfile ? p.dockerfile || "Dockerfile" : "missing";
    li.innerHTML = `<strong>${p.name}</strong> — Dockerfile: ${df} — updated ${p.modified}`;
    projectList.appendChild(li);

    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.name;
    buildProject.appendChild(opt);
  }
}

function renderQueuePanel(queue) {
  if (!queuePanel) return;
  const running = queue?.running;
  const queued = queue?.queued || [];
  let html = "";
  if (running) {
    html += `<p><strong>Running:</strong> ${running.id} — ${running.image}</p>`;
  } else {
    html += "<p><strong>Running:</strong> none</p>";
  }
  if (queued.length) {
    html += "<p><strong>Queued:</strong></p><ol>";
    for (const q of queued) {
      html += `<li>#${q.queue_position} ${q.id} — ${q.image}</li>`;
    }
    html += "</ol>";
  } else {
    html += "<p><em>Queue empty</em></p>";
  }
  queuePanel.innerHTML = html;
}

async function refreshQueue() {
  try {
    const res = await fetch("/api/builds/queue");
    const data = await res.json();
    renderQueuePanel(data);
    return data;
  } catch {
    return null;
  }
}

async function loadBuildHistory(selectJobId = null) {
  const res = await fetch("/api/builds");
  const data = await res.json();
  renderQueuePanel(data.queue);
  buildHistory.innerHTML = "";

  if (!data.builds.length) {
    buildHistory.innerHTML = "<li>No builds yet.</li>";
    return;
  }

  for (const b of data.builds) {
    const li = document.createElement("li");
    const pos = b.queue_position != null ? ` #${b.queue_position}` : "";
    const lines =
      b.log_lines != null ? ` — ${b.log_lines} lines` : "";
    const label = `${b.id}${pos} — ${b.image} — ${b.state}${lines}`;
    li.innerHTML = `<button type="button" class="linkish" data-job="${b.id}">${label}</button>`;
    if (b.id === selectJobId) {
      li.classList.add("selected");
    }
    buildHistory.appendChild(li);
  }

  buildHistory.querySelectorAll("button[data-job]").forEach((btn) => {
    btn.addEventListener("click", () => {
      showBuild(btn.getAttribute("data-job"), { follow: false });
    });
  });
}

async function showBuild(jobId, { follow = true } = {}) {
  watchedJobId = jobId;
  rememberJob(jobId);

  const data = await fetchBuildMeta(jobId);
  if (!data) {
    setStatus(buildStatus, `Build ${jobId} not found.`, "err");
    updateCancelButton(null, null);
    return;
  }

  if (!follow) {
    stopWatching();
  }

  await loadLogTail(jobId, { stickToBottom: true });

  const kind = stateClass(data.state);
  const active = data.state === "running" || data.state === "queued";
  const pos =
    data.queue_position != null ? ` (queue #${data.queue_position})` : "";
  setStatus(
    buildStatus,
    `${data.id} — ${data.state}${pos}${data.error ? ` — ${data.error}` : ""}`,
    kind
  );
  updateCancelButton(jobId, data.state);

  await loadBuildHistory(jobId);

  if (follow && active) {
    watchBuild(jobId, logView.endLine);
  } else if (!active) {
    if (watchedJobId === jobId) {
      rememberJob(null);
      updateCancelButton(null, null);
    }
  }
}

function watchBuild(jobId, fromLine = 0) {
  stopWatching();
  watchedJobId = jobId;
  rememberJob(jobId);
  logView.jobId = jobId;

  const url = `/api/builds/${jobId}/stream?from_line=${fromLine}`;
  activeStream = new EventSource(url);

  activeStream.onmessage = (ev) => {
    appendStreamLine(ev.data);
  };

  activeStream.addEventListener("done", async (ev) => {
    const { state, error: eventError } = parseDonePayload(ev.data);
    stopWatching();
    await loadLogTail(jobId, { stickToBottom: true });
    let meta = null;
    try {
      meta = await fetchBuildMeta(jobId);
    } catch {
      /* ignore */
    }
    const { msg, kind } = buildFailureStatus(
      jobId,
      state,
      eventError,
      meta
    );
    setStatus(buildStatus, msg, kind);
    rememberJob(null);
    updateCancelButton(null, null);
    loadBuildHistory(jobId);
    refreshQueue();
  });

  activeStream.onerror = () => {
    activeStream.close();
    activeStream = null;
    startPollFallback(jobId);
  };

  if (!queuePollTimer) {
    queuePollTimer = setInterval(refreshQueue, 3000);
  }
}

function startPollFallback(jobId) {
  setStatus(buildStatus, "SSE disconnected — polling logs…", "running");
  pollTimer = setInterval(async () => {
    const meta = await fetchBuildMeta(jobId);
    if (!meta) return;
    await loadLogTail(jobId, { stickToBottom: true });
    updateCancelButton(jobId, meta.state);
    if (meta.state === "success") {
      setStatus(buildStatus, "Build and push succeeded.", "ok");
      stopWatching();
      rememberJob(null);
      updateCancelButton(null, null);
      loadBuildHistory(jobId);
      refreshQueue();
    } else if (meta.state === "failed" || meta.state === "cancelled") {
      const { msg, kind } = buildFailureStatus(
        jobId,
        meta.state,
        meta.error,
        meta
      );
      setStatus(buildStatus, msg, kind);
      stopWatching();
      rememberJob(null);
      updateCancelButton(null, null);
      loadBuildHistory(jobId);
      refreshQueue();
    }
  }, 2000);
}

async function resumeActiveBuild() {
  let jobId = localStorage.getItem(STORAGE_KEY);

  try {
    const cur = await fetch("/api/builds/current");
    const curData = await cur.json();
    renderQueuePanel(curData.queue);
    if (curData.build) {
      jobId = curData.build.id;
    }
  } catch {
    /* ignore */
  }

  if (!jobId) return;

  const meta = await fetchBuildMeta(jobId);
  if (!meta) {
    rememberJob(null);
    return;
  }

  const active = meta.state === "running" || meta.state === "queued";
  await showBuild(jobId, { follow: active });
}

async function cancelCurrentBuild() {
  const jobId = cancelBuildBtn?.dataset?.job;
  if (!jobId) return;
  cancelBuildBtn.disabled = true;
  try {
    const res = await fetch(`/api/builds/${jobId}/cancel`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Cancel failed");
    setStatus(buildStatus, `Cancel requested for ${jobId}…`, "running");
    renderQueuePanel(data.queue);
    await loadBuildHistory(jobId);
  } catch (err) {
    setStatus(buildStatus, String(err.message || err), "err");
  } finally {
    cancelBuildBtn.disabled = false;
  }
}

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  setStatus(uploadStatus, "Uploading…");

  const name = document.getElementById("project-name").value.trim();
  const file = document.getElementById("project-zip").files[0];
  const body = new FormData();
  body.append("name", name);
  body.append("archive", file);

  try {
    const res = await fetch("/api/projects/upload", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    const dfInput = document.getElementById("build-dockerfile");
    if (data.dockerfile) {
      dfInput.value = data.dockerfile;
    }
    const hint = data.has_dockerfile
      ? `found at ${data.dockerfile}`
      : `missing (top-level: ${(data.files || []).join(", ") || "empty"})`;
    setStatus(
      uploadStatus,
      `Uploaded ${data.name}. Dockerfile ${hint}.`,
      data.has_dockerfile ? "ok" : "err"
    );
    await loadProjects();
  } catch (err) {
    setStatus(uploadStatus, String(err.message || err), "err");
  }
});

buildForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  resetLogView();

  const body = new FormData();
  body.append("project", buildProject.value);
  body.append("image", document.getElementById("build-image").value.trim());
  body.append(
    "dockerfile",
    document.getElementById("build-dockerfile").value.trim() || "Dockerfile"
  );

  try {
    const res = await fetch("/api/builds", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Build failed to start");
    renderQueuePanel(data.queue);
    const pos =
      data.queue_position != null ? ` (queue #${data.queue_position})` : "";
    const msg =
      data.state === "queued"
        ? `Build ${data.id} queued${pos}`
        : `Build ${data.id} started`;
    setStatus(buildStatus, msg, "running");
    await loadBuildHistory(data.id);
    const curRes = await fetch("/api/builds/current");
    const curData = await curRes.json();
    if (curData.build) {
      await loadLogTail(curData.build.id, { stickToBottom: true });
      watchBuild(curData.build.id, logView.endLine);
    } else {
      showBuild(data.id, { follow: false });
    }
  } catch (err) {
    setStatus(buildStatus, String(err.message || err), "err");
  }
});

refreshBtn.addEventListener("click", loadProjects);
refreshBuildsBtn.addEventListener("click", () => {
  loadBuildHistory(watchedJobId);
  refreshQueue();
  if (watchedJobId) {
    loadLogTail(watchedJobId, { stickToBottom: false });
  }
});
if (cancelBuildBtn) {
  cancelBuildBtn.addEventListener("click", cancelCurrentBuild);
}
if (kanikoVerbose) {
  kanikoVerbose.addEventListener("change", () => {
    saveKanikoVerbose(kanikoVerbose.checked);
  });
}

(async function init() {
  await loadSettings();
  await loadProjects();
  await loadBuildHistory();
  await refreshQueue();
  await resumeActiveBuild();
  if (!queuePollTimer) {
    queuePollTimer = setInterval(refreshQueue, 5000);
  }
})();
