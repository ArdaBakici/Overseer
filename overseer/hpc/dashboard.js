// The HPC's dashboard tabs (hpc/dashboard.py): read-only views of jobs, results, findings and a timeline.
"use strict";
const tg = window.Telegram && window.Telegram.WebApp;
if (tg) {
  tg.ready();
  tg.expand();
  document.documentElement.dataset.theme = tg.colorScheme === "dark" ? "dark" : "light";
  tg.onEvent && tg.onEvent("themeChanged", () => {
    document.documentElement.dataset.theme = tg.colorScheme === "dark" ? "dark" : "light";
  });
}
const main = document.getElementById("main");
const tip = document.getElementById("tip");
const state = { tab: "jobs", project: "", jobFilter: "all", results: { sort: "ended", dir: -1, q: "", verdict: "" },
                days: 7, projects: [] };

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;",
  '"': "&quot;", "'": "&#39;"}[c]));
const when = (ts) => ts ? new Date(ts * 1000).toLocaleString([], {month: "short", day: "numeric", hour: "2-digit",
  minute: "2-digit"}) : "";
const STATES = {
  RUNNING: ["running", "🏃"], PENDING: ["pending", "⏳"], COMPLETED: ["good", "✅"],
  FAILED: ["critical", "❌"], OUT_OF_MEMORY: ["critical", "💥"], NODE_FAIL: ["critical", "❌"],
  BOOT_FAIL: ["critical", "❌"], TIMEOUT: ["serious", "⏱"], DEADLINE: ["serious", "⏱"],
  CANCELLED: ["serious", "🚫"], PREEMPTED: ["serious", "↩️"],
};
const kind = (s) => (STATES[s] || ["pending", "•"])[0];
const icon = (s) => (STATES[s] || ["pending", "•"])[1];

async function api(path) {
  const res = await fetch(path, {headers: {"X-Telegram-Init-Data": tg ? tg.initData : ""}});
  if (res.status === 401) throw new Error("Open the dashboard from Telegram (the 📊 button in the chat with the bot).");
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res;
}
const getJSON = async (path) => (await api(path)).json();

function projectSelect(value, withMain) {
  const opts = [["", "All projects"], ...(withMain ? [["main", "main"]] : []),
                ...state.projects.map((p) => [p, p])];
  return `<select id="proj" aria-label="Project">${opts.map(([v, l]) =>
    `<option value="${esc(v)}"${v === value ? " selected" : ""}>${esc(l)}</option>`).join("")}</select>`;
}
function onProject(render) {
  const el = document.getElementById("proj");
  if (el) el.onchange = () => { state.project = el.value; render(); };
}
function fail(e) { main.innerHTML = `<div class="error">${esc(e.message || e)}</div>`; }

// ------------------------------------------------------------------ jobs
async function renderJobs() {
  const data = await getJSON("/api/overview");
  state.projects = data.projects;
  const filters = [["all", "All"], ["running", "Running"], ["queued", "Queued"], ["ended", "Ended"],
                   ["failed", "Failed"]];
  const keep = (j) => {
    if (state.project && j.project !== (state.project === "main" ? "" : state.project)) return false;
    const f = state.jobFilter;
    if (f === "running") return j.state === "RUNNING";
    if (f === "queued") return j.state === "PENDING";
    if (f === "ended") return !!j.ended;
    if (f === "failed") return !!j.ended && kind(j.state) !== "good";
    return true;
  };
  const jobs = data.jobs.filter(keep);
  main.innerHTML = `<div class="filters">${projectSelect(state.project, true)}
    ${filters.map(([v, l]) => `<button class="chip" data-f="${v}" aria-pressed="${state.jobFilter === v}">${l}</button>`).join("")}</div>
    ${jobs.length ? jobs.map(jobHTML).join("") : '<div class="empty">No jobs here.</div>'}`;
  onProject(renderJobs);
  main.querySelectorAll(".chip").forEach((b) => b.onclick = () => { state.jobFilter = b.dataset.f; renderJobs(); });
  main.querySelectorAll("details.job").forEach((d) => d.addEventListener("toggle", () => d.open && openJob(d)));
}
function jobHTML(j) {
  const sub = [j.project && `📁 ${j.project}`, j.agent && "🤖 agent", j.idle && "💤 GPU idle",
               j.start && `🕐 ${j.start}`, j.ended ? `ended ${when(j.ended)}` : `seen ${when(j.seen)}`].filter(Boolean);
  return `<details class="job" data-id="${esc(j.id)}" data-plots="${j.plots}">
    <summary><span aria-hidden="true">${icon(j.state)}</span>
      <span><span class="jid">${esc(j.id)}</span> ${esc(j.name)}<br><span class="muted small">${esc(sub.join(" · "))}</span></span>
      <span class="badge">${esc(j.state.toLowerCase())}</span></summary>
    <div class="card">${esc(j.card || j.report || "")}</div>
    <div class="carousel"></div>
    <button class="plain logbtn">📜 Log</button><pre class="log" hidden></pre>
  </details>`;
}
async function openJob(d) {
  const id = d.dataset.id, n = +d.dataset.plots, box = d.querySelector(".carousel");
  if (n && !box.dataset.ready) {
    box.dataset.ready = "1";
    let i = 0;
    const urls = [];
    const show = async () => {
      if (!urls[i]) urls[i] = URL.createObjectURL(await (await api(`/api/plot/${encodeURIComponent(id)}/${i}`)).blob());
      box.innerHTML = `<img alt="graph ${i + 1} of ${n} of job ${esc(id)}" src="${urls[i]}">` + (n > 1 ?
        `<div class="nav"><button class="plain prev" aria-label="previous graph">◀</button><span class="small">${i + 1}/${n}</span>
         <button class="plain next" aria-label="next graph">▶</button></div>` : "");
      const p = box.querySelector(".prev"), q = box.querySelector(".next");
      if (p) p.onclick = () => { i = (i + n - 1) % n; show().catch(fail); };
      if (q) q.onclick = () => { i = (i + 1) % n; show().catch(fail); };
    };
    show().catch((e) => box.textContent = e.message);
  }
  const btn = d.querySelector(".logbtn"), pre = d.querySelector(".log");
  btn.onclick = async () => {
    btn.disabled = true;
    pre.hidden = false;
    pre.textContent = "Reading the log on the HPC…";
    try {
      const r = await getJSON(`/api/log/${encodeURIComponent(id)}?lines=120`);
      pre.textContent = r.error ? r.error : `${r.path}\n\n${r.text || "(empty)"}`;
      pre.scrollTop = pre.scrollHeight;
    } catch (e) { pre.textContent = e.message; }
    btn.disabled = false;
  };
}

// ------------------------------------------------------------------ results
async function renderResults() {
  const data = await getJSON(`/api/results?project=${encodeURIComponent(state.project)}`);
  const r = state.results, keys = data.metric_keys;
  const verdicts = [...new Set(data.rows.map((x) => x.verdict).filter(Boolean))].sort();
  const cols = [["id", "id"], ["name", "name"], ["verdict", "verdict"], ...keys.map((k) => [`m:${k}`, k]),
                ["elapsed", "time"], ["ended", "ended"], ["conclusion", "conclusion"]];
  const val = (row, c) => c.startsWith("m:") ? row.metrics[c.slice(2)] : row[c];
  const q = r.q.toLowerCase();
  let rows = data.rows.filter((x) => (!r.verdict || (r.verdict === "-" ? !x.verdict : x.verdict === r.verdict)) &&
    (!q || [x.id, x.name, x.verdict, x.conclusion, x.hypothesis, x.tags].join(" ").toLowerCase().includes(q)));
  rows.sort((a, b) => {
    const x = val(a, r.sort), y = val(b, r.sort);
    if (x === undefined || x === "" || x === null) return 1;
    if (y === undefined || y === "" || y === null) return -1;
    const nx = parseFloat(x), ny = parseFloat(y);
    const cmp = (!isNaN(nx) && !isNaN(ny) && String(nx) === String(x).trim()) ? nx - ny : String(x).localeCompare(String(y));
    return cmp * r.dir;
  });
  main.innerHTML = `<div class="filters">${projectSelect(state.project, true)}
    <select id="verdict" aria-label="Verdict"><option value="">All verdicts</option><option value="-"${r.verdict === "-" ? " selected" : ""}>not judged</option>
    ${verdicts.map((v) => `<option${v === r.verdict ? " selected" : ""}>${esc(v)}</option>`).join("")}</select>
    <input type="search" id="q" placeholder="Search name, conclusion, tags" value="${esc(r.q)}"></div>
    <div class="muted small">${rows.length} of ${data.rows.length} result(s) · tap a column to sort, a row for details</div>
    ${rows.length ? `<div class="table-wrap"><table><thead><tr>${cols.map(([c, l]) =>
      `<th data-c="${esc(c)}"${r.sort === c ? ` aria-sort="${r.dir > 0 ? "ascending" : "descending"}"` : ""}>${esc(l)}</th>`).join("")}
    </tr></thead><tbody>${rows.map((x, i) => `<tr data-i="${i}">${cols.map(([c]) => {
      let v = val(x, c);
      if (c === "ended") v = when(v);
      const num = c.startsWith("m:") && v !== undefined && !isNaN(parseFloat(v));
      return `<td class="${num ? "num" : ""}${c === "conclusion" ? " wrap" : ""}">${esc(v ?? "")}</td>`;
    }).join("")}</tr>`).join("")}</tbody></table></div>` : '<div class="empty">No results here.</div>'}`;
  onProject(renderResults);
  document.getElementById("verdict").onchange = (e) => { r.verdict = e.target.value; renderResults(); };
  const qi = document.getElementById("q");
  qi.oninput = () => { r.q = qi.value; clearTimeout(qi._t); qi._t = setTimeout(() => renderResults().then(() => {
    const el = document.getElementById("q"); el.focus(); el.setSelectionRange(el.value.length, el.value.length); }), 300); };
  main.querySelectorAll("th").forEach((th) => th.onclick = () => {
    r.dir = r.sort === th.dataset.c ? -r.dir : (th.dataset.c.startsWith("m:") || th.dataset.c === "ended" ? -1 : 1);
    r.sort = th.dataset.c;
    renderResults();
  });
  main.querySelectorAll("tbody tr").forEach((tr) => tr.onclick = () => {
    const next = tr.nextElementSibling;
    if (next && next.classList.contains("more")) { next.remove(); return; }
    const x = rows[+tr.dataset.i];
    const more = document.createElement("tr");
    more.className = "more";
    more.innerHTML = `<td colspan="${cols.length}"><div class="detail">${[
      x.project && `📁 ${esc(x.project)}`, x.hypothesis && `<b>Hypothesis:</b> ${esc(x.hypothesis)}`,
      x.conclusion && `<b>Conclusion:</b> ${esc(x.conclusion)}`, x.tags && `<b>Tags:</b> ${esc(x.tags)}`,
      `<b>Metrics:</b> ${esc(Object.entries(x.metrics).map(([k, v]) => `${k}=${v}`).join(", ") || "none")}`,
      `${esc(x.state)}${x.elapsed ? " after " + esc(x.elapsed) : ""}`].filter(Boolean).join("<br>")}</div></td>`;
    tr.after(more);
  });
}

// ------------------------------------------------------------------ findings
async function renderFindings() {
  const data = await getJSON("/api/findings");
  const rows = data.rows.filter((f) => !state.project || f.project === (state.project === "main" ? "" : state.project));
  main.innerHTML = `<div class="filters">${projectSelect(state.project, true)}</div>
    ${rows.length ? rows.map((f) => `<div class="finding">${f.kind === "dead-end" ? "🚫 <b>Dead end</b>" : "💡 <b>Insight</b>"}
      <span class="muted small">${esc(f.id)}${f.project ? " · 📁 " + esc(f.project) : ""} · ${esc(when(f.created))}${f.job ? " · job " + esc(f.job) : ""}</span>
      <div>${esc(f.text)}</div></div>`).join("") : '<div class="empty">No findings yet.</div>'}`;
  onProject(renderFindings);
}

// ------------------------------------------------------------------ timeline
async function renderTimeline() {
  main.innerHTML = '<div class="empty">Reading SLURM\'s accounting…</div>';
  const data = await getJSON(`/api/timeline?days=${state.days}`);
  const now = data.clock ? new Date(data.clock) : new Date();
  const lo = new Date(now - state.days * 86400e3);
  let jobs = data.jobs.filter((j) => !state.project || j.project === (state.project === "main" ? "" : state.project))
    .map((j) => ({...j, s: new Date(j.start), e: j.end ? new Date(j.end) : now}))
    .filter((j) => j.e >= lo).sort((a, b) => b.s - a.s).slice(0, 60);
  const legend = [["running", "🏃 running"], ["good", "✅ completed"], ["critical", "❌ failed (hatched)"],
                  ["serious", "⏱ timeout / cancelled"]];
  main.innerHTML = `<div class="filters">${projectSelect(state.project, true)}
    <select id="days" aria-label="Period">${[1, 3, 7, 14, 30].map((d) =>
      `<option value="${d}"${d === state.days ? " selected" : ""}>last ${d} day${d > 1 ? "s" : ""}</option>`).join("")}</select></div>
    ${data.error ? `<div class="error">${esc(data.error)}</div>` : ""}
    <div class="legend">${legend.map(([k, l]) => `<span style="--c: var(--${k})">${l}</span>`).join("")}</div>
    <div class="muted small">${jobs.length} job(s) that ran${jobs.length === 60 ? " (the latest 60)" : ""} · cluster time · tap a bar</div>
    <div id="tl"></div>`;
  onProject(renderTimeline);
  document.getElementById("days").onchange = (e) => { state.days = +e.target.value; renderTimeline().catch(fail); };
  if (!jobs.length) { document.getElementById("tl").innerHTML = '<div class="empty">No jobs in this period.</div>'; return; }
  const first = Math.min(...jobs.map((j) => +j.s));  // the axis starts with the first job shown, not the period
  drawTimeline(jobs, new Date(Math.max(+lo, first - (+now - first) * 0.03)), now);
}
function drawTimeline(jobs, lo, hi) {
  const box = document.getElementById("tl");
  const W = box.clientWidth || 340, label = Math.min(150, W * 0.34), row = 22, top = 22, H = top + jobs.length * row + 8;
  const x = (t) => label + (Math.max(lo, Math.min(hi, t)) - lo) / (hi - lo) * (W - label - 4);
  // Ticks at the finest step that leaves ~52px per label.
  const span = hi - lo, room = Math.max(1, Math.floor((W - label) / 52)), H1 = 3600e3;
  const step = [H1, 2 * H1, 3 * H1, 6 * H1, 12 * H1, 24 * H1, 48 * H1, 72 * H1, 7 * 24 * H1]
    .find((st) => span / st <= room) || 7 * 24 * H1;
  const off = step >= 24 * H1 ? new Date(+lo).getTimezoneOffset() * 60e3 : 0;  // (day ticks at local midnight)
  const ticks = [];
  for (let t = Math.ceil((+lo - off) / step) * step + off; t <= +hi; t += step) ticks.push(new Date(t));
  const fmt = (d) => step >= 86400e3 ? d.toLocaleDateString([], {month: "short", day: "numeric"})
                                     : d.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
  const bars = jobs.map((j, i) => {
    const y = top + i * row, x0 = x(j.s), x1 = Math.max(x(j.e), x0 + 3), k = kind(j.state);
    const name = j.name.length > 18 ? j.name.slice(0, 17) + "…" : j.name;
    return `<text x="0" y="${y + 14}">${icon(j.state)} ${esc(name)}</text>
      <rect class="bar" data-i="${i}" x="${x0}" y="${y + 3}" width="${x1 - x0}" height="${row - 8}" rx="3"
        style="fill: ${k === "critical" ? "url(#hatch)" : `var(--${k})`}"/>`;
  }).join("");
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="Jobs over time">
    <defs><pattern id="hatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
      <rect width="6" height="6" style="fill: var(--critical)"/><line x1="0" y1="0" x2="0" y2="6" style="stroke: var(--bg); stroke-width: 2"/></pattern></defs>
    <g class="grid">${ticks.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${top - 4}" y2="${H}"/>`).join("")}</g>
    <g class="axis">${ticks.map((t) => `<text x="${x(t)}" y="12" text-anchor="middle">${esc(fmt(t))}</text>`).join("")}</g>
    ${bars}</svg>`;
  const show = (ev, j) => {
    const dur = (j.e - j.s) / 3600e3;
    tip.innerHTML = `<b>${icon(j.state)} ${esc(j.id)}</b> ${esc(j.name)}<br>${esc(j.state.toLowerCase())}${j.end ? "" : " (still running)"}
      · ${dur >= 1 ? dur.toFixed(1) + " h" : Math.round(dur * 60) + " min"}${j.gpus ? ` · ${j.gpus} GPU${j.gpus > 1 ? "s" : ""}` : ""}
      ${j.project ? "<br>📁 " + esc(j.project) : ""}${j.tasks ? "<br>📦 " + esc(j.tasks) : ""}
      <br><span class="muted">${esc(j.start.replace("T", " "))} → ${esc(j.end ? j.end.replace("T", " ") : "now")}</span>`;
    tip.style.display = "block";
    const r = tip.getBoundingClientRect();
    tip.style.left = Math.min(ev.clientX + 12, innerWidth - r.width - 8) + "px";
    tip.style.top = Math.min(ev.clientY + 12, innerHeight - r.height - 8) + "px";
  };
  box.querySelectorAll("rect.bar").forEach((b) => {
    const j = jobs[+b.dataset.i];
    b.addEventListener("pointerenter", (ev) => show(ev, j));
    b.addEventListener("pointermove", (ev) => show(ev, j));
    b.addEventListener("pointerleave", () => tip.style.display = "none");
    b.addEventListener("click", (ev) => show(ev, j));
  });
}
document.addEventListener("scroll", () => tip.style.display = "none", {passive: true});

// ------------------------------------------------------------------ tabs
const TABS = [["jobs", "Jobs", renderJobs], ["results", "Results", renderResults], ["findings", "Findings", renderFindings],
              ["timeline", "Timeline", renderTimeline]];
const nav = document.querySelector("nav");
nav.innerHTML = TABS.map(([key, label], i) =>
  `<button role="tab" data-tab="${key}" aria-selected="${i === 0}">${label}</button>`).join("");
nav.querySelectorAll("button").forEach((b) => b.onclick = () => {
  nav.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  state.tab = b.dataset.tab;
  tip.style.display = "none";
  TABS.find(([key]) => key === state.tab)[2]().catch(fail);
});
renderJobs().catch(fail);
