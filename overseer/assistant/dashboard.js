// The assistant's dashboard tabs (assistant/dashboard.py): today, tasks and the vault's changes. Read-only.
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
document.head.insertAdjacentHTML("beforeend", `<style>
  .card.task { display: flex; gap: .5em; align-items: flex-start; }
  .card.task > span:nth-child(2) { flex: 1; }
  main h3 { font-size: .95em; margin: 1em 0 .4em; }
</style>`);
const main = document.getElementById("main");
const state = { when: "open", q: "" };

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;",
  '"': "&quot;", "'": "&#39;"}[c]));
const when = (ts) => ts ? new Date(ts * 1000).toLocaleString([], {weekday: "short", month: "short", day: "numeric",
  hour: "2-digit", minute: "2-digit"}) : "";
const PRIO = {highest: "🔺", high: "⏫", medium: "🔼", low: "🔽", lowest: "⏬"};

async function getJSON(path) {
  const res = await fetch(path, {headers: {"X-Telegram-Init-Data": tg ? tg.initData : ""}});
  if (res.status === 401) throw new Error("Open the dashboard from Telegram (the 📊 button in the chat with the bot).");
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}
function fail(e) { main.innerHTML = `<div class="error">${esc(e.message || e)}</div>`; }

function relDay(iso) {
  if (!iso) return "";
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const d = new Date(iso + "T00:00:00");
  const days = Math.round((d - today) / 86400000);
  if (days === 0) return "today";
  if (days === 1) return "tomorrow";
  if (days < 0) return `${-days}d overdue`;
  return d.toLocaleDateString([], {weekday: "short", month: "short", day: "numeric"});
}
function taskHTML(t) {
  const date = t.due ? `📅 ${relDay(t.due)}` : t.scheduled ? `⏳ ${relDay(t.scheduled)}` : "";
  const late = t.due && relDay(t.due).includes("overdue");
  const sub = [date, t.recurrence && `🔁 ${t.recurrence}`, t.path].filter(Boolean);
  const mark = t.status === "x" ? "✅" : t.status === "-" ? "❌" : t.status === "/" ? "🔄" : "⬜";
  return `<div class="card task"><span aria-hidden="true">${mark}</span>
    <span>${esc(t.text)} ${PRIO[t.priority] || ""}<br><span class="muted small">${esc(sub.join(" · "))}</span></span>
    ${late ? '<span class="badge">late</span>' : ""}<span class="muted small jid">${esc(t.id)}</span></div>`;
}

// ------------------------------------------------------------------ today
async function renderToday() {
  const d = await getJSON("/api/today");
  const parts = [`<div class="filters"><span class="muted small">${esc(new Date().toLocaleDateString([],
    {weekday: "long", month: "long", day: "numeric"}))} · daily note ${esc(d.daily)}</span></div>`];
  if (d.conflicts.length) {
    parts.push(`<div class="error">Sync conflicts (two devices changed a note at once): ${esc(d.conflicts.join(", "))}</div>`);
  }
  parts.push("<h3>Due today and overdue</h3>");
  parts.push(d.due.length ? d.due.map(taskHTML).join("") : '<div class="empty">Nothing due. 🎉</div>');
  parts.push("<h3>Reminders</h3>");
  parts.push(d.reminders.length ? d.reminders.map((r) =>
    `<div class="card"><span aria-hidden="true">⏰</span> ${esc(when(r.at))} — ${esc(r.text)}</div>`).join("")
    : '<div class="empty">No reminders.</div>');
  if (d.upcoming.length) {
    parts.push("<h3>Coming up</h3>");
    parts.push(d.upcoming.map(taskHTML).join(""));
  }
  main.innerHTML = parts.join("");
}

// ------------------------------------------------------------------ tasks
async function renderTasks() {
  const d = await getJSON(`/api/tasks?when=${encodeURIComponent(state.when)}&q=${encodeURIComponent(state.q)}`);
  const filters = [["open", "Open"], ["week", "This week"], ["overdue", "Overdue"], ["upcoming", "Upcoming"],
                   ["undated", "No date"], ["done", "Done"]];
  main.innerHTML = `<div class="filters">
    <input id="q" type="search" placeholder="Search" aria-label="Search tasks" value="${esc(state.q)}">
    ${filters.map(([v, l]) => `<button class="chip" data-w="${v}" aria-pressed="${state.when === v}">${l}</button>`).join("")}</div>
    ${d.error ? `<div class="error">${esc(d.error)}</div>` : ""}
    ${d.tasks.length ? d.tasks.map(taskHTML).join("") + (d.total > d.tasks.length ?
      `<div class="muted small">${d.total - d.tasks.length} more</div>` : "") : '<div class="empty">No tasks here.</div>'}`;
  main.querySelectorAll(".chip").forEach((b) => b.onclick = () => { state.when = b.dataset.w; renderTasks().catch(fail); });
  const q = document.getElementById("q");
  q.onchange = () => { state.q = q.value; renderTasks().catch(fail); };
}

// ------------------------------------------------------------------ changes
async function renderChanges() {
  const d = await getJSON("/api/changes");
  if (!d.history) {
    main.innerHTML = '<div class="empty">The vault\'s history is off (vault.history, and git must be installed).</div>';
    return;
  }
  const mark = {A: "+", D: "−", M: "~"};
  main.innerHTML = d.changes.length ? d.changes.map((c) => `<details class="job"><summary>
      <span aria-hidden="true">${c.message === "Your edits" ? "✍️" : c.message.startsWith("Revert") ? "↩️" : "🤖"}</span>
      <span>${esc(c.message)}<br><span class="muted small">${esc(when(c.when))} · ${c.files.length + c.more} note(s)</span></span>
      </summary><div class="card">${c.files.map(([st, p]) => `${esc(mark[st[0]] || st)} ${esc(p)}`).join("<br>")}
      ${c.more ? `<br>… ${c.more} more` : ""}</div></details>`).join("")
    : '<div class="empty">No changes recorded yet.</div>';
}

// ------------------------------------------------------------------ tabs
const TABS = [["today", "Today", renderToday], ["tasks", "Tasks", renderTasks], ["changes", "Changes", renderChanges]];
const nav = document.querySelector("nav");
nav.innerHTML = TABS.map(([key, label], i) =>
  `<button role="tab" data-tab="${key}" aria-selected="${i === 0}">${label}</button>`).join("");
nav.querySelectorAll("button").forEach((b) => b.onclick = () => {
  nav.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  TABS.find(([key]) => key === b.dataset.tab)[2]().catch(fail);
});
renderToday().catch(fail);
