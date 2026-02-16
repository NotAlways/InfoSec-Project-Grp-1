const API_URL = "https://localhost:8000";
let currentNoteId = null;

// Load notes when page first loads
document.addEventListener("DOMContentLoaded", function () {
  console.log("Page loaded, initializing...");
  showSection("notes");
  loadNotes();
});

// ===== Security Dashboard (individual logs) =====
let DASH_RAW = [];
let DASH_PAGE = 1;
const DASH_PAGE_SIZE = 15;

// Candidate endpoints (we try in order). Keep frontend-only.
// Your backend can expose ANY ONE of these and it will work.
const DASH_ENDPOINTS = [
  "/security/logs",
  "/logs/me",
  "/me/logs",
  "/activity/me",
  "/user-activity/me",
  "/audit/me",
  "/user/activity",
  "/activity",
];

// Basic debounce helper
function debounce(fn, wait = 250) {
  let t = null;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), wait);
  };
}

function setDashStatus(msg) {
  const el = document.getElementById("dashStatus");
  if (el) el.textContent = msg || "";
}

function normalizeEvent(row) {
  // Accepts many backend shapes:
  // - user_activity rows: {login_time, ip_address, action, success, ...}
  // - audit_events rows: {created_at, event_type, ip, note_id, trash_id, details, ...}
  const time =
    row.time || row.created_at || row.login_time || row.timestamp || row.at || null;

  const type =
    row.type || row.event_type || row.action || row.event || "event";

  // success can be: true/false, "true"/"false", 1/0, null (unknown)
  let success = row.success;
  if (typeof success === "string") success = (success.toLowerCase() === "true");
  if (typeof success === "number") success = (success === 1);

  const ip =
    row.ip || row.ip_address || row.client_ip || row.remote_addr || "";

  const noteId =
    row.note_id ?? row.noteId ?? row.original_note_id ?? null;

  const trashId =
    row.trash_id ?? row.trashId ?? null;

  const ua =
    row.user_agent || row.ua || row.device || "";

  const detailsObj = row.details || row.meta || {};
  const detailsStr =
    (typeof detailsObj === "string")
      ? detailsObj
      : JSON.stringify(detailsObj);

  // combine for searching
  const searchBlob = [
    type,
    ip,
    ua,
    noteId,
    trashId,
    detailsStr,
    row.device_fingerprint,
    row.reason,
    row.message,
  ]
    .filter(v => v !== null && v !== undefined)
    .join(" ")
    .toLowerCase();

  return {
    raw: row,
    time,
    type,
    success: (success === true ? true : (success === false ? false : null)),
    ip,
    noteId,
    trashId,
    ua,
    detailsStr,
    searchBlob
  };
}

async function fetchDashboardLogs() {
  setDashStatus("Loading logs...");
  for (const path of DASH_ENDPOINTS) {
    try {
      const res = await fetch(`${API_URL}${path}`, {
        method: "GET",
        credentials: "include",
      });
      if (!res.ok) continue;

      const data = await res.json();

      // allow {items: [...]} or just [...]
      const items = Array.isArray(data) ? data : (data.items || data.logs || []);
      if (!Array.isArray(items)) continue;

      setDashStatus(`Loaded from ${path}`);
      return items;
    } catch (e) {
      // try next endpoint
    }
  }
  setDashStatus("No logs endpoint found. (Frontend is ready, backend must expose one endpoint.)");
  return [];
}

function inferBucket(typeStr) {
  const t = (typeStr || "").toLowerCase();
  if (t.includes("login")) return "login";
  if (t.includes("password") || t.includes("reset")) return "password_reset";
  if (t.includes("trash") || t.includes("purge") || t.includes("restore")) return "trash";
  if (t.includes("note")) return "note";
  if (t.includes("admin")) return "admin";
  return "other";
}

function applyDashboardFilters() {
  const q = (document.getElementById("dashSearch")?.value || "").trim().toLowerCase();
  const typeFilter = document.getElementById("dashType")?.value || "";
  const successFilter = document.getElementById("dashSuccess")?.value || "";
  const fromVal = document.getElementById("dashFrom")?.value || "";
  const toVal = document.getElementById("dashTo")?.value || "";

  const fromDate = fromVal ? new Date(fromVal + "T00:00:00") : null;
  const toDate = toVal ? new Date(toVal + "T23:59:59") : null;

  return DASH_RAW.filter(ev => {
    // Search
    if (q && !ev.searchBlob.includes(q)) return false;

    // Type bucket filter
    if (typeFilter) {
      const bucket = inferBucket(ev.type);
      if (bucket !== typeFilter) return false;
    }

    // Success filter
    if (successFilter !== "") {
      const want = (successFilter === "true");
      if (ev.success === null) return false;
      if (ev.success !== want) return false;
    }

    // Date range filter
    if (fromDate || toDate) {
      const t = ev.time ? new Date(ev.time) : null;
      if (!t || isNaN(t.getTime())) return false;
      if (fromDate && t < fromDate) return false;
      if (toDate && t > toDate) return false;
    }

    return true;
  });
}

function renderDashboard() {
  const rowsEl = document.getElementById("dashRows");
  if (!rowsEl) return;

  const filtered = applyDashboardFilters();

  // Sort newest first (best effort)
  filtered.sort((a, b) => {
    const ta = a.time ? new Date(a.time).getTime() : 0;
    const tb = b.time ? new Date(b.time).getTime() : 0;
    return tb - ta;
  });

  // KPIs
  const total = filtered.length;
  const ok = filtered.filter(x => x.success === true).length;
  const bad = filtered.filter(x => x.success === false).length;

  const last = filtered[0]?.time ? new Date(filtered[0].time).toLocaleString() : "—";
  const kpiTotal = document.getElementById("kpiTotal");
  const kpiSuccess = document.getElementById("kpiSuccess");
  const kpiFailed = document.getElementById("kpiFailed");
  const kpiLast = document.getElementById("kpiLast");

  if (kpiTotal) kpiTotal.textContent = String(total);
  if (kpiSuccess) kpiSuccess.textContent = String(ok);
  if (kpiFailed) kpiFailed.textContent = String(bad);
  if (kpiLast) kpiLast.textContent = last;

  // Pagination
  const pages = Math.max(1, Math.ceil(filtered.length / DASH_PAGE_SIZE));
  if (DASH_PAGE > pages) DASH_PAGE = pages;
  if (DASH_PAGE < 1) DASH_PAGE = 1;

  const start = (DASH_PAGE - 1) * DASH_PAGE_SIZE;
  const slice = filtered.slice(start, start + DASH_PAGE_SIZE);

  const pageInfo = document.getElementById("dashPageInfo");
  if (pageInfo) pageInfo.textContent = `Page ${DASH_PAGE} / ${pages}`;

  if (!slice.length) {
    rowsEl.innerHTML = `<tr><td colspan="6" class="muted">No logs match your filters.</td></tr>`;
    return;
  }

  rowsEl.innerHTML = "";
  slice.forEach(ev => {
    const dt = ev.time ? new Date(ev.time) : null;
    const timeStr = (dt && !isNaN(dt.getTime())) ? dt.toLocaleString() : "—";

    const bucket = inferBucket(ev.type);
    const typeLabel = ev.type || bucket;

    let badge = `<span class="badge badge-neutral">UNKNOWN</span>`;
    if (ev.success === true) badge = `<span class="badge badge-ok">OK</span>`;
    if (ev.success === false) badge = `<span class="badge badge-bad">FAIL</span>`;

    const ipStr = ev.ip || "—";
    const objRef = (ev.noteId != null) ? `note:${ev.noteId}` : ((ev.trashId != null) ? `trash:${ev.trashId}` : "—");

    // Details: keep it short
    let details = (ev.detailsStr || "").trim();
    if (!details || details === "{}") details = "—";
    if (details.length > 220) details = details.slice(0, 220) + "…";

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${timeStr}</td>
      <td>${escapeHtml(typeLabel)}</td>
      <td>${badge}</td>
      <td>${escapeHtml(ipStr)}</td>
      <td>${escapeHtml(String(objRef))}</td>
      <td>${escapeHtml(details)}</td>
    `;
    rowsEl.appendChild(tr);
  });
}

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadDashboard() {
  // Only load when dashboard exists
  if (!document.getElementById("dashboard")) return;

  // Try to fetch logs
  const raw = await fetchDashboardLogs();

  // Normalize
  DASH_RAW = raw.map(normalizeEvent);

  // Reset to page 1 on refresh
  DASH_PAGE = 1;
  renderDashboard();
}

function dashPrevPage() {
  DASH_PAGE = Math.max(1, DASH_PAGE - 1);
  renderDashboard();
}

function dashNextPage() {
  DASH_PAGE = DASH_PAGE + 1;
  renderDashboard();
}

function resetDashboardFilters() {
  const s = document.getElementById("dashSearch");
  const t = document.getElementById("dashType");
  const ok = document.getElementById("dashSuccess");
  const f = document.getElementById("dashFrom");
  const to = document.getElementById("dashTo");

  if (s) s.value = "";
  if (t) t.value = "";
  if (ok) ok.value = "";
  if (f) f.value = "";
  if (to) to.value = "";

  DASH_PAGE = 1;
  renderDashboard();
}

// Wire up dashboard inputs (once)
document.addEventListener("DOMContentLoaded", function () {
  const onChange = debounce(() => {
    DASH_PAGE = 1;
    renderDashboard();
  }, 200);

  const ids = ["dashSearch", "dashType", "dashSuccess", "dashFrom", "dashTo"];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("input", onChange);
    if (el) el.addEventListener("change", onChange);
  });
});

/* ✅ NEW: Ctrl+C Copy Stamping (best-effort)
- Works ONLY when user copies text inside the NoteVault page.
- Does NOT work if they copy outside the page, screenshot, etc.
*/
document.addEventListener("copy", (e) => {
  try {
    const notesSection = document.getElementById("notes");
    if (!notesSection) return;

    // Only stamp when Notes section is active/visible
    const isHidden = (notesSection.style.display === "none");
    if (isHidden) return;

    const sel = window.getSelection();
    const selectedText = sel ? sel.toString() : "";
    if (!selectedText || !selectedText.trim()) return;

    const user = window.NV_USER || {};
    const email = user.email || "unknown";
    const username = user.username || "unknown";
    const ts = new Date().toLocaleString();

    const stamped =
      `${selectedText}\n\n— Copied from NoteVault by ${email} (${username}) on ${ts}`;

    // Override clipboard content
    e.clipboardData.setData("text/plain", stamped);
    e.preventDefault();
  } catch (err) {
    // If anything fails, allow normal copy (no stamping)
  }
});
// ✅ END NEW

function openTrash() {
  showSection("trash");
  loadTrash();
}

async function loadTrash() {
  const trashList = document.getElementById("trashList");
  trashList.innerHTML = "<p class='muted'>Loading...</p>";

  try {
    // ✅ Backend endpoint is GET /trash
    const res = await fetch(`${API_URL}/trash`, { credentials: "include" });
    if (!res.ok) {
      trashList.innerHTML = "<p class='muted'>Failed to load Recycle Bin. Please login again</p>";
      return;
    }

    const items = await res.json();
    if (!items.length) {
      trashList.innerHTML = "<p class='muted'>Recycle Bin is empty</p>";
      return;
    }

    trashList.innerHTML = "";
    items.forEach(t => {
      const div = document.createElement("div");
      div.className = "trash-item";
      div.innerHTML = `
        <h3>${t.title}</h3>
        <div class="trash-meta">Deleted by ${t.deleted_by_email} on ${new Date(t.deleted_at).toLocaleString()}</div>
        <div class="trash-actions">
          <button class="btn-restore" onclick="restoreTrash(${t.trash_id})">Restore</button>
          <button class="btn-danger" onclick="purgeTrash(${t.trash_id})">Permanent Delete</button>
        </div>
      `;
      trashList.appendChild(div);
    });
  } catch (e) {
    trashList.innerHTML = "<p class='muted'>Error loading Recycle Bin</p>";
  }
}

function showSection(sectionId) {
  const sections = document.querySelectorAll(".section");
  sections.forEach(sec => sec.style.display = "none");
  document.getElementById(sectionId).style.display = "block";
  if (sectionId === "notes") {
    loadNotes();
  }
}

async function saveNote() {
  const title = document.getElementById("noteTitle").value;
  const content = document.getElementById("noteContent").value;

  if (!title.trim() || !content.trim()) {
    alert("Please fill in both title and content");
    return;
  }

  try {
    if (currentNoteId) {
      // Update existing note
      console.log(`Updating note ${currentNoteId}...`);
      const response = await fetch(`${API_URL}/notes/${currentNoteId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, content })
      });
      console.log(`Response status: ${response.status}`);
      if (response.ok) {
        alert("Note updated!");
        clearNoteForm();
        loadNotes();
      } else {
        const error = await response.text();
        console.error("Update failed:", error);
        alert(`Failed to update note: ${response.status} - ${error}`);
      }
    } else {
      // Create new note
      console.log("Creating new note...");
      const response = await fetch(`${API_URL}/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, content })
      });
      console.log(`Response status: ${response.status}`);
      if (response.ok) {
        alert("Note saved!");
        clearNoteForm();
        loadNotes();
      } else {
        const error = await response.text();
        console.error("Create failed:", error);
        alert(`Failed to save note: ${response.status} - ${error}`);
      }
    }
  } catch (error) {
    console.error("Network/Parse Error:", error);
    alert(`Error saving note: ${error.message}`);
  }
}

async function loadNotes() {
  try {
    console.log("Loading notes from " + API_URL + "/notes");
    const response = await fetch(`${API_URL}/notes`);
    console.log(`GET /notes response status: ${response.status}`);

    if (!response.ok) {
      const error = await response.text();
      console.error(`Failed to load notes: ${response.status} - ${error}`);
      return;
    }

    const notes = await response.json();
    console.log(`Loaded ${notes.length} notes`);
    const noteList = document.getElementById("noteList");
    noteList.innerHTML = "";

    notes.forEach(note => {
      const noteItem = document.createElement("div");
      noteItem.className = "note-item";
      noteItem.innerHTML = `
      <h3>${note.title}</h3>
      <p>${note.content}</p>
      <button onclick="editNote(${note.id})">Edit</button>
      <button onclick="deleteNote(${note.id})">Delete</button>
      <button onclick="copyNote(${note.id})">Copy</button>
      <button onclick="exportNote(${note.id})">Export PDF</button>
    `;
      noteList.appendChild(noteItem);
    });
  } catch (error) {
    console.error("Network/Parse Error loading notes:", error);
  }
}

async function editNote(noteId) {
  try {
    const response = await fetch(`${API_URL}/notes/${noteId}`);
    const note = await response.json();
    document.getElementById("noteTitle").value = note.title;
    document.getElementById("noteContent").value = note.content;
    currentNoteId = noteId;
    document.getElementById("saveBtn").textContent = "Update Note";
  } catch (error) {
    console.error("Error loading note:", error);
  }
}

async function deleteNote(noteId) {
  if (!confirm("Are you sure you want to delete this note?")) return;

  try {
    const response = await fetch(`${API_URL}/notes/${noteId}`, {
      method: "DELETE",
      credentials: "include" // ✅ REQUIRED so session cookie is sent
    });

    if (response.ok) {
      alert("Note moved to Recycle Bin!");
      loadNotes();
    } else {
      const err = await response.text();
      console.error("Delete failed:", response.status, err);
      alert("Failed to delete note: " + err);
    }
  } catch (error) {
    console.error("Error:", error);
    alert("Error deleting note");
  }
}

function clearNoteForm() {
  document.getElementById("noteTitle").value = "";
  document.getElementById("noteContent").value = "";
  currentNoteId = null;
  document.getElementById("saveBtn").textContent = "Save Note";
}

function logout() {
  const confirmed = confirm("Are you sure you want to log out of NoteVault?");

  if (confirmed) {
    // InfoSec: Clear storage to prevent session leakage on shared computers
    localStorage.clear();
    sessionStorage.clear();

    // Redirect to the backend route to destroy the session cookie
    window.location.href = "/logout";
  }
}

async function copyNote(noteId) {
  try {
    const res = await fetch(`${API_URL}/notes/${noteId}/copy-text`, {
      method: "GET",
      credentials: "include" // IMPORTANT: send session cookie
    });

    if (!res.ok) {
      const err = await res.text();
      console.error("Copy-text failed:", res.status, err);
      alert("Failed to copy. Please login again");
      return;
    }

    const data = await res.json();
    const text = data.text || "";

    await navigator.clipboard.writeText(text);
    alert("Copied!");
  } catch (e) {
    console.error("Copy error:", e);
    alert("Copy failed. Please try again");
  }
}

async function exportNote(noteId) {
  try {
    const res = await fetch(`${API_URL}/notes/${noteId}/export-pdf`, {
      method: "GET",
      credentials: "include"
    });

    if (!res.ok) {
      const err = await res.text();
      console.error("Export failed:", res.status, err);
      alert("Failed to export. Please login again");
      return;
    }

    const blob = await res.blob();
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `note_${noteId}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();

    window.URL.revokeObjectURL(url);
  } catch (e) {
    console.error("Export error:", e);
    alert("Export failed. Please try again");
  }
}

/* ✅ UPDATED: Trash actions must call /trash/{trash_id}/... */
async function restoreTrash(trashId) {
  const res = await fetch(`${API_URL}/trash/${trashId}/restore`, {
    method: "POST",
    credentials: "include"
  });

  if (res.ok) {
    loadTrash();
    loadNotes(); // show restored note again
  } else {
    const err = await res.text();
    alert("Restore failed: " + err);
  }
}

async function purgeTrash(trashId) {
  if (!confirm("Permanently delete this note? This cannot be undone")) return;

  const res = await fetch(`${API_URL}/trash/${trashId}/purge`, {
    method: "DELETE",
    credentials: "include"
  });

  if (res.ok) loadTrash();
  else {
    const err = await res.text();
    alert("Permanent delete failed: " + err);
  }
}