const API_URL = "https://localhost:8000";
let currentNoteId = null;

// Load notes when page first loads
document.addEventListener("DOMContentLoaded", function () {
  console.log("Page loaded, initializing...");
  showSection("notes");
  loadNotes();
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
      alert("Note deleted!");
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