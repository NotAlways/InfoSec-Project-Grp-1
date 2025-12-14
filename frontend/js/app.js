const API_URL = "http://localhost:8000";
let currentNoteId = null;

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
      const response = await fetch(`${API_URL}/notes/${currentNoteId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, content })
      });
      if (response.ok) {
        alert("Note updated!");
        clearNoteForm();
        loadNotes();
      } else {
        alert("Failed to update note");
      }
    } else {
      // Create new note
      const response = await fetch(`${API_URL}/notes`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, content })
      });
      if (response.ok) {
        alert("Note saved!");
        clearNoteForm();
        loadNotes();
      } else {
        alert("Failed to save note");
      }
    }
  } catch (error) {
    console.error("Error:", error);
    alert("Error saving note");
  }
}

async function loadNotes() {
  try {
    const response = await fetch(`${API_URL}/notes`);
    const notes = await response.json();
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
      `;
      noteList.appendChild(noteItem);
    });
  } catch (error) {
    console.error("Error loading notes:", error);
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
  if (confirm("Are you sure you want to delete this note?")) {
    try {
      const response = await fetch(`${API_URL}/notes/${noteId}`, {
        method: "DELETE"
      });
      if (response.ok) {
        alert("Note deleted!");
        loadNotes();
      } else {
        alert("Failed to delete note");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Error deleting note");
    }
  }
}

function clearNoteForm() {
  document.getElementById("noteTitle").value = "";
  document.getElementById("noteContent").value = "";
  currentNoteId = null;
  document.getElementById("saveBtn").textContent = "Save Note";
}

function logout() {
  alert("Logged out!");
  // TODO: clear session token and redirect to login
}