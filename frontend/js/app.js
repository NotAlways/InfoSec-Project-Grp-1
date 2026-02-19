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

/**
 * Convert a Base64 string to a Uint8Array (for WebCrypto).
 */
function b64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/**
 * Decrypt one AES-256-GCM ciphertext string using the WebCrypto API.
 *
 * @param {string} encryptedStr  Server format: "nonceB64:ciphertextB64:tagB64"
 * @param {CryptoKey} cryptoKey  AES-GCM CryptoKey (imported from raw DEK bytes)
 * @returns {Promise<string>}    Decrypted plaintext
 */
async function aesGcmDecrypt(encryptedStr, cryptoKey) {
  if (!encryptedStr || !encryptedStr.includes(":")) {
    return encryptedStr; // Already plaintext (legacy note)
  }
  const [nonceB64, ctB64, tagB64] = encryptedStr.split(":");
  const nonce      = b64ToBytes(nonceB64);
  const ciphertext = b64ToBytes(ctB64);
  const tag        = b64ToBytes(tagB64);

  // WebCrypto expects ciphertext + tag concatenated (matches Python AESGCM output)
  const fullCt = new Uint8Array(ciphertext.length + tag.length);
  fullCt.set(ciphertext, 0);
  fullCt.set(tag, ciphertext.length);

  const plainBuf = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: nonce },
    cryptoKey,
    fullCt,
  );
  return new TextDecoder().decode(plainBuf);
}

/**
 * Import a raw AES-256 key (from the server's dek_b64) into a WebCrypto CryptoKey.
 *
 * @param {string} dekB64  Base64-encoded 32-byte AES-256 key
 * @returns {Promise<CryptoKey>}
 */
async function importDEK(dekB64) {
  const rawKey = b64ToBytes(dekB64);
  return crypto.subtle.importKey(
    "raw",
    rawKey,
    { name: "AES-GCM" },
    false,          // Not extractable after import
    ["decrypt"],
  );
}

/**
 * Fetch the encrypted payload for a note and decrypt ENTIRELY in the browser.
 * The server only unwraps the DEK and forwards ciphertext – no plaintext visible.
 *
 * @param {number} noteId
 * @returns {Promise<{title: string, content: string, meta: object}>}
 */
async function decryptNoteClientSide(noteId) {
  const res = await fetch(`${API_URL}/notes/${noteId}/encrypted-payload`, {
    credentials: "include",
  });
  if (!res.ok) throw new Error(`Payload fetch failed: ${res.status}`);
  const payload = await res.json();

  // Import DEK into browser's non-extractable WebCrypto key store
  const cryptoKey = await importDEK(payload.dek_b64);

  // Decrypt title: use encrypted_title if present, else fallback to plaintext_title
  let title = payload.plaintext_title || `Note ${noteId}`;
  if (payload.encrypted_title) {
    title = await aesGcmDecrypt(payload.encrypted_title, cryptoKey);
  }

  // Decrypt content
  const content = await aesGcmDecrypt(payload.encrypted_content, cryptoKey);

  return {
    title,
    content,
    hasSig:    payload.has_signature,
    signerId:  payload.signer_id,
    createdAt: payload.created_at,
    updatedAt: payload.updated_at,
  };
}

/**
 * "Secure View" – renders a single note using client-side decryption.
 * Called when user clicks "Secure View" on a note card.
 */
async function secureViewNote(noteId) {
  const modal = document.getElementById("secureViewModal");
  const body  = document.getElementById("secureViewBody");
  if (!modal || !body) return;

  body.innerHTML = `
    <div class="sv-loading">
      <span class="sv-spinner"></span> Decrypting in browser…
    </div>`;
  modal.style.display = "flex";

  try {
    const note = await decryptNoteClientSide(noteId);
    body.innerHTML = `
      <div class="sv-header">
        <h2 class="sv-title">${escHtml(note.title)}</h2>
        ${note.hasSig
          ? `<span class="sig-badge sig-pending" id="sigBadge-${noteId}"
                   onclick="verifySig(${noteId})" title="Click to verify signature">
               Signed – click to verify
             </span>`
          : `<span class="sig-badge sig-none">Unsigned</span>`}
      </div>
      <div class="sv-meta">
        Created: ${note.createdAt ? new Date(note.createdAt).toLocaleString() : "—"} &nbsp;|&nbsp;
        Updated: ${note.updatedAt ? new Date(note.updatedAt).toLocaleString() : "—"}
      </div>
      <div class="sv-content">${escHtml(note.content)}</div>
      <div class="sv-zk-notice">
        Decrypted locally via WebCrypto
      </div>
      <div class="sv-actions">
        <button onclick="downloadNoteTxt(${noteId})" class="btn-download">Download .txt</button>
        <button onclick="closeSVModal()" class="btn-close-sv">Close</button>
      </div>`;
  } catch (err) {
    body.innerHTML = `<p class="sv-error">Decryption failed: ${escHtml(err.message)}</p>`;
    console.error("Secure view error:", err);
  }
}

/** Close the Secure View modal. */
function closeSVModal() {
  const modal = document.getElementById("secureViewModal");
  if (modal) modal.style.display = "none";
}

/** Simple HTML escaping to prevent XSS in decrypted content. */
function escHtml(str) {
  if (!str) return "";
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Close modal when clicking the backdrop
document.addEventListener("click", function (e) {
  const modal = document.getElementById("secureViewModal");
  if (modal && e.target === modal) closeSVModal();
});

/**
 * Generate Ed25519 signing keypair for the logged-in user.
 */
async function generateSigningKeys() {
  const btn = document.getElementById("genKeysBtn");
  if (btn) btn.disabled = true;

  try {
    const res = await fetch(`${API_URL}/auth/generate-signing-keys`, {
      method: "POST",
      credentials: "include",
    });
    const data = await res.json();
    if (res.ok) {
      showSigningStatus(
        `Signing keys generated. Public key: ${data.public_key.substring(0, 20)}…`,
        "success",
      );
    } else {
      showSigningStatus(`${data.detail || "Key generation failed"}`, "warn");
    }
  } catch (err) {
    showSigningStatus(`Error: ${err.message}`, "error");
  } finally {
    if (btn) btn.disabled = false;
  }
}

/**
 * Sign a note with the user's Ed25519 private key (server-side signing call).
 */
async function signNote(noteId) {
  try {
    const res = await fetch(`${API_URL}/notes/${noteId}/sign`, {
      method: "POST",
      credentials: "include",
    });
    const data = await res.json();
    if (res.ok) {
      // Update badge in the note card
      const badge = document.getElementById(`sigBadge-${noteId}`);
      if (badge) {
        badge.className = "sig-badge sig-pending";
        badge.textContent = "Signed – click to verify";
        badge.onclick = () => verifySig(noteId);
      }
      // If Secure View modal is open, refresh the badge there too
      const modalBadge = document.getElementById(`sigBadge-${noteId}`);
      if (modalBadge) modalBadge.className = "sig-badge sig-pending";

      alert(`Note #${noteId} signed successfully.\nSHA-256: ${data.content_sha256}`);
    } else {
      alert(`Signing failed: ${data.detail || JSON.stringify(data)}`);
    }
  } catch (err) {
    alert(`Error signing note: ${err.message}`);
  }
}

/**
 * Verify the digital signature on a note and update the badge.
 */
async function verifySig(noteId) {
  const badge = document.getElementById(`sigBadge-${noteId}`);
  if (badge) {
    badge.textContent = "Verifying…";
    badge.className   = "sig-badge sig-pending";
  }

  try {
    const res  = await fetch(`${API_URL}/notes/${noteId}/verify-signature`, {
      credentials: "include",
    });
    const data = await res.json();

    if (data.verified) {
      if (badge) {
        badge.className   = "sig-badge sig-verified";
        badge.textContent = `Verified – signed by ${data.signer}`;
        badge.title       = `Public key: ${(data.public_key || "").substring(0, 32)}…`;
      }
    } else {
      if (badge) {
        badge.className   = "sig-badge sig-invalid";
        badge.textContent = `${data.reason || "Signature invalid"}`;
      }
    }
  } catch (err) {
    if (badge) {
      badge.className   = "sig-badge sig-invalid";
      badge.textContent = "Verification error";
    }
  }
}

function showSigningStatus(msg, type) {
  const el = document.getElementById("signingStatus");
  if (!el) return;
  el.textContent  = msg;
  el.className    = `signing-status signing-status--${type}`;
  el.style.display = "block";
  setTimeout(() => (el.style.display = "none"), 6000);
}

/**
 * Upload a file via the secure ingest pipeline.
 * Replaces the stub uploadFile() referenced in index.html.
 */
async function uploadFile() {
  const input    = document.getElementById("fileInput");
  const statusEl = document.getElementById("uploadStatus");
  const progBar  = document.getElementById("uploadProgress");

  if (!input || !input.files.length) {
    alert("Please select a file first.");
    return;
  }

  const file    = input.files[0];
  const formData = new FormData();
  formData.append("file", file);

  if (statusEl)  statusEl.textContent = "Uploading & scanning…";
  if (progBar)   progBar.style.display = "block";

  try {
    const res  = await fetch(`${API_URL}/files/upload`, {
      method:      "POST",
      credentials: "include",
      body:        formData,
    });
    const data = await res.json();

    if (res.ok) {
      if (statusEl) {
        statusEl.textContent = `Uploaded (id=${data.id}, SHA-256: ${data.sha256.substring(0, 16)}…)`;
        statusEl.className   = "upload-status upload-status--ok";
      }
      input.value = "";
      loadFileList();
    } else {
      const reason = data.detail || JSON.stringify(data);
      if (statusEl) {
        statusEl.textContent = `Rejected: ${reason}`;
        statusEl.className   = "upload-status upload-status--err";
      }
    }
  } catch (err) {
    if (statusEl) {
      statusEl.textContent = `Error: ${err.message}`;
      statusEl.className   = "upload-status upload-status--err";
    }
  } finally {
    if (progBar) progBar.style.display = "none";
  }
}

/**
 * Load the list of the user's uploaded files and render them.
 */
async function loadFileList() {
  const listEl = document.getElementById("fileList");
  if (!listEl) return;

  try {
    const res  = await fetch(`${API_URL}/files`, { credentials: "include" });
    const files = await res.json();

    if (!files.length) {
      listEl.innerHTML = "<p class='no-files'>No files uploaded yet.</p>";
      return;
    }

    listEl.innerHTML = files.map(f => {
      const sizeKB = (f.file_size / 1024).toFixed(1);
      const ts     = f.uploaded_at ? new Date(f.uploaded_at).toLocaleString() : "—";
      return `
        <div class="file-item" id="file-${f.id}">
          <div class="file-info">
            <span class="file-name">${escHtml(f.original_name)}</span>
            <span class="file-meta">${sizeKB} KB &bull; ${ts}</span>
            <span class="file-integrity" title="SHA-256 of encrypted payload">
              ${f.sha256.substring(0, 16)}…
            </span>
          </div>
          <div class="file-actions">
            <button onclick="downloadUploadedFile(${f.id})" class="btn-dl">Download</button>
            <button onclick="deleteUploadedFile(${f.id})" class="btn-del-file">Delete</button>
          </div>
        </div>`;
    }).join("");
  } catch (err) {
    listEl.innerHTML = `<p class='file-error'>Failed to load files: ${err.message}</p>`;
  }
}

/**
 * Download a previously uploaded, encrypted file.
 * Server decrypts and streams it; browser receives plaintext bytes.
 */
async function downloadUploadedFile(fileId) {
  try {
    const res = await fetch(`${API_URL}/files/${fileId}/download`, {
      credentials: "include",
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Download failed" }));
      alert(`Download failed: ${err.detail}`);
      return;
    }

    // Extract filename from Content-Disposition header
    const cd       = res.headers.get("Content-Disposition") || "";
    const match    = cd.match(/filename="([^"]+)"/);
    const filename = match ? match[1] : `file_${fileId}`;

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Error downloading file: ${err.message}`);
  }
}

/**
 * Delete an uploaded file (also zeros the DEK for cryptographic deletion).
 */
async function deleteUploadedFile(fileId) {
  if (!confirm("Permanently delete this file?")) return;
  try {
    const res = await fetch(`${API_URL}/files/${fileId}`, {
      method: "DELETE",
      credentials: "include",
    });
    if (res.ok) {
      document.getElementById(`file-${fileId}`)?.remove();
    } else {
      alert("Delete failed.");
    }
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}

/**
 * Download a note as a plain-text file.
 * Content is decrypted locally in the browser before writing to the file.
 *
 * NOTE: "Export PDF" in the original code also triggers a browser download
 * (Content-Disposition: attachment).  This function adds a lighter-weight
 * .txt download that uses client-side decryption instead of server-side.
 */
async function downloadNoteTxt(noteId) {
  try {
    const note     = await decryptNoteClientSide(noteId);
    const ts       = new Date().toLocaleString();
    const content  = [
      `Title: ${note.title}`,
      `Exported: ${ts}`,
      `Signed by: ${note.signerId || "unsigned"}`,
      "",
      note.content,
      "",
      `— Downloaded from NoteVault by ${(window.NV_USER || {}).email || "user"} on ${ts}`,
    ].join("\n");

    const blob  = new Blob([content], { type: "text/plain" });
    const url   = URL.createObjectURL(blob);
    const a     = document.createElement("a");
    a.href      = url;
    a.download  = `note_${noteId}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Download failed: ${err.message}`);
  }
}

/**
 * Enhances each note card in #noteList with security action buttons after
 * the original loadNotes() has rendered them.
 */
function enhanceNoteCards() {
  const noteList = document.getElementById("noteList");
  if (!noteList) return;

  noteList.querySelectorAll(".note-item").forEach(card => {
    // Avoid double-enhancing
    if (card.dataset.enhanced) return;
    card.dataset.enhanced = "true";

    // Extract note ID from existing Edit button (onclick="editNote(N)")
    const editBtn = [...card.querySelectorAll("button")].find(
      b => b.textContent.trim() === "Edit",
    );
    if (!editBtn) return;
    const match = editBtn.getAttribute("onclick").match(/\d+/);
    if (!match) return;
    const noteId = parseInt(match[0], 10);

    // Signature badge (initially "Unsigned" – changes after verification)
    const sigBadge = document.createElement("span");
    sigBadge.className = "sig-badge sig-none";
    sigBadge.id        = `sigBadge-${noteId}`;
    sigBadge.textContent = "Unsigned";

    // Secure View button
    const svBtn = document.createElement("button");
    svBtn.textContent = "Secure View";
    svBtn.title       = "Decrypt and view this note entirely in the browser";
    svBtn.onclick     = () => secureViewNote(noteId);

    // Sign button
    const signBtn = document.createElement("button");
    signBtn.textContent = "Sign";
    signBtn.title       = "Sign this note with your Ed25519 key";
    signBtn.onclick     = () => signNote(noteId);

    // Download .txt button
    const dlBtn = document.createElement("button");
    dlBtn.textContent = "Download";
    dlBtn.title       = "Download note as .txt (client-side decryption)";
    dlBtn.onclick     = () => downloadNoteTxt(noteId);

    card.appendChild(sigBadge);
    card.appendChild(svBtn);
    card.appendChild(signBtn);
    card.appendChild(dlBtn);
  });
}

// Observe DOM mutations so we catch notes rendered after page load
const _noteListObserver = new MutationObserver(enhanceNoteCards);
document.addEventListener("DOMContentLoaded", () => {
  const noteList = document.getElementById("noteList");
  if (noteList) {
    _noteListObserver.observe(noteList, { childList: true, subtree: true });
  }
  // Load file list when Files section becomes visible
  document.querySelectorAll('[onclick*="files"]').forEach(el => {
    el.addEventListener("click", loadFileList);
  });
});

// Patch showSection() to also load files when the Files tab is opened
const _originalShowSection = window.showSection;
window.showSection = function (sectionId) {
  _originalShowSection(sectionId);
  if (sectionId === "files") {
    loadFileList();
  }
};