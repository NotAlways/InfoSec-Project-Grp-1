function showSection(sectionId) {
  const sections = document.querySelectorAll(".section");
  sections.forEach(sec => sec.style.display = "none");
  document.getElementById(sectionId).style.display = "block";
}

function logout() {
  alert("Logged out!");
  // TODO: clear session token and redirect to login
}