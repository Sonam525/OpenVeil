/* OpenVeil project page: enlarge an image in place, and copy code blocks.
   The page works without this script: every image is also a plain link. */

function setupZoom() {
  const dlg = document.getElementById("zoom");
  if (!dlg || typeof dlg.showModal !== "function") return;
  const img = dlg.querySelector("img"), cap = dlg.querySelector(".zoom-cap");
  document.querySelectorAll("a[data-zoom]").forEach(a => {
    a.addEventListener("click", e => {
      if (e.metaKey || e.ctrlKey || e.shiftKey) return;
      e.preventDefault();
      const small = a.querySelector("img"), fc = a.closest("figure")?.querySelector("figcaption");
      img.src = a.getAttribute("href");
      img.alt = small ? small.alt : "";
      cap.textContent = fc ? fc.textContent : "";
      dlg.showModal();
    });
  });
  dlg.addEventListener("click", e => { if (e.target === dlg) dlg.close(); });
  dlg.addEventListener("close", () => { img.removeAttribute("src"); });
}

function setupCopy() {
  if (!navigator.clipboard) return;
  document.querySelectorAll("pre").forEach(pre => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "copy"; b.textContent = "Copy";
    b.addEventListener("click", () => {
      const text = pre.querySelector("code").innerText.replace(/[ \t]+#.*$/gm, "");
      navigator.clipboard.writeText(text).then(() => {
        b.textContent = "Copied";
        setTimeout(() => { b.textContent = "Copy"; }, 1500);
      }, () => {});
    });
    pre.appendChild(b);
  });
}

setupZoom();
setupCopy();
