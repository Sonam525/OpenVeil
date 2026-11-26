/* OpenVeil project page: enlarge an image in place, copy code blocks, and play the
   silent screen recording while it is in view.
   The page works without this script: every image is also a plain link, and the
   recording has its own controls. */

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

function setupRecording() {
  // Play the muted recording while it is on screen, like an animated image. Not when the
  // visitor asks for reduced motion, and not again after they have paused it themselves.
  const v = document.querySelector("video[data-inview]");
  if (!v || !("IntersectionObserver" in window)) return;
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  let byScript = false, userPaused = false;
  v.addEventListener("pause", () => { if (!byScript) userPaused = true; byScript = false; });
  v.addEventListener("play", () => { userPaused = false; });
  new IntersectionObserver(entries => {
    const seen = entries[entries.length - 1].isIntersecting;
    if (seen && v.paused && !userPaused) {
      v.muted = true;
      const p = v.play();
      if (p) p.catch(() => {});
    } else if (!seen && !v.paused) {
      byScript = true;
      v.pause();
    }
  }, { threshold: 0.6 }).observe(v);
}

setupZoom();
setupCopy();
setupRecording();
