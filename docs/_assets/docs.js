/* NewPage docs — shared behaviour. Generated into each doc; edit
   docs/_assets/docs.js and run docs/_assets/sync-design-system.py.

   Two jobs: the light/dark toggle, and TOC section highlighting. Light is the
   default and stays the default even when the OS prefers dark — newpage.io has
   no dark mode, so dark here is opt-in only. Docs render fine with JS off. */
(function () {
  /* ---- theme toggle ---- */
  var KEY = 'newpage-docs-theme';
  var root = document.documentElement;

  function apply(theme) {
    if (theme === 'dark') root.setAttribute('data-theme', 'dark');
    else root.removeAttribute('data-theme');
    var btn = document.querySelector('.theme-toggle');
    if (btn) {
      btn.textContent = theme === 'dark' ? '☀' : '☾';
      btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light' : 'Switch to dark');
      btn.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
    }
  }

  var stored;
  try { stored = localStorage.getItem(KEY); } catch (e) { stored = null; }
  apply(stored === 'dark' ? 'dark' : 'light');

  document.addEventListener('click', function (e) {
    var btn = e.target.closest && e.target.closest('.theme-toggle');
    if (!btn) return;
    var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    apply(next);
    try { localStorage.setItem(KEY, next); } catch (err) { /* private mode */ }
  });

  /* ---- TOC highlighting ---- */
  var toc = document.querySelector('.toc');
  if (!toc || !('IntersectionObserver' in window)) return;

  var links = {};
  var targets = [];
  Array.prototype.forEach.call(toc.querySelectorAll('a[href^="#"]'), function (a) {
    var el = document.getElementById(decodeURIComponent(a.hash.slice(1)));
    if (!el) return;
    links[el.id] = a;
    targets.push(el);
  });
  if (!targets.length) return;

  var visible = new Set();
  function paint() {
    var current = null;
    for (var i = 0; i < targets.length; i++) {
      if (visible.has(targets[i].id)) { current = targets[i].id; break; }
    }
    Object.keys(links).forEach(function (id) {
      links[id].classList.toggle('on', id === current);
    });
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) visible.add(e.target.id);
      else visible.delete(e.target.id);
    });
    paint();
  }, { rootMargin: '-8% 0px -70% 0px', threshold: 0 });

  targets.forEach(function (t) { io.observe(t); });
})();
