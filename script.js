function toggleMenu() {
  document.getElementById('burger').classList.toggle('open');
  document.getElementById('mobileMenu').classList.toggle('open');
  document.body.style.overflow = document.getElementById('mobileMenu').classList.contains('open') ? 'hidden' : '';
}

const obs = new IntersectionObserver(es => {
  es.forEach(e => {
    if (e.isIntersecting) e.target.classList.add('vis');
  });
}, { threshold: 0.08, rootMargin: '0px 0px -30px 0px' });
document.querySelectorAll('.reveal, .stagger-kids, .slide-left, .slide-right, .scale-in').forEach(el => obs.observe(el));

const countObs = new IntersectionObserver(es => {
  es.forEach(e => {
    if (!e.isIntersecting) return;
    const el = e.target, t = parseInt(el.dataset.target);
    if (!t || el.dataset.counted) return;
    el.dataset.counted = '1';
    const dur = 1500, st = performance.now();
    function tick(now) {
      const p = Math.min((now - st) / dur, 1), ea = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(t * ea).toLocaleString() + (t === 1000 ? '+' : t === 100 ? '%' : '');
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}, { threshold: 0.5 });
document.querySelectorAll('.count-up[data-target]').forEach(el => countObs.observe(el));

(function() {
  const f = document.getElementById('heroParticles');
  if (!f) return;
  for (let i = 0; i < 20; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    p.style.left = Math.random() * 100 + '%';
    p.style.bottom = '-10px';
    p.style.animationDuration = (4 + Math.random() * 6) + 's';
    p.style.animationDelay = Math.random() * 8 + 's';
    p.style.width = p.style.height = (1 + Math.random() * 2) + 'px';
    f.appendChild(p);
  }
})();

(function() {
  const c = document.getElementById('radarChart');
  if (!c) return;
  const ctx = c.getContext('2d'), dpr = window.devicePixelRatio || 1;
  c.width = 420 * dpr;
  c.height = 420 * dpr;
  c.style.width = '420px';
  c.style.height = '420px';
  ctx.scale(dpr, dpr);
  const cx = 210, cy = 210, R = 155, labels = ['Goal Reach', 'Speed', 'Coins', 'Kills', 'Survival'], n = labels.length;
  const data = [
    { c: '#E7575A', d: [0.95, 0.6, 0.35, 0.2, 0.8] },
    { c: '#6ec6e6', d: [0.75, 1, 0.1, 0.15, 0.45] },
    { c: '#ffd24d', d: [0.55, 0.3, 1, 0.25, 0.65] },
    { c: '#ff4d6a', d: [0.4, 0.5, 0.2, 1, 0.35] },
    { c: '#50c878', d: [0.8, 0.55, 0.7, 0.55, 0.7] }
  ];
  const a = i => (Math.PI * 2 * i / n) - Math.PI / 2;
  
  function drawBase() {
    for (let r = 1; r <= 5; r++) {
      const rad = (r / 5) * R;
      ctx.beginPath();
      for (let i = 0; i <= n; i++) {
        const x = cx + Math.cos(a(i % n)) * rad, y = cy + Math.sin(a(i % n)) * rad;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.strokeStyle = r === 5 ? 'rgba(61, 26, 36, 0.6)' : 'rgba(61, 26, 36, 0.25)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    for (let i = 0; i < n; i++) {
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(a(i)) * R, cy + Math.sin(a(i)) * R);
      ctx.strokeStyle = 'rgba(61, 26, 36, 0.35)';
      ctx.stroke();
    }
    ctx.font = '600 11px "Outfit", sans-serif';
    ctx.fillStyle = '#b0a0a8';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) {
      ctx.fillText(labels[i], cx + Math.cos(a(i)) * (R + 22), cy + Math.sin(a(i)) * (R + 22));
    }
  }
  
  function drawPoly(t) {
    data.forEach(p => {
      ctx.beginPath();
      p.d.forEach((v, i) => {
        const x = cx + Math.cos(a(i)) * v * R * t, y = cy + Math.sin(a(i)) * v * R * t;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fillStyle = p.c + '15';
      ctx.fill();
      ctx.strokeStyle = p.c + 'aa';
      ctx.lineWidth = 2;
      ctx.stroke();
      p.d.forEach((v, i) => {
        ctx.beginPath();
        ctx.arc(cx + Math.cos(a(i)) * v * R * t, cy + Math.sin(a(i)) * v * R * t, 3, 0, Math.PI * 2);
        ctx.fillStyle = p.c;
        ctx.fill();
      });
    });
  }
  
  drawBase();
  let prog = 0;
  const rObs = new IntersectionObserver(es => {
    if (es[0].isIntersecting && prog === 0) {
      prog = 0.01;
      anim();
    }
  }, { threshold: 0.3 });
  rObs.observe(c);
  
  function anim() {
    if (prog >= 1) return;
    prog = Math.min(1, prog + 0.025);
    const e = 1 - Math.pow(1 - prog, 3);
    ctx.clearRect(0, 0, 420, 420);
    drawBase();
    drawPoly(e);
    requestAnimationFrame(anim);
  }
})();

window.addEventListener('scroll', () => {
  document.querySelector('nav').style.background = window.scrollY > 80 ? 'rgba(12, 6, 8, 0.96)' : 'rgba(12, 6, 8, 0.9)';
});