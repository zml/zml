// =============================================================================
// serve/page.zig — HTML for the compile-once generator
// =============================================================================

pub const html =
    \\<!doctype html>
    \\<html lang="en">
    \\<head>
    \\  <meta charset="utf-8"/>
    \\  <meta name="viewport" content="width=device-width, initial-scale=1"/>
    \\  <title>ZML Super</title>
    \\  <meta name="color-scheme" content="dark light"/>
    \\  <link rel="preconnect" href="https://fonts.googleapis.com"/>
    \\  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
    \\  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet"/>
    \\  <style>
    \\    :root {
    \\      color-scheme: light dark;
    \\      --font: "Inter", ui-sans-serif, system-ui, sans-serif;
    \\      --display: "Inter", ui-sans-serif, system-ui, sans-serif;
    \\      --cyan: #25dff1; --violet: #825df2; --magenta: #f456b4;
    \\      --bg: #07050c; --paper: #120e1a;
    \\      --glow-a: rgba(37,223,241,.22); --glow-b: rgba(244,86,180,.18); --glow-c: rgba(130,93,242,.32);
    \\      --card: rgba(18,14,28,.72); --line: rgba(196,176,255,.14); --line-hover: rgba(196,176,255,.45);
    \\      --muted: #9b90b3; --text: #f7f3fc;
    \\      --acc: #c4a6ff; --acc-soft: #e4d4ff; --ok: #3ee08f; --wait: #ffc14d; --err: #ff8a80;
    \\      --field: rgba(8,6,14,.88); --pick-on: rgba(130,93,242,.22);
    \\      --btn: #825df2; --btn-hi: #9b78ff; --btn-text: #fff;
    \\      --rail: rgba(196,176,255,.18); --dot-bg: #2a2238; --dot-line: #4a3d66; --bar: #1a1426;
    \\      --shadow: rgba(4,1,12,.55); --hair: linear-gradient(90deg, #25dff1, #825df2, #f456b4);
    \\      --radius: 16px; --radius-sm: 10px;
    \\    }
    \\    @media (prefers-color-scheme: light) {
    \\      :root {
    \\        --bg: #f5f5f3; --paper: #ffffff;
    \\        --glow-a: rgba(37,223,241,.14); --glow-b: rgba(244,86,180,.12); --glow-c: rgba(116,62,255,.16);
    \\        --card: #ffffff; --line: #e0e0dd; --line-hover: #111111;
    \\        --muted: #8e8e8a; --text: #111111;
    \\        --acc: #743eff; --acc-soft: #c4b0ff; --ok: #1a9d5c; --wait: #c48412; --err: #c62828;
    \\        --field: #fafaf8; --pick-on: rgba(116,62,255,.08);
    \\        --btn: #111111; --btn-hi: #111111; --btn-text: #f5f5f3;
    \\        --rail: #e0e0dd; --dot-bg: #ecece8; --dot-line: #c8c8c2; --bar: #ecece8;
    \\        --shadow: rgba(17,17,17,.06);
    \\        --radius: 16px; --radius-sm: 10px;
    \\      }
    \\    }
    \\    html[data-theme="dark"] {
    \\      color-scheme: dark;
    \\      --bg: #07050c; --paper: #120e1a;
    \\      --glow-a: rgba(37,223,241,.22); --glow-b: rgba(244,86,180,.18); --glow-c: rgba(130,93,242,.32);
    \\      --card: rgba(18,14,28,.72); --line: rgba(196,176,255,.14); --line-hover: rgba(196,176,255,.45);
    \\      --muted: #9b90b3; --text: #f7f3fc;
    \\      --acc: #c4a6ff; --acc-soft: #e4d4ff; --ok: #3ee08f; --wait: #ffc14d; --err: #ff8a80;
    \\      --field: rgba(8,6,14,.88); --pick-on: rgba(130,93,242,.22);
    \\      --btn: #825df2; --btn-hi: #9b78ff; --btn-text: #fff;
    \\      --rail: rgba(196,176,255,.18); --dot-bg: #2a2238; --dot-line: #4a3d66; --bar: #1a1426;
    \\      --shadow: rgba(4,1,12,.55);
    \\    }
    \\    html[data-theme="light"] {
    \\      color-scheme: light;
    \\      --bg: #f5f5f3; --paper: #ffffff;
    \\      --glow-a: rgba(37,223,241,.14); --glow-b: rgba(244,86,180,.12); --glow-c: rgba(116,62,255,.16);
    \\      --card: #ffffff; --line: #e0e0dd; --line-hover: #111111;
    \\      --muted: #8e8e8a; --text: #111111;
    \\      --acc: #743eff; --acc-soft: #c4b0ff; --ok: #1a9d5c; --wait: #c48412; --err: #c62828;
    \\      --field: #fafaf8; --pick-on: rgba(116,62,255,.08);
    \\      --btn: #111111; --btn-hi: #111111; --btn-text: #f5f5f3;
    \\      --rail: #e0e0dd; --dot-bg: #ecece8; --dot-line: #c8c8c2; --bar: #ecece8;
    \\      --shadow: rgba(17,17,17,.06);
    \\    }
    \\    * { box-sizing: border-box; }
    \\    html, body { margin: 0; min-height: 100%; }
    \\    body {
    \\      position: relative; color: var(--text); font: 15px/1.55 var(--font);
    \\      background:
    \\        linear-gradient(180deg, color-mix(in srgb, var(--bg) 88%, #000) 0%, var(--bg) 28%, var(--bg) 100%);
    \\      isolation: isolate;
    \\    }
    \\    body::before, body::after {
    \\      content: ""; position: fixed; inset: auto; z-index: -1; pointer-events: none;
    \\      width: 58vw; height: 58vw; border-radius: 50%; filter: blur(80px);
    \\    }
    \\    body::before {
    \\      top: -20vw; left: -14vw;
    \\      background: radial-gradient(circle at 30% 36%, var(--glow-a), transparent 58%),
    \\                  radial-gradient(circle at 52% 48%, var(--glow-c), transparent 68%);
    \\    }
    \\    body::after {
    \\      right: -18vw; top: -12vw;
    \\      background: radial-gradient(circle at 70% 66%, var(--glow-b), transparent 56%),
    \\                  radial-gradient(circle at 40% 40%, var(--glow-c), transparent 70%);
    \\    }
    \\    header {
    \\      display: flex; justify-content: space-between; align-items: center; gap: 16px;
    \\      padding: 0 40px; height: 72px; border-bottom: 1px solid var(--line);
    \\      background: color-mix(in srgb, var(--bg) 78%, transparent); backdrop-filter: blur(22px);
    \\    }
    \\    .brand { display: flex; align-items: center; gap: 14px; min-width: 0; }
    \\    .mark { position: relative; width: 36px; height: 36px; flex: none; }
    \\    .mark::before {
    \\      content: ""; position: absolute; inset: -70%; border-radius: 50%; z-index: -1;
    \\      background:
    \\        radial-gradient(circle at 30% 36%, rgba(30,225,255,.55), transparent 54%),
    \\        radial-gradient(circle at 70% 66%, rgba(255,64,183,.5), transparent 56%),
    \\        radial-gradient(circle at 52% 48%, rgba(116,62,255,.36), transparent 68%);
    \\      filter: blur(10px); animation: mark-glow 2.8s ease-in-out infinite;
    \\    }
    \\    .mark svg { display: block; width: 36px; height: 36px; }
    \\    .word { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
    \\    .tag {
    \\      margin: 0; font-family: var(--display); font-size: 16px; font-weight: 700;
    \\      letter-spacing: -.02em; line-height: 1.2; color: var(--text);
    \\    }
    \\    .sub {
    \\      margin: 0; font-size: 11px; font-weight: 500; letter-spacing: .12em;
    \\      text-transform: uppercase; color: var(--muted);
    \\    }
    \\    .pill { display: flex; align-items: center; gap: 10px; font-size: 12px; font-weight: 500; color: var(--muted); }
    \\    .theme {
    \\      width: 38px; height: 38px; border: 1px solid var(--line); border-radius: var(--radius-sm);
    \\      background: var(--field); color: var(--text); cursor: pointer;
    \\      display: grid; place-items: center;
    \\    }
    \\    .theme:hover { border-color: var(--line-hover); }
    \\    .theme svg { width: 16px; height: 16px; }
    \\    .theme .moon { display: none; }
    \\    html[data-theme="dark"] .theme .sun { display: none; }
    \\    html[data-theme="dark"] .theme .moon { display: block; }
    \\    @media (prefers-color-scheme: dark) {
    \\      html:not([data-theme]) .theme .sun { display: none; }
    \\      html:not([data-theme]) .theme .moon { display: block; }
    \\    }
    \\    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
    \\    .dot.on { background: var(--ok); box-shadow: 0 0 12px var(--ok); }
    \\    .dot.run { background: var(--violet); animation: glow 1.4s ease-in-out infinite; }
    \\    .dot.wait { background: var(--wait); }
    \\    @keyframes glow { 50% { opacity: .55; } }
    \\    @keyframes mark-glow {
    \\      0%, 100% { opacity: .75; transform: scale(.92); }
    \\      50% { opacity: 1; transform: scale(1.08); }
    \\    }
    \\    .wire { height: 2px; background: var(--hair); opacity: .85; }
    \\    main {
    \\      width: min(100%, 1480px); margin: 0 auto; padding: 40px 40px 88px;
    \\      display: grid; grid-template-columns: minmax(300px, 420px) minmax(0, 1fr); gap: 40px; align-items: start;
    \\    }
    \\    .lead {
    \\      margin: 0 0 8px; font-family: var(--display); font-size: clamp(32px, 4vw, 48px); font-weight: 800;
    \\      letter-spacing: -.025em; line-height: 1.15;
    \\    }
    \\    .lead em {
    \\      font-style: normal;
    \\      background: var(--hair); -webkit-background-clip: text; background-clip: text; color: transparent;
    \\    }
    \\    .dek { margin: 0 0 22px; max-width: 34rem; color: var(--muted); font-size: 14px; line-height: 1.7; }
    \\    .card {
    \\      background: var(--card); border: 1px solid var(--line); border-radius: var(--radius);
    \\      padding: 22px; box-shadow: 0 30px 80px var(--shadow);
    \\      backdrop-filter: blur(18px);
    \\    }
    \\    label {
    \\      display: block; font-size: 11px; font-weight: 600; letter-spacing: .12em;
    \\      text-transform: uppercase; color: var(--muted); margin: 0 0 8px;
    \\    }
    \\    textarea, input {
    \\      width: 100%; border: 1px solid var(--line); background: var(--field); color: inherit;
    \\      border-radius: var(--radius-sm); padding: 12px 14px; font: 15px/1.5 var(--font);
    \\    }
    \\    textarea { min-height: 148px; resize: vertical; }
    \\    textarea:focus, input:focus { outline: 2px solid color-mix(in srgb, var(--violet) 55%, transparent); border-color: transparent; }
    \\    .picks { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 18px; }
    \\    .seg { display: flex; gap: 0; margin: 0 0 18px; border: 1px solid var(--line); border-radius: var(--radius-sm); overflow: hidden; background: var(--field); }
    \\    .seg .pick {
    \\      flex: 1; margin: 0; border: 0; border-radius: 0; border-right: 1px solid var(--line);
    \\      padding: 10px 8px; text-align: center; background: transparent;
    \\    }
    \\    .seg .pick:last-child { border-right: 0; }
    \\    .seg .pick.on { background: var(--pick-on); color: var(--acc); }
    \\    .pick {
    \\      border: 1px solid var(--line); background: transparent; color: var(--text);
    \\      border-radius: 999px; padding: 7px 12px; font: 12px/1 var(--font); font-weight: 600;
    \\      letter-spacing: .02em; cursor: pointer;
    \\    }
    \\    .pick:hover { border-color: var(--line-hover); }
    \\    .pick.on { border-color: var(--violet); color: var(--acc); background: var(--pick-on); }
    \\    .pick:disabled { opacity: .35; cursor: not-allowed; }
    \\    .row { display: flex; gap: 12px; align-items: end; margin-top: 14px; flex-wrap: wrap; }
    \\    .seed { width: 132px; flex: none; }
    \\    #go {
    \\      border: 0; border-radius: var(--radius-sm); min-height: 46px; padding: 0 22px; flex: 1;
    \\      font: 12px/1 var(--font); font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
    \\      background: linear-gradient(90deg, var(--violet), var(--magenta)); color: #fff; cursor: pointer;
    \\      box-shadow: 0 8px 28px rgba(130,93,242,.28);
    \\    }
    \\    #go:hover:not(:disabled) { filter: brightness(1.06); }
    \\    #go:disabled { opacity: .45; cursor: wait; box-shadow: none; }
    \\    #run { display: none; margin-top: 18px; }
    \\    #run.show { display: block; }
    \\    .rail { display: flex; justify-content: space-between; gap: 4px; position: relative; }
    \\    .rail::before { content: ""; position: absolute; left: 8%; right: 8%; top: 13px; height: 1px; background: var(--rail); }
    \\    .st { flex: 1; text-align: center; position: relative; font-size: 10px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
    \\    .st i { display: block; width: 10px; height: 10px; margin: 8px auto 8px; border-radius: 50%; background: var(--dot-bg); border: 2px solid var(--dot-line); }
    \\    .st.done { color: var(--ok); }
    \\    .st.done i { background: var(--ok); border-color: var(--ok); box-shadow: 0 0 10px var(--ok); }
    \\    .st.active { color: var(--acc); }
    \\    .st.active i { background: var(--violet); border-color: var(--acc-soft); box-shadow: 0 0 16px var(--violet); }
    \\    .bar { margin: 14px 0 8px; height: 4px; border-radius: 99px; background: var(--bar); overflow: hidden; }
    \\    .bar span { display: block; height: 100%; width: 0; background: var(--hair); transition: width .25s ease; }
    \\    #now { margin: 0; font-size: 14px; font-weight: 600; letter-spacing: .01em; }
    \\    #now.err { color: var(--err); }
    \\    .stage { min-height: 280px; }
    \\    #vid { display: none; width: 100%; border-radius: calc(var(--radius) - 2px); background: #000; aspect-ratio: 16 / 9; }
    \\    #vid.on { display: block; }
    \\    .frame {
    \\      position: relative;
    \\      border: 1px solid var(--line); border-radius: var(--radius); background: #07060c;
    \\      min-height: 320px; display: grid; place-items: center; overflow: hidden;
    \\      box-shadow: inset 0 0 80px rgba(130,93,242,.08);
    \\    }
    \\    .frame:has(#vid.on) { display: block; min-height: 0; }
    \\    .frame:has(#vid.on) .placeholder { display: none; }
    \\    .save {
    \\      position: absolute; top: 14px; right: 14px; z-index: 2;
    \\      display: none; align-items: center; gap: 8px;
    \\      border: 1px solid rgba(255,255,255,.22); border-radius: 999px;
    \\      background: rgba(12, 9, 20, .72); color: #fff;
    \\      padding: 8px 14px; cursor: pointer;
    \\      font: 11px/1 var(--font); font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    \\      backdrop-filter: blur(12px);
    \\    }
    \\    .save svg { width: 14px; height: 14px; flex: none; }
    \\    .frame:has(#vid.on) .save { display: inline-flex; }
    \\    .save:hover { background: rgba(12, 9, 20, .92); border-color: rgba(255,255,255,.4); }
    \\    .save:disabled { opacity: .5; cursor: wait; }
    \\    .placeholder { margin: 0; padding: 56px 28px; text-align: center; color: var(--muted); font-size: 14px; }
    \\    .placeholder strong {
    \\      display: block; margin-bottom: 8px; color: var(--text);
    \\      font-family: var(--display); font-size: 15px; letter-spacing: -.02em; text-transform: none; font-weight: 800;
    \\    }
    \\    video { width: 100%; border-radius: calc(var(--radius) - 2px); background: #000; aspect-ratio: 16 / 9; }
    \\    .note { margin: 14px 0 0; font-size: 13px; color: var(--muted); }
    \\    .clips h2 {
    \\      margin: 28px 0 8px; font-size: 11px; font-weight: 600; letter-spacing: .1em;
    \\      text-transform: uppercase; color: var(--muted);
    \\    }
    \\    .hint { margin: 0 0 12px; font-size: 13px; color: var(--muted); }
    \\    .hint.hide { display: none; }
    \\    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
    \\    .grid video { margin: 0; cursor: pointer; border: 1px solid var(--line); border-radius: var(--radius-sm); transition: transform .18s ease, border-color .18s ease; }
    \\    .grid video:hover { transform: translateY(-2px); border-color: var(--line-hover); }
    \\    footer {
    \\      display: flex; justify-content: flex-end; align-items: center;
    \\      padding: 0 40px; height: 52px; border-top: 1px solid var(--line); color: var(--muted);
    \\      font-size: 12px; letter-spacing: .08em; text-transform: uppercase;
    \\    }
    \\    .boot {
    \\      position: fixed; inset: 0; z-index: 40; display: grid; place-items: center;
    \\      background: color-mix(in srgb, var(--bg) 94%, #000); backdrop-filter: blur(28px);
    \\    }
    \\    .boot.hide { display: none; }
    \\    .boot-panel {
    \\      width: min(520px, calc(100% - 40px)); padding: 32px 28px;
    \\      border: 1px solid var(--line); border-radius: var(--radius); background: var(--paper);
    \\      box-shadow: 0 30px 80px var(--shadow);
    \\    }
    \\    .boot-kicker {
    \\      margin: 0 0 10px; font-size: 11px; font-weight: 600; letter-spacing: .14em;
    \\      text-transform: uppercase; color: var(--muted);
    \\    }
    \\    .boot-title {
    \\      margin: 0 0 12px; font-family: var(--display); font-size: clamp(26px, 3.6vw, 36px); font-weight: 800;
    \\      letter-spacing: -.025em; line-height: 1.2;
    \\    }
    \\    .boot-title em {
    \\      font-style: normal; background: var(--hair);
    \\      -webkit-background-clip: text; background-clip: text; color: transparent;
    \\    }
    \\    .boot-now { margin: 0 0 18px; font-size: 15px; color: var(--muted); }
    \\    .sku-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 0 0 18px; }
    \\    .sku-chip {
    \\      border: 1px solid var(--line); border-radius: var(--radius-sm); padding: 8px 6px;
    \\      text-align: center; font-size: 11px; font-weight: 700; letter-spacing: .04em;
    \\      color: var(--muted); background: var(--field);
    \\    }
    \\    .sku-chip.run { color: var(--acc); border-color: var(--violet); box-shadow: 0 0 0 1px color-mix(in srgb, var(--violet) 40%, transparent); }
    \\    .sku-chip.done { color: var(--ok); border-color: color-mix(in srgb, var(--ok) 55%, var(--line)); }
    \\    .sku-chip.skip { opacity: .4; text-decoration: line-through; }
    \\    .opts { margin: 0 0 16px; }
    \\    @media (prefers-reduced-motion: reduce) {
    \\      .mark::before { animation: none; }
    \\    }
    \\    @media (max-width: 900px) {
    \\      header, footer { padding: 0 20px; }
    \\      main { grid-template-columns: 1fr; gap: 28px; padding: 24px 20px 72px; }
    \\      .lead { font-size: 40px; }
    \\      .st { font-size: 9px; }
    \\      .sku-grid { grid-template-columns: repeat(2, 1fr); }
    \\    }
    \\  </style>
    \\</head>
    \\<body>
    \\<div id="boot" class="boot">
    \\  <div class="boot-panel">
    \\    <p class="boot-kicker">Compile once · then generate</p>
    \\    <p class="boot-title" id="boot-title">Warming <em>Super</em></p>
    \\    <p class="boot-now" id="boot-now">Starting</p>
    \\    <div class="sku-grid" id="sku-grid"></div>
    \\    <div class="bar"><span id="boot-fill"></span></div>
    \\  </div>
    \\</div>
    \\<header>
    \\  <div class="brand">
    \\    <span class="mark" aria-hidden="true">
    \\      <svg viewBox="0 0 32 32" fill="none">
    \\        <defs>
    \\          <linearGradient id="zg" x1="4" y1="4" x2="28" y2="28" gradientUnits="userSpaceOnUse">
    \\            <stop stop-color="#25dff1"/>
    \\            <stop offset=".52" stop-color="#825df2"/>
    \\            <stop offset="1" stop-color="#f456b4"/>
    \\          </linearGradient>
    \\        </defs>
    \\        <path fill="url(#zg)" d="M6 6h20v5.1L13.8 21H26v5H6v-5.1L18.2 11H6V6z"/>
    \\      </svg>
    \\    </span>
    \\    <div class="word">
    \\      <p class="tag">ZML Super</p>
    \\      <p class="sub">Draft · refine · serve</p>
    \\    </div>
    \\  </div>
    \\  <div class="pill">
    \\    <button type="button" class="theme" id="theme" aria-label="Toggle theme">
    \\      <svg class="sun" viewBox="0 0 16 16" fill="none" aria-hidden="true"><circle cx="8" cy="8" r="3.1" stroke="currentColor" stroke-width="1.5"/><path d="M8 1.4v1.6M8 13v1.6M1.4 8h1.6M13 8h1.6M3.2 3.2l1.1 1.1M11.7 11.7l1.1 1.1M3.2 12.8l1.1-1.1M11.7 4.3l1.1-1.1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
    \\      <svg class="moon" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M13.2 9.1A5.2 5.2 0 0 1 6.9 2.8 5.4 5.4 0 1 0 13.2 9.1Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>
    \\    </button>
    \\    <i class="dot" id="dot"></i><span id="live">Connecting</span>
    \\  </div>
    \\</header>
    \\<div class="wire" aria-hidden="true"></div>
    \\<main>
    \\  <section>
    \\    <p class="lead">Make a <em>clip</em></p>
    \\    <p class="dek" id="dek">H3 draft, LTX refine. One job at a time.</p>
    \\    <div class="card">
    \\      <label>Examples</label>
    \\      <div class="picks" id="picks"></div>
    \\      <div class="opts">
    \\        <label>Length</label>
    \\        <div class="seg" id="lens"></div>
    \\        <label>Canvas</label>
    \\        <div class="seg" id="sizes"></div>
    \\      </div>
    \\      <label for="prompt">Prompt</label>
    \\      <textarea id="prompt" placeholder="Describe the shot…"></textarea>
    \\      <div class="row">
    \\        <div class="seed">
    \\          <label for="seed">Seed</label>
    \\          <input id="seed" type="number" value="7" min="0"/>
    \\        </div>
    \\        <button id="go" type="button" disabled>Generate</button>
    \\      </div>
    \\      <div id="run">
    \\        <div class="rail" id="rail"></div>
    \\        <div class="bar"><span id="fill"></span></div>
    \\        <p id="now"></p>
    \\      </div>
    \\    </div>
    \\  </section>
    \\  <section class="stage">
    \\    <div class="frame">
    \\      <p class="placeholder" id="blank"><strong>Stage is empty</strong>Generate a shot and it lands here.</p>
    \\      <video id="vid" controls playsinline></video>
    \\      <button type="button" class="save" id="save" aria-label="Save MP4">
    \\        <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
    \\          <path d="M8 2v8m0 0L5 7.5M8 10l3-2.5M3 13.5h10" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
    \\        </svg>
    \\        Save MP4
    \\      </button>
    \\    </div>
    \\    <div class="clips">
    \\      <h2>Your clips</h2>
    \\      <p class="hint" id="empty">Clips stay in this browser.</p>
    \\      <div id="grid" class="grid"></div>
    \\    </div>
    \\  </section>
    \\</main>
    \\<footer>ZML</footer>
    \\<script>
    \\const STEPS = [
    \\  { id: "text", title: "Text" },
    \\  { id: "draft", title: "Draft" },
    \\  { id: "decode", title: "Decode" },
    \\  { id: "vae", title: "VAE" },
    \\  { id: "refine", title: "Refine" },
    \\  { id: "output", title: "Output" },
    \\  { id: "encode", title: "Encode" },
    \\];
    \\const EXAMPLES = [
    \\  { name: "Tokyo dusk", seed: 7, text: "A close handheld shot of a woman in a wool coat talking to camera on a wet Tokyo side street at dusk, neon shop signs and passing umbrellas behind her, warm practical lights on her face, shallow depth of field, 24mm, photoreal, no text." },
    \\  { name: "Window portrait", seed: 11, text: "A locked-off close-up of a woman in a cream linen shirt standing in bright window light, sharp eyes and skin texture, clean background, 85mm, shallow depth of field, photoreal, no text." },
    \\  { name: "Alpine lake", seed: 21, text: "A slow drone over a crystal alpine lake at sunrise, snow peaks mirrored in still water, razor-sharp pine trees, 24mm, bright clean air, photoreal, no text." },
    \\  { name: "Red coupe", seed: 3, text: "A locked-off shot of a red sports car on wet asphalt in hard noon sun, chrome and paint reflections crystal clear, 35mm, photoreal, no text." },
    \\  { name: "Sunlit kitchen", seed: 14, text: "A chef in a white jacket plates food in a bright kitchen, steam catching backlight, sharp knives and stainless steel, 50mm, photoreal, no text." },
    \\  { name: "Coastal rock", seed: 19, text: "Waves crash on black volcanic rock in bright sunlight, water spray sharp and clear, locked-off 24mm, photoreal, no text." },
    \\  { name: "Glass lobby", seed: 5, text: "A man in a navy suit walking through a sunlit glass lobby, sharp marble and architecture, 35mm, photoreal, no text." },
    \\  { name: "Rain leaf", seed: 8, text: "Macro shot of raindrops on a green leaf in a sunlit garden, extreme detail, shallow focus, 100mm, photoreal, no text." },
    \\];
    \\const prompt = document.getElementById("prompt");
    \\const seed = document.getElementById("seed");
    \\const go = document.getElementById("go");
    \\const run = document.getElementById("run");
    \\const rail = document.getElementById("rail");
    \\const fill = document.getElementById("fill");
    \\const now = document.getElementById("now");
    \\const vid = document.getElementById("vid");
    \\const save = document.getElementById("save");
    \\const live = document.getElementById("live");
    \\const dot = document.getElementById("dot");
    \\const grid = document.getElementById("grid");
    \\const empty = document.getElementById("empty");
    \\const picks = document.getElementById("picks");
    \\const themeBtn = document.getElementById("theme");
    \\const boot = document.getElementById("boot");
    \\const bootTitle = document.getElementById("boot-title");
    \\const bootNow = document.getElementById("boot-now");
    \\const bootFill = document.getElementById("boot-fill");
    \\const skuGrid = document.getElementById("sku-grid");
    \\const dek = document.getElementById("dek");
    \\const lens = document.getElementById("lens");
    \\const sizes = document.getElementById("sizes");
    \\let SKUS = [];
    \\let current = null;
    \\function isHd(s) { return !!s.hd; }
    \\function skuId() { return current ? current.id : ""; }
    \\function families() {
    \\  const out = [];
    \\  const seen = new Set();
    \\  for (const s of SKUS) {
    \\    const key = isHd(s) ? "hd" : "super";
    \\    if (seen.has(key)) continue;
    \\    seen.add(key);
    \\    out.push({ hd: isHd(s), label: (isHd(s) ? "HD " : "Super ") + s.width + "×" + s.height });
    \\  }
    \\  return out;
    \\}
    \\function durations(hd) {
    \\  return [...new Set(SKUS.filter(s => isHd(s) === hd).map(s => s.seconds))].sort((a, b) => a - b);
    \\}
    \\function findSku(seconds, hd) {
    \\  return SKUS.find(s => s.seconds === seconds && isHd(s) === hd) || null;
    \\}
    \\function preferSku() {
    \\  return SKUS.find(s => s.preferred) || SKUS[0] || null;
    \\}
    \\function renderOpts() {
    \\  if (SKUS.length === 0) return;
    \\  if (!current || !SKUS.some(s => s.id === current.id)) current = preferSku();
    \\  if (!current) return;
    \\  const hd = isHd(current);
    \\  const secs = durations(hd);
    \\  const lensHtml = secs.map(L => '<button type="button" class="pick' + (L === current.seconds ? ' on' : '') + '" data-len="' + L + '">' + L + 's</button>').join("");
    \\  if (lens.innerHTML !== lensHtml) lens.innerHTML = lensHtml;
    \\  else [...lens.children].forEach(btn => btn.classList.toggle("on", Number(btn.dataset.len) === current.seconds));
    \\  const sizeHtml = families().map(f => '<button type="button" class="pick' + (f.hd === hd ? ' on' : '') + '" data-hd="' + (f.hd ? "1" : "0") + '">' + f.label + '</button>').join("");
    \\  if (sizes.innerHTML !== sizeHtml) sizes.innerHTML = sizeHtml;
    \\  else [...sizes.children].forEach(btn => btn.classList.toggle("on", (btn.dataset.hd === "1") === hd));
    \\  setText(dek, current.seconds + "s · " + current.width + "×" + current.height + " · one job at a time.");
    \\}
    \\lens.addEventListener("click", ev => {
    \\  const btn = ev.target.closest(".pick");
    \\  if (!btn || !current) return;
    \\  const next = findSku(Number(btn.dataset.len), isHd(current));
    \\  if (next) current = next;
    \\  renderOpts();
    \\});
    \\sizes.addEventListener("click", ev => {
    \\  const btn = ev.target.closest(".pick");
    \\  if (!btn || !current) return;
    \\  const wantHd = btn.dataset.hd === "1";
    \\  current = findSku(current.seconds, wantHd) || SKUS.find(s => isHd(s) === wantHd) || current;
    \\  renderOpts();
    \\});
    \\function systemDark() { return matchMedia("(prefers-color-scheme: dark)").matches; }
    \\function themeNow() { return document.documentElement.dataset.theme || (systemDark() ? "dark" : "light"); }
    \\function applyTheme(mode) {
    \\  if (mode === "light" || mode === "dark") document.documentElement.dataset.theme = mode;
    \\  else delete document.documentElement.dataset.theme;
    \\  localStorage.setItem("theme", mode || "");
    \\}
    \\applyTheme(localStorage.getItem("theme"));
    \\themeBtn.addEventListener("click", () => applyTheme(themeNow() === "dark" ? "light" : "dark"));
    \\matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    \\  if (!document.documentElement.dataset.theme) applyTheme("");
    \\});
    \\let mine = false;
    \\let timer = null;
    \\let lastIds = "";
    \\let chosen = 0;
    \\rail.innerHTML = STEPS.map(s => '<div class="st" data-id="' + s.id + '"><i></i>' + s.title + '</div>').join("");
    \\picks.innerHTML = EXAMPLES.map((ex, i) => '<button type="button" class="pick" data-i="' + i + '">' + ex.name + '</button>').join("");
    \\function useExample(i) {
    \\  const ex = EXAMPLES[i];
    \\  if (!ex) return;
    \\  chosen = i;
    \\  prompt.value = ex.text;
    \\  seed.value = String(ex.seed);
    \\  [...picks.children].forEach((el, n) => el.classList.toggle("on", n === i));
    \\}
    \\picks.addEventListener("click", ev => {
    \\  const btn = ev.target.closest(".pick");
    \\  if (btn) useExample(Number(btn.dataset.i));
    \\});
    \\prompt.addEventListener("input", () => {
    \\  [...picks.children].forEach(el => el.classList.remove("on"));
    \\  chosen = -1;
    \\});
    \\useExample(0);
    \\function setClass(el, name) { if (el.className !== name) el.className = name; }
    \\function setText(el, text) { if (el.textContent !== text) el.textContent = text; }
    \\function showHero(src, sku) {
    \\  if (!src) return;
    \\  if (sku) vid.dataset.sku = sku;
    \\  if (vid.dataset.src !== src) {
    \\    vid.dataset.src = src;
    \\    vid.muted = false;
    \\    vid.src = src;
    \\  }
    \\  vid.classList.add("on");
    \\  vid.play().catch(() => {});
    \\}
    \\function clipName(src) {
    \\  const path = src || "";
    \\  const mark = "/video/";
    \\  const at = path.lastIndexOf(mark);
    \\  const rest = at < 0 ? "clip" : path.slice(at + mark.length);
    \\  const id = rest.replace(".mp4", "").replace(/[^0-9a-f]/gi, "") || "clip";
    \\  const sku = (vid.dataset.sku || "clip").replace(/[^a-z0-9-]+/gi, "");
    \\  const t = new Date();
    \\  const p = n => String(n).padStart(2, "0");
    \\  const stamp = t.getFullYear() + p(t.getMonth() + 1) + p(t.getDate()) + "-" + p(t.getHours()) + p(t.getMinutes()) + p(t.getSeconds());
    \\  return "zml-super-" + sku + "-" + stamp + "-" + id.slice(0, 8) + ".mp4";
    \\}
    \\async function saveMp4() {
    \\  const src = vid.dataset.src || vid.currentSrc || vid.src;
    \\  if (!src) return;
    \\  const name = clipName(src);
    \\  save.disabled = true;
    \\  try {
    \\    const res = await fetch(src, { credentials: "same-origin" });
    \\    if (!res.ok) throw new Error("download failed");
    \\    const blob = await res.blob();
    \\    const url = URL.createObjectURL(blob);
    \\    const a = document.createElement("a");
    \\    a.href = url;
    \\    a.download = name;
    \\    document.body.appendChild(a);
    \\    a.click();
    \\    a.remove();
    \\    setTimeout(() => URL.revokeObjectURL(url), 1500);
    \\  } catch (e) {
    \\    const a = document.createElement("a");
    \\    a.href = src;
    \\    a.download = name;
    \\    a.click();
    \\  }
    \\  save.disabled = false;
    \\}
    \\save.addEventListener("click", ev => { ev.preventDefault(); ev.stopPropagation(); saveMp4(); });
    \\function paint(s) {
    \\  const stage = s.stage || "idle";
    \\  const idx = STEPS.findIndex(x => x.id === stage);
    \\  [...rail.children].forEach((el, i) => {
    \\    setClass(el, "st" + (idx < 0 ? "" : i < idx ? " done" : i === idx ? " active" : ""));
    \\  });
    \\  const pct = Math.max(0, Math.min(100, s.pct || 0)) + "%";
    \\  if (fill.style.width !== pct) fill.style.width = pct;
    \\  setClass(now, "");
    \\  setText(now, s.label || "");
    \\  run.classList.add("show");
    \\}
    \\function paintQueue(ahead) {
    \\  run.classList.add("show");
    \\  [...rail.children].forEach(el => setClass(el, "st"));
    \\  if (fill.style.width !== "0%") fill.style.width = "0%";
    \\  setClass(now, "");
    \\  setText(now, ahead ? "In queue · " + ahead + " ahead" : "Next up");
    \\}
    \\function renderVideos(ids) {
    \\  const next = (ids || []).map(String);
    \\  const key = next.join(",");
    \\  empty.classList.toggle("hide", next.length > 0);
    \\  if (key === lastIds) return;
    \\  const have = new Map([...grid.children].map(el => [el.dataset.id, el]));
    \\  next.forEach(id => {
    \\    if (have.has(id)) return;
    \\    const v = document.createElement("video");
    \\    v.dataset.id = id;
    \\    v.src = "/video/" + id + ".mp4";
    \\    v.muted = false;
    \\    v.playsInline = true;
    \\    v.preload = "metadata";
    \\    v.addEventListener("click", () => showHero(v.currentSrc || v.src));
    \\    grid.appendChild(v);
    \\  });
    \\  next.forEach((id, i) => {
    \\    const el = grid.querySelector('[data-id="' + id + '"]');
    \\    if (el && grid.children[i] !== el) grid.insertBefore(el, grid.children[i] || null);
    \\  });
    \\  [...grid.children].forEach(el => {
    \\    if (!next.includes(el.dataset.id)) el.remove();
    \\  });
    \\  lastIds = key;
    \\}
    \\function skuLabel(id) {
    \\  const row = SKUS.find(s => s.id === id);
    \\  if (row) return row.seconds + "s " + (row.hd ? "HD" : "Super");
    \\  return id || "";
    \\}
    \\function paintSkuGrid(items) {
    \\  const rows = items || [];
    \\  const html = rows.map(s => '<div class="sku-chip ' + (s.state || "pending") + '">' + skuLabel(s.id) + '</div>').join("");
    \\  if (skuGrid.innerHTML !== html) skuGrid.innerHTML = html;
    \\}
    \\let lastCompileSkus = [];
    \\function paintBoot(s) {
    \\  boot.classList.remove("hide");
    \\  go.disabled = true;
    \\  const failed = s.phase === "failed";
    \\  bootTitle.innerHTML = failed ? "Compile failed" : "Warming <em>Super</em>";
    \\  setText(bootNow, s.label || (failed ? "See server logs" : "Starting"));
    \\  if (Array.isArray(s.compile_skus) && s.compile_skus.length) lastCompileSkus = s.compile_skus;
    \\  paintSkuGrid(s.compile_skus && s.compile_skus.length ? s.compile_skus : lastCompileSkus);
    \\  const pct = Math.max(0, Math.min(100, s.pct || 0)) + "%";
    \\  if (bootFill.style.width !== pct) bootFill.style.width = pct;
    \\  setClass(dot, failed ? "dot" : "dot run");
    \\  setText(live, (s.label || (failed ? "Failed" : "Compiling")) + (s.devices ? " · " + s.devices + " GPU" : ""));
    \\}
    \\async function poll() {
    \\  try {
    \\    const s = await (await fetch("/api/status", { credentials: "same-origin" })).json();
    \\    const phase = s.phase || (s.ready ? "ready" : "compiling");
    \\    if (phase !== "ready") {
    \\      paintBoot(s);
    \\      renderVideos(s.videos || []);
    \\      return;
    \\    }
    \\    if (Array.isArray(s.skus) && s.skus.length) {
    \\      SKUS = s.skus;
    \\      renderOpts();
    \\    }
    \\    if (!boot.classList.contains("hide")) {
    \\      boot.classList.add("hide");
    \\      go.disabled = false;
    \\      tickFast(mine);
    \\    }
    \\    const you = s.you || "idle";
    \\    const q = s.queue || 0;
    \\    const gpu = " · " + s.devices + " GPU";
    \\    if (you === "running") {
    \\      setClass(dot, "dot run");
    \\      setText(live, (s.label || "Generating") + gpu);
    \\      paint(s);
    \\    } else if (you === "queued") {
    \\      setClass(dot, "dot wait");
    \\      const ahead = s.ahead || 0;
    \\      setText(live, (ahead ? "In queue · " + ahead + " ahead" : "Next up") + gpu);
    \\      paintQueue(ahead);
    \\    } else {
    \\      setClass(dot, s.busy ? "dot wait" : "dot on");
    \\      setText(live, (s.busy ? (q > 1 ? "In use · " + (q - 1) + " waiting" : "In use") : "Ready") + gpu);
    \\    }
    \\    renderVideos(s.videos || []);
    \\  } catch (e) {
    \\    setClass(dot, "dot");
    \\    setText(live, "Offline");
    \\    if (!boot.classList.contains("hide")) {
    \\      paintBoot({ phase: "failed", label: "Server offline", compile_skus: [], pct: 0 });
    \\    }
    \\  }
    \\}
    \\function tickFast(on) {
    \\  if (timer) clearInterval(timer);
    \\  timer = setInterval(poll, on ? 350 : 4000);
    \\}
    \\async function generate() {
    \\  if (!boot.classList.contains("hide")) return;
    \\  const text = prompt.value.trim();
    \\  if (!text) { run.classList.add("show"); setClass(now, "err"); setText(now, "Enter a prompt"); return; }
    \\  mine = true;
    \\  go.disabled = true;
    \\  run.classList.add("show");
    \\  paint({ stage: "text", pct: 2, label: "Starting", busy: true });
    \\  tickFast(true);
    \\  try {
    \\    const res = await fetch("/generate", {
    \\      method: "POST",
    \\      credentials: "same-origin",
    \\      headers: { "content-type": "application/json" },
    \\      body: JSON.stringify({ prompt: text, seed: Number(seed.value) || 42, sku: skuId() }),
    \\    });
    \\    const data = await res.json();
    \\    if (!data.ok) throw new Error(data.error || res.statusText);
    \\    showHero(data.video, skuId());
    \\    paint({ stage: "encode", pct: 100, label: (data.infer_ms / 1000).toFixed(2) + "s", busy: false });
    \\    await poll();
    \\  } catch (err) {
    \\    setClass(now, "err");
    \\    setText(now, String(err.message || err));
    \\  } finally {
    \\    mine = false;
    \\    go.disabled = false;
    \\    tickFast(false);
    \\  }
    \\}
    \\go.addEventListener("click", generate);
    \\poll();
    \\tickFast(true);
    \\</script>
    \\</body>
    \\</html>
;
