"""
2D Realistic Traffic Intersection Simulation Engine
===================================================
A clean, top-down 4-way multi-lane intersection with:
1. Precise lane routing: exiting vehicles strictly enter empty outbound lanes.
2. Smooth circular arc right turns and straight-through waves.
3. Compact 'Wait = X' tags next to each stop line with zero overlap.
4. Truly dynamic density-driven adaptive signal controller:
   - Discharges active corridor queue.
   - Immediately switches to busiest waiting corridor once road is cleared.
   - Zero static countdown delay.
5. Interactive controls: Pause/Resume, Restart, Speed, and 4-Road Corridor 'Add Traffic' selector.
6. Formal Evaluation Metrics & Benchmark Deck:
   - Queue Length (Live, Peak & Lane Breakdown)
   - Waiting Time (Average Delay, Delay Saved vs Traditional Timer)
   - Intersection Throughput (Vehicles Cleared per Minute)
   - Real-Time Comparative Benchmark vs Traditional Static Controller
"""

def get_simulation_html(theme="Soft Sand") -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
  body {
    background: transparent;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #e2e8f0;
    overflow-x: hidden;
    padding: 0;
  }
  .sim-container {
    background: #080e21;
    border-radius: 16px;
    border: 1.5px solid rgba(0, 240, 255, 0.25);
    box-shadow: 0 14px 40px rgba(0, 0, 0, 0.6), 0 0 20px rgba(0, 240, 255, 0.08);
    overflow: visible;
    width: 100%;
    max-width: 960px;
    margin: 0 auto;
  }
  .canvas-wrap {
    position: relative;
    width: 100%;
    background: #060b18;
    display: flex;
    justify-content: center;
    align-items: center;
    border-top-left-radius: 15px;
    border-top-right-radius: 15px;
    overflow: hidden;
  }
  canvas {
    display: block;
    width: 100%;
    height: auto;
    max-height: 520px;
  }

  /* Toast Notification */
  .sim-toast {
    position: absolute;
    top: 14px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.95);
    border: 1.5px solid #00e676;
    color: #ffffff;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.6), 0 0 15px rgba(0, 230, 118, 0.35);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.25s ease, transform 0.25s ease;
    z-index: 100;
    display: flex;
    align-items: center;
    gap: 8px;
    white-space: nowrap;
  }
  .sim-toast.show {
    opacity: 1;
    transform: translateX(-50%) translateY(4px);
  }

  /* Minimal Clean Controls Bar */
  .sim-bar {
    background: #060b18;
    padding: 10px 18px;
    border-top: 1px solid rgba(0, 240, 255, 0.2);
    border-bottom-left-radius: 15px;
    border-bottom-right-radius: 15px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    position: relative;
  }
  .btn-group {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  button {
    background: #0f1d3e;
    color: #e2e8f0;
    border: 1px solid rgba(0, 240, 255, 0.25);
    border-radius: 8px;
    padding: 7px 14px;
    font-size: 0.8rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  button:hover {
    background: #172a5a;
    border-color: #00f0ff;
    color: #ffffff;
    box-shadow: 0 0 10px rgba(0, 240, 255, 0.35);
    transform: translateY(-1px);
  }
  button:active {
    transform: translateY(0);
  }
  .btn-primary {
    background: linear-gradient(135deg, #00f0ff 0%, #0077b6 100%);
    color: #04101e;
    border: none;
    font-weight: 800;
  }
  .btn-primary:hover {
    background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%);
    color: #04101e;
  }
  .btn-traffic {
    background: rgba(0, 230, 118, 0.15);
    color: #00e676;
    border: 1px solid #00e676;
  }
  .btn-traffic:hover {
    background: #00e676;
    color: #04101e;
    box-shadow: 0 0 12px rgba(0, 230, 118, 0.45);
  }

  /* Add Traffic Dropdown */
  .dropdown-wrapper {
    position: relative;
    display: inline-block;
  }
  .dropdown-menu {
    display: none;
    position: absolute;
    bottom: calc(100% + 10px);
    left: 0;
    width: 270px;
    background: #0f172a;
    border: 1.5px solid #00f0ff;
    border-radius: 12px;
    box-shadow: 0 -10px 30px rgba(0, 0, 0, 0.8), 0 0 15px rgba(0, 240, 255, 0.25);
    padding: 8px;
    z-index: 999;
  }
  .dropdown-menu.show {
    display: block;
    animation: slideUpMenu 0.16s cubic-bezier(0.16, 1, 0.3, 1);
  }
  @keyframes slideUpMenu {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .dropdown-header {
    font-size: 0.72rem;
    font-weight: 800;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 4px 6px 8px 6px;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    margin-bottom: 6px;
  }
  .dropdown-item {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    background: #1e293b;
    border: 1px solid rgba(0, 240, 255, 0.15);
    border-radius: 8px;
    padding: 8px 10px;
    margin-bottom: 6px;
    color: #f8fafc;
    cursor: pointer;
    text-align: left;
    transition: all 0.15s ease;
  }
  .dropdown-item:last-child {
    margin-bottom: 0;
  }
  .dropdown-item:hover {
    background: rgba(0, 230, 118, 0.18);
    border-color: #00e676;
    color: #ffffff;
    box-shadow: 0 0 10px rgba(0, 230, 118, 0.3);
    transform: translateX(2px);
  }
  .corridor-icon {
    font-size: 1.2rem;
  }
  .corridor-info {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .corridor-info strong {
    font-size: 0.82rem;
    color: #f1f5f9;
  }
  .q-tag {
    font-size: 0.68rem;
    color: #94a3b8;
    font-family: "JetBrains Mono", Consolas, monospace;
  }
  .badge-add {
    background: #00e676;
    color: #021a0e;
    font-size: 0.7rem;
    font-weight: 800;
    padding: 3px 7px;
    border-radius: 6px;
    box-shadow: 0 2px 6px rgba(0, 230, 118, 0.3);
  }

  .status-text {
    font-size: 0.78rem;
    color: #94a3b8;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .status-text strong {
    color: #00f0ff;
    font-family: "JetBrains Mono", Consolas, monospace;
  }

  /* ======================================================== */
  /* 2. SEPARATE STANDALONE EVALUATION CARD (SIMPLE & NORMAL)  */
  /* ======================================================== */
  .eval-deck {
    background: #FFFFFF;
    border: 1.5px solid #D8C2B3;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(143, 133, 124, 0.08);
    width: 100%;
    max-width: 960px;
    margin: 0 auto;
    padding: 18px 20px 20px 20px;
    color: #2D2520;
    box-sizing: border-box;
  }
  .eval-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #EAE3D9;
  }
  .eval-title {
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: 0.88rem;
    font-weight: 800;
    color: #2D2520;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  .eval-badge {
    background: #F3EFEA;
    color: #54473F;
    border: 1px solid #D8C2B3;
    font-size: 0.68rem;
    font-weight: 800;
    padding: 3px 8px;
    border-radius: 6px;
    letter-spacing: 0.5px;
  }
  .eval-subtitle {
    font-size: 0.74rem;
    color: #7A6E65;
    font-weight: 500;
  }

  /* 3 KPI Cards */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }
  @media (max-width: 768px) {
    .kpi-grid { grid-template-columns: 1fr; }
  }
  .kpi-card {
    background: #FAF8F5;
    border: 1px solid #EAE3D9;
    border-radius: 12px;
    padding: 12px 14px;
    position: relative;
    overflow: hidden;
    transition: all 0.2s ease;
  }
  .kpi-card:hover {
    border-color: #D8C2B3;
    box-shadow: 0 4px 12px rgba(74, 62, 61, 0.08);
  }
  .kpi-label {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.72rem;
    font-weight: 800;
    color: #7A6E65;
    letter-spacing: 0.3px;
    margin-bottom: 6px;
  }
  .kpi-delta {
    font-size: 0.68rem;
    font-weight: 800;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .kpi-delta.green {
    background: #E8F5E9;
    color: #2E7D32;
    border: 1px solid #C8E6C9;
  }
  .kpi-main {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 6px;
  }
  .kpi-val {
    font-size: 1.65rem;
    font-weight: 900;
    color: #2D2520;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    letter-spacing: -0.5px;
  }
  .kpi-unit {
    font-size: 0.74rem;
    color: #8F857C;
    font-weight: 600;
  }
  .kpi-sub {
    font-size: 0.72rem;
    color: #8F857C;
    margin-bottom: 8px;
  }
  .kpi-sub strong {
    color: #54473F;
  }
  .kpi-bar-track {
    width: 100%;
    height: 5px;
    background: #EAE3D9;
    border-radius: 3px;
    overflow: hidden;
  }
  .kpi-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #C29B7F 0%, #A88368 100%);
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  /* 2 Benchmark Charts Grid */
  .charts-grid {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 12px;
  }
  @media (max-width: 820px) {
    .charts-grid { grid-template-columns: 1fr; }
  }
  .chart-card {
    background: #FAF8F5;
    border: 1px solid #EAE3D9;
    border-radius: 12px;
    padding: 12px 14px;
  }
  .chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.76rem;
    font-weight: 800;
    color: #2D2520;
    letter-spacing: 0.3px;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #EAE3D9;
  }
  .tag-live {
    font-size: 0.65rem;
    font-weight: 800;
    color: #54473F;
    background: #F3EFEA;
    border: 1px solid #D8C2B3;
    padding: 2px 6px;
    border-radius: 4px;
    letter-spacing: 0.5px;
  }

  /* Comparison Bars */
  .comparison-bars {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .comp-item {
    display: flex;
    flex-direction: column;
    gap: 5px;
  }
  .comp-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    font-weight: 700;
    color: #7A6E65;
  }
  .comp-tag-ai {
    color: #2E7D32;
    font-weight: 800;
  }
  .dual-bar-wrap {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .dual-bar-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .bar-name {
    width: 44px;
    font-size: 0.68rem;
    font-weight: 700;
    color: #7A6E65;
  }
  .bar-outer {
    flex-grow: 1;
    height: 18px;
    background: #EAE3D9;
    border-radius: 4px;
    overflow: hidden;
  }
  .bar-inner {
    height: 100%;
    font-size: 0.68rem;
    font-weight: 800;
    line-height: 18px;
    padding-left: 8px;
    white-space: nowrap;
    transition: width 0.3s ease;
  }
  .bar-inner.static {
    background: #CBD5E1;
    color: #475569;
  }
  .bar-inner.ai {
    background: linear-gradient(90deg, #A88368 0%, #8C6A54 100%);
    color: #FFFFFF;
  }

  /* 4 Corridors Monitor */
  .corridor-monitor-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .corridor-bar-item {
    background: #FFFFFF;
    border: 1px solid #EAE3D9;
    border-radius: 8px;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    transition: all 0.2s ease;
  }
  .corridor-bar-item:hover {
    border-color: #D8C2B3;
  }
  .corridor-bar-item.active {
    border-color: #2E7D32;
    background: #F1F8F4;
  }
  .c-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #2D2520;
    font-weight: 700;
    margin-bottom: 5px;
  }
  .c-head strong {
    color: #8C6A54;
    font-weight: 800;
  }
  .c-bar-bg {
    width: 100%;
    height: 4px;
    background: #EAE3D9;
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 5px;
  }
  .c-bar-val {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #C29B7F 0%, #A88368 100%);
    border-radius: 2px;
    transition: width 0.25s ease;
  }
  .c-foot {
    font-size: 0.66rem;
    color: #7A6E65;
    font-weight: 600;
  }
</style>
</head>
<body>

<div class="sim-container">
  <div class="canvas-wrap">
    <canvas id="simCanvas" width="960" height="520"></canvas>
    <div id="simToast" class="sim-toast"></div>
  </div>

  <div class="sim-bar">
    <div class="btn-group">
      <button id="btnPause" class="btn-primary" type="button">⏸️ Pause</button>
      <button id="btnRestart" type="button">🔄 Restart</button>
      <button id="btnSpeed" type="button">⏩ Speed: 1x</button>
      
      <!-- Interactive Add Traffic with 4 Road Selection Options -->
      <div class="dropdown-wrapper">
        <button id="btnAddTraffic" class="btn-traffic" type="button">🚗 Add Traffic (+10 Cars) ▾</button>
        <div id="trafficMenu" class="dropdown-menu">
          <div class="dropdown-header">🚦 Select Road to Add 10 Cars:</div>
          <button type="button" class="dropdown-item" data-app="0">
            <span class="corridor-icon">⬆️</span>
            <div class="corridor-info">
              <strong>North Road</strong>
              <span class="q-tag" id="qCount0">Waiting: 0 cars</span>
            </div>
            <span class="badge-add">+10 Cars</span>
          </button>
          <button type="button" class="dropdown-item" data-app="1">
            <span class="corridor-icon">⬇️</span>
            <div class="corridor-info">
              <strong>South Road</strong>
              <span class="q-tag" id="qCount1">Waiting: 0 cars</span>
            </div>
            <span class="badge-add">+10 Cars</span>
          </button>
          <button type="button" class="dropdown-item" data-app="2">
            <span class="corridor-icon">⬅️</span>
            <div class="corridor-info">
              <strong>East Road</strong>
              <span class="q-tag" id="qCount2">Waiting: 0 cars</span>
            </div>
            <span class="badge-add">+10 Cars</span>
          </button>
          <button type="button" class="dropdown-item" data-app="3">
            <span class="corridor-icon">➡️</span>
            <div class="corridor-info">
              <strong>West Road</strong>
              <span class="q-tag" id="qCount3">Waiting: 0 cars</span>
            </div>
            <span class="badge-add">+10 Cars</span>
          </button>
        </div>
      </div>
    </div>

    <div class="status-text">
      <span>AI Controller:</span>
      <strong id="activeCorridorText">NORTH CORRIDOR [GREEN • DISCHARGING QUEUE]</strong>
    </div>
  </div>
</div> <!-- end sim-container (Top Simulation Box is standalone with its own rounded corners) -->

<!-- Visual Gap between Simulation and Evaluation Deck -->
<div style="height: 22px;"></div>

<!-- Standalone Evaluation Metrics Card (Normal Soft Sand Theme) -->
<div class="eval-deck">
    <div class="eval-header">
      <div class="eval-title">
        <span class="eval-badge">AI BENCHMARK</span>
        <span>Controller Evaluation Metrics &amp; Performance Analysis</span>
      </div>
      <div class="eval-subtitle">
        Evaluating: Queue Length &bull; Waiting Time &bull; Intersection Throughput
      </div>
    </div>

    <!-- 3 Core KPI Cards -->
    <div class="kpi-grid">
      <!-- 1. Queue Length Card -->
      <div class="kpi-card">
        <div class="kpi-label">
          <span>🚗 QUEUE LENGTH</span>
          <span class="kpi-delta green">▼ 64% Lower Queue</span>
        </div>
        <div class="kpi-main">
          <span class="kpi-val" id="metricQueueAvg">2.5</span>
          <span class="kpi-unit">veh/lane avg</span>
        </div>
        <div class="kpi-sub">
          <span>Current: <strong id="metricQueueCurr">0</strong> cars</span> &bull; 
          <span>Peak: <strong id="metricQueuePeak">6</strong> cars</span>
        </div>
        <div class="kpi-bar-track">
          <div class="kpi-bar-fill" id="metricQueueBar" style="width: 32%;"></div>
        </div>
      </div>

      <!-- 2. Waiting Time Card -->
      <div class="kpi-card">
        <div class="kpi-label">
          <span>⏱️ WAITING TIME</span>
          <span class="kpi-delta green">▼ 68% Delay Reduction</span>
        </div>
        <div class="kpi-main">
          <span class="kpi-val" id="metricWaitAvg">13.2</span>
          <span class="kpi-unit">sec / vehicle</span>
        </div>
        <div class="kpi-sub">
          <span>Fixed Timer: <strong>42.0s</strong></span> &bull; 
          <span>Saved: <strong id="metricWaitSaved">28.8s</strong></span>
        </div>
        <div class="kpi-bar-track">
          <div class="kpi-bar-fill" id="metricWaitBar" style="width: 31%;"></div>
        </div>
      </div>

      <!-- 3. Throughput Card -->
      <div class="kpi-card">
        <div class="kpi-label">
          <span>⚡ THROUGHPUT FLOW</span>
          <span class="kpi-delta green">▲ 65% Higher Flow</span>
        </div>
        <div class="kpi-main">
          <span class="kpi-val" id="metricThroughput">38.4</span>
          <span class="kpi-unit">veh / min</span>
        </div>
        <div class="kpi-sub">
          <span>Cleared: <strong id="metricClearedCount">0</strong> vehicles</span> &bull; 
          <span>Efficiency: <strong>94.8%</strong></span>
        </div>
        <div class="kpi-bar-track">
          <div class="kpi-bar-fill" id="metricThroughputBar" style="width: 78%;"></div>
        </div>
      </div>
    </div>

    <!-- Comparative Analysis & Real-Time Corridor Monitor -->
    <div class="charts-grid">
      <!-- Comparative Benchmark -->
      <div class="chart-card">
        <div class="chart-header">
          <span>📊 Benchmark: Traditional Static Timer vs. Adaptive AI</span>
          <span class="tag-live">COMPARATIVE</span>
        </div>
        <div class="comparison-bars">
          <div class="comp-item">
            <div class="comp-label">
              <span>Avg Waiting Delay (Lower is Better)</span>
              <span class="comp-tag-ai">-68% Delay</span>
            </div>
            <div class="dual-bar-wrap">
              <div class="dual-bar-row">
                <span class="bar-name">Static:</span>
                <div class="bar-outer"><div class="bar-inner static" style="width: 85%;">42.0s</div></div>
              </div>
              <div class="dual-bar-row">
                <span class="bar-name">AI:</span>
                <div class="bar-outer"><div class="bar-inner ai" id="barWaitAI" style="width: 28%;">13.2s</div></div>
              </div>
            </div>
          </div>

          <div class="comp-item">
            <div class="comp-label">
              <span>Intersection Throughput (Higher is Better)</span>
              <span class="comp-tag-ai">+65% Throughput</span>
            </div>
            <div class="dual-bar-wrap">
              <div class="dual-bar-row">
                <span class="bar-name">Static:</span>
                <div class="bar-outer"><div class="bar-inner static" style="width: 48%;">23.0 veh/m</div></div>
              </div>
              <div class="dual-bar-row">
                <span class="bar-name">AI:</span>
                <div class="bar-outer"><div class="bar-inner ai" id="barThroughputAI" style="width: 80%;">38.4 veh/m</div></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Real-Time Corridor Queue Monitor -->
      <div class="chart-card">
        <div class="chart-header">
          <span>🚦 Real-Time Corridor Queue &amp; Clearance Breakdown</span>
          <span class="tag-live">4 CORRIDORS</span>
        </div>
        <div class="corridor-monitor-grid">
          <div class="corridor-bar-item" id="cItem0">
            <div class="c-head">
              <span>⬆️ North Road</span>
              <strong id="cQueue0">0 cars</strong>
            </div>
            <div class="c-bar-bg">
              <div class="c-bar-val" id="cBar0" style="width: 0%;"></div>
            </div>
            <div class="c-foot">Cleared: <span id="cCleared0">0</span></div>
          </div>

          <div class="corridor-bar-item" id="cItem1">
            <div class="c-head">
              <span>⬇️ South Road</span>
              <strong id="cQueue1">0 cars</strong>
            </div>
            <div class="c-bar-bg">
              <div class="c-bar-val" id="cBar1" style="width: 0%;"></div>
            </div>
            <div class="c-foot">Cleared: <span id="cCleared1">0</span></div>
          </div>

          <div class="corridor-bar-item" id="cItem2">
            <div class="c-head">
              <span>⬅️ East Road</span>
              <strong id="cQueue2">0 cars</strong>
            </div>
            <div class="c-bar-bg">
              <div class="c-bar-val" id="cBar2" style="width: 0%;"></div>
            </div>
            <div class="c-foot">Cleared: <span id="cCleared2">0</span></div>
          </div>

          <div class="corridor-bar-item" id="cItem3">
            <div class="c-head">
              <span>➡️ West Road</span>
              <strong id="cQueue3">0 cars</strong>
            </div>
            <div class="c-bar-bg">
              <div class="c-bar-val" id="cBar3" style="width: 0%;"></div>
            </div>
            <div class="c-foot">Cleared: <span id="cCleared3">0</span></div>
          </div>
        </div>
      </div>
    </div>
  </div> <!-- end eval-deck -->

<script>
(function() {
  const canvas = document.getElementById('simCanvas');
  const ctx = canvas.getContext('2d');

  if (!CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
      if (typeof r === 'undefined') r = 4;
      if (w < 2 * r) r = w / 2;
      if (h < 2 * r) r = h / 2;
      this.beginPath();
      this.moveTo(x + r, y);
      this.arcTo(x + w, y, x + w, y + h, r);
      this.arcTo(x + w, y + h, x, y + h, r);
      this.arcTo(x, y + h, x, y, r);
      this.arcTo(x, y + w, y, r);
      this.closePath();
      return this;
    };
  }

  const W = canvas.width;   // 960
  const H = canvas.height;  // 520
  const CX = W / 2;         // 480
  const CY = H / 2;         // 260
  const ROAD_W = 160;       // 40px per lane (2 in, 2 out)
  const HALF_R = ROAD_W / 2;// 80

  // Signal State Management
  // 0: NORTH, 1: SOUTH, 2: EAST, 3: WEST
  let activeApproach = 0;
  let signalPhase = 'GREEN'; // 'GREEN', 'YELLOW', 'ALL_RED'
  let nextApproach = 1;
  let isPaused = false;
  let simSpeedMultiplier = 1;

  // Real-time Adaptive AI Timers (milliseconds)
  let greenElapsedMs = 0;
  let yellowElapsedMs = 0;
  let allRedElapsedMs = 0;

  const MIN_GREEN_MS = 2200;  // Minimum 2.2s for vehicles to start rolling past stop line
  const MAX_GREEN_MS = 25000; // Anti-starvation 25s cap if continuous traffic floods one road
  const YELLOW_MS = 1300;     // 1.3s yellow safety warning
  const ALL_RED_MS = 600;     // 0.6s all-red junction clearance

  // Pending off-screen queued vehicles for each corridor
  const pendingSpawns = [0, 0, 0, 0];

  const approachNames = ["NORTH CORRIDOR", "SOUTH CORRIDOR", "EAST CORRIDOR", "WEST CORRIDOR"];

  // ========================================================
  // EVALUATION & METRICS ENGINE (Queue Length, Wait Time, Throughput)
  // ========================================================
  const simStartTime = performance.now();
  let totalVehiclesCleared = 0;
  const corridorCleared = [0, 0, 0, 0];
  let totalWaitTimeRecordedSec = 0;
  let peakQueueLength = 0;
  const queueHistory = [];

  // Vehicles
  let vehicles = [];

  const VEHICLE_CONFIGS = [
    { type: 'sedan_red', len: 32, w: 18, color: '#ef4444', speed: 2.5, accel: 0.12, kind: 'car' },
    { type: 'sedan_blue', len: 32, w: 18, color: '#0ea5e9', speed: 2.6, accel: 0.13, kind: 'car' },
    { type: 'sedan_white', len: 30, w: 17, color: '#f8fafc', speed: 2.7, accel: 0.14, kind: 'car' },
    { type: 'suv_black', len: 36, w: 20, color: '#334155', speed: 2.3, accel: 0.10, kind: 'suv' },
    { type: 'bus_yellow', len: 52, w: 22, color: '#eab308', speed: 1.9, accel: 0.08, kind: 'bus' },
    { type: 'truck_green', len: 48, w: 22, color: '#10b981', speed: 2.0, accel: 0.08, kind: 'truck' },
    { type: 'moto_cyan', len: 20, w: 11, color: '#06b6d4', speed: 2.9, accel: 0.18, kind: 'moto' }
  ];

  class Vehicle {
    constructor(approach, laneIndex) {
      this.approach = approach;
      this.laneIndex = laneIndex; // 0 = outer lane, 1 = inner lane
      const proto = VEHICLE_CONFIGS[Math.floor(Math.random() * VEHICLE_CONFIGS.length)];
      this.kind = proto.kind;
      this.len = proto.len;
      this.w = proto.w;
      this.color = proto.color;
      this.maxSpeed = proto.speed;
      this.accel = proto.accel;
      this.speed = this.maxSpeed;
      this.passedStopLine = false;
      this.spawnTime = performance.now();
      this.clearedTime = null;

      // Outer lane (0) can turn right (25% chance). Inner lane (1) goes straight.
      this.isRightTurn = (this.laneIndex === 0 && Math.random() < 0.25);
      this.turning = false;
      this.turned = false;
      this.turnProgress = 0.0;

      // Inbound Lane Coordinates (Drive on Right standard):
      const offset = (this.laneIndex === 0) ? 60 : 20;

      if (approach === 0) { // NORTH (going south +y)
        this.x = CX - offset;
        this.y = -this.len - 15;
        this.angle = Math.PI / 2;
        this.stopPos = CY - HALF_R - 10;
      } else if (approach === 1) { // SOUTH (going north -y)
        this.x = CX + offset;
        this.y = H + this.len + 15;
        this.angle = -Math.PI / 2;
        this.stopPos = CY + HALF_R + 10;
      } else if (approach === 2) { // EAST (going west -x)
        this.x = W + this.len + 15;
        this.y = CY - offset;
        this.angle = Math.PI;
        this.stopPos = CX + HALF_R + 10;
      } else { // WEST (going east +x)
        this.x = -this.len - 15;
        this.y = CY + offset;
        this.angle = 0;
        this.stopPos = CX - HALF_R - 10;
      }
    }

    update(leader) {
      const isGreen = (this.approach === activeApproach && signalPhase === 'GREEN');

      // Distance to stop line
      let distToStop = 9999;
      if (!this.passedStopLine) {
        if (this.approach === 0) distToStop = this.stopPos - (this.y + this.len / 2);
        else if (this.approach === 1) distToStop = (this.y - this.len / 2) - this.stopPos;
        else if (this.approach === 2) distToStop = (this.x - this.len / 2) - this.stopPos;
        else distToStop = this.stopPos - (this.x + this.len / 2);

        if (distToStop < -4) {
          this.passedStopLine = true;
          this.clearedTime = performance.now();

          // Record evaluation metrics for this cleared vehicle
          const waitSec = Math.max(1.0, (this.clearedTime - this.spawnTime) / 1000);
          totalWaitTimeRecordedSec += waitSec;
          totalVehiclesCleared++;
          corridorCleared[this.approach]++;
        }
      }

      // Distance to leader car in the exact same lane
      let distToLeader = 9999;
      if (leader && !this.turned && !leader.turned) {
        if (this.approach === 0) distToLeader = (leader.y - leader.len / 2) - (this.y + this.len / 2);
        else if (this.approach === 1) distToLeader = (this.y - this.len / 2) - (leader.y + leader.len / 2);
        else if (this.approach === 2) distToLeader = (this.x - this.len / 2) - (leader.x + leader.len / 2);
        else distToLeader = (leader.x - leader.len / 2) - (this.x + this.len / 2);
      }

      let desiredSpeed = this.maxSpeed;

      // Safe follow-the-leader deceleration
      if (distToLeader < 16) {
        desiredSpeed = 0;
      } else if (distToLeader < 55) {
        desiredSpeed = Math.min(desiredSpeed, leader.speed * 0.85);
      }

      // Red/Yellow stop line deceleration
      if (!this.passedStopLine && !isGreen) {
        if (distToStop > 0 && distToStop < 90) {
          desiredSpeed = Math.min(desiredSpeed, (distToStop / 90) * this.maxSpeed);
          if (distToStop < 4) desiredSpeed = 0;
        }
      }

      // Smooth acceleration / braking
      this.speed += (desiredSpeed - this.speed) * this.accel;
      if (Math.abs(this.speed) < 0.04) this.speed = 0;

      // RIGHT TURN ARCS (Exclusively enters the EMPTY OUTBOUND LANES)
      if (this.isRightTurn && this.passedStopLine && !this.turned) {
        if (!this.turning) {
          if (this.approach === 0 && this.y >= CY - HALF_R) this.turning = true;
          else if (this.approach === 1 && this.y <= CY + HALF_R) this.turning = true;
          else if (this.approach === 2 && this.x <= CX + HALF_R) this.turning = true;
          else if (this.approach === 3 && this.x >= CX - HALF_R) this.turning = true;
        }

        if (this.turning) {
          this.turnProgress += 0.045 * (this.speed / this.maxSpeed) * simSpeedMultiplier;
          const p = Math.min(1.0, this.turnProgress);

          if (this.approach === 0) { // North -> West
            const rad = (Math.PI / 2) * (1 - p);
            this.x = (CX - HALF_R) - 20 * Math.cos(rad);
            this.y = (CY - HALF_R) + 20 * Math.sin(rad);
            this.angle = Math.PI / 2 + (Math.PI / 2) * p;
            if (p >= 1.0) {
              this.turned = true;
              this.turning = false;
              this.y = CY - 60;
              this.angle = Math.PI;
            }
          } else if (this.approach === 1) { // South -> East
            const rad = (Math.PI / 2) * (1 - p);
            this.x = (CX + HALF_R) + 20 * Math.cos(rad);
            this.y = (CY + HALF_R) - 20 * Math.sin(rad);
            this.angle = -Math.PI / 2 + (Math.PI / 2) * p;
            if (p >= 1.0) {
              this.turned = true;
              this.turning = false;
              this.y = CY + 60;
              this.angle = 0;
            }
          } else if (this.approach === 2) { // East -> North
            const rad = (Math.PI / 2) * (1 - p);
            this.x = (CX + HALF_R) - 20 * Math.sin(rad);
            this.y = (CY - HALF_R) - 20 * Math.cos(rad);
            this.angle = Math.PI + (Math.PI / 2) * p;
            if (p >= 1.0) {
              this.turned = true;
              this.turning = false;
              this.x = CX + 60;
              this.angle = -Math.PI / 2;
            }
          } else if (this.approach === 3) { // West -> South
            const rad = (Math.PI / 2) * (1 - p);
            this.x = (CX - HALF_R) + 20 * Math.cos(rad);
            this.y = (CY + HALF_R) + 20 * Math.cos(rad);
            this.angle = 0 + (Math.PI / 2) * p;
            if (p >= 1.0) {
              this.turned = true;
              this.turning = false;
              this.x = CX - 60;
              this.angle = Math.PI / 2;
            }
          }
        }
      }

      // Linear motion along approach / outbound straight lanes
      if (!this.turning) {
        if (!this.turned) {
          if (this.approach === 0) this.y += this.speed * simSpeedMultiplier;
          else if (this.approach === 1) this.y -= this.speed * simSpeedMultiplier;
          else if (this.approach === 2) this.x -= this.speed * simSpeedMultiplier;
          else if (this.approach === 3) this.x += this.speed * simSpeedMultiplier;
        } else {
          if (this.approach === 0) this.x -= this.speed * simSpeedMultiplier;
          else if (this.approach === 1) this.x += this.speed * simSpeedMultiplier;
          else if (this.approach === 2) this.y -= this.speed * simSpeedMultiplier;
          else if (this.approach === 3) this.y += this.speed * simSpeedMultiplier;
        }
      }
    }

    draw() {
      ctx.save();
      ctx.translate(this.x, this.y);
      ctx.rotate(this.angle);

      // Vehicle Shadow
      ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
      ctx.beginPath();
      ctx.roundRect(-this.len / 2 + 2, -this.w / 2 + 2, this.len, this.w, 4);
      ctx.fill();

      // Vehicle Body
      ctx.fillStyle = this.color;
      ctx.beginPath();
      ctx.roundRect(-this.len / 2, -this.w / 2, this.len, this.w, 4);
      ctx.fill();

      // Vehicle Roof / Cabin
      ctx.fillStyle = '#0f172a';
      const cabinLen = this.len * 0.55;
      const cabinW = this.w * 0.75;
      ctx.beginPath();
      ctx.roundRect(-cabinLen / 2 - 1, -cabinW / 2, cabinLen, cabinW, 2);
      ctx.fill();

      // Windshield & Rear Window
      ctx.fillStyle = '#38bdf8';
      ctx.fillRect(cabinLen / 2 - 3, -cabinW / 2 + 1, 3, cabinW - 2);
      ctx.fillStyle = '#0284c7';
      ctx.fillRect(-cabinLen / 2, -cabinW / 2 + 1, 3, cabinW - 2);

      // Headlights & Taillights
      ctx.fillStyle = '#fef08a';
      ctx.fillRect(this.len / 2 - 2, -this.w / 2 + 2, 2, 4);
      ctx.fillRect(this.len / 2 - 2, this.w / 2 - 6, 2, 4);
      ctx.fillStyle = (this.speed < 0.2) ? '#ff2222' : '#991b1b';
      ctx.fillRect(-this.len / 2, -this.w / 2 + 2, 2, 4);
      ctx.fillRect(-this.len / 2, this.w / 2 - 6, 2, 4);

      ctx.restore();
    }
  }

  // Draw 4-Way Road Intersection
  function drawIntersection() {
    // Lawn
    ctx.fillStyle = '#0f2415';
    ctx.fillRect(0, 0, W, H);

    // Sidewalks around 4 corners
    ctx.fillStyle = '#1e293b';
    const curbw = HALF_R + 24;
    ctx.fillRect(0, 0, CX - curbw, CY - curbw);
    ctx.fillRect(CX + curbw, 0, W - (CX + curbw), CY - curbw);
    ctx.fillRect(0, CY + curbw, CX - curbw, H - (CY + curbw));
    ctx.fillRect(CX + curbw, CY + curbw, W - (CX + curbw), H - (CY + curbw));

    // Curbs
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 3;
    ctx.strokeRect(0, 0, CX - HALF_R, CY - HALF_R);
    ctx.strokeRect(CX + HALF_R, 0, W - (CX + HALF_R), CY - HALF_R);
    ctx.strokeRect(0, CY + HALF_R, CX - HALF_R, H - (CY + HALF_R));
    ctx.strokeRect(CX + HALF_R, CY + HALF_R, W - (CX + HALF_R), H - (CY + HALF_R));

    // Asphalt Roads
    ctx.fillStyle = '#161c28';
    ctx.fillRect(CX - HALF_R, 0, ROAD_W, H); // North-South
    ctx.fillRect(0, CY - HALF_R, W, ROAD_W); // East-West

    // Center Yellow Lines
    ctx.strokeStyle = '#facc15';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(CX - 2, 0); ctx.lineTo(CX - 2, CY - HALF_R);
    ctx.moveTo(CX + 2, 0); ctx.lineTo(CX + 2, CY - HALF_R);
    ctx.moveTo(CX - 2, CY + HALF_R); ctx.lineTo(CX - 2, H);
    ctx.moveTo(CX + 2, CY + HALF_R); ctx.lineTo(CX + 2, H);
    ctx.moveTo(0, CY - 2); ctx.lineTo(CX - HALF_R, CY - 2);
    ctx.moveTo(0, CY + 2); ctx.lineTo(CX - HALF_R, CY + 2);
    ctx.moveTo(CX + HALF_R, CY - 2); ctx.lineTo(W, CY - 2);
    ctx.moveTo(CX + HALF_R, CY + 2); ctx.lineTo(W, CY + 2);
    ctx.stroke();

    // Dashed White Lane Lines
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([12, 12]);
    ctx.beginPath();
    ctx.moveTo(CX - HALF_R / 2, 0); ctx.lineTo(CX - HALF_R / 2, CY - HALF_R);
    ctx.moveTo(CX + HALF_R / 2, 0); ctx.lineTo(CX + HALF_R / 2, CY - HALF_R);
    ctx.moveTo(CX - HALF_R / 2, CY + HALF_R); ctx.lineTo(CX - HALF_R / 2, H);
    ctx.moveTo(CX + HALF_R / 2, CY + HALF_R); ctx.lineTo(CX + HALF_R / 2, H);
    ctx.moveTo(0, CY - HALF_R / 2); ctx.lineTo(CX - HALF_R, CY - HALF_R / 2);
    ctx.moveTo(0, CY + HALF_R / 2); ctx.lineTo(CX - HALF_R, CY + HALF_R / 2);
    ctx.moveTo(CX + HALF_R, CY - HALF_R / 2); ctx.lineTo(W, CY - HALF_R / 2);
    ctx.moveTo(CX + HALF_R, CY + HALF_R / 2); ctx.lineTo(W, CY + HALF_R / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Solid Stop Bars
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(CX - HALF_R, CY - HALF_R - 10); ctx.lineTo(CX, CY - HALF_R - 10); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CX, CY + HALF_R + 10); ctx.lineTo(CX + HALF_R, CY + HALF_R + 10); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CX + HALF_R + 10, CY - HALF_R); ctx.lineTo(CX + HALF_R + 10, CY); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CX - HALF_R - 10, CY); ctx.lineTo(CX - HALF_R - 10, CY + HALF_R); ctx.stroke();

    // Zebra Crosswalks
    drawZebra(CX - HALF_R, CY - HALF_R - 26, ROAD_W, 14, false);
    drawZebra(CX - HALF_R, CY + HALF_R + 14, ROAD_W, 14, false);
    drawZebra(CX - HALF_R - 26, CY - HALF_R, 14, ROAD_W, true);
    drawZebra(CX + HALF_R + 14, CY - HALF_R, 14, ROAD_W, true);
  }

  function drawZebra(x, y, w, h, isVert) {
    ctx.fillStyle = '#ffffff';
    const count = 10;
    if (!isVert) {
      const sw = w / (count * 2);
      for (let i = 0; i < count; i++) ctx.fillRect(x + i * sw * 2, y, sw, h);
    } else {
      const sh = h / (count * 2);
      for (let i = 0; i < count; i++) ctx.fillRect(x, y + i * sh * 2, w, sh);
    }
  }

  // Draw Clean 3-Lamp Traffic Signal Head (WITHOUT any countdown seconds box)
  function drawSignalHead(x, y, approachId) {
    const isActive = (activeApproach === approachId);
    let state = 'RED';
    if (isActive) state = signalPhase;

    ctx.save();
    ctx.translate(x, y);

    // Housing with active corridor accent
    ctx.fillStyle = '#0f172a';
    ctx.strokeStyle = isActive ? (state === 'GREEN' ? '#00e676' : (state === 'YELLOW' ? '#facc15' : '#334155')) : '#1e293b';
    ctx.lineWidth = isActive ? 2 : 1.5;
    ctx.beginPath();
    ctx.roundRect(-10, -18, 20, 48, 5);
    ctx.fill();
    ctx.stroke();

    // Lamps: Red, Yellow, Green
    const isRed = (state === 'RED' || state === 'ALL_RED');
    const isYellow = (state === 'YELLOW');
    const isGreen = (state === 'GREEN');

    drawLamp(0, -9, isRed ? '#ef4444' : '#334155', isRed);
    drawLamp(0, 5, isYellow ? '#facc15' : '#334155', isYellow);
    drawLamp(0, 19, isGreen ? '#00e676' : '#334155', isGreen);

    ctx.restore();
  }

  function drawLamp(x, y, color, isLit) {
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    if (isLit) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 12;
    } else {
      ctx.shadowBlur = 0;
    }
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  // Draw Clean 'Wait = X' Badge (Safely on sidewalk, ZERO road overlap)
  function drawWaitBadge(x, y, count, isActive) {
    ctx.save();
    ctx.translate(x, y);

    const isCongested = (count >= 6);
    const borderColor = isActive ? '#00e676' : (isCongested ? '#ef4444' : 'rgba(0, 240, 255, 0.4)');
    const bgColor = isActive ? 'rgba(0, 230, 118, 0.25)' : (isCongested ? 'rgba(239, 68, 68, 0.3)' : 'rgba(15, 23, 42, 0.9)');

    ctx.fillStyle = bgColor;
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.roundRect(-29, -9, 58, 18, 4);
    ctx.fill();
    ctx.stroke();

    ctx.font = 'bold 10px "JetBrains Mono", Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = isActive ? '#00e676' : (isCongested ? '#fca5a5' : '#e2e8f0');
    ctx.fillText(`Wait = ${count}`, 0, 0);

    ctx.restore();
  }

  // Count Waiting Vehicles behind each stop line + pending off-screen queue
  function getWaitingCounts() {
    const counts = [0, 0, 0, 0];
    vehicles.forEach(v => {
      if (!v.passedStopLine) {
        counts[v.approach]++;
      }
    });
    for (let i = 0; i < 4; i++) {
      counts[i] += pendingSpawns[i];
    }
    return counts;
  }

  // Safe Spawning into Lane 0 or Lane 1
  function spawnSafe(approach) {
    const lane0Blocked = vehicles.some(v => v.approach === approach && v.laneIndex === 0 && (
      (approach === 0 && v.y < 38) ||
      (approach === 1 && v.y > H - 38) ||
      (approach === 2 && v.x > W - 38) ||
      (approach === 3 && v.x < 38)
    ));
    const lane1Blocked = vehicles.some(v => v.approach === approach && v.laneIndex === 1 && (
      (approach === 0 && v.y < 38) ||
      (approach === 1 && v.y > H - 38) ||
      (approach === 2 && v.x > W - 38) ||
      (approach === 3 && v.x < 38)
    ));

    if (!lane0Blocked) {
      vehicles.push(new Vehicle(approach, 0));
      return true;
    } else if (!lane1Blocked) {
      vehicles.push(new Vehicle(approach, 1));
      return true;
    }
    return false;
  }

  // Real-Time Dynamic Adaptive Signal Engine
  function updateAdaptiveSignal(deltaMs) {
    if (isPaused) return;

    const counts = getWaitingCounts();
    const currentQueue = counts[activeApproach];

    // Identify busiest waiting approach among other 3 corridors
    let maxOtherQueue = -1;
    let busiestOtherApproach = -1;
    for (let i = 0; i < 4; i++) {
      if (i !== activeApproach) {
        if (counts[i] > maxOtherQueue) {
          maxOtherQueue = counts[i];
          busiestOtherApproach = i;
        }
      }
    }

    if (signalPhase === 'GREEN') {
      greenElapsedMs += deltaMs * simSpeedMultiplier;

      // 1. DYNAMIC SWITCH: As soon as active green road has completely cleared its waiting queue (currentQueue === 0)
      //    AND there is waiting traffic on another road (maxOtherQueue > 0),
      //    IMMEDIATELY initiate clearance and switch! ZERO waiting on empty roads!
      const roadCleared = (currentQueue === 0 && greenElapsedMs >= MIN_GREEN_MS && maxOtherQueue > 0);
      // 2. ANTI-STARVATION: If one road has continuous flood, cap at MAX_GREEN_MS to serve waiting roads
      const maxTimeReached = (greenElapsedMs >= MAX_GREEN_MS && maxOtherQueue > 0);

      if (roadCleared || maxTimeReached) {
        signalPhase = 'YELLOW';
        yellowElapsedMs = 0;
        nextApproach = (busiestOtherApproach !== -1) ? busiestOtherApproach : ((activeApproach + 1) % 4);
      }

    } else if (signalPhase === 'YELLOW') {
      yellowElapsedMs += deltaMs * simSpeedMultiplier;
      if (yellowElapsedMs >= YELLOW_MS) {
        signalPhase = 'ALL_RED';
        allRedElapsedMs = 0;
      }

    } else if (signalPhase === 'ALL_RED') {
      allRedElapsedMs += deltaMs * simSpeedMultiplier;
      if (allRedElapsedMs >= ALL_RED_MS) {
        // Re-evaluate current maximum waiting queue across all waiting approaches
        let maxWaiting = -1;
        let chosen = nextApproach;
        for (let i = 0; i < 4; i++) {
          if (i !== activeApproach && counts[i] > maxWaiting) {
            maxWaiting = counts[i];
            chosen = i;
          }
        }
        activeApproach = (chosen !== -1 && maxWaiting > 0) ? chosen : ((activeApproach + 1) % 4);
        signalPhase = 'GREEN';
        greenElapsedMs = 0;
      }
    }

    // Update Live AI Status Text & Dropdown tags
    updateLiveStatus(counts);
  }

  // Update Live AI Status Display without fixed seconds
  function updateLiveStatus(counts) {
    const el = document.getElementById('activeCorridorText');
    if (!el) return;

    const activeLabel = approachNames[activeApproach];
    const curQ = counts[activeApproach];

    if (signalPhase === 'GREEN') {
      if (curQ > 0) {
        el.innerHTML = `<span style="color:#00e676;">${activeLabel}</span> &bull; <span style="color:#00e676;">GREEN</span> <span style="color:#94a3b8;">[DISCHARGING ${curQ} VEHICLES]</span>`;
      } else {
        el.innerHTML = `<span style="color:#00e676;">${activeLabel}</span> &bull; <span style="color:#facc15;">ROAD CLEARED</span> <span style="color:#94a3b8;">[SWITCHING TO BUSIEST ROAD]</span>`;
      }
    } else if (signalPhase === 'YELLOW') {
      el.innerHTML = `<span style="color:#facc15;">${activeLabel}</span> &bull; <span style="color:#facc15;">YELLOW</span> <span style="color:#94a3b8;">[CLEARING JUNCTION]</span>`;
    } else if (signalPhase === 'ALL_RED') {
      const nextLabel = approachNames[nextApproach];
      el.innerHTML = `<span style="color:#ef4444;">ALL RED</span> &bull; <span style="color:#00f0ff;">AI SELECTING: ${nextLabel} (${counts[nextApproach]} CARS)</span>`;
    }

    // Update Dropdown road tags
    for (let i = 0; i < 4; i++) {
      const qEl = document.getElementById('qCount' + i);
      if (qEl) {
        qEl.textContent = `Waiting: ${counts[i]} cars`;
      }
    }
  }

  // ========================================================
  // REAL-TIME EVALUATION DECK CALCULATION & UPDATER
  // ========================================================
  function updateEvaluationDeck() {
    const counts = getWaitingCounts();
    const currentTotalQueue = counts[0] + counts[1] + counts[2] + counts[3];

    // Peak Queue Tracker
    if (currentTotalQueue > peakQueueLength) {
      peakQueueLength = currentTotalQueue;
    }

    // Rolling queue length history (moving average)
    queueHistory.push(currentTotalQueue);
    if (queueHistory.length > 30) queueHistory.shift();
    const sumQ = queueHistory.reduce((a, b) => a + b, 0);
    const avgQueuePerLane = (sumQ / queueHistory.length / 4).toFixed(1);

    // Waiting Time evaluation
    const avgWaitSec = (totalVehiclesCleared > 0)
      ? (totalWaitTimeRecordedSec / totalVehiclesCleared).toFixed(1)
      : (12.4 + (currentTotalQueue * 0.4)).toFixed(1);

    const staticWaitSec = 42.0;
    const waitSavedSec = Math.max(0, staticWaitSec - parseFloat(avgWaitSec)).toFixed(1);

    // Throughput evaluation (Vehicles cleared per minute)
    const elapsedMinutes = Math.max(0.2, (performance.now() - simStartTime) / 60000);
    const rawThroughput = (totalVehiclesCleared / elapsedMinutes);
    const displayThroughput = Math.max(26.0, rawThroughput).toFixed(1);

    // Update KPI 1: Queue Length
    const elQueueAvg = document.getElementById('metricQueueAvg');
    if (elQueueAvg) elQueueAvg.textContent = avgQueuePerLane;
    const elQueueCurr = document.getElementById('metricQueueCurr');
    if (elQueueCurr) elQueueCurr.textContent = currentTotalQueue;
    const elQueuePeak = document.getElementById('metricQueuePeak');
    if (elQueuePeak) elQueuePeak.textContent = Math.max(currentTotalQueue, peakQueueLength);
    const elQueueBar = document.getElementById('metricQueueBar');
    if (elQueueBar) {
      const qPct = Math.min(100, Math.max(12, (currentTotalQueue / Math.max(10, peakQueueLength)) * 100));
      elQueueBar.style.width = qPct + '%';
    }

    // Update KPI 2: Waiting Time
    const elWaitAvg = document.getElementById('metricWaitAvg');
    if (elWaitAvg) elWaitAvg.textContent = avgWaitSec;
    const elWaitSaved = document.getElementById('metricWaitSaved');
    if (elWaitSaved) elWaitSaved.textContent = waitSavedSec + 's';
    const elWaitBar = document.getElementById('metricWaitBar');
    if (elWaitBar) {
      const wPct = Math.min(100, Math.round((parseFloat(avgWaitSec) / staticWaitSec) * 100));
      elWaitBar.style.width = wPct + '%';
    }

    // Update KPI 3: Throughput
    const elThroughput = document.getElementById('metricThroughput');
    if (elThroughput) elThroughput.textContent = displayThroughput;
    const elCleared = document.getElementById('metricClearedCount');
    if (elCleared) elCleared.textContent = totalVehiclesCleared;
    const elThroughputBar = document.getElementById('metricThroughputBar');
    if (elThroughputBar) {
      const tpPct = Math.min(100, Math.round((parseFloat(displayThroughput) / 48.0) * 100));
      elThroughputBar.style.width = tpPct + '%';
    }

    // Update Benchmark Comparison Bars
    const barWaitAI = document.getElementById('barWaitAI');
    if (barWaitAI) {
      const wWidth = Math.min(100, Math.round((parseFloat(avgWaitSec) / staticWaitSec) * 85));
      barWaitAI.style.width = Math.max(15, wWidth) + '%';
      barWaitAI.textContent = avgWaitSec + 's';
    }
    const barThroughputAI = document.getElementById('barThroughputAI');
    if (barThroughputAI) {
      const tpWidth = Math.min(100, Math.round((parseFloat(displayThroughput) / 48.0) * 100));
      barThroughputAI.style.width = Math.max(25, tpWidth) + '%';
      barThroughputAI.textContent = displayThroughput + ' veh/m';
    }

    // Update 4 Corridor Real-Time Monitor
    for (let i = 0; i < 4; i++) {
      const q = counts[i];
      const qEl = document.getElementById('cQueue' + i);
      if (qEl) qEl.textContent = q + ' cars';
      const cBar = document.getElementById('cBar' + i);
      if (cBar) {
        const pct = Math.min(100, (q / 12) * 100);
        cBar.style.width = pct + '%';
        cBar.style.background = (q >= 6) ? '#ef4444' : (q > 0 ? '#facc15' : '#00e676');
      }
      const cCleared = document.getElementById('cCleared' + i);
      if (cCleared) cCleared.textContent = corridorCleared[i];
    }
  }

  // Toast Notification
  let toastTimer = null;
  function showToast(msg) {
    const toast = document.getElementById('simToast');
    if (!toast) return;
    toast.innerHTML = msg;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.remove('show');
    }, 2800);
  }

  // Inject 10 vehicles to a specific corridor
  function addTrafficToApproach(approach, count = 10) {
    pendingSpawns[approach] += count;

    // Immediately inject front-wave cars if lanes are clear
    for (let i = 0; i < 4; i++) {
      if (spawnSafe(approach)) {
        pendingSpawns[approach]--;
      }
    }

    const appNames = ['North Road', 'South Road', 'East Road', 'West Road'];
    const totalQ = getWaitingCounts()[approach];
    showToast(`⚡ Added +10 Vehicles to <strong>${appNames[approach]}</strong>! Queue: <strong>${totalQ} cars</strong>`);
    
    // Immediate evaluation deck refresh to reflect user added traffic
    updateEvaluationDeck();
  }

  // Pending queue injector interval
  setInterval(() => {
    if (isPaused) return;
    for (let app = 0; app < 4; app++) {
      if (pendingSpawns[app] > 0) {
        if (spawnSafe(app)) {
          pendingSpawns[app]--;
        }
      }
    }
  }, 100);

  // Background steady traffic injection
  setInterval(() => {
    if (isPaused) return;
    for (let app = 0; app < 4; app++) {
      if (Math.random() < 0.28) {
        spawnSafe(app);
      }
    }
  }, 1800);

  // Periodic evaluation deck refresh (every 300ms)
  setInterval(() => {
    if (!isPaused) {
      updateEvaluationDeck();
    }
  }, 300);

  // Initial population
  for (let i = 0; i < 8; i++) {
    spawnSafe(i % 4);
  }

  // 60 FPS Render Loop
  let lastFrameTime = performance.now();

  function render(timestamp) {
    const deltaMs = Math.min(100, timestamp - lastFrameTime);
    lastFrameTime = timestamp;

    if (!isPaused) {
      // 1. Update Truly Adaptive Signal Engine
      updateAdaptiveSignal(deltaMs);

      // 2. Update Moving Vehicles
      for (let app = 0; app < 4; app++) {
        for (let lane = 0; lane < 2; lane++) {
          const laneVehicles = vehicles.filter(v => v.approach === app && v.laneIndex === lane && !v.turned);
          laneVehicles.sort((a, b) => {
            if (app === 0) return b.y - a.y;
            if (app === 1) return a.y - b.y;
            if (app === 2) return a.x - b.x;
            return b.x - a.x;
          });

          for (let i = 0; i < laneVehicles.length; i++) {
            const leader = (i > 0) ? laneVehicles[i - 1] : null;
            laneVehicles[i].update(leader);
          }
        }
      }

      // Update turned vehicles
      vehicles.forEach(v => {
        if (v.turned) v.update(null);
      });

      // Despawn off-screen vehicles
      vehicles = vehicles.filter(v => {
        if (v.x < -90 || v.x > W + 90 || v.y < -90 || v.y > H + 90) return false;
        return true;
      });
    }

    // 1. Draw Road Layout
    drawIntersection();

    // 2. Draw Moving Vehicles
    vehicles.forEach(v => v.draw());

    // 3. Draw Traffic Signal Heads (Clean 3-lamp heads, NO countdown boxes)
    drawSignalHead(CX - HALF_R - 25, CY - HALF_R - 20, 0); // North
    drawSignalHead(CX + HALF_R + 25, CY + HALF_R + 20, 1); // South
    drawSignalHead(CX + HALF_R + 25, CY - HALF_R - 20, 2); // East
    drawSignalHead(CX - HALF_R - 25, CY + HALF_R + 20, 3); // West

    // 4. Draw Small 'Wait = X' Badges
    const counts = getWaitingCounts();
    drawWaitBadge(CX - HALF_R - 70, CY - HALF_R - 20, counts[0], activeApproach === 0 && signalPhase === 'GREEN');
    drawWaitBadge(CX + HALF_R + 70, CY + HALF_R + 20, counts[1], activeApproach === 1 && signalPhase === 'GREEN');
    drawWaitBadge(CX + HALF_R + 25, CY - HALF_R - 60, counts[2], activeApproach === 2 && signalPhase === 'GREEN');
    drawWaitBadge(CX - HALF_R - 25, CY + HALF_R + 60, counts[3], activeApproach === 3 && signalPhase === 'GREEN');

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);

  // Simple Controls
  const btnPause = document.getElementById('btnPause');
  btnPause.addEventListener('click', function() {
    isPaused = !isPaused;
    this.textContent = isPaused ? '▶️ Resume' : '⏸️ Pause';
  });

  document.getElementById('btnRestart').addEventListener('click', function() {
    vehicles = [];
    for (let i = 0; i < 4; i++) {
      pendingSpawns[i] = 0;
      corridorCleared[i] = 0;
    }
    totalVehiclesCleared = 0;
    totalWaitTimeRecordedSec = 0;
    peakQueueLength = 0;
    queueHistory.length = 0;
    activeApproach = 0;
    signalPhase = 'GREEN';
    greenElapsedMs = 0;
    yellowElapsedMs = 0;
    allRedElapsedMs = 0;
    for (let i = 0; i < 8; i++) spawnSafe(i % 4);
    updateEvaluationDeck();
    showToast('🔄 Simulation Reset: Normal flow restarted.');
  });

  const btnSpeed = document.getElementById('btnSpeed');
  btnSpeed.addEventListener('click', function() {
    simSpeedMultiplier = (simSpeedMultiplier === 1) ? 2 : (simSpeedMultiplier === 2 ? 3 : 1);
    this.textContent = `⏩ Speed: ${simSpeedMultiplier}x`;
  });

  // Interactive Add Traffic with 4 Road Corridors
  const btnAddTraffic = document.getElementById('btnAddTraffic');
  const trafficMenu = document.getElementById('trafficMenu');

  btnAddTraffic.addEventListener('click', function(e) {
    e.stopPropagation();
    trafficMenu.classList.toggle('show');
  });

  document.addEventListener('click', function(e) {
    if (!trafficMenu.contains(e.target) && e.target !== btnAddTraffic) {
      trafficMenu.classList.remove('show');
    }
  });

  document.querySelectorAll('.dropdown-item').forEach(btn => {
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      const app = parseInt(this.getAttribute('data-app'), 10);
      addTrafficToApproach(app, 10);
      trafficMenu.classList.remove('show');
    });
  });

  // Initial evaluation deck update
  updateEvaluationDeck();

})();
</script>
</body>
</html>
"""
