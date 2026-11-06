/**
 * FlowPulse ITMS - Production Frontend Controller
 * Connects to FastAPI backend, orchestrates real-time telemetry updates,
 * Chart.js forecasting visualizations, camera switching, and incident management.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global State
  const state = {
    activeSource: "synthetic_highway",
    telemetry: null,
    forecastChart: null,
    historyChart: null,
    historyBuffer: {
      labels: [],
      speeds: [],
      congestions: []
    }
  };

  // DOM Elements
  const liveClockEl = document.getElementById("live-clock");
  const valFps = document.getElementById("val-fps");
  const valLosLetter = document.getElementById("val-los-letter");
  const losBadgeBox = document.getElementById("los-badge-box");
  const valLosTitle = document.getElementById("val-los-title");
  const valLosDescription = document.getElementById("val-los-description");
  const valCongestionPct = document.getElementById("val-congestion-pct");
  const barCongestionFill = document.getElementById("bar-congestion-fill");
  
  const valInboundCount = document.getElementById("val-inbound-count");
  const valOutboundCount = document.getElementById("val-outbound-count");
  const valAvgSpeed = document.getElementById("val-avg-speed");
  const valActiveVehicles = document.getElementById("val-active-vehicles");

  const countCars = document.getElementById("count-cars");
  const countBuses = document.getElementById("count-buses");
  const countTrucks = document.getElementById("count-trucks");
  const countMotorcycles = document.getElementById("count-motorcycles");

  const laneFill1 = document.getElementById("lane-fill-1");
  const laneFill2 = document.getElementById("lane-fill-2");
  const laneFill3 = document.getElementById("lane-fill-3");
  const laneFill4 = document.getElementById("lane-fill-4");
  const laneVal1 = document.getElementById("lane-val-1");
  const laneVal2 = document.getElementById("lane-val-2");
  const laneVal3 = document.getElementById("lane-val-3");
  const laneVal4 = document.getElementById("lane-val-4");

  const incidentsTbody = document.getElementById("incidents-tbody");
  const badgeIncidentCount = document.getElementById("badge-incident-count");

  // Camera Tabs
  const camTabs = document.querySelectorAll(".cam-tab");
  const badgeActiveCameraName = document.getElementById("badge-active-camera-name");

  // Overlays
  const toggleBoxes = document.getElementById("toggle-boxes");
  const toggleTrails = document.getElementById("toggle-trails");
  const toggleSpeed = document.getElementById("toggle-speed");
  const toggleTripwires = document.getElementById("toggle-tripwires");
  const toggleLanes = document.getElementById("toggle-lanes");
  const toggleHud = document.getElementById("toggle-hud");

  // Modals
  const modalTripwire = document.getElementById("modal-tripwire-calibration");
  const btnCalibrateTripwires = document.getElementById("btn-calibrate-tripwires");
  const btnCloseTripwireModal = document.getElementById("btn-close-tripwire-modal");
  const btnCancelTripwires = document.getElementById("btn-cancel-tripwires");
  const btnSaveTripwires = document.getElementById("btn-save-tripwires");
  const sliderLineIn = document.getElementById("slider-line-in");
  const sliderLineOut = document.getElementById("slider-line-out");
  const valSliderLineIn = document.getElementById("val-slider-line-in");
  const valSliderLineOut = document.getElementById("val-slider-line-out");

  const modalUpload = document.getElementById("modal-upload");
  const btnOpenUpload = document.getElementById("btn-open-upload");
  const btnCloseUploadModal = document.getElementById("btn-close-upload-modal");
  const uploadDropzone = document.getElementById("upload-dropzone");
  const fileVideoInput = document.getElementById("file-video-input");
  const uploadProgressWrap = document.getElementById("upload-progress-wrap");
  const uploadProgressFill = document.getElementById("upload-progress-fill");
  const uploadProgressPct = document.getElementById("upload-progress-pct");
  const tabCamCustom = document.getElementById("tab-cam-custom");

  // Actions
  const btnExportCsv = document.getElementById("btn-export-csv");
  const btnSnapshotFrame = document.getElementById("btn-snapshot-frame");
  const btnResetCounters = document.getElementById("btn-reset-counters");
  const toastContainer = document.getElementById("toast-container");

  // Initialize UTC Clock
  function updateClock() {
    const now = new Date();
    liveClockEl.textContent = now.toUTCString().replace("GMT", "UTC");
  }
  setInterval(updateClock, 1000);
  updateClock();

  // Toast Notification
  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = "toast";
    if (type === "success") toast.style.borderLeftColor = "var(--accent-emerald)";
    if (type === "warning") toast.style.borderLeftColor = "var(--accent-amber)";
    if (type === "error") toast.style.borderLeftColor = "var(--accent-crimson)";

    toast.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
      <span>${message}</span>
    `;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(100%)";
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  // Initialize Charts
  function initCharts() {
    // 1. Forecast Chart
    const ctxForecast = document.getElementById("forecastChart").getContext("2d");
    const gradient = ctxForecast.createLinearGradient(0, 0, 0, 220);
    gradient.addColorStop(0, "rgba(6, 182, 212, 0.45)");
    gradient.addColorStop(1, "rgba(6, 182, 212, 0.0)");

    state.forecastChart = new Chart(ctxForecast, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Predicted Hourly Flow (vph)",
            data: [],
            borderColor: "#06b6d4",
            backgroundColor: gradient,
            borderWidth: 2.5,
            fill: true,
            tension: 0.38,
            pointBackgroundColor: "#06b6d4",
            pointRadius: 3,
            pointHoverRadius: 6
          },
          {
            label: "Upper 95% Confidence",
            data: [],
            borderColor: "rgba(148, 163, 184, 0.25)",
            borderDash: [4, 4],
            borderWidth: 1.2,
            fill: false,
            pointRadius: 0
          },
          {
            label: "Lower 95% Confidence",
            data: [],
            borderColor: "rgba(148, 163, 184, 0.25)",
            borderDash: [4, 4],
            borderWidth: 1.2,
            fill: false,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: { color: "#94a3b8", font: { family: "Inter", size: 10 } }
          },
          tooltip: {
            backgroundColor: "#0f172a",
            titleFont: { family: "Outfit", size: 12 },
            bodyFont: { family: "JetBrains Mono", size: 11 },
            borderColor: "rgba(255,255,255,0.1)",
            borderWidth: 1
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.04)" },
            ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 9 }, maxTicksLimit: 12 }
          },
          y: {
            grid: { color: "rgba(255, 255, 255, 0.04)" },
            ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 9 } }
          }
        }
      }
    });

    // 2. Real-Time Telemetry History Chart (Speed & Congestion)
    const ctxHistory = document.getElementById("telemetryHistoryChart").getContext("2d");
    state.historyChart = new Chart(ctxHistory, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Corridor Speed (km/h)",
            data: [],
            borderColor: "#3b82f6",
            borderWidth: 2,
            yAxisID: "ySpeed",
            tension: 0.3,
            pointRadius: 0
          },
          {
            label: "Congestion Index (%)",
            data: [],
            borderColor: "#f59e0b",
            borderWidth: 2,
            yAxisID: "yCong",
            tension: 0.3,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: { color: "#94a3b8", font: { family: "Inter", size: 10 } }
          }
        },
        scales: {
          x: {
            grid: { color: "rgba(255, 255, 255, 0.03)" },
            ticks: { display: false }
          },
          ySpeed: {
            type: "linear",
            position: "left",
            min: 0,
            max: 120,
            grid: { color: "rgba(255, 255, 255, 0.04)" },
            ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 9 }, stepSize: 30 }
          },
          yCong: {
            type: "linear",
            position: "right",
            min: 0,
            max: 100,
            grid: { drawOnChartArea: false },
            ticks: { color: "#64748b", font: { family: "JetBrains Mono", size: 9 }, stepSize: 25 }
          }
        }
      }
    });
  }

  // Poll Telemetry
  async function fetchTelemetry() {
    try {
      const res = await fetch("/api/telemetry");
      if (!res.ok) return;
      const data = await res.json();
      if (!data.success) return;

      const t = data.telemetry;
      state.telemetry = t;

      // Update Top Badges & FPS
      valFps.textContent = t.fps.toFixed(1);

      // Update LOS Card
      valLosLetter.textContent = t.los_level;
      losBadgeBox.style.background = t.los_color;
      valLosTitle.textContent = t.los_description;
      valCongestionPct.textContent = `${t.congestion_index.toFixed(1)}%`;
      barCongestionFill.style.width = `${Math.min(100, Math.max(3, t.congestion_index))}%`;
      barCongestionFill.style.background = t.los_color;

      // LOS Description texts
      const losDescriptions = {
        A: "Traffic moving at free-flow speed. High maneuverability with minimal vehicular interaction.",
        B: "Reasonably free flow. Speeds slightly restricted by surrounding vehicles; high comfort.",
        C: "Stable traffic flow. Passing opportunities restricted; drivers must adjust speeds moderately.",
        D: "Approaching unstable flow. High density; speeds drop considerably during minor fluctuations.",
        E: "High congestion near capacity limit. Fluctuating stop-and-go waves with frequent slowdowns.",
        F: "Forced breakdown & gridlock. Demand exceeds corridor throughput; severe delays."
      };
      valLosDescription.textContent = losDescriptions[t.los_level] || "Monitoring real-time traffic corridor.";

      // Update Real-Time Throughput Counters
      valInboundCount.textContent = t.total_in.toLocaleString();
      valOutboundCount.textContent = t.total_out.toLocaleString();
      valAvgSpeed.innerHTML = `${t.avg_speed_kmh.toFixed(1)} <span style="font-size: 0.9rem;">km/h</span>`;
      valActiveVehicles.textContent = t.active_vehicles;

      // Vehicle Classification Pills
      const cc = t.class_counts || {};
      countCars.textContent = cc.car || 0;
      countBuses.textContent = cc.bus || 0;
      countTrucks.textContent = cc.truck || 0;
      countMotorcycles.textContent = (cc.motorcycle || 0) + (cc.motorbike || 0);

      // Lane Occupancy Barometers
      const lc = t.lane_counts || {};
      const totalLaneVehicles = Math.max(1, (lc[1] || 0) + (lc[2] || 0) + (lc[3] || 0) + (lc[4] || 0));
      laneVal1.textContent = lc[1] || 0;
      laneVal2.textContent = lc[2] || 0;
      laneVal3.textContent = lc[3] || 0;
      laneVal4.textContent = lc[4] || 0;

      laneFill1.style.width = `${((lc[1] || 0) / totalLaneVehicles) * 100}%`;
      laneFill2.style.width = `${((lc[2] || 0) / totalLaneVehicles) * 100}%`;
      laneFill3.style.width = `${((lc[3] || 0) / totalLaneVehicles) * 100}%`;
      laneFill4.style.width = `${((lc[4] || 0) / totalLaneVehicles) * 100}%`;

      // Update Real-time History Chart
      const nowStr = new Date().toLocaleTimeString();
      state.historyBuffer.labels.push(nowStr);
      state.historyBuffer.speeds.push(t.avg_speed_kmh);
      state.historyBuffer.congestions.push(t.congestion_index);

      if (state.historyBuffer.labels.length > 25) {
        state.historyBuffer.labels.shift();
        state.historyBuffer.speeds.shift();
        state.historyBuffer.congestions.shift();
      }

      state.historyChart.data.labels = state.historyBuffer.labels;
      state.historyChart.data.datasets[0].data = state.historyBuffer.speeds;
      state.historyChart.data.datasets[1].data = state.historyBuffer.congestions;
      state.historyChart.update("none");

    } catch (e) {
      console.error("Telemetry error:", e);
    }
  }

  // Fetch Predictive Forecast
  async function fetchForecast() {
    try {
      const res = await fetch("/api/forecast");
      if (!res.ok) return;
      const json = await res.json();
      if (!json.success) return;

      const f = json.forecast;
      state.forecastChart.data.labels = f.timestamps;
      state.forecastChart.data.datasets[0].data = f.forecast_flow_vph;
      state.forecastChart.data.datasets[1].data = f.upper_bound;
      state.forecastChart.data.datasets[2].data = f.lower_bound;
      state.forecastChart.update();

      // Peak info
      const p = f.peak_analysis;
      document.getElementById("val-peak-hour").textContent = `${p.max_predicted_hour} (${p.max_predicted_flow.toLocaleString()} vph)`;
      document.getElementById("val-advisory-recommendation").textContent = `Recommendation: ${p.recommended_action}`;
    } catch (e) {
      console.error("Forecast error:", e);
    }
  }

  // Fetch Incidents
  async function fetchIncidents() {
    try {
      const res = await fetch("/api/incidents");
      if (!res.ok) return;
      const json = await res.json();
      if (!json.success) return;

      const incs = json.incidents || [];
      const activeIncs = incs.filter(i => !i.resolved);
      badgeIncidentCount.textContent = `${activeIncs.length} Active Anomaly Alerts`;
      badgeIncidentCount.style.borderColor = activeIncs.length > 0 ? "rgba(239, 68, 68, 0.4)" : "var(--border-dim)";

      if (incs.length === 0) {
        incidentsTbody.innerHTML = `
          <tr>
            <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 24px;">
              Monitoring corridor in real time. No safety incidents currently detected.
            </td>
          </tr>
        `;
        return;
      }

      incidentsTbody.innerHTML = incs.map(inc => `
        <tr style="${inc.resolved ? 'opacity: 0.45;' : ''}">
          <td style="font-family: var(--font-mono); color: var(--text-muted);">${inc.timestamp.split(" ")[1] || inc.timestamp}</td>
          <td><strong>${inc.type.replace(/_/g, " ")}</strong></td>
          <td><span class="severity-pill severity-${inc.severity}">${inc.severity}</span></td>
          <td style="font-family: var(--font-mono);">${inc.track_id ? '#' + inc.track_id : 'N/A'}</td>
          <td style="font-family: var(--font-mono);">${inc.lane_id ? 'Lane ' + inc.lane_id : 'All'}</td>
          <td style="color: var(--text-secondary);">${inc.description}</td>
          <td>
            ${inc.resolved 
              ? '<span style="color: var(--accent-emerald); font-size: 0.75rem;">Resolved</span>' 
              : `<button class="btn-resolve" onclick="window.resolveIncident(${inc.id})">Resolve</button>`}
          </td>
        </tr>
      `).join("");

    } catch (e) {
      console.error("Incidents error:", e);
    }
  }

  // Global Resolve Incident Callback
  window.resolveIncident = async function(id) {
    try {
      const res = await fetch(`/api/incidents/${id}/resolve`, { method: "POST" });
      if (res.ok) {
        showToast(`Incident #${id} resolved`, "success");
        fetchIncidents();
      }
    } catch (e) {
      showToast("Failed to resolve incident", "error");
    }
  };

  // Switch Active Camera Feed
  async function switchCamera(sourceType, camName) {
    try {
      const res = await fetch("/api/source", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_type: sourceType })
      });
      if (res.ok) {
        state.activeSource = sourceType;
        camTabs.forEach(t => t.classList.remove("active"));
        const targetTab = document.querySelector(`.cam-tab[data-source="${sourceType}"]`);
        if (targetTab) targetTab.classList.add("active");

        badgeActiveCameraName.textContent = camName.toUpperCase();
        showToast(`Switched feed to: ${camName}`, "info");

        // Reconnect stream image
        const img = document.getElementById("main-video-feed");
        img.src = `/api/stream?t=${Date.now()}`;
      }
    } catch (e) {
      showToast("Error switching camera source", "error");
    }
  }

  // Event: Camera Tab Clicks
  camTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const src = tab.getAttribute("data-source");
      const name = tab.textContent.trim();
      switchCamera(src, name);
    });
  });

  // Event: HUD Overlays Toggles
  function updateDisplayOptions() {
    fetch("/api/display", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        show_boxes: toggleBoxes.checked,
        show_trails: toggleTrails.checked,
        show_speed: toggleSpeed.checked,
        show_tripwires: toggleTripwires.checked,
        show_lanes: toggleLanes.checked,
        show_hud: toggleHud.checked
      })
    });
  }

  [toggleBoxes, toggleTrails, toggleSpeed, toggleTripwires, toggleLanes, toggleHud].forEach(el => {
    el.addEventListener("change", updateDisplayOptions);
  });

  // Modal: Tripwires Calibration
  btnCalibrateTripwires.addEventListener("click", () => {
    modalTripwire.classList.add("active");
  });

  function closeTripwireModal() {
    modalTripwire.classList.remove("active");
  }
  btnCloseTripwireModal.addEventListener("click", closeTripwireModal);
  btnCancelTripwires.addEventListener("click", closeTripwireModal);

  sliderLineIn.addEventListener("input", (e) => {
    valSliderLineIn.textContent = `${e.target.value} px`;
  });
  sliderLineOut.addEventListener("input", (e) => {
    valSliderLineOut.textContent = `${e.target.value} px`;
  });

  btnSaveTripwires.addEventListener("click", async () => {
    const line_in_y = parseInt(sliderLineIn.value);
    const line_out_y = parseInt(sliderLineOut.value);
    try {
      const res = await fetch("/api/tripwires", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ line_in_y, line_out_y })
      });
      if (res.ok) {
        showToast("Tripwire lines calibrated successfully", "success");
        closeTripwireModal();
      }
    } catch (e) {
      showToast("Failed to calibrate tripwires", "error");
    }
  });

  // Modal: Video Upload
  btnOpenUpload.addEventListener("click", () => {
    modalUpload.classList.add("active");
  });

  function closeUploadModal() {
    modalUpload.classList.remove("active");
    uploadProgressWrap.style.display = "none";
  }
  btnCloseUploadModal.addEventListener("click", closeUploadModal);

  uploadDropzone.addEventListener("click", () => fileVideoInput.click());

  uploadDropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadDropzone.style.borderColor = "var(--accent-cyan)";
  });

  uploadDropzone.addEventListener("dragleave", () => {
    uploadDropzone.style.borderColor = "rgba(6, 182, 212, 0.35)";
  });

  uploadDropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadDropzone.style.borderColor = "rgba(6, 182, 212, 0.35)";
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileVideoInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  });

  function handleFileUpload(file) {
    const formData = new FormData();
    formData.append("file", file);

    uploadProgressWrap.style.display = "block";
    uploadProgressFill.style.width = "40%";
    uploadProgressPct.textContent = "40%";

    fetch("/api/upload", {
      method: "POST",
      body: formData
    })
      .then(res => {
        if (!res.ok) throw new Error("Upload failed");
        return res.json();
      })
      .then(data => {
        uploadProgressFill.style.width = "100%";
        uploadProgressPct.textContent = "100%";
        showToast(`Uploaded: ${file.name}`, "success");
        tabCamCustom.style.display = "inline-flex";
        badgeActiveCameraName.textContent = `CUSTOM: ${file.name.toUpperCase()}`;
        camTabs.forEach(t => t.classList.remove("active"));
        tabCamCustom.classList.add("active");

        setTimeout(() => {
          closeUploadModal();
          const img = document.getElementById("main-video-feed");
          img.src = `/api/stream?t=${Date.now()}`;
        }, 800);
      })
      .catch(err => {
        showToast("Video upload failed", "error");
        uploadProgressWrap.style.display = "none";
      });
  }

  // Export CSV
  btnExportCsv.addEventListener("click", () => {
    window.location.href = "/api/export";
    showToast("Audit telemetry CSV downloaded", "info");
  });

  // Snapshot Current Video Frame
  btnSnapshotFrame.addEventListener("click", () => {
    const img = document.getElementById("main-video-feed");
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth || 960;
    canvas.height = img.naturalHeight || 540;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const link = document.createElement("a");
    link.download = `FlowPulse_Traffic_Snapshot_${Date.now()}.jpg`;
    link.href = canvas.toDataURL("image/jpeg", 0.92);
    link.click();
    showToast("Snapshot saved to downloads", "success");
  });

  // Reset Tripwire Counters
  btnResetCounters.addEventListener("click", () => {
    switchCamera(state.activeSource, badgeActiveCameraName.textContent);
    showToast("Tripwire counters reset to zero", "info");
  });

  // Initialize
  initCharts();
  fetchForecast();
  fetchIncidents();
  setInterval(fetchTelemetry, 750);
  setInterval(fetchForecast, 30000);
  setInterval(fetchIncidents, 4000);
});
