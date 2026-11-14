(function () {
  const dataUrl = window.SITE_DATA_URL || "data/results.json";

  const OFFICIAL_TASKS = new Set([
    "l-bracket-stress",
    "suspension-rocker-multiload",
    "aerospace-rib-obstacles",
  ]);

  function fmtReward(val) {
    return typeof val === "number" ? val.toFixed(4) : "pending";
  }

  function renderGapChart(data) {
    const container = document.getElementById("gap-chart");
    container.innerHTML = "";

    for (const task of data.tasks.filter((t) => OFFICIAL_TASKS.has(t.task_id))) {
      const taskRuns = data.harbor_runs.filter(
        (r) =>
          r.task_id === task.task_id &&
          r.agent !== "oracle" &&
          typeof r.harbor_reward === "number"
      );
      const bestAgent = taskRuns.reduce(
        (best, cur) => (!best || cur.harbor_reward > best.harbor_reward ? cur : best),
        null
      );

      const oracleReward = task.oracle.harbor_reward ?? 0;
      const agentReward = bestAgent ? bestAgent.harbor_reward : null;
      const baselineReward = task.do_nothing.harbor_reward ?? 0;

      const card = document.createElement("div");
      card.className = "gap-task";
      card.innerHTML = `
        <div class="gap-task-title">
          <span>${task.title}</span>
          <span class="mono muted">target vol &le; ${task.target_volume}</span>
        </div>
        <div class="bar-row">
          <span>Reference Implementation</span>
          <div class="bar-track"><div class="bar-fill bar-oracle" style="width:${(oracleReward * 100).toFixed(1)}%"></div></div>
          <span class="mono">${fmtReward(oracleReward)}</span>
        </div>
        <div class="bar-row">
          <span>${bestAgent ? bestAgent.agent : "Frontier Agent (pending)"}</span>
          <div class="bar-track"><div class="bar-fill bar-agent" style="width:${agentReward !== null ? (agentReward * 100).toFixed(1) : 0}%"></div></div>
          <span class="mono">${fmtReward(agentReward)}</span>
        </div>
        <div class="bar-row">
          <span>Do-Nothing Baseline</span>
          <div class="bar-track"><div class="bar-fill bar-baseline" style="width:${(baselineReward * 100).toFixed(1)}%"></div></div>
          <span class="mono">${fmtReward(baselineReward)}</span>
        </div>
      `;
      container.appendChild(card);
    }
  }

  function renderRunsTable(data) {
    const wrap = document.getElementById("runs-table-wrap");
    const officialRuns = data.harbor_runs.filter((r) =>
      OFFICIAL_TASKS.has(r.task_id)
    );
    if (!officialRuns.length) {
      wrap.innerHTML = `<p class="muted">No Harbor trials found in <code>jobs/</code> yet.</p>`;
      return;
    }

    const rows = officialRuns
      .map((r) => {
        const rewardStr = fmtReward(r.harbor_reward);
        const cls =
          r.harbor_reward == null
            ? "pill-warn"
            : r.harbor_reward >= 0.8
            ? "pill-good"
            : r.harbor_reward > 0
            ? "pill-warn"
            : "pill-bad";
        return `
          <tr>
            <td class="mono">${r.task_id || "—"}</td>
            <td>${r.agent}</td>
            <td class="mono ${cls}">${rewardStr}</td>
            <td class="mono">${r.volume_fraction ?? "—"}</td>
            <td class="mono">${r.max_stress ?? "—"}</td>
            <td class="mono muted">${r.trial}</td>
          </tr>
        `;
      })
      .join("");

    wrap.innerHTML = `
      <h3>Harbor Trial Log (<code>jobs/</code>)</h3>
      <table>
        <thead>
          <tr>
            <th>Task</th>
            <th>Agent</th>
            <th>Reward</th>
            <th>Volume</th>
            <th>Max Stress</th>
            <th>Trial</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  function setupExplorer(data) {
    const taskSelect = document.getElementById("task-select");
    const viewSelect = document.getElementById("view-select");
    const compGrid = document.getElementById("comparison-grid");
    const replaySourceSelect = document.getElementById("replay-source-select");
    const replayPrev = document.getElementById("replay-prev");
    const replayPlay = document.getElementById("replay-play");
    const replayNext = document.getElementById("replay-next");
    const replaySlider = document.getElementById("replay-slider");
    const replayStepLabel = document.getElementById("replay-step-label");
    const replayMetrics = document.getElementById("replay-metrics");
    const replayImages = document.getElementById("replay-images");
    const preloadedUrls = new Set();
    let playTimer = null;

    function preloadTrajectory(frames) {
      for (const f of frames) {
        if (!f.images) continue;
        for (const url of [f.images.geometry, f.images.stress]) {
          if (url && !preloadedUrls.has(url)) {
            preloadedUrls.add(url);
            const img = new Image();
            img.src = url;
          }
        }
      }
    }

    function ensureReplayImgSlots() {
      if (replayImages.querySelector("[data-slot='geometry']")) return;
      if (!replayImages.dataset.mobileActive) {
        replayImages.dataset.mobileActive = "geometry";
      }
      replayImages.innerHTML = `
        <div class="replay-slot-geometry"><img data-slot="geometry" alt="Step geometry" /></div>
        <div class="replay-slot-stress"><img data-slot="stress" alt="Step stress" /></div>
      `;
    }

    function stopPlayback() {
      if (playTimer !== null) {
        clearInterval(playTimer);
        playTimer = null;
      }
      replayPlay.innerHTML = "&#9654;";
    }

    function startPlayback() {
      stopPlayback();
      const max = Number(replaySlider.max) || 0;
      if (max <= 0) return;
      replayPlay.innerHTML = "&#9208;";
      playTimer = setInterval(() => {
        const cur = Number(replaySlider.value) || 0;
        const next = cur >= max ? 0 : cur + 1;
        replaySlider.value = String(next);
        renderReplayFrame();
      }, 950);
    }

    const orderedTasks = [
      ...data.tasks.filter((t) => OFFICIAL_TASKS.has(t.task_id)),
      ...data.tasks.filter((t) => !OFFICIAL_TASKS.has(t.task_id)),
    ];
    taskSelect.innerHTML = orderedTasks
      .map((t) => `<option value="${t.task_id}">${t.title}</option>`)
      .join("");

    function makeVizCard(title, sub, imgSrc, slotClass) {
      const div = document.createElement("div");
      div.className = `viz-card ${slotClass || ""}`.trim();
      div.innerHTML = `
        <strong>${title}</strong>
        <div class="viz-meta">${sub}</div>
        ${
          imgSrc
            ? `<img src="${imgSrc}" alt="${title}" loading="lazy" />`
            : `<p class="muted">Render skipped or unavailable.</p>`
        }
      `;
      return div;
    }

    function getReplaySources(task) {
      const sources = [
        {
          id: "oracle",
          label: `Reference (${fmtReward(task.oracle.harbor_reward)})`,
          frames: task.oracle_replay || [],
        },
      ];
      for (const run of data.harbor_runs) {
        if (run.task_id === task.task_id && run.replay && run.replay.length) {
          sources.push({
            id: run.trial,
            label: `${run.agent} (${fmtReward(run.harbor_reward)})`,
            frames: run.replay,
          });
        }
      }
      return sources;
    }

    function renderReplayFrame() {
      const task = data.tasks.find((t) => t.task_id === taskSelect.value);
      const sources = getReplaySources(task);
      const src = sources.find((s) => s.id === replaySourceSelect.value) || sources[0];
      const frames = src ? src.frames : [];

      if (!frames.length) {
        stopPlayback();
        replayStepLabel.textContent = "Step 0 / 0";
        replayPrev.disabled = true;
        replayNext.disabled = true;
        replayPlay.disabled = true;
        replayMetrics.innerHTML = `<span class="muted">No cutout steps recorded.</span>`;
        replayImages.innerHTML = "";
        return;
      }

      const idx = Math.min(Number(replaySlider.value) || 0, frames.length - 1);
      const frame = frames[idx];
      replayPrev.disabled = idx <= 0;
      replayNext.disabled = idx >= frames.length - 1;
      replayPlay.disabled = frames.length <= 1;
      replayStepLabel.textContent = `Step ${frame.step} / ${frame.total_steps}`;

      replayMetrics.innerHTML = `
        <div class="stat-item">
          <span class="stat-label">Volume</span>
          <span class="stat-value">${frame.volume_fraction}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Compliance</span>
          <span class="stat-value">${frame.compliance}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Max Stress</span>
          <span class="stat-value">${frame.max_stress}</span>
        </div>
      `;

      ensureReplayImgSlots();
      const geoImg = replayImages.querySelector("[data-slot='geometry']");
      const stressImg = replayImages.querySelector("[data-slot='stress']");
      if (frame.images.geometry) {
        geoImg.src = frame.images.geometry;
        geoImg.parentElement.style.display = "";
      } else {
        geoImg.parentElement.style.display = "none";
      }
      if (frame.images.stress) {
        stressImg.src = frame.images.stress;
        stressImg.parentElement.style.display = "";
      } else {
        stressImg.parentElement.style.display = "none";
      }
    }

    function renderComparison() {
      const task = data.tasks.find((t) => t.task_id === taskSelect.value);
      const view = viewSelect.value;
      if (!compGrid.dataset.mobileActive) {
        compGrid.dataset.mobileActive = "simp";
      }
      compGrid.innerHTML = "";

      compGrid.appendChild(
        makeVizCard(
          "Topological Optimization (C*)",
          `<span>Vol: ${task.simp.volume_fraction}</span><span>Comp: ${task.simp.compliance}</span>`,
          task.simp.images[view],
          "comp-slot-simp"
        )
      );
      compGrid.appendChild(
        makeVizCard(
          "Reference Implementation",
          `<span>Reward: ${fmtReward(task.oracle.harbor_reward)}</span><span>Vol: ${task.oracle.volume_fraction}</span>`,
          task.oracle.images[view],
          "comp-slot-oracle"
        )
      );

      const agentRuns = data.harbor_runs.filter(
        (r) =>
          r.task_id === task.task_id &&
          r.agent !== "oracle" &&
          r.images &&
          r.images[view]
      );
      const bestAgent = agentRuns.reduce(
        (best, cur) =>
          !best || (cur.harbor_reward ?? -1) > (best.harbor_reward ?? -1) ? cur : best,
        null
      );

      if (bestAgent) {
        compGrid.appendChild(
          makeVizCard(
            bestAgent.agent,
            `<span>Reward: ${fmtReward(bestAgent.harbor_reward)}</span><span>Vol: ${bestAgent.volume_fraction ?? "—"}</span>`,
            bestAgent.images[view],
            "comp-slot-agent"
          )
        );
      } else {
        compGrid.appendChild(
          makeVizCard(
            "Frontier Agent",
            `<span>Status: pending / no design.json</span>`,
            null,
            "comp-slot-agent"
          )
        );
      }

      const sources = getReplaySources(task);
      replaySourceSelect.innerHTML = sources
        .map((s) => `<option value="${s.id}">${s.label}</option>`)
        .join("");
      const firstFrames = sources[0] ? sources[0].frames : [];
      preloadTrajectory(firstFrames);
      replaySlider.max = Math.max(0, firstFrames.length - 1);
      replaySlider.value = "0";
      renderReplayFrame();
      if (firstFrames.length > 1) {
        startPlayback();
      }
    }

    taskSelect.addEventListener("change", () => {
      stopPlayback();
      renderComparison();
    });
    viewSelect.addEventListener("change", renderComparison);
    replaySourceSelect.addEventListener("change", () => {
      stopPlayback();
      const task = data.tasks.find((t) => t.task_id === taskSelect.value);
      const sources = getReplaySources(task);
      const src = sources.find((s) => s.id === replaySourceSelect.value) || sources[0];
      const frames = src ? src.frames : [];
      preloadTrajectory(frames);
      replaySlider.max = Math.max(0, frames.length - 1);
      replaySlider.value = "0";
      renderReplayFrame();
      if (frames.length > 1) {
        startPlayback();
      }
    });
    replaySlider.addEventListener("input", () => {
      stopPlayback();
      renderReplayFrame();
    });
    replayPrev.addEventListener("click", () => {
      stopPlayback();
      const cur = Number(replaySlider.value) || 0;
      if (cur > 0) {
        replaySlider.value = String(cur - 1);
        renderReplayFrame();
      }
    });
    replayNext.addEventListener("click", () => {
      stopPlayback();
      const cur = Number(replaySlider.value) || 0;
      const max = Number(replaySlider.max) || 0;
      if (cur < max) {
        replaySlider.value = String(cur + 1);
        renderReplayFrame();
      }
    });
    replayPlay.addEventListener("click", () => {
      if (playTimer !== null) {
        stopPlayback();
      } else {
        startPlayback();
      }
    });

    document.querySelectorAll(".comp-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        compGrid.dataset.mobileActive = btn.dataset.compView;
        document.querySelectorAll(".comp-tab").forEach((b) => {
          b.classList.toggle("active", b === btn);
        });
      });
    });

    document.querySelectorAll(".replay-tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        const view = btn.dataset.mobileView;
        replayImages.dataset.mobileActive = view;
        document.querySelectorAll(".replay-tab").forEach((b) => {
          b.classList.toggle("active", b === btn);
        });
      });
    });

    renderComparison();
  }

  fetch(dataUrl)
    .then((res) => res.json())
    .then((data) => {
      renderGapChart(data);
      renderRunsTable(data);
      setupExplorer(data);
    })
    .catch((err) => {
      console.error("Failed to load site data:", err);
    });
})();
