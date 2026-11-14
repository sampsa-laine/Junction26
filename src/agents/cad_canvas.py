"""Interactive HTML5 canvas component with cursor crosshair and polygon/CAD cutout tools."""

from __future__ import annotations

import math
from typing import Any

import streamlit as st

from src.agents.csg import CADModel
from src.agents.problem import DesignProblem

_CAD_CANVAS_HTML = """
<div class="cad-canvas-wrap">
  <div class="cad-toolbar">
    <span class="cad-status" id="cad-status">Move cursor over domain</span>
    <div class="cad-actions">
      <button type="button" class="cad-btn cad-btn-primary" id="cad-finish-btn" style="display:none;">Finish Polygon</button>
      <button type="button" class="cad-btn" id="cad-undo-pt-btn" style="display:none;">Undo Point</button>
      <button type="button" class="cad-btn" id="cad-clear-pt-btn" style="display:none;">Cancel</button>
    </div>
  </div>
  <canvas id="cad-canvas"></canvas>
  <div class="cad-hint" id="cad-hint"></div>
</div>
"""

_CAD_CANVAS_CSS = """
.cad-canvas-wrap {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  border: 1px solid rgba(128, 128, 128, 0.25);
  border-radius: 8px;
  padding: 8px 10px;
  background: #ffffff;
}
.cad-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
  min-height: 28px;
}
.cad-status {
  font-size: 12.5px;
  font-weight: 600;
  color: #1e293b;
  font-variant-numeric: tabular-nums;
}
.cad-actions {
  display: flex;
  gap: 6px;
}
.cad-btn {
  font-size: 12px;
  font-weight: 500;
  padding: 3px 9px;
  border-radius: 5px;
  border: 1px solid #cbd5e1;
  background: #f8fafc;
  color: #1e293b;
  cursor: pointer;
}
.cad-btn:hover {
  background: #e2e8f0;
}
.cad-btn-primary {
  background: #2563eb;
  border-color: #2563eb;
  color: #ffffff;
}
.cad-btn-primary:hover {
  background: #1d4ed8;
}
#cad-canvas {
  display: block;
  width: 100%;
  height: auto;
  cursor: crosshair;
  border-radius: 4px;
  user-select: none;
}
.cad-hint {
  margin-top: 5px;
  font-size: 11.5px;
  color: #64748b;
}
"""

_CAD_CANVAS_JS = """
export default function(component) {
  const { data, setTriggerValue, parentElement } = component;
  if (!data) return;

  const canvas = parentElement.getElementById("cad-canvas");
  const statusEl = parentElement.getElementById("cad-status");
  const hintEl = parentElement.getElementById("cad-hint");
  const finishBtn = parentElement.getElementById("cad-finish-btn");
  const undoPtBtn = parentElement.getElementById("cad-undo-pt-btn");
  const clearPtBtn = parentElement.getElementById("cad-clear-pt-btn");
  if (!canvas) return;

  if (!parentElement.__cadState) {
    parentElement.__cadState = {
      data: data,
      polyPts: [],
      anchor: null,
      hoverWorld: null,
      mouseDownPos: null,
      isDragging: false,
      bound: false,
    };
  }
  const state = parentElement.__cadState;
  const prevTool = state.data ? state.data.tool : null;
  const prevProblem = state.data ? state.data.problemKey : null;
  state.data = data;

  if (prevTool !== data.tool || prevProblem !== data.problemKey) {
    state.polyPts = [];
    state.anchor = null;
    state.isDragging = false;
  }

  const PAD_L = 44;
  const PAD_R = 16;
  const PAD_T = 16;
  const PAD_B = 34;

  function getLayout() {
    const w = state.data.width;
    const h = state.data.height;
    const cw = 920;
    const plotW = cw - PAD_L - PAD_R;
    const plotH = Math.max(260, Math.min(500, Math.round(plotW * (h / w))));
    const ch = plotH + PAD_T + PAD_B;
    return { w, h, cw, ch, plotW, plotH };
  }

  function worldToPx(wx, wy, L) {
    return {
      x: PAD_L + (wx / L.w) * L.plotW,
      y: PAD_T + L.plotH - (wy / L.h) * L.plotH,
    };
  }

  function pxToWorld(px, py, L) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = L.cw / rect.width;
    const scaleY = L.ch / rect.height;
    const cx = (px - rect.left) * scaleX;
    const cy = (py - rect.top) * scaleY;
    let wx = ((cx - PAD_L) / L.plotW) * L.w;
    let wy = ((PAD_T + L.plotH - cy) / L.plotH) * L.h;
    if (wx < 0 || wx > L.w || wy < 0 || wy > L.h) {
      return null;
    }
    const snap = state.data.gridSnap || 0;
    if (snap > 0) {
      wx = Math.round(wx / snap) * snap;
      wy = Math.round(wy / snap) * snap;
    }
    wx = Math.max(0, Math.min(L.w, Number(wx.toFixed(3))));
    wy = Math.max(0, Math.min(L.h, Number(wy.toFixed(3))));
    return { x: wx, y: wy, cx, cy };
  }

  function emitAction(payload) {
    setTriggerValue("action", {
      ...payload,
      nonce: `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    });
  }

  function commitPolygon() {
    if (state.polyPts.length < 3) return;
    const pts = state.polyPts.map((p) => [p.x, p.y]);
    state.polyPts = [];
    emitAction({
      type: "polygon",
      points: pts,
      fillet_radius: state.data.filletRadius || 0,
    });
    redraw();
  }

  function updateToolbar() {
    const tool = state.data.tool;
    const isPoly = tool === "polygon";
    const hasPts = isPoly && state.polyPts.length > 0;
    finishBtn.style.display = isPoly && state.polyPts.length >= 3 ? "inline-block" : "none";
    undoPtBtn.style.display = hasPts ? "inline-block" : "none";
    clearPtBtn.style.display = hasPts ? "inline-block" : "none";

    if (tool === "polygon") {
      hintEl.textContent = "Polygon: Click vertices to draw cutout. Double-click, click start vertex, or press Enter to finish. Right-click/Backspace undoes last point; Esc cancels.";
    } else if (tool === "circle") {
      hintEl.textContent = "Circle: Click center then click/drag to set radius. Esc cancels.";
    } else if (tool === "rect") {
      hintEl.textContent = "Rectangle: Click first corner then click/drag to opposite corner. Esc cancels.";
    } else {
      hintEl.textContent = "Slot: Click start point then click/drag to end point. Esc cancels.";
    }

    if (state.hoverWorld) {
      let msg = `Cursor: x = ${state.hoverWorld.x.toFixed(2)}, y = ${state.hoverWorld.y.toFixed(2)}`;
      if (tool === "polygon" && state.polyPts.length > 0) {
        msg += `  |  Vertices: ${state.polyPts.length}`;
      } else if (state.anchor && tool === "circle") {
        const r = Math.hypot(state.hoverWorld.x - state.anchor.x, state.hoverWorld.y - state.anchor.y);
        msg += `  |  Center (${state.anchor.x.toFixed(1)}, ${state.anchor.y.toFixed(1)}), r = ${r.toFixed(2)}`;
      } else if (state.anchor && tool === "rect") {
        const rw = Math.abs(state.hoverWorld.x - state.anchor.x);
        const rh = Math.abs(state.hoverWorld.y - state.anchor.y);
        msg += `  |  Size: ${rw.toFixed(2)} x ${rh.toFixed(2)}`;
      } else if (state.anchor && tool === "slot") {
        const len = Math.hypot(state.hoverWorld.x - state.anchor.x, state.hoverWorld.y - state.anchor.y);
        msg += `  |  Length: ${len.toFixed(2)}, r = ${(state.data.slotRadius || 2.5).toFixed(2)}`;
      }
      statusEl.textContent = msg;
    } else {
      statusEl.textContent = `Domain: ${state.data.width} x ${state.data.height} — Move cursor over domain`;
    }
  }

  function redraw() {
    const L = getLayout();
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== L.cw * dpr || canvas.height !== L.ch * dpr) {
      canvas.width = L.cw * dpr;
      canvas.height = L.ch * dpr;
    }
    const ctx = canvas.getContext("2d");
    ctx.save();
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, L.cw, L.ch);

    // Plot background
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(PAD_L, PAD_T, L.plotW, L.plotH);

    // Grid lines
    const step = L.w >= 80 ? 10 : 5;
    ctx.lineWidth = 1;
    ctx.font = "11px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillStyle = "#475569";

    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let gx = 0; gx <= L.w + 1e-6; gx += step) {
      const p = worldToPx(gx, 0, L);
      ctx.strokeStyle = "#e2e8f0";
      ctx.beginPath();
      ctx.moveTo(p.x, PAD_T);
      ctx.lineTo(p.x, PAD_T + L.plotH);
      ctx.stroke();
      ctx.fillText(String(Math.round(gx)), p.x, PAD_T + L.plotH + 6);
    }

    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let gy = 0; gy <= L.h + 1e-6; gy += step) {
      const p = worldToPx(0, gy, L);
      ctx.strokeStyle = "#e2e8f0";
      ctx.beginPath();
      ctx.moveTo(PAD_L, p.y);
      ctx.lineTo(PAD_L + L.plotW, p.y);
      ctx.stroke();
      ctx.fillText(String(Math.round(gy)), PAD_L - 6, p.y);
    }

    // Void keepouts
    for (const ko of state.data.keepouts) {
      if (ko.type !== "void") continue;
      ctx.save();
      ctx.fillStyle = "rgba(239, 68, 68, 0.08)";
      ctx.strokeStyle = "rgba(239, 68, 68, 0.55)";
      ctx.setLineDash([5, 4]);
      ctx.lineWidth = 1.5;
      if (ko.shape === "rect") {
        const [x, y, rw, rh] = ko.params;
        const p1 = worldToPx(x, y + rh, L);
        const p2 = worldToPx(x + rw, y, L);
        ctx.fillRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
        ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
      } else if (ko.shape === "circle") {
        const [cx, cy, r] = ko.params;
        const c = worldToPx(cx, cy, L);
        const rx = (r / L.w) * L.plotW;
        ctx.beginPath();
        ctx.arc(c.x, c.y, rx, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    }

    // Material geometry (outer rings + cutouts as holes)
    ctx.save();
    ctx.fillStyle = "rgba(59, 130, 246, 0.24)";
    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 2;
    for (const poly of state.data.polygons) {
      ctx.beginPath();
      poly.outer.forEach((pt, idx) => {
        const p = worldToPx(pt[0], pt[1], L);
        if (idx === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      ctx.closePath();
      for (const hole of poly.holes) {
        hole.forEach((pt, idx) => {
          const p = worldToPx(pt[0], pt[1], L);
          if (idx === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        });
        ctx.closePath();
      }
      ctx.fill("evenodd");
      ctx.stroke();
    }
    ctx.restore();

    // Solid keepouts
    for (const ko of state.data.keepouts) {
      if (ko.type !== "solid") continue;
      ctx.save();
      ctx.fillStyle = "rgba(16, 185, 129, 0.18)";
      ctx.strokeStyle = "rgba(5, 150, 105, 0.75)";
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.5;
      if (ko.shape === "rect") {
        const [x, y, rw, rh] = ko.params;
        const p1 = worldToPx(x, y + rh, L);
        const p2 = worldToPx(x + rw, y, L);
        ctx.fillRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
        ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
      } else if (ko.shape === "circle") {
        const [cx, cy, r] = ko.params;
        const c = worldToPx(cx, cy, L);
        const rx = (r / L.w) * L.plotW;
        ctx.beginPath();
        ctx.arc(c.x, c.y, rx, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    }

    // Supports
    for (const s of state.data.supports) {
      const p = worldToPx(s.x, s.y, L);
      ctx.fillStyle = "#059669";
      ctx.beginPath();
      ctx.moveTo(p.x, p.y - 5);
      ctx.lineTo(p.x - 5, p.y + 5);
      ctx.lineTo(p.x + 5, p.y + 5);
      ctx.closePath();
      ctx.fill();
    }

    // Loads
    for (const ld of state.data.loads) {
      const p = worldToPx(ld.x, ld.y, L);
      const mag = Math.hypot(ld.fx, ld.fy) || 1;
      const dx = (ld.fx / mag) * 24;
      const dy = -(ld.fy / mag) * 24;
      ctx.strokeStyle = "#dc2626";
      ctx.fillStyle = "#dc2626";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(p.x + dx, p.y + dy);
      ctx.stroke();
      const ang = Math.atan2(dy, dx);
      ctx.beginPath();
      ctx.moveTo(p.x + dx, p.y + dy);
      ctx.lineTo(p.x + dx - 7 * Math.cos(ang - 0.45), p.y + dy - 7 * Math.sin(ang - 0.45));
      ctx.lineTo(p.x + dx - 7 * Math.cos(ang + 0.45), p.y + dy - 7 * Math.sin(ang + 0.45));
      ctx.closePath();
      ctx.fill();
    }

    // Active tool preview
    const tool = state.data.tool;
    if (tool === "polygon" && state.polyPts.length > 0) {
      ctx.save();
      ctx.fillStyle = "rgba(239, 68, 68, 0.20)";
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 2;
      ctx.beginPath();
      state.polyPts.forEach((pt, idx) => {
        const p = worldToPx(pt.x, pt.y, L);
        if (idx === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      });
      if (state.hoverWorld) {
        const hp = worldToPx(state.hoverWorld.x, state.hoverWorld.y, L);
        ctx.lineTo(hp.x, hp.y);
      }
      if (state.polyPts.length >= 2) {
        ctx.closePath();
        ctx.fill();
      }
      ctx.stroke();

      state.polyPts.forEach((pt, idx) => {
        const p = worldToPx(pt.x, pt.y, L);
        ctx.fillStyle = idx === 0 ? "#16a34a" : "#dc2626";
        ctx.beginPath();
        ctx.arc(p.x, p.y, idx === 0 ? 5 : 4, 0, Math.PI * 2);
        ctx.fill();
      });

      if (state.polyPts.length >= 3 && state.hoverWorld) {
        const p0 = worldToPx(state.polyPts[0].x, state.polyPts[0].y, L);
        const ph = worldToPx(state.hoverWorld.x, state.hoverWorld.y, L);
        if (Math.hypot(ph.x - p0.x, ph.y - p0.y) <= 12) {
          ctx.strokeStyle = "#16a34a";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(p0.x, p0.y, 9, 0, Math.PI * 2);
          ctx.stroke();
        }
      }
      ctx.restore();
    } else if (state.anchor && state.hoverWorld) {
      ctx.save();
      ctx.fillStyle = "rgba(239, 68, 68, 0.20)";
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      const p1 = worldToPx(state.anchor.x, state.anchor.y, L);
      const p2 = worldToPx(state.hoverWorld.x, state.hoverWorld.y, L);

      if (tool === "circle") {
        const rPx = Math.hypot(p2.x - p1.x, p2.y - p1.y);
        ctx.beginPath();
        ctx.arc(p1.x, p1.y, rPx, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      } else if (tool === "rect") {
        const rx = Math.min(p1.x, p2.x);
        const ry = Math.min(p1.y, p2.y);
        const rw = Math.abs(p2.x - p1.x);
        const rh = Math.abs(p2.y - p1.y);
        ctx.fillRect(rx, ry, rw, rh);
        ctx.strokeRect(rx, ry, rw, rh);
      } else if (tool === "slot") {
        const sr = ((state.data.slotRadius || 2.5) / L.w) * L.plotW;
        ctx.lineCap = "round";
        ctx.lineWidth = Math.max(4, sr * 2);
        ctx.strokeStyle = "rgba(239, 68, 68, 0.35)";
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
      }
      ctx.restore();
    }

    // Plot border
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 1.2;
    ctx.strokeRect(PAD_L, PAD_T, L.plotW, L.plotH);

    // Crosshair + coordinate pill
    if (state.hoverWorld) {
      const cp = worldToPx(state.hoverWorld.x, state.hoverWorld.y, L);
      ctx.save();
      ctx.strokeStyle = "rgba(15, 23, 42, 0.45)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(cp.x, PAD_T);
      ctx.lineTo(cp.x, PAD_T + L.plotH);
      ctx.moveTo(PAD_L, cp.y);
      ctx.lineTo(PAD_L + L.plotW, cp.y);
      ctx.stroke();

      const label = `(${state.hoverWorld.x.toFixed(1)}, ${state.hoverWorld.y.toFixed(1)})`;
      ctx.font = "600 11px -apple-system, BlinkMacSystemFont, sans-serif";
      const tw = ctx.measureText(label).width + 10;
      const th = 20;
      const bx = cp.x + tw + 14 > PAD_L + L.plotW ? cp.x - tw - 8 : cp.x + 8;
      const by = cp.y - th - 8 < PAD_T ? cp.y + 8 : cp.y - th - 8;
      ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
      ctx.fillRect(bx, by, tw, th);
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(label, bx + 5, by + th / 2);
      ctx.restore();
    }

    ctx.restore();
    updateToolbar();
  }

  if (!state.bound) {
    state.bound = true;

    canvas.addEventListener("mousemove", (e) => {
      const L = getLayout();
      state.hoverWorld = pxToWorld(e.clientX, e.clientY, L);
      if (state.mouseDownPos && state.anchor) {
        const dist = Math.hypot(e.clientX - state.mouseDownPos.x, e.clientY - state.mouseDownPos.y);
        if (dist > 6) state.isDragging = true;
      }
      redraw();
    });

    canvas.addEventListener("mouseleave", () => {
      state.hoverWorld = null;
      redraw();
    });

    canvas.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      if (state.data.tool === "polygon" && state.polyPts.length > 0) {
        state.polyPts.pop();
        redraw();
      } else if (state.anchor) {
        state.anchor = null;
        state.isDragging = false;
        redraw();
      }
    });

    canvas.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      const L = getLayout();
      const pt = pxToWorld(e.clientX, e.clientY, L);
      if (!pt) return;
      const tool = state.data.tool;

      if (tool === "polygon") {
        if (state.polyPts.length >= 3) {
          const p0 = worldToPx(state.polyPts[0].x, state.polyPts[0].y, L);
          const pc = worldToPx(pt.x, pt.y, L);
          if (Math.hypot(pc.x - p0.x, pc.y - p0.y) <= 12) {
            commitPolygon();
            return;
          }
        }
        state.polyPts.push({ x: pt.x, y: pt.y });
        redraw();
        return;
      }

      if (!state.anchor) {
        state.anchor = { x: pt.x, y: pt.y };
        state.mouseDownPos = { x: e.clientX, y: e.clientY };
        state.isDragging = false;
        redraw();
      } else {
        finishTwoPointShape(pt);
      }
    });

    canvas.addEventListener("mouseup", (e) => {
      if (e.button !== 0) return;
      if (state.isDragging && state.anchor && state.data.tool !== "polygon") {
        const L = getLayout();
        const pt = pxToWorld(e.clientX, e.clientY, L);
        if (pt) finishTwoPointShape(pt);
      }
      state.mouseDownPos = null;
      state.isDragging = false;
    });

    canvas.addEventListener("dblclick", (e) => {
      e.preventDefault();
      if (state.data.tool === "polygon" && state.polyPts.length >= 3) {
        // Remove duplicate vertex from second click if identical to previous
        if (state.polyPts.length >= 4) {
          const last = state.polyPts[state.polyPts.length - 1];
          const prev = state.polyPts[state.polyPts.length - 2];
          if (Math.hypot(last.x - prev.x, last.y - prev.y) < 1e-6) {
            state.polyPts.pop();
          }
        }
        if (state.polyPts.length >= 3) {
          commitPolygon();
        }
      }
    });

    function finishTwoPointShape(pt) {
      const a = state.anchor;
      const tool = state.data.tool;
      state.anchor = null;
      state.isDragging = false;
      if (!a) return;

      if (tool === "circle") {
        const r = Math.hypot(pt.x - a.x, pt.y - a.y);
        if (r > 0.25) {
          emitAction({ type: "circle", cx: a.x, cy: a.y, r });
        }
      } else if (tool === "rect") {
        const x = Math.min(a.x, pt.x);
        const y = Math.min(a.y, pt.y);
        const w = Math.abs(pt.x - a.x);
        const h = Math.abs(pt.y - a.y);
        if (w > 0.25 && h > 0.25) {
          emitAction({
            type: "rect",
            x,
            y,
            w,
            h,
            fillet_radius: state.data.filletRadius || 0,
          });
        }
      } else if (tool === "slot") {
        const len = Math.hypot(pt.x - a.x, pt.y - a.y);
        if (len > 0.25) {
          emitAction({
            type: "slot",
            x1: a.x,
            y1: a.y,
            x2: pt.x,
            y2: pt.y,
            r: state.data.slotRadius || 2.5,
          });
        }
      }
      redraw();
    }

    finishBtn.addEventListener("click", () => commitPolygon());
    undoPtBtn.addEventListener("click", () => {
      state.polyPts.pop();
      redraw();
    });
    clearPtBtn.addEventListener("click", () => {
      state.polyPts = [];
      state.anchor = null;
      redraw();
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (state.polyPts.length > 0 || state.anchor) {
          state.polyPts = [];
          state.anchor = null;
          state.isDragging = false;
          redraw();
        }
      } else if (e.key === "Enter" && state.data.tool === "polygon" && state.polyPts.length >= 3) {
        commitPolygon();
      } else if (e.key === "Backspace" && state.data.tool === "polygon" && state.polyPts.length > 0) {
        state.polyPts.pop();
        redraw();
      }
    });
  }

  redraw();
}
"""

_components_v2: Any = getattr(st.components, "v2")
CAD_CANVAS_COMPONENT = _components_v2.component(
    "cad_interactive_canvas",
    html=_CAD_CANVAS_HTML,
    css=_CAD_CANVAS_CSS,
    js=_CAD_CANVAS_JS,
)


def build_canvas_payload(
    problem: DesignProblem,
    model: CADModel,
    *,
    tool: str,
    fillet_radius: float,
    slot_radius: float,
    grid_snap: float,
    problem_key: str,
) -> dict[str, Any]:
    """Serializes the current CADModel and problem overlays for the JS crosshair canvas."""
    polygons: list[dict[str, Any]] = []
    for poly in model._extract_polygons(model.current_geometry):
        outer = [[float(x), float(y)] for x, y in poly.exterior.coords[:-1]]
        holes = [
            [[float(x), float(y)] for x, y in interior.coords[:-1]]
            for interior in poly.interiors
        ]
        if len(outer) >= 3:
            polygons.append({"outer": outer, "holes": holes})

    keepouts = [
        {
            "shape": ko.shape,
            "params": [float(v) for v in ko.params],
            "type": ko.keepout_type.value,
        }
        for ko in problem.keepouts
    ]
    supports = [{"x": float(s.x), "y": float(s.y)} for s in problem.supports]
    loads = [
        {
            "x": float(ld.x),
            "y": float(ld.y),
            "fx": float(ld.fx),
            "fy": float(ld.fy),
        }
        for ld in problem.all_loads
    ]
    return {
        "problemKey": problem_key,
        "width": float(problem.width),
        "height": float(problem.height),
        "tool": tool,
        "filletRadius": float(fillet_radius),
        "slotRadius": float(slot_radius),
        "gridSnap": float(grid_snap),
        "polygons": polygons,
        "keepouts": keepouts,
        "supports": supports,
        "loads": loads,
    }


def apply_canvas_action(
    model: CADModel, action_payload: dict[str, Any]
) -> tuple[bool, str | None]:
    """Applies a completed interactive canvas cutout action to a CADModel."""
    action_type = str(action_payload.get("type", ""))
    if action_type == "polygon":
        raw_pts = action_payload.get("points", [])
        pts = [(float(p[0]), float(p[1])) for p in raw_pts if len(p) >= 2]
        fillet = float(action_payload.get("fillet_radius", 0.0))
        if model.cut_polygon(pts, fillet_radius=fillet):
            return True, None
        return (
            False,
            "Polygon cutout rejected (self-intersecting, outside material, or inside solid keepout).",
        )
    if action_type == "circle":
        cx = float(action_payload["cx"])
        cy = float(action_payload["cy"])
        r = max(float(action_payload["r"]), float(model.problem.min_feature_radius))
        if math.isfinite(r) and model.cut_circle(cx, cy, r):
            return True, None
        return False, "Cutout rejected (outside material or inside solid keepout)."
    if action_type == "rect":
        x = float(action_payload["x"])
        y = float(action_payload["y"])
        w = float(action_payload["w"])
        h = float(action_payload["h"])
        fillet = float(action_payload.get("fillet_radius", 0.0))
        if model.cut_rect(x, y, w, h, fillet_radius=fillet):
            return True, None
        return False, "Cutout rejected (outside material or inside solid keepout)."
    if action_type == "slot":
        x1 = float(action_payload["x1"])
        y1 = float(action_payload["y1"])
        x2 = float(action_payload["x2"])
        y2 = float(action_payload["y2"])
        r = max(float(action_payload["r"]), float(model.problem.min_feature_radius))
        if model.cut_slot(x1, y1, x2, y2, r):
            return True, None
        return (
            False,
            "Cutout rejected (degenerate slot, outside material, or inside solid keepout).",
        )
    return False, f"Unknown canvas action type: {action_type}"
