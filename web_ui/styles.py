"""Web UI style definitions."""

from __future__ import annotations

# Color theme constants
COLORS = {
    "primary": "#21409a",
    "accent": "#d94a2d",
    "accent_soft": "#edf3ff",
    "text_main": "#12203d",
    "text_sub": "#5a6987",
    "border": "#d9e2f4",
    "card_bg": "rgba(255,255,255,0.9)",
    "page_bg": "linear-gradient(180deg, #f3f7fd 0%, #ebf1fb 52%, #e5edf9 100%)",
}

# Main CSS styles
APP_CSS = """
:root {
  --page-bg: linear-gradient(180deg, #f3f7fd 0%, #ebf1fb 52%, #e5edf9 100%);
  --page-glow-a: rgba(48, 86, 179, 0.16);
  --page-glow-b: rgba(217, 74, 45, 0.10);
  --card-bg: rgba(255,255,255,0.9);
  --border-color: #d9e2f4;
  --text-main: #12203d;
  --text-sub: #5a6987;
  --accent: #21409a;
  --accent-strong: #152a6a;
  --accent-soft: #edf3ff;
  --warm-accent: #d94a2d;
  --font-sans: "Avenir Next", "Segoe UI Variable", "Segoe UI", "Noto Sans SC",
    "Noto Sans CJK SC", "Source Han Sans SC", "Source Han Sans CN", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei", "Arial Unicode MS", "Helvetica Neue", Arial, sans-serif;
}
html, body {
  background: var(--page-bg);
}
.gradio-container {
  --block-background-fill: rgba(255,255,255,0.78);
  --block-border-color: rgba(166, 184, 224, 0.58);
  --block-border-width: 1px;
  --block-label-background-fill: rgba(248,251,255,0.96);
  --block-label-border-color: rgba(166, 184, 224, 0.44);
  --block-label-text-color: var(--text-main);
  --block-radius: 12px;
  --block-shadow: none;
  --border-color-primary: var(--border-color);
  --button-border-width: 1px;
  --button-large-radius: 10px;
  --button-medium-radius: 10px;
  --button-primary-background-fill: var(--accent);
  --button-primary-background-fill-hover: var(--accent-strong);
  --button-primary-border-color: rgba(33, 64, 154, 0.18);
  --button-primary-border-color-hover: rgba(33, 64, 154, 0.28);
  --button-secondary-background-fill: rgba(255,255,255,0.92);
  --button-secondary-background-fill-hover: rgba(245,248,255,0.98);
  --button-secondary-border-color: rgba(166, 184, 224, 0.52);
  --button-secondary-border-color-hover: rgba(33, 64, 154, 0.30);
  --button-secondary-shadow: none;
  --button-secondary-shadow-hover: 0 8px 18px rgba(25, 40, 78, 0.08);
  --button-secondary-text-color: var(--text-main);
  --checkbox-border-color: rgba(166, 184, 224, 0.78);
  --checkbox-border-color-focus: rgba(33, 64, 154, 0.58);
  --checkbox-border-color-hover: rgba(33, 64, 154, 0.34);
  --checkbox-border-color-selected: var(--accent);
  --checkbox-border-radius: 6px;
  --input-background-fill: rgba(255,255,255,0.96);
  --input-background-fill-focus: #ffffff;
  --input-border-color: rgba(166, 184, 224, 0.72);
  --input-border-color-focus: rgba(33, 64, 154, 0.52);
  --input-border-color-hover: rgba(33, 64, 154, 0.28);
  --input-border-width: 1px;
  --input-radius: 10px;
  --input-shadow: inset 0 1px 0 rgba(255,255,255,0.78);
  --input-shadow-focus: 0 0 0 3px rgba(33, 64, 154, 0.10);
  --panel-border-color: rgba(166, 184, 224, 0.44);
  --panel-border-width: 1px;
  --table-border-color: rgba(166, 184, 224, 0.50);
  background:
    radial-gradient(circle at top left, var(--page-glow-a), transparent 30%),
    radial-gradient(circle at top right, var(--page-glow-b), transparent 28%),
    var(--page-bg);
  font-family: var(--font-sans) !important;
  color: var(--text-main);
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

.gradio-container, .gradio-container * {
  font-family: var(--font-sans) !important;
}
.gradio-container .block,
.gradio-container .group,
.gradio-container .form,
.gradio-container .panel,
.gradio-container fieldset {
  border-color: rgba(166, 184, 224, 0.44) !important;
  box-shadow: none !important;
}
.block-title h1, .block-title h2, .block-title h3 {
  color: var(--text-main);
}
.hero {
  position: relative;
  overflow: hidden;
  background:
    linear-gradient(
      135deg,
      rgba(255,255,255,0.96) 0%,
      rgba(236,243,255,0.98) 48%,
      rgba(227,236,255,0.94) 100%
    );
  color: var(--text-main);
  border-radius: 28px;
  padding: 26px 30px;
  margin-bottom: 16px;
  box-shadow: 0 24px 54px rgba(28, 53, 114, 0.12);
  border: 1px solid rgba(113, 143, 226, 0.24);
}
.hero::before,
.hero::after {
  content: "";
  position: absolute;
  border-radius: 999px;
  pointer-events: none;
}
.hero::before {
  width: 260px;
  height: 260px;
  right: -90px;
  top: -120px;
  background: radial-gradient(circle, rgba(33, 64, 154, 0.18) 0%, rgba(33, 64, 154, 0) 68%);
}
.hero::after {
  width: 180px;
  height: 180px;
  left: -40px;
  bottom: -90px;
  background: radial-gradient(circle, rgba(217, 74, 45, 0.12) 0%, rgba(217, 74, 45, 0) 72%);
}
.hero-brand {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  gap: 22px;
}
.hero-copy {
  max-width: 920px;
}
.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  color: var(--accent-strong);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.progress-text[data-runtime-text] {
  font-size: 0 !important;
  line-height: 0 !important;
}
.progress-text[data-runtime-text]::before {
  content: attr(data-runtime-text);
  color: var(--body-text-color);
  font-size: 13px;
  line-height: 1.4;
}
.hero-kicker::before {
  content: "";
  width: 34px;
  height: 1px;
  background: linear-gradient(90deg, var(--warm-accent) 0%, rgba(217, 74, 45, 0.2) 100%);
}
.hero h1 {
  margin: 0 0 10px 0;
  font-size: clamp(30px, 4vw, 42px);
  line-height: 1.05;
  font-weight: 800;
  letter-spacing: -0.04em;
  color: var(--text-main);
}
.hero-logo {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: 0 0 84px;
  width: 84px;
  height: 84px;
  border-radius: 26px;
  background: linear-gradient(180deg, rgba(255,255,255,0.88) 0%, rgba(221,233,255,0.98) 100%);
  box-shadow:
    0 18px 34px rgba(33, 64, 154, 0.18),
    inset 0 1px 0 rgba(255,255,255,0.9);
  border: 1px solid rgba(109, 141, 247, 0.26);
}
.hero-logo svg {
  width: 62px;
  height: 62px;
  overflow: visible;
}
.hero-logo .compass-needle {
  transform-box: fill-box;
  transform-origin: center;
  animation: compass-sway 6s ease-in-out infinite;
}
.hero p {
  margin: 0;
  max-width: 860px;
  font-size: 15px;
  line-height: 1.8;
  color: #43526f;
}
.op-table-wide-device th:nth-child(6),
.op-table-wide-device td:nth-child(6) {
  min-width: 120px;
}
.op-table-narrow-name th:nth-child(1),
.op-table-narrow-name td:nth-child(1) {
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.section-card {
  background: var(--card-bg);
  border: 1px solid var(--border-color);
  border-radius: 20px;
  padding: 14px 16px 10px 16px;
  box-shadow: 0 14px 28px rgba(25, 40, 78, 0.08);
  margin-bottom: 12px;
}
.recommendation-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(237,243,255,0.92) 100%);
  border: 1px solid rgba(33, 64, 154, 0.18);
  border-left: 4px solid var(--accent);
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 12px;
  box-shadow: 0 10px 22px rgba(25, 40, 78, 0.07);
}
.recommendation-card h3,
.recommendation-card p {
  margin-top: 0;
}
.section-card .wrap table,
.section-card .table-wrap table {
  font-size: 12px;
}
.section-card .dataframe,
.section-card .table-wrap {
  min-height: 0 !important;
}
.section-card h2 {
  margin: 0 0 8px 0;
  font-size: 20px;
  font-weight: 700;
  color: var(--text-main);
}
.section-card p {
  margin: 0;
  color: var(--text-sub);
  font-size: 13px;
  line-height: 1.7;
}
.field-hint, .field-hint p {
  color: var(--text-sub) !important;
  font-size: 12px !important;
  line-height: 1.6;
  margin-top: -4px !important;
}
.preview-summary {
  margin-top: 8px !important;
  margin-bottom: 6px !important;
}
.preview-summary h3 {
  font-size: 15px !important;
  line-height: 1.25 !important;
  margin: 4px 0 6px 0 !important;
}
.preview-summary p,
.preview-summary li {
  font-size: 12px !important;
  line-height: 1.45 !important;
}
.memory-analysis-row {
  align-items: stretch;
}
.memory-analysis-row .plot,
.memory-analysis-row .table-wrap,
.memory-analysis-row .dataframe {
  min-height: 360px !important;
}
.memory-table .table-wrap {
  max-height: 360px !important;
}
.sim-mode-tabs button[role="tab"] {
  min-height: 50px;
  padding: 0 24px;
  margin-right: 14px;
  border-radius: 16px;
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.01em;
}
.sim-mode-tabs [role="tablist"] {
  gap: 14px;
  margin-bottom: 16px;
}
.sim-mode-tabs button[role="tab"][aria-selected="true"] {
  box-shadow: 0 10px 20px rgba(33, 64, 154, 0.12);
}
.gradio-container [role="tablist"] {
  border-color: rgba(166, 184, 224, 0.40) !important;
}
.gradio-container button[role="tab"] {
  border-color: transparent !important;
  color: var(--text-sub) !important;
}
.gradio-container button[role="tab"][aria-selected="true"] {
  background: rgba(237,243,255,0.92) !important;
  color: var(--accent-strong) !important;
  border-color: rgba(166, 184, 224, 0.44) !important;
}
.gradio-container details,
.gradio-container .accordion {
  border: 1px solid rgba(166, 184, 224, 0.56) !important;
  border-radius: 12px !important;
  background: rgba(255,255,255,0.74) !important;
  box-shadow: none !important;
  overflow: hidden;
}
.gradio-container details > summary,
.gradio-container .accordion > .label-wrap {
  min-height: 36px;
  border: 0 !important;
  background: rgba(248,251,255,0.94) !important;
  box-shadow: none !important;
  color: var(--text-main) !important;
}
.gradio-container details[open] > summary,
.gradio-container .accordion.open > .label-wrap {
  border-bottom: 1px solid rgba(166, 184, 224, 0.38) !important;
}
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea,
.gradio-container select {
  border-color: rgba(166, 184, 224, 0.72) !important;
  border-radius: 10px !important;
  background-color: rgba(255,255,255,0.96) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.78) !important;
}
.gradio-container input[type="text"]:focus,
.gradio-container input[type="number"]:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
  border-color: rgba(33, 64, 154, 0.50) !important;
  box-shadow: 0 0 0 3px rgba(33, 64, 154, 0.10) !important;
}
.gradio-container input[type="checkbox"],
.gradio-container input[type="radio"] {
  border-color: rgba(166, 184, 224, 0.78) !important;
  box-shadow: none !important;
}
.gradio-container input[type="checkbox"]:checked,
.gradio-container input[type="radio"]:checked {
  border-color: var(--accent) !important;
  background-color: var(--accent) !important;
}
.gradio-container button:not([role="tab"]) {
  border-color: rgba(166, 184, 224, 0.52) !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}
.gradio-container button:not([role="tab"]):hover {
  border-color: rgba(33, 64, 154, 0.30) !important;
  box-shadow: 0 8px 18px rgba(25, 40, 78, 0.08) !important;
}
.gradio-container .primary,
.gradio-container button.primary {
  border-color: rgba(33, 64, 154, 0.18) !important;
  background: var(--accent) !important;
  color: #ffffff !important;
}
.gradio-container .primary:hover,
.gradio-container button.primary:hover {
  border-color: rgba(33, 64, 154, 0.28) !important;
  background: var(--accent-strong) !important;
}
.gradio-container .table-wrap,
.gradio-container .dataframe {
  border-color: rgba(166, 184, 224, 0.50) !important;
  border-radius: 10px !important;
}
.gradio-container .table-wrap table,
.gradio-container .dataframe table {
  border-color: rgba(166, 184, 224, 0.44) !important;
}
.gradio-container .table-wrap th,
.gradio-container .table-wrap td,
.gradio-container .dataframe th,
.gradio-container .dataframe td {
  border-color: rgba(166, 184, 224, 0.36) !important;
}
.progress-shell {
  background: rgba(255,255,255,0.92);
  border: 1px solid var(--border-color);
  border-radius: 16px;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.progress-title {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 8px;
}
.progress-title strong {
  color: var(--text-main);
  font-size: 15px;
}
.progress-title span {
  color: var(--text-sub);
  font-size: 13px;
}
.progress-track {
  width: 100%;
  height: 12px;
  border-radius: 999px;
  background: #e6ebf5;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent) 0%, var(--warm-accent) 100%);
}
.progress-caption {
  margin-top: 8px;
  color: var(--text-sub);
  font-size: 12px;
  line-height: 1.5;
}
@keyframes compass-sway {
  0%,
  100% { transform: rotate(-8deg); }
  50% { transform: rotate(8deg); }
}
@media (max-width: 900px) {
  .hero {
    padding: 22px 20px;
    border-radius: 24px;
  }
  .hero-brand {
    align-items: flex-start;
  }
  .hero-logo {
    flex-basis: 72px;
    width: 72px;
    height: 72px;
    border-radius: 22px;
  }
  .hero-logo svg {
    width: 54px;
    height: 54px;
  }
}
@media (max-width: 640px) {
  .hero-brand {
    flex-direction: column;
    gap: 16px;
  }
  .hero-copy {
    max-width: none;
  }
  .hero-kicker {
    letter-spacing: 0.14em;
  }
  .hero p {
    font-size: 14px;
    line-height: 1.7;
  }
}
/* ASCII chart output - use monospace font for proper alignment */
.ascii-chart-output {
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
  font-size: 13px !important;
  line-height: 1.2 !important;
  letter-spacing: 0 !important;
  white-space: pre !important;
  overflow-x: auto !important;
}
.ascii-chart-output textarea {
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
  font-size: 13px !important;
  line-height: 1.2 !important;
  letter-spacing: 0 !important;
  white-space: pre !important;
  tab-size: 8 !important;
}
"""
