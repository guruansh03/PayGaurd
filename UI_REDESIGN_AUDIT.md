# PayGuard — UI Redesign Audit
**Date:** 2026-05-14
**Status:** ✅ IMPLEMENTED — all items below applied to app.py, analysis.py, visualization.py, tests
**Replaces:** UPI Shield branding and current app.py UI

---

## Decisions locked

| # | Decision | Value |
|---|----------|-------|
| D1 | New name | PayGuard |
| D2 | Feel | Monitoring tool (always-on, live feed) — NOT upload-and-explore |
| D3 | Sidebar | Keep all pages, collapse better |
| D4 | Dashboard hero | Big live score feed (transactions rolling) + One big risk gauge + recent flags |
| D5 | Color | Bold and modern — white + strong accent |
| D6 | Current problems | Too many sidebar pages visible, cluttered dashboard, too technical/boring |

---

## Current problems (from user)

1. **Sidebar bloat** — 6 pages all visible at once (Dashboard, Upload & Scan, Live Score, Model Metrics, Visualizations, About). Feels like a settings menu, not a product.
2. **Dashboard clutter** — Model Comparison table, Fraud Pattern Breakdown, Score Distribution all competing at same visual weight. No clear hierarchy.
3. **Too technical/boring** — Space Mono everywhere, red/black only palette, raw metric labels, no personality.
4. **Wrong mental model** — Currently feels like an analysis tool (run models, inspect outputs). Should feel like a monitoring dashboard (things are happening, you're watching).

---

## Proposed structure

### Sidebar
- Collapse to icons by default, expand on hover (CSS only in Streamlit — use `st.sidebar` width trick)
- 5 pages max, grouped:
  - 🏠 Overview (rename Dashboard)
  - 📡 Live Score
  - 📤 Scan (rename Upload & Scan)
  - 📊 Analytics (merge Model Metrics + Visualizations)
  - ℹ️ About
- Remove standalone Visualizations page — fold charts into Analytics
- Active page indicator: left border accent, not background fill

### Dashboard / Overview page
**Hero zone (top 40% of screen):**
- Left: Single large risk gauge (Plotly indicator, 0–1 scale, colored zones: green <0.4, amber 0.4–0.7, red >0.7) showing current ensemble threshold position
- Right: Live transaction feed — scrolling table of last 20 scored transactions with score badge, pattern tag, timestamp. Auto-refreshes every 30s via `st.rerun()`.

**Below hero:**
- Row of 4 KPI cards: Transactions Today / Flagged / Flag Rate / Top Pattern
- Fraud Pattern Breakdown bar chart (keep, just make it smaller, horizontal, cleaner)
- Score Distribution (keep, move to Analytics page or make collapsible)

**Remove from Dashboard:**
- Model Comparison table → move to Analytics
- Raw confusion matrix → move to Analytics

### Live Score page
- Cleaner form layout: 2-column grid instead of single column
- Advanced overrides expander stays but cleaner labels
- Score result: bigger verdict banner, less raw numbers upfront
- Add: pattern explanation in plain English below score (not just SHAP tags)

### Analytics page (merged Metrics + Visualizations)
- Tabbed: Model Performance | SHAP | Charts
- Model Comparison table here (not Dashboard)
- Confusion matrix here
- Score distribution here
- 5-fold CV results here

### Scan page (Upload & Scan)
- Simplified: drag-drop zone prominent, model selector below
- Results table: cleaner, flagged rows highlighted with left border accent (not full red bg)

---

## Color system

**Accent:** Indigo `#4C3EE8` (primary actions, badges, gauge fill)
**Danger:** `#E03E3E` (fraud flags, high scores)
**Warning:** `#E8900A` (medium scores, caution)
**Success:** `#1D7A4F` (normal, low scores)
**Background:** `#FFFFFF` page, `#F7F8FC` sidebar + cards
**Text primary:** `#0F1117`
**Text secondary:** `#6B7280`
**Border:** `#E5E7EB` (0.5px)

**Replace entirely:**
- Current `#e63946` red accent → Indigo `#4C3EE8`
- Current `#0f1117` dark bg → White `#FFFFFF`
- Current Space Mono everywhere → Inter for UI, Space Mono only for scores/numbers

---

## Typography

| Element | Font | Size | Weight |
|---------|------|------|--------|
| Page title | Inter | 24px | 500 |
| Section header | Inter | 14px uppercase | 500 |
| Body / labels | Inter | 14px | 400 |
| KPI numbers | Space Mono | 28px | 700 |
| Score badge | Space Mono | 16px | 500 |
| Table data | Inter | 13px | 400 |

---

## Branding changes

| Location | Current | New |
|----------|---------|-----|
| Sidebar title | 🛡️ UPI Shield | 🔷 PayGuard |
| Sidebar subtitle | Fraud Detection System | Real-time UPI fraud monitoring |
| Page title (browser tab) | UPI Shield | PayGuard |
| About page header | About UPI Shield | About PayGuard |
| Dataset description | "UPI Shield Fraud Detection" | "PayGuard — 293k UPI transactions" |
| Footer | "Unsupervised ML · 293k UPI transactions" | "PayGuard · Unsupervised ML · 293k transactions" |

---

## Files to change

| File | Changes |
|------|---------|
| `app.py` | Full UI overhaul — sidebar, Dashboard, Analytics merge, branding |
| `config.py` | No changes needed |
| `src/` files | No changes needed |

---

## Implementation order

1. Branding find-replace (UPI Shield → PayGuard) — 10 min
2. CSS overhaul — replace dark theme with white + indigo — 30 min
3. Sidebar restructure — collapse, rename, reorder pages — 20 min
4. Dashboard hero — add gauge + live feed, demote table — 45 min
5. Analytics page — merge Metrics + Visualizations, add tabs — 30 min
6. Live Score form — 2-col layout, cleaner verdict — 20 min
7. Scan page — drag-drop prominence, table cleanup — 20 min

**Total estimated:** ~3 hours coding

---

## Open questions before coding

| # | Question |
|---|----------|
| Q1 | Live feed on Dashboard — simulate with existing anomaly_scores.csv data or add real `st.rerun()` loop? |
| Q2 | Gauge on Dashboard — show current threshold position or last-batch top score? |
| Q3 | Sidebar collapse — full icon-only collapse or just tighter spacing? Streamlit sidebar width is limited. |
| Q4 | Analytics tabs — keep 5-fold CV results visible or hide behind expander? |

---

*Audit complete. No code changed. Resume implementation in new conversation.*
