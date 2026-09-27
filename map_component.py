"""
Zoom-preserving Plotly map for Streamlit.

st.plotly_chart hashes the entire figure JSON into its element ID (even when a
key is given), so every data change — a date-slider move, an auto-play step —
remounts the chart and throws away the user's pan/zoom before Plotly's
uirevision can apply. This component keeps a single Plotly div alive per key
and updates it in place with Plotly.react, carrying the user's view edits
forward, so zoom survives date changes and even a remount (e.g. toggling the
control panel). County clicks come back to Python as a one-shot trigger value.

plotly.js and the county GeoJSON are served once from ./static (Streamlit
static serving, enabled in .streamlit/config.toml) rather than being resent
with every update; the plotly.js CDN is the fallback when static serving is off.
"""

import json
import os
import shutil
from pathlib import Path

import plotly
import streamlit as st
from plotly.offline import get_plotlyjs_version
from streamlit.components.v2.get_bidi_component_manager import get_bidi_component_manager

STATIC_DIR = Path(__file__).parent / "static"
_STATIC_URL = "app/static/"
_PLOTLY_JS_VERSION = get_plotlyjs_version()
_PLOTLY_JS_NAME = f"plotly-{_PLOTLY_JS_VERSION}.min.js"
_PLOTLY_CDN_URL = f"https://cdn.plot.ly/plotly-{_PLOTLY_JS_VERSION}.min.js"
_GEOJSON_NAME = "geojson-counties-fips.json"

_JS = r"""
const CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d"],
};

// The user's geo view edits (from plotly_relayout), per component key. Kept at
// module scope so a remounted chart restores the view; reset whenever the
// Python-side view key changes (metric / filter / scale), which is the
// intended "start from the default view" signal.
const views = new Map();
let plotlyLoading = null;

function viewFor(key, viewKey) {
  let view = views.get(key);
  if (!view || view.viewKey !== viewKey) {
    view = { viewKey, edits: {} };
    views.set(key, view);
  }
  return view;
}

function applyEdits(layout, edits) {
  for (const [path, value] of Object.entries(edits)) {
    const parts = path.split(".");
    let node = layout;
    for (const part of parts.slice(0, -1)) {
      if (!node[part] || typeof node[part] !== "object") node[part] = {};
      node = node[part];
    }
    node[parts[parts.length - 1]] = value;
  }
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = () =>
      window.Plotly ? resolve(window.Plotly) : reject(new Error(`no Plotly in ${src}`));
    script.onerror = () => {
      script.remove();
      reject(new Error(`could not load ${src}`));
    };
    document.head.appendChild(script);
  });
}

// Try each source in order (local static copy first, then the CDN).
function loadPlotly(sources) {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  if (!plotlyLoading) {
    plotlyLoading = sources.reduce(
      (prev, src) => prev.catch(() => loadScript(src)),
      Promise.reject(new Error("no plotly.js source"))
    );
    plotlyLoading.catch(() => { plotlyLoading = null; });
  }
  return plotlyLoading;
}

export default function ({ data, parentElement, setTriggerValue }) {
  // Streamlit calls this again on every data change with the same parent, so
  // the Plotly div is created once and then updated in place.
  let el = parentElement.querySelector(":scope > .geo-map");
  if (!el) {
    el = document.createElement("div");
    el.className = "geo-map";
    el.style.width = "100%";
    parentElement.appendChild(el);
  }
  const fig = JSON.parse(data.figure);
  el.style.height = `${fig.layout.height || 600}px`;
  el._latest = fig;
  el._key = data.key;
  el._viewKey = data.view_key;
  el._setTrigger = setTriggerValue;

  loadPlotly(data.plotly_src)
    .then((Plotly) => {
      el._queue = (el._queue || Promise.resolve())
        .then(() => {
          // Skip frames superseded by a newer update, or drawn after unmount
          if (el._latest !== fig || !el.isConnected) return;
          applyEdits(fig.layout, viewFor(el._key, el._viewKey).edits);
          const firstDraw = !el._plotted;
          return Plotly.react(el, fig.data, fig.layout, CONFIG).then(() => {
            if (!firstDraw) return;
            el._plotted = true;
            el.on("plotly_click", (ev) => {
              const pt = ev && ev.points && ev.points[0];
              if (!pt) return;
              el._setTrigger("click", {
                location: pt.location ?? null,
                customdata: pt.customdata ?? null,
              });
            });
            el.on("plotly_relayout", (ev) => {
              const view = viewFor(el._key, el._viewKey);
              for (const [path, value] of Object.entries(ev || {})) {
                if (path.startsWith("geo")) view.edits[path] = value;
              }
            });
            el._resizeObserver = new ResizeObserver(() => Plotly.Plots.resize(el));
            el._resizeObserver.observe(el);
          });
        })
        .catch((err) => console.error("geo_map:", err));
    })
    .catch((err) => {
      el.textContent = `Map could not load (${err.message}).`;
    });

  return () => {
    if (el._resizeObserver) el._resizeObserver.disconnect();
    if (window.Plotly) window.Plotly.purge(el);
  };
}
"""

_COMPONENT_NAME = "geo_map"
_mount_component = None


def _geo_map_component():
    """Register the component on first use, and again if the runtime's
    registry was replaced (AppTest builds a fresh runtime for every run)."""
    global _mount_component
    if _mount_component is None or get_bidi_component_manager().get(_COMPONENT_NAME) is None:
        _mount_component = st.components.v2.component(
            _COMPONENT_NAME,
            js=_JS,
            # Plotly injects its modebar/hover CSS into document.head, which a
            # shadow root would block.
            isolate_styles=False,
        )
    return _mount_component


def _write_atomic(path: Path, write) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write(tmp)
    os.replace(tmp, path)


@st.cache_resource(show_spinner=False)
def _publish_static_assets(_geojson, geojson_available: bool) -> dict:
    """
    Copy plotly.js (matching the installed plotly.py) and the county GeoJSON
    into ./static once per server process. Returns the plotly.js sources to
    try in order and the GeoJSON URL (None when it must stay inline).
    """
    assets = {"plotly_src": [_PLOTLY_CDN_URL], "geojson_url": None}
    if not st.get_option("server.enableStaticServing"):
        return assets
    try:
        STATIC_DIR.mkdir(exist_ok=True)
        js_path = STATIC_DIR / _PLOTLY_JS_NAME
        if not js_path.exists():
            bundled = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
            _write_atomic(js_path, lambda p: shutil.copyfile(bundled, p))
        assets["plotly_src"] = [_STATIC_URL + _PLOTLY_JS_NAME, _PLOTLY_CDN_URL]

        if geojson_available:
            geo_path = STATIC_DIR / _GEOJSON_NAME
            if not geo_path.exists():
                _write_atomic(
                    geo_path,
                    lambda p: p.write_text(json.dumps(_geojson), encoding="utf-8"),
                )
            assets["geojson_url"] = _STATIC_URL + _GEOJSON_NAME
    except OSError:
        pass  # read-only checkout: fall back to the CDN / inline GeoJSON
    return assets


def render_geo_map(fig, *, key: str, view_key: str, geojson=None):
    """
    Render a Plotly geo figure that keeps the user's pan/zoom across updates.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The map figure. Its traces' GeoJSON is swapped for a static URL when
        one is available, so the boundaries are downloaded once per browser.
    key : str
        Stable Streamlit key — the chart's identity across reruns.
    view_key : str
        Changes whenever the default view should be restored (metric, filter,
        scale). While it stays the same, the user's view is preserved.
    geojson : dict or str or None
        The GeoJSON the figure was built with (dict to publish, URL to keep).

    Returns
    -------
    dict or None
        ``{"location": ..., "customdata": [...]}`` on the run triggered by a
        county click, otherwise None.
    """
    assets = _publish_static_assets(geojson, isinstance(geojson, dict))
    if assets["geojson_url"]:
        fig.update_traces(geojson=assets["geojson_url"])

    # st.plotly_chart would apply Streamlit's theme; outside it, use a neutral
    # template and a light hover label so the map keeps the same look.
    fig.update_layout(
        template="plotly_white",
        hoverlabel=dict(bgcolor="white", bordercolor="#D0D5DD",
                        font=dict(color="#31333F")),
    )

    result = _geo_map_component()(
        key=key,
        data={
            "figure": fig.to_json(),
            "key": key,
            "view_key": view_key,
            "plotly_src": assets["plotly_src"],
        },
        on_click_change=lambda: None,
    )
    return getattr(result, "click", None)
