import argparse
import html
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import (
    atomic_write_json,
    atomic_write_text,
    configure_utf8_stdio,
    extract_ytm_video_id,
    find_local_plotly_js,
    get_track_provider,
    is_real_audio_features,
    is_valid_spotify_id,
    load_json_safe,
)


def _safe_json_for_script(obj):
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _format_tooltip_telemetry(info):
    parts = []
    if info.get("year"):
        parts.append(f"Year: {html.escape(str(info['year']), quote=True)}")
    af = info.get("audio_features")
    if is_real_audio_features(af):
        bpm = af.get("bpm")
        eng = af.get("energy")
        val = af.get("valence")
        dance = af.get("danceability")
        metrics = []
        if bpm is not None:
            metrics.append(f"{bpm} BPM")
        if eng is not None:
            metrics.append(f"Energy {eng}")
        if val is not None:
            metrics.append(f"Valence {val}")
        if dance is not None:
            metrics.append(f"Dance {dance}")
        if metrics:
            parts.append("Audio: " + html.escape(" | ".join(metrics), quote=True))
    return "<br>".join(parts)


def generate_html_visualizer(
    track_coords,
    track_clusters,
    track_ids,
    metadata,
    clusters,
    center_coords,
    html_path,
    explained_variance=None,
    track_coords_3d=None,
    center_coords_3d=None,
    explained_variance_3d=None,
    era_trajectory=None,
    offline=False,
    projection_method="pca",
):
    colors = ["#00E676", "#00E5FF", "#E040FB", "#FFEA00", "#FF6D00", "#FF1744", "#7C4DFF"]
    traces_2d = []
    traces_3d = []
    cards_html = []
    max_weight = 1.0

    for c_idx, c_info in enumerate(clusters):
        cid = c_info["cluster_id"]
        color = colors[c_idx % len(colors)]
        raw_vibe = c_info.get("vibe_label", f"Cluster {cid}")
        vibe_esc = html.escape(str(raw_vibe), quote=True)
        weighted_pct = c_info.get("weighted_share_pct", c_info.get("raw_percentage", c_info.get("weight_percentage", 0.0)))
        raw_count = c_info.get("track_count", 0)

        idx_list = [i for i, cl in enumerate(track_clusters) if cl == cid]

        x_vals = [float(track_coords[i][0]) for i in idx_list]
        y_vals = [float(track_coords[i][1]) for i in idx_list]
        if track_coords_3d is not None and len(track_coords_3d) == len(track_coords):
            z_vals = [float(track_coords_3d[i][2]) for i in idx_list]
            x3_vals = [float(track_coords_3d[i][0]) for i in idx_list]
            y3_vals = [float(track_coords_3d[i][1]) for i in idx_list]
        else:
            x3_vals = x_vals
            y3_vals = y_vals
            z_vals = [0.0 for _ in idx_list]

        sizes = []
        opacities = []
        hover_texts = []
        customdata = []

        for i in idx_list:
            tid = str(track_ids[i])
            info = metadata.get(tid, {})
            raw_name = str(info.get("name", "Unknown"))
            raw_artist = str(info.get("artists", "Unknown"))
            raw_genres = str(info.get("genres") or "None")
            raw_playlists = str(info.get("playlists") or "None")

            name = html.escape(raw_name, quote=True)
            artist = html.escape(raw_artist, quote=True)
            genres = html.escape(raw_genres, quote=True)
            playlists = html.escape(raw_playlists, quote=True)
            weight = float(info.get("recency_weight", 1.0))
            if weight > max_weight:
                max_weight = weight

            m_size = min(20, max(5, int(4 + (weight ** 0.6) * 2)))
            m_opacity = min(1.0, max(0.35, round(0.35 + (weight / 30.0), 2)))

            sizes.append(m_size)
            opacities.append(m_opacity)
            sp_id = (
                tid
                if is_valid_spotify_id(tid)
                else (info.get("spotify_id") if is_valid_spotify_id(info.get("spotify_id")) else "")
            )
            vid = extract_ytm_video_id(info.get("video_id") or tid) or ""
            prov = get_track_provider({"id": tid, "provider": info.get("provider"), "spotify_id": sp_id, "video_id": vid})
            customdata.append(
                {
                    "s": f"{raw_name} {raw_artist} {raw_genres}".lower(),
                    "w": round(weight, 3),
                    "id": sp_id,
                    "vid": vid,
                    "p": prov,
                }
            )

            w_tag = (
                "🔥 High Priority (Recent)"
                if weight >= 10.0
                else ("⚡ Medium Priority" if weight >= 3.0 else "💤 Low Priority (Historical)")
            )

            ht = (
                f"<b>{name}</b> - {artist}"
                f"<br>Vibe: {vibe_esc}"
                f"<br>Genres: {genres}"
                f"<br>Playlists: {playlists}"
            )
            af_line = _format_tooltip_telemetry(info)
            if af_line:
                ht += f"<br>{af_line}"
            ht += f"<br>Weight: {weight:.1f} ({w_tag})"
            if sp_id:
                ht += "<br><i>Click dot to open in Spotify</i>"
            elif vid:
                ht += "<br><i>Click dot to open in YouTube Music</i>"
            hover_texts.append(ht)

        cluster_legend_name = f"C{cid}: {raw_vibe} ({weighted_pct}% Taste Share | {raw_count} songs)"

        traces_2d.append(
            {
                "x": x_vals,
                "y": y_vals,
                "mode": "markers",
                "type": "scatter",
                "name": cluster_legend_name,
                "text": hover_texts,
                "customdata": customdata,
                "hoverinfo": "text",
                "marker": {
                    "size": sizes,
                    "color": color,
                    "opacity": opacities,
                    "line": {"color": "#000000", "width": 0.5},
                },
            }
        )
        traces_3d.append(
            {
                "x": x3_vals,
                "y": y3_vals,
                "z": z_vals,
                "mode": "markers",
                "type": "scatter3d",
                "name": cluster_legend_name,
                "text": hover_texts,
                "customdata": customdata,
                "hoverinfo": "text",
                "marker": {
                    "size": [max(3, int(s * 0.55)) for s in sizes],
                    "color": color,
                    "opacity": opacities,
                },
            }
        )

        tel = c_info.get("telemetry") or {}
        tel_badges = []
        if tel.get("avg_bpm") is not None:
            tel_badges.append(f"Avg {html.escape(str(tel['avg_bpm']), quote=True)} BPM")
        if tel.get("avg_energy") is not None:
            tel_badges.append(f"Energy {html.escape(str(tel['avg_energy']), quote=True)}")
        if tel.get("avg_valence") is not None:
            tel_badges.append(f"Valence {html.escape(str(tel['avg_valence']), quote=True)}")
        tel_html = (
            f"<div class='card-metrics'>{' · '.join(tel_badges)}</div>"
            if tel_badges
            else "<div class='card-metrics'>Acoustic telemetry pending</div>"
        )
        anchors_li = "".join(
            f"<li><b>{html.escape(str(a.get('name', '')), quote=True)}</b> — "
            f"{html.escape(str(a.get('artist', '')), quote=True)}</li>"
            for a in (c_info.get("anchor_tracks") or [])[:3]
        )
        cards_html.append(
            f"""
            <div class="cluster-card" style="border-top: 3px solid {color};">
                <div class="card-title"><span class="badge" style="background:{color};">C{cid}</span> {vibe_esc}</div>
                <div class="card-sub">{weighted_pct}% Recency Share · {raw_count} tracks</div>
                {tel_html}
                <ul class="anchors">{anchors_li}</ul>
            </div>
            """
        )

    out_idx = [i for i, cl in enumerate(track_clusters) if cl == -1]
    if out_idx:
        out_texts = [
            f"<b>{html.escape(str(metadata.get(str(track_ids[i]), {}).get('name', 'Unknown')), quote=True)}</b>"
            f" - {html.escape(str(metadata.get(str(track_ids[i]), {}).get('artists', 'Unknown')), quote=True)}"
            f"<br>Vibe: Unclustered / Eclectic"
            for i in out_idx
        ]
        traces_2d.append(
            {
                "x": [float(track_coords[i][0]) for i in out_idx],
                "y": [float(track_coords[i][1]) for i in out_idx],
                "mode": "markers",
                "type": "scatter",
                "name": f"Unclustered / Eclectic ({len(out_idx)})",
                "text": out_texts,
                "hoverinfo": "text",
                "marker": {"size": 6, "color": "#7a7d88", "opacity": 0.42},
            }
        )
        if track_coords_3d is not None and len(track_coords_3d) == len(track_coords):
            traces_3d.append(
                {
                    "x": [float(track_coords_3d[i][0]) for i in out_idx],
                    "y": [float(track_coords_3d[i][1]) for i in out_idx],
                    "z": [float(track_coords_3d[i][2]) for i in out_idx],
                    "mode": "markers",
                    "type": "scatter3d",
                    "name": f"Unclustered / Eclectic ({len(out_idx)})",
                    "text": out_texts,
                    "hoverinfo": "text",
                    "marker": {"size": 3, "color": "#7a7d88", "opacity": 0.42},
                }
            )

    if len(center_coords) > 0:
        cx_vals = [float(pt[0]) for pt in center_coords]
        cy_vals = [float(pt[1]) for pt in center_coords]
        c_texts = [
            f"<b>Cluster {c['cluster_id']} Center:</b> {html.escape(str(c.get('vibe_label', '')), quote=True)}"
            for c in clusters
        ]
        traces_2d.append(
            {
                "x": cx_vals,
                "y": cy_vals,
                "mode": "markers+text",
                "type": "scatter",
                "name": "Cluster Centers",
                "text": [f"C{c['cluster_id']}" for c in clusters],
                "textposition": "top center",
                "hoverinfo": "text",
                "hovertext": c_texts,
                "marker": {
                    "size": 16,
                    "color": "#FFFFFF",
                    "symbol": "diamond",
                    "line": {"color": "#000000", "width": 2},
                },
            }
        )
        if center_coords_3d is not None and len(center_coords_3d) == len(center_coords):
            cx3 = [float(pt[0]) for pt in center_coords_3d]
            cy3 = [float(pt[1]) for pt in center_coords_3d]
            cz3 = [float(pt[2]) for pt in center_coords_3d]
        else:
            cx3, cy3, cz3 = cx_vals, cy_vals, [0.0 for _ in cx_vals]
        traces_3d.append(
            {
                "x": cx3,
                "y": cy3,
                "z": cz3,
                "mode": "markers+text",
                "type": "scatter3d",
                "name": "Cluster Centers",
                "text": [f"C{c['cluster_id']}" for c in clusters],
                "hoverinfo": "text",
                "hovertext": c_texts,
                "marker": {"size": 8, "color": "#FFFFFF", "symbol": "diamond"},
            }
        )

    ev1 = float(explained_variance[0]) * 100.0 if explained_variance and len(explained_variance) >= 1 else 0.0
    ev2 = float(explained_variance[1]) * 100.0 if explained_variance and len(explained_variance) >= 2 else 0.0
    x_title = f"PCA Dimension 1 ({ev1:.1f}% variance)" if ev1 > 0 else "PCA Dimension 1"
    y_title = f"PCA Dimension 2 ({ev2:.1f}% variance)" if ev2 > 0 else "PCA Dimension 2"

    ev3 = float(explained_variance_3d[2]) * 100.0 if explained_variance_3d and len(explained_variance_3d) >= 3 else 0.0
    z_title = f"PCA Dimension 3 ({ev3:.1f}% variance)" if ev3 > 0 else "PCA Dimension 3"
    slider_max = max(20, int(min(100, round(max_weight))))

    if era_trajectory:
        traces_2d.append(
            {
                "x": [e["x"] for e in era_trajectory],
                "y": [e["y"] for e in era_trajectory],
                "mode": "lines+markers+text",
                "type": "scatter",
                "name": "Era Evolution Path",
                "text": [str(e["year"]) for e in era_trajectory],
                "textposition": "top center",
                "hoverinfo": "text",
                "hovertext": [f"<b>{e['year']}</b>: {e['plays']} plays ({e['unique_tracks']} tracks)" for e in era_trajectory],
                "line": {"color": "#FFB300", "width": 3, "dash": "dot"},
                "marker": {"size": 11, "color": "#FFB300", "symbol": "circle-open", "line": {"width": 2, "color": "#FFFFFF"}},
            }
        )
        traces_3d.append(
            {
                "x": [e.get("x3", e["x"]) for e in era_trajectory],
                "y": [e.get("y3", e["y"]) for e in era_trajectory],
                "z": [e.get("z", 0.0) for e in era_trajectory],
                "mode": "lines+markers+text",
                "type": "scatter3d",
                "name": "Era Evolution Path",
                "text": [str(e["year"]) for e in era_trajectory],
                "hoverinfo": "text",
                "hovertext": [f"<b>{e['year']}</b>: {e['plays']} plays" for e in era_trajectory],
                "line": {"color": "#FFB300", "width": 4},
                "marker": {"size": 6, "color": "#FFB300"},
            }
        )

    plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
    if offline:
        local_js = find_local_plotly_js()
        if local_js:
            try:
                with open(local_js, "r", encoding="utf-8", errors="replace") as pf:
                    plotly_tag = f"<script>{pf.read().replace('</script>', '<\\/script>')}</script>"
            except OSError:
                pass

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Music Taste Analyzer - Mathematical Taste Vector Map</title>
    {plotly_tag}
    <style>
        body {{
            background-color: #0d0e15;
            color: #e0e0e0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #1e1e2f 0%, #0d0e15 100%);
            border-radius: 12px;
            margin-bottom: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            border: 1px solid #2a2b3d;
        }}
        h1 {{ color: #00E5FF; margin: 0 0 10px 0; font-size: 26px; letter-spacing: 1px; }}
        p {{ color: #a0a0b0; margin: 0; }}
        .legend-note {{ font-size: 13px; color: #707088; margin-top: 6px; }}
        .toolbar {{
            display: flex;
            gap: 12px;
            align-items: center;
            justify-content: center;
            margin-bottom: 14px;
            flex-wrap: wrap;
        }}
        #searchBox {{
            width: 300px;
            max-width: 90vw;
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid #2a2b3d;
            background: #13141f;
            color: #ffffff;
            font-size: 14px;
            outline: none;
        }}
        #searchBox:focus {{ border-color: #00E5FF; }}
        .slider-wrap {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: #cfd3e3;
            background: #13141f;
            padding: 7px 12px;
            border-radius: 8px;
            border: 1px solid #2a2b3d;
        }}
        #matchCount {{ font-size: 13px; color: #00E5FF; min-width: 130px; }}
        .btn {{
            padding: 9px 14px;
            border-radius: 8px;
            border: 1px solid #2a2b3d;
            background: #1e1e2f;
            color: #e0e0e0;
            cursor: pointer;
            font-size: 13px;
        }}
        .btn:hover {{ border-color: #00E5FF; color: #ffffff; }}
        #offlineBanner {{
            display: none;
            background: #3a161b;
            color: #ffb3ba;
            border: 1px solid #ff4d6d;
            padding: 14px;
            border-radius: 10px;
            margin-bottom: 14px;
            text-align: center;
        }}
        #plot {{
            width: 100%;
            height: 70vh;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #2a2b3d;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            margin-bottom: 20px;
        }}
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 14px;
        }}
        .cluster-card {{
            background: #13141f;
            border: 1px solid #2a2b3d;
            border-radius: 10px;
            padding: 14px;
        }}
        .card-title {{ font-size: 15px; font-weight: 600; color: #ffffff; margin-bottom: 6px; }}
        .badge {{
            display: inline-block;
            color: #0d0e15;
            font-weight: 700;
            font-size: 11px;
            padding: 2px 7px;
            border-radius: 999px;
            margin-right: 6px;
        }}
        .card-sub {{ font-size: 12px; color: #9ea4b8; margin-bottom: 6px; }}
        .card-metrics {{ font-size: 12px; color: #00E5FF; margin-bottom: 8px; }}
        .anchors {{ margin: 0; padding-left: 18px; font-size: 12px; color: #cfd3e3; line-height: 1.5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🌌 MUSIC TASTE ANALYZER - RECENCY-WEIGHTED TASTE VECTOR MAP</h1>
        <p>PCA Projection of <b>{len(track_coords)} tracks</b> across <b>{len(clusters)} Vibe Clusters</b>.</p>
        <div class="legend-note">💡 Point Size &amp; Opacity reflect <b>Recency Weight</b>. Click any track dot to open it in Spotify!</div>
    </div>
    <div id="offlineBanner">⚠️ Plotly visualization library failed to load (offline or CDN blocked). Cluster summaries remain available below.</div>
    <div class="toolbar">
        <input id="searchBox" type="text" placeholder="Search track, artist, or genre to highlight..." />
        <div class="slider-wrap">
            <span>Min Weight:</span>
            <input id="minWeightSlider" type="range" min="0" max="{slider_max}" value="0" step="1" />
            <span id="minWeightVal">0</span>
        </div>
        <button class="btn" id="modeToggleBtn" type="button">Switch to 3D View</button>
        <button class="btn" id="clearSearchBtn" type="button">Reset Filters</button>
        <span id="matchCount"></span>
    </div>
    <div id="plot"></div>
    <div class="cards-grid">
        {"".join(cards_html)}
    </div>
    <script>
        const traces2d = {_safe_json_for_script(traces_2d)};
        const traces3d = {_safe_json_for_script(traces_3d)};
        let is3d = false;

        const origOpacities2d = traces2d.map(t => (t.marker && Array.isArray(t.marker.opacity)) ? [...t.marker.opacity] : null);
        const origSizes2d = traces2d.map(t => (t.marker && Array.isArray(t.marker.size)) ? [...t.marker.size] : null);
        const origOpacities3d = traces3d.map(t => (t.marker && Array.isArray(t.marker.opacity)) ? [...t.marker.opacity] : null);
        const origSizes3d = traces3d.map(t => (t.marker && Array.isArray(t.marker.size)) ? [...t.marker.size] : null);

        const layout2d = {{
            paper_bgcolor: '#0d0e15',
            plot_bgcolor: '#13141f',
            font: {{ color: '#e0e0e0' }},
            xaxis: {{ title: {_safe_json_for_script(x_title)}, gridcolor: '#2a2b3d', zerolinecolor: '#3a3b50' }},
            yaxis: {{ title: {_safe_json_for_script(y_title)}, gridcolor: '#2a2b3d', zerolinecolor: '#3a3b50' }},
            legend: {{ orientation: 'h', y: -0.15, font: {{ size: 12 }} }},
            margin: {{ l: 55, r: 40, t: 30, b: 85 }},
            hoverlabel: {{ bgcolor: '#1e1e2f', bordercolor: '#00E5FF', font: {{ family: 'sans-serif', size: 13 }} }}
        }};
        const layout3d = {{
            paper_bgcolor: '#0d0e15',
            font: {{ color: '#e0e0e0' }},
            scene: {{
                bgcolor: '#13141f',
                xaxis: {{ title: {_safe_json_for_script(x_title)}, gridcolor: '#2a2b3d' }},
                yaxis: {{ title: {_safe_json_for_script(y_title)}, gridcolor: '#2a2b3d' }},
                zaxis: {{ title: {_safe_json_for_script(z_title)}, gridcolor: '#2a2b3d' }}
            }},
            legend: {{ orientation: 'h', y: -0.08, font: {{ size: 12 }} }},
            margin: {{ l: 20, r: 20, t: 30, b: 50 }},
            hoverlabel: {{ bgcolor: '#1e1e2f', bordercolor: '#00E5FF', font: {{ family: 'sans-serif', size: 13 }} }}
        }};

        if (typeof Plotly === 'undefined') {{
            document.getElementById('offlineBanner').style.display = 'block';
        }} else {{
            const plotDiv = document.getElementById('plot');
            const searchBox = document.getElementById('searchBox');
            const minWeightSlider = document.getElementById('minWeightSlider');
            const minWeightVal = document.getElementById('minWeightVal');
            const matchCountEl = document.getElementById('matchCount');
            const modeBtn = document.getElementById('modeToggleBtn');

            function attachClickHandler() {{
                plotDiv.on('plotly_click', (evt) => {{
                    if (!evt || !evt.points || !evt.points.length) return;
                    const cd = evt.points[0].customdata;
                    if (cd && cd.id) {{
                        window.open('https://open.spotify.com/track/' + encodeURIComponent(cd.id), '_blank', 'noopener,noreferrer');
                    }} else if (cd && cd.vid) {{
                        window.open('https://music.youtube.com/watch?v=' + encodeURIComponent(cd.vid), '_blank', 'noopener,noreferrer');
                    }}
                }});
            }}

            function applyFilters() {{
                const q = searchBox.value.trim().toLowerCase();
                const minW = parseFloat(minWeightSlider.value || '0');
                minWeightVal.textContent = String(minW);
                const activeTraces = is3d ? traces3d : traces2d;
                const baseOps = is3d ? origOpacities3d : origOpacities2d;
                const baseSzs = is3d ? origSizes3d : origSizes2d;
                let matches = 0;
                const clusterTraceIndices = [];
                const nextOpacities = [];
                const nextSizes = [];

                activeTraces.forEach((trace, idx) => {{
                    if (!Array.isArray(trace.customdata) || !baseOps[idx]) return;
                    clusterTraceIndices.push(idx);
                    if (!q && minW <= 0) {{
                        nextOpacities.push(baseOps[idx]);
                        nextSizes.push(baseSzs[idx]);
                    }} else {{
                        const opArr = [];
                        const szArr = [];
                        trace.customdata.forEach((meta, ptIdx) => {{
                            const textOk = !q || meta.s.includes(q);
                            const weightOk = meta.w >= minW;
                            if (textOk && weightOk) {{
                                matches += 1;
                                opArr.push(q ? 1.0 : baseOps[idx][ptIdx]);
                                szArr.push(q ? Math.max(8, baseSzs[idx][ptIdx] + (is3d ? 2 : 4)) : baseSzs[idx][ptIdx]);
                            }} else {{
                                opArr.push(0.05);
                                szArr.push(Math.max(3, baseSzs[idx][ptIdx] - 2));
                            }}
                        }});
                        nextOpacities.push(opArr);
                        nextSizes.push(szArr);
                    }}
                }});

                if (clusterTraceIndices.length > 0) {{
                    Plotly.restyle(plotDiv, {{ 'marker.opacity': nextOpacities, 'marker.size': nextSizes }}, clusterTraceIndices);
                }}
                matchCountEl.textContent = (q || minW > 0) ? (matches + ' matching tracks') : '';
            }}

            function renderCurrentMode() {{
                Plotly.newPlot(plotDiv, is3d ? traces3d : traces2d, is3d ? layout3d : layout2d, {{ responsive: true }});
                attachClickHandler();
                applyFilters();
            }}

            renderCurrentMode();
            searchBox.addEventListener('input', applyFilters);
            minWeightSlider.addEventListener('input', applyFilters);
            modeBtn.addEventListener('click', () => {{
                is3d = !is3d;
                modeBtn.textContent = is3d ? 'Switch to 2D View' : 'Switch to 3D View';
                renderCurrentMode();
            }});
            document.getElementById('clearSearchBtn').addEventListener('click', () => {{
                searchBox.value = '';
                minWeightSlider.value = '0';
                applyFilters();
            }});
        }}
    </script>
</body>
</html>"""

    atomic_write_text(html_path, html_content)


def main():
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Music Taste Analyzer - 2D/3D Taste Vector Map")
    parser.add_argument("--projection", choices=["pca", "umap"], default="pca", help="Dimensionality reduction algorithm")
    parser.add_argument("--offline", action="store_true", help="Inline local Plotly JS inside taste_map.html if available")
    args = parser.parse_args()

    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity

    base_dir = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(base_dir, "taste_vectors.npz")
    meta_path = os.path.join(base_dir, "track_metadata.json")
    summary_path = os.path.join(base_dir, "taste_profile_summary.json")

    if not os.path.exists(npz_path) or not os.path.exists(meta_path) or not os.path.exists(summary_path):
        print("❌ Error: Missing vector core files.")
        print("Please run Stage 2 first to generate your taste vectors!")
        sys.exit(1)

    with np.load(npz_path) as data:
        raw_embeddings = np.array(data["embeddings"])
        raw_track_ids = [str(tid) for tid in data["track_ids"]]

    metadata = load_json_safe(meta_path)
    summary = load_json_safe(summary_path)
    if not isinstance(metadata, dict) or not isinstance(summary, dict) or len(raw_embeddings) == 0:
        print("❌ Error: Vector core artifacts are empty or corrupted.")
        sys.exit(1)

    clusters = summary.get("clusters", [])
    if not clusters:
        print("❌ Error: No taste clusters found in taste_profile_summary.json.")
        sys.exit(1)

    # Exclude tracks filtered out by session avoid lists during clustering (keep unclustered -1 points)
    active_indices = [
        idx
        for idx, tid in enumerate(raw_track_ids)
        if not metadata.get(tid, {}).get("excluded_from_taste")
    ]
    excluded_count = len(raw_track_ids) - len(active_indices)
    if not active_indices:
        active_indices = list(range(len(raw_track_ids)))
        excluded_count = 0

    embeddings = raw_embeddings[active_indices]
    track_ids = [raw_track_ids[i] for i in active_indices]

    master_centroid = np.array(summary.get("master_taste_centroid"), dtype=float)
    cluster_centers = np.array([c["centroid_vector"] for c in clusters], dtype=float)

    print("==================================================")
    print(" 🌌 MUSIC TASTE ANALYZER - MATHEMATICAL TASTE VECTOR MAP")
    print("==================================================")
    print(f" Active Vectorized Tracks : {len(embeddings)} (Excluded: {excluded_count})")
    print(f" Embedding Dimensionality : {embeddings.shape[1]}-D Dense Space")
    print(f" Vibe Clusters Discovered : {len(clusters)}")
    print("-" * 50)

    projection_method = args.projection
    yearly_raw = summary.get("yearly_centroids") or []
    yearly_vecs = [np.array(y["centroid_vector"], dtype=float) for y in yearly_raw if y.get("centroid_vector")]

    if projection_method == "umap" and len(embeddings) >= 15:
        try:
            import umap

            n_nb = min(15, max(2, len(embeddings) - 1))
            reducer2 = umap.UMAP(n_components=2, metric="cosine", n_neighbors=n_nb, random_state=42)
            track_coords = reducer2.fit_transform(embeddings)
            centroid_coord = reducer2.transform([master_centroid])[0]
            center_coords = reducer2.transform(cluster_centers)
            yearly_2d = reducer2.transform(np.array(yearly_vecs)) if yearly_vecs else []
            explained_variance = [0.0, 0.0]

            reducer3 = umap.UMAP(n_components=3, metric="cosine", n_neighbors=n_nb, random_state=42)
            track_coords_3d = reducer3.fit_transform(embeddings)
            centroid_coord_3d = reducer3.transform([master_centroid])[0]
            center_coords_3d = reducer3.transform(cluster_centers)
            yearly_3d = reducer3.transform(np.array(yearly_vecs)) if yearly_vecs else []
            explained_variance_3d = [0.0, 0.0, 0.0]
            print("      ✅ Computed 2D & 3D UMAP nonlinear projections.")
        except Exception as e:
            print(f"      ⚠️ UMAP unavailable ({e}); falling back to PCA.")
            projection_method = "pca"

    if projection_method == "pca" or len(embeddings) < 15:
        projection_method = "pca"
        if len(embeddings) >= 2 and embeddings.shape[1] >= 2:
            pca2 = PCA(n_components=2, random_state=42)
            track_coords = pca2.fit_transform(embeddings)
            centroid_coord = pca2.transform([master_centroid])[0]
            center_coords = pca2.transform(cluster_centers)
            yearly_2d = pca2.transform(np.array(yearly_vecs)) if yearly_vecs else []
            explained_variance = pca2.explained_variance_ratio_.tolist()
        else:
            track_coords = np.zeros((len(embeddings), 2), dtype=float)
            centroid_coord = np.zeros(2, dtype=float)
            center_coords = np.zeros((len(clusters), 2), dtype=float)
            yearly_2d = [[0.0, 0.0] for _ in yearly_vecs]
            explained_variance = [0.0, 0.0]

        if len(embeddings) >= 3 and embeddings.shape[1] >= 3:
            pca3 = PCA(n_components=3, random_state=42)
            track_coords_3d = pca3.fit_transform(embeddings)
            centroid_coord_3d = pca3.transform([master_centroid])[0]
            center_coords_3d = pca3.transform(cluster_centers)
            yearly_3d = pca3.transform(np.array(yearly_vecs)) if yearly_vecs else []
            explained_variance_3d = pca3.explained_variance_ratio_.tolist()
        else:
            track_coords_3d = np.hstack([track_coords, np.zeros((len(embeddings), 1), dtype=float)])
            centroid_coord_3d = np.append(centroid_coord, 0.0)
            center_coords_3d = np.hstack([center_coords, np.zeros((len(clusters), 1), dtype=float)])
            yearly_3d = [list(pt) + [0.0] for pt in yearly_2d]
            explained_variance_3d = explained_variance + [0.0]

    era_trajectory = []
    valid_yearly = [y for y in yearly_raw if y.get("centroid_vector")]
    for idx, y in enumerate(valid_yearly):
        pt2 = yearly_2d[idx] if idx < len(yearly_2d) else [0.0, 0.0]
        pt3 = yearly_3d[idx] if idx < len(yearly_3d) else [pt2[0], pt2[1], 0.0]
        era_trajectory.append(
            {
                "year": y["year"],
                "plays": y.get("plays", 0),
                "unique_tracks": y.get("unique_tracks", 0),
                "x": round(float(pt2[0]), 5),
                "y": round(float(pt2[1]), 5),
                "x3": round(float(pt3[0]), 5),
                "y3": round(float(pt3[1]), 5),
                "z": round(float(pt3[2]), 5),
            }
        )

    valid_cids = {c["cluster_id"] for c in clusters}
    cid_order = [c["cluster_id"] for c in clusters]
    sims = cosine_similarity(embeddings, cluster_centers)
    track_clusters = []
    for idx, tid in enumerate(track_ids):
        t_meta = metadata.get(tid, {})
        stored_cid = t_meta.get("cluster_id")
        if isinstance(stored_cid, int) and (
            stored_cid in valid_cids or (stored_cid == -1 and not t_meta.get("excluded_from_taste"))
        ):
            track_clusters.append(stored_cid)
        else:
            best_pos = int(np.argmax(sims[idx]))
            track_clusters.append(cid_order[best_pos])

    GRID_W, GRID_H = 54, 16
    grid = [[" " for _ in range(GRID_W)] for _ in range(GRID_H)]
    x_min, x_max = track_coords[:, 0].min(), track_coords[:, 0].max()
    y_min, y_max = track_coords[:, 1].min(), track_coords[:, 1].max()

    symbols = ["●", "▲", "★", "◆", "■", "✦", "▼"]
    cid_to_sym = {c["cluster_id"]: symbols[idx % len(symbols)] for idx, c in enumerate(clusters)}

    for (x, y), cl in zip(track_coords, track_clusters):
        gx = int((x - x_min) / (x_max - x_min + 1e-9) * (GRID_W - 1))
        gy = int((y_max - y) / (y_max - y_min + 1e-9) * (GRID_H - 1))
        gx = max(0, min(GRID_W - 1, gx))
        gy = max(0, min(GRID_H - 1, gy))
        grid[gy][gx] = cid_to_sym.get(cl, "●")

    cx = int((centroid_coord[0] - x_min) / (x_max - x_min + 1e-9) * (GRID_W - 1))
    cy = int((y_max - centroid_coord[1]) / (y_max - y_min + 1e-9) * (GRID_H - 1))
    cx = max(0, min(GRID_W - 1, cx))
    cy = max(0, min(GRID_H - 1, cy))
    grid[cy][cx] = "⊕"

    print("\n  2D PCA Vector Space Projection:")
    print("  +" + "-" * GRID_W + "+")
    for row in grid:
        print("  |" + "".join(row) + "|")
    print("  +" + "-" * GRID_W + "+")
    legend_parts = [f"{cid_to_sym[c['cluster_id']]} C{c['cluster_id']}" for c in clusters]
    print("  Legend: ⊕ Master Centroid | " + "  ".join(legend_parts) + "\n")
    print("-" * 50)

    for c in clusters:
        cid = c["cluster_id"]
        vibe = c.get("vibe_label", f"Cluster {cid}")
        weighted_share = c.get("weighted_share_pct", c.get("weight_percentage", 0.0))
        raw_pct = c.get("raw_percentage", 0.0)
        count = c["track_count"]
        sym = cid_to_sym.get(cid, "●")
        tel = c.get("telemetry") or {}

        print(f"\n {sym} Cluster {cid}: [{vibe}]")
        print(f"   • Recency-Weighted Taste Share : {weighted_share}% (Active Listening Volume)")
        print(f"   • Historical Track Count       : {count} tracks ({raw_pct}% raw library)")
        if tel.get("avg_bpm") is not None:
            print(
                f"   • Acoustic Profile             : Avg {tel['avg_bpm']} BPM | "
                f"Energy {tel.get('avg_energy')} | Valence {tel.get('avg_valence')}"
            )
        print("   • Top Anchor Tracks:")
        for t in c.get("anchor_tracks", [])[:3]:
            print(f"     - {t['name']} by {t['artist']} (sim: {t['similarity']})")

    # Export structured map JSON for the Web UI (2D, 3D, radar telemetry, sliders)
    map_clusters = []
    for idx, c in enumerate(clusters):
        map_clusters.append(
            {
                "cluster_id": c["cluster_id"],
                "vibe_label": c.get("vibe_label", f"Cluster {c['cluster_id']}"),
                "weighted_share_pct": c.get("weighted_share_pct", 0.0),
                "raw_percentage": c.get("raw_percentage", 0.0),
                "track_count": c.get("track_count", 0),
                "top_genres": c.get("top_genres", []),
                "anchor_tracks": c.get("anchor_tracks", [])[:5],
                "telemetry": c.get("telemetry") or {},
                "center_2d": [round(float(center_coords[idx][0]), 5), round(float(center_coords[idx][1]), 5)],
                "center_3d": [
                    round(float(center_coords_3d[idx][0]), 5),
                    round(float(center_coords_3d[idx][1]), 5),
                    round(float(center_coords_3d[idx][2]), 5),
                ],
            }
        )

    map_tracks = []
    for idx, tid in enumerate(track_ids):
        info = metadata.get(tid, {})
        af = info.get("audio_features")
        real_af = af if is_real_audio_features(af) else {}
        sp_id = (
            tid
            if is_valid_spotify_id(tid)
            else (info.get("spotify_id") if is_valid_spotify_id(info.get("spotify_id")) else None)
        )
        vid = extract_ytm_video_id(info.get("video_id") or tid)
        prov = get_track_provider({"id": tid, "provider": info.get("provider"), "spotify_id": sp_id, "video_id": vid})
        map_tracks.append(
            {
                "id": tid,
                "name": info.get("name", "Unknown"),
                "artist": info.get("artists", "Unknown"),
                "genres": info.get("genres", ""),
                "playlists": info.get("playlists", ""),
                "year": info.get("year", ""),
                "provider": prov,
                "video_id": vid,
                "cluster_id": int(track_clusters[idx]),
                "recency_weight": round(float(info.get("recency_weight", 1.0)), 3),
                "skip_ratio": info.get("skip_ratio"),
                "spotify_url": f"https://open.spotify.com/track/{sp_id}" if sp_id else None,
                "youtube_url": f"https://music.youtube.com/watch?v={vid}" if vid else None,
                "bpm": real_af.get("bpm"),
                "energy": real_af.get("energy"),
                "valence": real_af.get("valence"),
                "danceability": real_af.get("danceability"),
                "acousticness": real_af.get("acousticness"),
                "x": round(float(track_coords[idx][0]), 5),
                "y": round(float(track_coords[idx][1]), 5),
                "z": round(float(track_coords_3d[idx][2]), 5),
                "x3": round(float(track_coords_3d[idx][0]), 5),
                "y3": round(float(track_coords_3d[idx][1]), 5),
            }
        )

    map_data = {
        "projection_method": projection_method,
        "explained_variance_2d": [round(float(v), 4) for v in explained_variance],
        "explained_variance_3d": [round(float(v), 4) for v in explained_variance_3d],
        "session_filters": summary.get("session_filters") or {},
        "total_tracks": len(map_tracks),
        "excluded_count": excluded_count,
        "master_centroid_2d": [round(float(centroid_coord[0]), 5), round(float(centroid_coord[1]), 5)],
        "master_centroid_3d": [
            round(float(centroid_coord_3d[0]), 5),
            round(float(centroid_coord_3d[1]), 5),
            round(float(centroid_coord_3d[2]), 5),
        ],
        "era_trajectory": era_trajectory,
        "clusters": map_clusters,
        "tracks": map_tracks,
    }
    map_data_path = os.path.join(base_dir, "taste_map_data.json")
    atomic_write_json(map_data_path, map_data, indent=None)

    html_path = os.path.abspath(os.path.join(ROOT_DIR, "taste_map.html"))
    generate_html_visualizer(
        track_coords,
        track_clusters,
        track_ids,
        metadata,
        clusters,
        center_coords,
        html_path,
        explained_variance=explained_variance,
        track_coords_3d=track_coords_3d,
        center_coords_3d=center_coords_3d,
        explained_variance_3d=explained_variance_3d,
        era_trajectory=era_trajectory,
        offline=args.offline,
        projection_method=projection_method,
    )
    print("\n" + "=" * 50)
    print("🎨 Saved Interactive 2D/3D Vector Map & Data:")
    print(f"   • {html_path}")
    print(f"   • {map_data_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
