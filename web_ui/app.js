(() => {
    "use strict";

    const state = {
        status: null,
        avoidArtists: [],
        avoidGenres: [],
        penalizedArtists: {},
        currentPlaylist: null,
        mapData: null,
        is3d: false,
        drillClusterId: null,
        mapViewCache: new Map(),
        mapCacheSource: null,
        mapLegendClusterIds: {},
        overviewRenderedKey: null,
        drillRenderedKey: null,
        analytics: null,
        sse: null,
        feedback: { liked: {}, disliked: {} },
        presets: {},
        blendFriendCard: null,
        dragSrcIdx: null,
        studioMode: "prompt",
        librarySearchTimer: null,
        activeProvider: "spotify",
        ytmOauthDeviceCode: null,
        selectedLlmModelKey: "auto",
        llmCatalogData: null,
        statusInFlight: false,
        mapFilterTimer: null,
        mapTraceIndex: [],
        mapMaxWeight: 0,
    };

    const CLUSTER_COLORS = [
        "#e5a04b",
        "#5b9bd5",
        "#4cc38a",
        "#d96b6b",
        "#9b87c2",
        "#52b7b1",
        "#c98b5b",
        "#7fa8d7",
        "#a8c26c",
        "#d68cb8",
        "#6ea8fe",
        "#e899dc",
        "#a3d977",
        "#f4a261",
        "#72d5cb",
        "#c9a0dc",
    ];

    const SPOTIFY_ID_RE = /^[A-Za-z0-9]{10,40}$/;
    const YTM_VID_RE = /^[A-Za-z0-9_-]{11}$/;

    function extractYtmVid(val) {
        const s = String(val || "").trim();
        if (s.startsWith("yt:") && YTM_VID_RE.test(s.slice(3))) return s.slice(3);
        if (YTM_VID_RE.test(s)) return s;
        return "";
    }

    function escapeHtml(val, quote = true) {
        const text = String(val ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");
        return quote ? text.replace(/"/g, "&quot;").replace(/'/g, "&#39;") : text;
    }

    function plotlyText(val) {
        // Plotly renders SVG text, where HTML quote entities are displayed literally.
        const lines = String(val ?? "").replace(/\s+/g, " ").match(/.{1,48}(?:\s|$)|\S{1,48}/gu) || [];
        return lines.map((line) => escapeHtml(line.trim(), false)).join("<br>");
    }

    function safeHttpUrl(raw) {
        const s = String(raw || "").trim();
        return /^https?:\/\//i.test(s) ? s : "";
    }

    let youtubeApiPromise;
    function loadYoutubeApi() {
        if (window.YT?.Player) return Promise.resolve(window.YT);
        if (!youtubeApiPromise) {
            youtubeApiPromise = new Promise((resolve, reject) => {
                const fail = () => {
                    clearTimeout(timer);
                    youtubeApiPromise = null;
                    reject(new Error("YouTube player could not load."));
                };
                const timer = setTimeout(fail, 12000);
                window.onYouTubeIframeAPIReady = () => {
                    clearTimeout(timer);
                    resolve(window.YT);
                };
                const script = document.createElement("script");
                script.src = "https://www.youtube.com/iframe_api";
                script.onerror = fail;
                document.head.appendChild(script);
            });
        }
        return youtubeApiPromise;
    }

    async function openPreviewTrack(trackOrId, title) {
        let spId = "";
        let vid = "";
        let prov = "";
        let trackName = "";
        let trackArtist = "";
        let label = title || "Track Preview";

        if (trackOrId && typeof trackOrId === "object") {
            const rawSp = String(trackOrId.spotify_id || "").trim();
            const rawId = String(trackOrId.id || "").trim();
            if (rawSp && SPOTIFY_ID_RE.test(rawSp)) {
                spId = rawSp;
            } else if (rawId && !rawId.startsWith("yt:") && SPOTIFY_ID_RE.test(rawId)) {
                spId = rawId;
            }
            vid = extractYtmVid(trackOrId.video_id || rawId);
            prov = String(trackOrId.provider || "").toLowerCase();
            trackName = String(trackOrId.name || "").trim();
            trackArtist = String(trackOrId.artist || "").trim();
            if (!title) {
                label = trackArtist ? `${trackName || "Track"} — ${trackArtist}` : (trackName || "Track Preview");
            }
        } else {
            const raw = String(trackOrId || "").trim();
            vid = extractYtmVid(raw);
            if (!raw.startsWith("yt:") && SPOTIFY_ID_RE.test(raw)) spId = raw;
        }

        if (!trackName && label.includes(" — ")) {
            const parts = label.split(" — ");
            trackName = parts[0].trim();
            if (!trackArtist) trackArtist = parts.slice(1).join(" — ").trim();
        }

        const preferYtm = Boolean(vid && (!spId || state.activeProvider === "ytmusic" || prov === "ytmusic"));
        closePreviewDock();
        const mySeq = state.previewSeq;
        const iframe = document.getElementById("previewIframe");
        const youtubeWrap = document.getElementById("previewYoutubeWrap");
        const audioWrap = document.getElementById("previewAudioWrap");
        const audioEl = document.getElementById("previewAudio");
        const audioMeta = document.getElementById("previewAudioMeta");
        const extLink = document.getElementById("previewExternalLink");
        const notice = document.getElementById("previewEmbedNotice");
        const titleEl = document.getElementById("previewTrackTitle");
        titleEl.textContent = label;
        titleEl.title = label;

        let fallbackStarted = false;
        const useAudioFallback = async (reason) => {
            if (state.previewSeq !== mySeq || fallbackStarted) return;
            fallbackStarted = true;
            state.youtubePlayer?.destroy();
            state.youtubePlayer = null;
            youtubeWrap.replaceChildren();
            youtubeWrap.style.display = "none";
            notice.textContent = reason;
            notice.style.display = "block";
            const unavailable = () => {
                if (state.previewSeq !== mySeq) return;
                audioEl.onerror = null;
                audioEl.pause();
                audioEl.removeAttribute("src");
                audioEl.load();
                audioWrap.style.display = "none";
                notice.textContent = `${reason} No audio preview available. Use Open in YT Music above.`;
            };
            if (!trackName) return unavailable();
            audioWrap.style.display = "flex";
            audioMeta.textContent = "Loading fallback audio preview…";
            try {
                const qs = new URLSearchParams({ title: trackName, artist: trackArtist });
                const res = await apiJson(`/api/preview-audio?${qs.toString()}`);
                if (state.previewSeq !== mySeq) return;
                const previewUrl = safeHttpUrl(res?.preview_url);
                if (res?.found && previewUrl) {
                    audioMeta.textContent = `30s audio preview${res.album ? ` · ${res.album}` : ""}`;
                    audioEl.onerror = unavailable;
                    audioEl.src = previewUrl;
                    audioEl.load();
                    return;
                }
            } catch {
                if (state.previewSeq !== mySeq) return;
            }
            unavailable();
        };

        if (extLink) {
            const extHref = preferYtm && vid
                ? `https://music.youtube.com/watch?v=${encodeURIComponent(vid)}`
                : (spId ? `https://open.spotify.com/track/${encodeURIComponent(spId)}` : (vid ? `https://music.youtube.com/watch?v=${encodeURIComponent(vid)}` : ""));
            if (extHref) {
                extLink.href = extHref;
                extLink.textContent = preferYtm || (!spId && vid) ? "Open in YT Music ↗" : "Open in Spotify ↗";
                extLink.style.display = "inline-flex";
            } else {
                extLink.style.display = "none";
            }
        }

        if (preferYtm) {
            youtubeWrap.style.display = "block";
            notice.textContent = "Loading YouTube preview…";
            notice.style.display = "block";
            document.getElementById("previewDock").style.display = "block";
            try {
                const yt = await loadYoutubeApi();
                if (state.previewSeq !== mySeq) return;
                const mount = document.createElement("div");
                youtubeWrap.replaceChildren(mount);
                state.youtubePlayer = new yt.Player(mount, {
                    width: "100%",
                    height: "200",
                    videoId: vid,
                    playerVars: { autoplay: 0, rel: 0, playsinline: 1, origin: window.location.origin },
                    events: {
                        onReady: () => {
                            if (state.previewSeq === mySeq && !fallbackStarted) notice.style.display = "none";
                        },
                        onError: (event) => {
                            if (state.previewSeq !== mySeq) return;
                            const code = Number(event.data);
                            const reasons = {
                                2: "Invalid YouTube video ID or player parameter.",
                                5: "YouTube could not play this video in the browser.",
                                100: "This YouTube video was removed or is private.",
                                101: "This upload does not allow embedded playback.",
                                150: "This upload does not allow embedded playback.",
                                153: "YouTube could not identify this app (missing referrer/client information).",
                            };
                            const reason = `YouTube error ${code}: ${reasons[code] || "Playback failed."}`;
                            notice.dataset.youtubeError = String(code);
                            console.warn("[Preview]", reason, { videoId: vid, title: label });
                            return useAudioFallback(reason);
                        },
                    },
                });
            } catch {
                if (state.previewSeq !== mySeq) return;
                await useAudioFallback("YouTube player could not load.");
            }
            return;
        }

        if (spId) {
            document.getElementById("previewTrackTitle").textContent = label;
            audioWrap.style.display = "none";
            iframe.style.display = "block";
            iframe.height = "80";
            iframe.src = `https://open.spotify.com/embed/track/${encodeURIComponent(spId)}?utm_source=generator&theme=0`;
            document.getElementById("previewDock").style.display = "block";
            return;
        }
        showToast("No valid Spotify ID or YouTube Music video ID available for preview.", "err");
    }

    function closePreviewDock() {
        state.previewSeq = (state.previewSeq || 0) + 1;
        state.youtubePlayer?.destroy();
        state.youtubePlayer = null;
        const youtubeWrap = document.getElementById("previewYoutubeWrap");
        youtubeWrap.replaceChildren();
        youtubeWrap.style.display = "none";
        const audioEl = document.getElementById("previewAudio");
        if (audioEl) {
            audioEl.onerror = null;
            audioEl.pause();
            audioEl.removeAttribute("src");
            audioEl.load();
        }
        document.getElementById("previewAudioWrap").style.display = "none";
        document.getElementById("previewDock").style.display = "none";
        const iframe = document.getElementById("previewIframe");
        iframe.src = "about:blank";
        iframe.style.display = "none";
        const notice = document.getElementById("previewEmbedNotice");
        notice.style.display = "none";
        delete notice.dataset.youtubeError;
    }

    async function handleSaveLibraryButton(btn) {
        if (!btn) return;
        btn.disabled = true;
        try {
            await apiJson("/api/spotify/library", {
                method: "POST",
                body: { action: "save", uris: [`spotify:track:${btn.dataset.id}`] },
            });
            btn.textContent = "♥";
            showToast(`Saved "${btn.dataset.title || "track"}" to Spotify Library!`, "ok");
        } catch (err) {
            showToast(err.message, "err");
        } finally {
            btn.disabled = false;
        }
    }

    function openPromptModal(title, desc, defaultValue, onSave) {
        const modal = document.getElementById("promptModal");
        const input = document.getElementById("promptModalInput");
        document.getElementById("promptModalTitle").textContent = title;
        document.getElementById("promptModalDesc").textContent = desc || "";
        input.value = defaultValue || "";
        modal.style.display = "flex";
        input.focus();

        const okBtn = document.getElementById("promptModalOkBtn");
        const cancelBtn = document.getElementById("promptModalCancelBtn");
        const cleanup = () => {
            modal.style.display = "none";
            okBtn.onclick = null;
            cancelBtn.onclick = null;
            input.onkeydown = null;
        };
        cancelBtn.onclick = cleanup;
        okBtn.onclick = async () => {
            const val = input.value.trim();
            cleanup();
            await onSave(val);
        };
        input.onkeydown = (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                okBtn.click();
            } else if (e.key === "Escape") {
                e.preventDefault();
                cleanup();
            }
        };
    }

    function showToast(msg, kind = "info") {
        const container = document.getElementById("toastContainer");
        const el = document.createElement("div");
        el.className = `toast ${kind}`;
        el.textContent = msg;
        container.appendChild(el);
        setTimeout(() => {
            el.remove();
        }, 4200);
    }

    async function apiJson(url, options = {}) {
        const opts = { ...options };
        if (opts.body && typeof opts.body !== "string" && !(opts.body instanceof Blob)) {
            opts.headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
            opts.body = JSON.stringify(opts.body);
        }
        const resp = await fetch(url, opts);
        let data = {};
        const ct = resp.headers.get("Content-Type") || "";
        if (ct.includes("application/json")) {
            data = await resp.json();
        }
        if (!resp.ok) {
            throw new Error(data.error || `Request failed (${resp.status})`);
        }
        return data;
    }

    function switchTab(tabName) {
        document.querySelectorAll(".nav-btn").forEach((b) => {
            b.classList.toggle("active", b.dataset.tab === tabName);
        });
        document.querySelectorAll(".tab-panel").forEach((p) => {
            p.classList.toggle("active", p.id === `tab-${tabName}`);
        });
        const contentArea = document.querySelector(".content-area");
        if (contentArea) contentArea.scrollTop = 0;
        if (tabName === "map") loadTasteMap();
        if (tabName === "analytics") {
            loadPlaylistsArchive();
            loadAnalytics();
        }
        if (tabName === "preferences") {
            loadPreferences();
            loadAutomation();
            loadFeedbackMemory();
        }
        if (tabName === "doctor") loadDoctor();
    }

    // =====================================================
    // STATUS, PIPELINE & GUARDRAILS
    // =====================================================
    async function refreshStatus(force = false) {
        if (!force && (document.hidden || state.statusInFlight)) return;
        state.statusInFlight = true;
        try {
            const data = await apiJson("/api/status");
            state.status = data;
            state.avoidArtists = data.studio.session_avoid_artists || [];
            state.avoidGenres = data.studio.session_avoid_genres || [];
            if (data.active_provider) {
                state.activeProvider = data.active_provider;
                const provSel = document.getElementById("activeProviderSelect");
                if (provSel) provSel.value = state.activeProvider;
            }

            const hw = data.hardware || {};
            const cpuTxt = hw.cpu_percent != null ? `CPU ${hw.cpu_percent}% · RAM ${hw.ram_used_gb}/${hw.ram_total_gb}GB` : "Telemetry ready";
            document.getElementById("hwPill").textContent = `${hw.silicon || "CPU"} | ${cpuTxt}`;

            const st = data.studio || {};
            const warmParts = [];
            const llmShort = st.llm_label || "Local LLM";
            if (st.embed_warm) warmParts.push("MiniLM");
            if (st.llm_warm) warmParts.push(llmShort);
            document.getElementById("warmPill").textContent = warmParts.length
                ? `Warm: ${warmParts.join(" + ")}`
                : "Engine: Cold";
            const modelPill = document.getElementById("studioActiveModelPill");
            if (modelPill) {
                modelPill.textContent = `LLM: ${llmShort}`;
            }

            renderGuardrailsUI();
            renderPipelineCards(data);
            populateClusterSelect(st.clusters || []);
            updateJobIndicators(data.job || {});
        } catch (err) {
            console.error(err);
        } finally {
            state.statusInFlight = false;
        }
    }

    function renderPipelineCards(data) {
        const auth = data.auth || {};
        const pipe = data.pipeline || {};
        const s1 = pipe.stage1 || {};
        const s1Ytm = pipe.stage1_ytm || {};
        const ytmAuth = auth.ytmusic || {};
        const s2 = pipe.stage2 || {};
        const tm = pipe.taste_map || {};
        const eh = pipe.extended_history || {};

        const cards = [
            {
                label: "Spotify OAuth",
                badge: auth.token_cached ? ["Authenticated", "badge-ok"] : (auth.client_id_configured ? ["Ready to Link", "badge-warn"] : ["Missing ID", "badge-bad"]),
                val: auth.token_cached ? "Token Active" : "Not Linked",
                sub: auth.client_id_configured ? "PKCE Client ID configured" : "Set Client ID in Preferences",
            },
            {
                label: "Stage 1 Spotify",
                badge: s1.ready ? ["Ready", "badge-ok"] : ["Missing", "badge-warn"],
                val: s1.ready ? `${s1.tracks.toLocaleString()} Tracks` : "Run Stage 1",
                sub: s1.ready
                    ? `${s1.audio_features.toLocaleString()} with ReccoBeats audio${(s1.saved_albums || s1.followed_artists) ? ` · ${s1.saved_albums || 0} albums · ${s1.followed_artists || 0} followed` : ""}`
                    : "Fetches Spotify + ReccoBeats",
            },
            {
                label: "Stage 1 YT Music",
                badge: s1Ytm.ready ? ["Ready", "badge-ok"] : (ytmAuth.configured ? ["Auth Ready", "badge-info"] : ["Not Linked", "badge-warn"]),
                val: s1Ytm.ready ? `${(s1Ytm.tracks || 0).toLocaleString()} Tracks` : (ytmAuth.configured ? "Run 1b Sync" : "Link in Prefs"),
                sub: ytmAuth.configured
                    ? `Auth: ${ytmAuth.method === "browser" ? "Browser Headers" : "Google OAuth"}${s1Ytm.ready ? ` · ${(s1Ytm.audio_features || 0).toLocaleString()} ReccoBeats audio` : ""}`
                    : "Configure Browser Headers or OAuth",
            },
            {
                label: "Stage 2 Vector Core",
                badge: !s2.ready ? ["Missing", "badge-warn"] : (s2.stale ? ["Stale — Rebuild", "badge-warn"] : ["Fresh", "badge-ok"]),
                val: s2.ready ? `${s2.clusters} Vibe Clusters` : "Run Stage 2",
                sub: s2.session_filters && s2.session_filters.excluded_track_count
                    ? `${s2.session_filters.excluded_track_count} tracks excluded by guardrails`
                    : "384-D OpenVINO Embeddings",
            },
            {
                label: "Taste Map (2D/3D)",
                badge: !tm.ready ? ["Missing", "badge-warn"] : (tm.stale ? ["Outdated", "badge-warn"] : ["Ready", "badge-ok"]),
                val: tm.ready ? "PCA Projection Ready" : "Not Built",
                sub: "Interactive Spotify scatter & radar",
            },
            {
                label: "Extended History",
                badge: eh.imported ? ["Imported", "badge-ok"] : (eh.raw_files ? ["Files Ready", "badge-info"] : ["None", "badge-warn"]),
                val: eh.imported ? `${eh.unique_tracks.toLocaleString()} Tracks` : `${eh.raw_files} JSON Files`,
                sub: eh.imported ? "Time-decay & analytics active" : "Place JSONs in extended_history/",
            },
        ];

        const grid = document.getElementById("pipelineCards");
        grid.innerHTML = cards
            .map(
                (c) => `
            <div class="status-card">
                <div class="status-card-top">
                    <span class="status-label">${escapeHtml(c.label)}</span>
                    <span class="badge ${c.badge[1]}">${escapeHtml(c.badge[0])}</span>
                </div>
                <div class="status-val">${escapeHtml(c.val)}</div>
                <div class="status-sub">${escapeHtml(c.sub)}</div>
            </div>`
            )
            .join("");
    }

    function populateClusterSelect(clusters) {
        const sel = document.getElementById("studioCluster");
        const prev = sel.value;
        sel.innerHTML = `<option value="">Global Master Taste Centroid</option>` +
            clusters
                .map(
                    (c) =>
                        `<option value="${escapeHtml(c.cluster_id)}">C${escapeHtml(c.cluster_id)}: ${escapeHtml(
                            c.vibe_label
                        )} (${escapeHtml(c.weighted_share_pct)}%)</option>`
                )
                .join("");
        if (prev && Array.from(sel.options).some((o) => o.value === prev)) {
            sel.value = prev;
        }
    }

    function renderGuardrailsUI() {
        const total = state.avoidArtists.length + state.avoidGenres.length;
        document.getElementById("guardPill").textContent = `Guardrails: ${total} active`;

        const artWrap = document.getElementById("dashAvoidArtistChips");
        artWrap.innerHTML = state.avoidArtists
            .map(
                (a, idx) =>
                    `<span class="chip chip-artist">${escapeHtml(a)} <button type="button" data-kind="artist" data-idx="${idx}" aria-label="Remove">×</button></span>`
            )
            .join("");

        const genWrap = document.getElementById("dashAvoidGenreChips");
        genWrap.innerHTML = state.avoidGenres
            .map(
                (g, idx) =>
                    `<span class="chip chip-genre">${escapeHtml(g)} <button type="button" data-kind="genre" data-idx="${idx}" aria-label="Remove">×</button></span>`
            )
            .join("");

        const summaryEl = document.getElementById("studioGuardrailsSummary");
        if (total === 0) {
            summaryEl.textContent = "None (All artists & genres allowed)";
        } else {
            const parts = [];
            if (state.avoidArtists.length) parts.push(`Artists: ${state.avoidArtists.join(", ")}`);
            if (state.avoidGenres.length) parts.push(`Genres: ${state.avoidGenres.join(", ")}`);
            summaryEl.textContent = parts.join(" | ");
        }
    }

    async function pushSessionGuardrails(persist = false) {
        const res = await apiJson("/api/session/exclusions", {
            method: "POST",
            body: {
                avoid_artists: state.avoidArtists,
                avoid_genres: state.avoidGenres,
                persist,
            },
        });
        state.avoidArtists = res.avoid_artists || [];
        state.avoidGenres = res.avoid_genres || [];
        renderGuardrailsUI();
        showToast(persist ? "Guardrails saved to session & persistent defaults." : "Session guardrails updated.", "ok");
    }

    // =====================================================
    // JOB RUNNER & SSE CONSOLE
    // =====================================================
    function updateJobIndicators(job) {
        const status = job.status || "idle";
        const label = job.label || "Idle";
        const running = status === "running";

        const dot1 = document.getElementById("jobDot");
        const dot2 = document.getElementById("drawerJobDot");
        [dot1, dot2].forEach((d) => {
            d.className = `job-dot ${running ? "running" : status === "completed" ? "completed" : status === "failed" ? "failed" : ""}`;
        });

        document.getElementById("jobPillText").textContent = running ? `Running: ${label}` : `Job Console (${status})`;
        document.getElementById("drawerJobTitle").textContent = `Live Job Console — ${label} [${status}]`;
        document.getElementById("btnCancelJob").style.display = running ? "inline-flex" : "none";

        ["btnFullSync", "btnRunStage1", "btnRunStage1Ytm", "btnRunStage2", "btnRunMap", "btnRunHistory", "btnRebuildWithGuardrails", "btnRefreshMap", "mapProjectionSelect"].forEach(
            (id) => {
                const el = document.getElementById(id);
                if (el) el.disabled = running;
            }
        );
    }

    function appendConsoleLine(line) {
        const out = document.getElementById("consoleOutput");
        if (out.textContent === "Waiting for pipeline activity...") {
            out.textContent = "";
        }
        out.textContent += (out.textContent ? "\n" : "") + line;
        out.scrollTop = out.scrollHeight;
    }

    function connectJobStream() {
        if (state.sse) state.sse.close();
        const es = new EventSource("/api/jobs/stream");
        state.sse = es;

        es.addEventListener("snapshot", (ev) => {
            const snap = JSON.parse(ev.data);
            const out = document.getElementById("consoleOutput");
            out.textContent = (snap.logs && snap.logs.length) ? snap.logs.join("\n") : "Waiting for pipeline activity...";
            out.scrollTop = out.scrollHeight;
            updateJobIndicators(snap.job || {});
        });

        es.addEventListener("log", (ev) => {
            const payload = JSON.parse(ev.data);
            appendConsoleLine(payload.line || "");
        });

        es.addEventListener("state", (ev) => {
            const job = JSON.parse(ev.data);
            updateJobIndicators(job);
            if (job.status && job.status !== "running") {
                refreshStatus();
                if (job.status === "completed") showToast(`${job.label} completed!`, "ok");
                if (job.status === "failed") showToast(`${job.label} failed (exit ${job.exit_code}).`, "err");
            }
        });
    }

    async function startJob(jobType, extra = {}) {
        try {
            document.getElementById("consoleDrawer").classList.remove("collapsed");
            document.getElementById("consoleOutput").textContent = "";
            await apiJson("/api/jobs/run", {
                method: "POST",
                body: { job: jobType, ...extra },
            });
            showToast(`Started ${jobType}...`, "info");
        } catch (err) {
            showToast(err.message, "err");
        }
    }

    // =====================================================
    // PLAYLIST STUDIO
    // =====================================================
    function switchStudioMode(mode) {
        state.studioMode = mode || "prompt";
        document.querySelectorAll(".subtab-btn").forEach((b) => {
            b.classList.toggle("active", b.dataset.studioMode === state.studioMode);
        });
        document.getElementById("studioModePrompt").style.display = state.studioMode === "prompt" ? "block" : "none";
        document.getElementById("studioModeBridge").style.display = state.studioMode === "bridge" ? "block" : "none";
        document.getElementById("studioModeBlend").style.display = state.studioMode === "blend" ? "block" : "none";
    }

    function renderTransferReport(res, label = "Cross-Platform Transfer") {
        const banner = document.getElementById("transferReportBanner");
        if (!banner) return;
        if (!res) {
            banner.style.display = "none";
            banner.innerHTML = "";
            return;
        }
        const matched = Number(res.matched_count || 0);
        const total = Number(res.total_tracks || matched);
        const unmatched = Array.isArray(res.unmatched) ? res.unmatched : [];
        const destName = res.target_provider === "spotify" ? "Spotify" : "YouTube Music";
        const unmatchedHtml = unmatched.length
            ? `<div class="muted mt-8"><b>Unmatched (${escapeHtml(unmatched.length)}):</b> ` +
              unmatched
                  .slice(0, 12)
                  .map((u) => `${escapeHtml(u.name || "Unknown")} — ${escapeHtml(u.artist || "Unknown")}`)
                  .join(" · ") +
              (unmatched.length > 12 ? ` (+${escapeHtml(unmatched.length - 12)} more)` : "") +
              `</div>`
            : "";
        const safeLinkUrl = safeHttpUrl(res.url);
        const linkHtml = safeLinkUrl
            ? ` · <a class="track-link" href="${escapeHtml(safeLinkUrl)}" target="_blank" rel="noopener noreferrer">Open ${escapeHtml(destName)} Playlist ↗</a>`
            : "";
        banner.innerHTML = `<strong>${escapeHtml(label)}:</strong> Matched <b>${escapeHtml(matched)}/${escapeHtml(total)}</b> tracks on ${escapeHtml(destName)}${linkHtml}${unmatchedHtml}`;
        banner.style.display = "block";
    }

    function renderPlaylistResult(playlist, candidateCount = null) {
        state.currentPlaylist = playlist;
        renderTransferReport(null);
        const card = document.getElementById("studioResultCard");
        if (!playlist || !playlist.tracks) {
            card.style.display = "none";
            return;
        }
        card.style.display = "block";
        document.getElementById("resTitleInput").value = playlist.playlist_name || "Generated Playlist";
        document.getElementById("resDescInput").value = playlist.playlist_description || "";

        const tracks = playlist.tracks || [];
        const bpms = tracks.map((t) => t.bpm).filter((v) => typeof v === "number");
        const engs = tracks.map((t) => t.energy).filter((v) => typeof v === "number");
        const vals = tracks.map((t) => t.valence).filter((v) => typeof v === "number");
        const libCnt = tracks.filter((t) => t.source === "Local Library").length;
        const newCnt = tracks.length - libCnt;

        const badges = [
            `<span class="badge badge-info">${escapeHtml(tracks.length)} tracks</span>`,
            `<span class="badge badge-lib">${escapeHtml(libCnt)} Library</span>`,
            `<span class="badge badge-new">${escapeHtml(newCnt)} New/Other</span>`,
            `<span class="badge badge-info">Flow: ${escapeHtml(playlist.flow_mode || "mmr")}</span>`,
        ];
        if (bpms.length) {
            const avgBpm = (bpms.reduce((a, b) => a + b, 0) / bpms.length).toFixed(1);
            badges.push(`<span class="badge badge-ok">Avg ${escapeHtml(avgBpm)} BPM</span>`);
        }
        if (engs.length) {
            const avgEng = (engs.reduce((a, b) => a + b, 0) / engs.length).toFixed(2);
            badges.push(`<span class="badge badge-ok">Energy ${escapeHtml(avgEng)}</span>`);
        }
        if (vals.length) {
            const avgVal = (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(2);
            badges.push(`<span class="badge badge-ok">Valence ${escapeHtml(avgVal)}</span>`);
        }
        if (candidateCount) {
            badges.push(`<span class="badge badge-info">Pool: ${escapeHtml(candidateCount)}</span>`);
        }
        document.getElementById("resMetaBadges").innerHTML = badges.join("");
        document.getElementById("resequenceSelect").value = playlist.flow_mode || "mmr";

        const pid = encodeURIComponent(playlist.id || "last");
        document.getElementById("btnDlCsv").href = `/api/playlists/${pid}/download?format=csv`;
        document.getElementById("btnDlM3u").href = `/api/playlists/${pid}/download?format=m3u`;
        document.getElementById("btnDlMd").href = `/api/playlists/${pid}/download?format=md`;

        const likedMap = (state.feedback && state.feedback.liked) || {};
        const dislikedMap = (state.feedback && state.feedback.disliked) || {};

        const tbody = document.getElementById("studioTracksBody");
        tbody.innerHTML = tracks
            .map((t, idx) => {
                const srcBadge = t.source === "Local Library" ? "badge-lib" : "badge-new";
                const rawProv = String(t.provider || "").toLowerCase();
                const vid = extractYtmVid(t.video_id || t.id);
                const spId =
                    t.spotify_id && SPOTIFY_ID_RE.test(String(t.spotify_id))
                        ? String(t.spotify_id)
                        : (t.id && !String(t.id).startsWith("yt:") && SPOTIFY_ID_RE.test(String(t.id)) ? String(t.id) : "");
                const prov = rawProv || (spId && vid ? "both" : (vid ? "ytmusic" : "spotify"));
                const provBadgeClass = prov === "both" ? "badge-both" : (prov === "ytmusic" ? "badge-ytm" : "badge-sp");
                const provLabel = prov === "both" ? "Both" : (prov === "ytmusic" ? "YT Music" : "Spotify");

                const rawHrefUrl =
                    state.activeProvider === "ytmusic" && t.youtube_url
                        ? t.youtube_url
                        : (t.spotify_url || t.youtube_url || "");
                const hrefUrl = safeHttpUrl(rawHrefUrl);
                const link = hrefUrl
                    ? `<a class="track-link" href="${escapeHtml(hrefUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(t.name)}</a>`
                    : escapeHtml(t.name);
                const canPreview = Boolean(spId || vid);
                const isLiked = Boolean(t.id && likedMap[t.id]);
                const isDisliked = Boolean(t.id && dislikedMap[t.id]);
                return `
                <tr draggable="true" data-idx="${idx}">
                    <td><span class="drag-handle" title="Drag row to reorder">⋮⋮</span></td>
                    <td>${idx + 1}</td>
                    <td>${link}</td>
                    <td>${escapeHtml(t.artist)}</td>
                    <td>
                        <span class="badge ${srcBadge}">${escapeHtml(t.source || "")}</span>
                        <span class="badge ${provBadgeClass}">${escapeHtml(provLabel)}</span>
                    </td>
                    <td>${t.bpm != null ? escapeHtml(t.bpm) : "-"}</td>
                    <td>${t.energy != null ? escapeHtml(t.energy) : "-"}</td>
                    <td>${t.valence != null ? escapeHtml(t.valence) : "-"}</td>
                    <td>${t.camelot ? escapeHtml(t.camelot) : "-"}</td>
                    <td>
                        <div class="btn-row" style="gap:4px;">
                            ${canPreview ? `<button class="btn btn-xs preview-btn" data-sp-id="${escapeHtml(spId)}" data-vid="${escapeHtml(vid)}" data-provider="${escapeHtml(prov)}" data-title="${escapeHtml(t.name)} — ${escapeHtml(t.artist)}" type="button" title="Preview">▶</button>` : ""}
                            ${spId ? `<button class="btn btn-xs save-lib-btn" data-id="${escapeHtml(spId)}" data-title="${escapeHtml(t.name)}" type="button" title="Save to Spotify Library">♡</button>` : ""}
                            <button class="btn btn-xs fb-btn like-btn ${isLiked ? "active-like" : ""}" data-idx="${idx}" type="button" title="Like track">+</button>
                            <button class="btn btn-xs fb-btn dislike-btn ${isDisliked ? "active-dislike" : ""}" data-idx="${idx}" type="button" title="Dislike track (block future)">−</button>
                            <button class="btn btn-xs swap-btn" data-idx="${idx}" type="button" title="Swap track">↻</button>
                            <button class="btn btn-xs btn-danger remove-btn" data-idx="${idx}" type="button" title="Remove track">✕</button>
                        </div>
                    </td>
                </tr>`;
            })
            .join("");
    }

    async function loadLastPlaylistIntoStudio() {
        try {
            const res = await apiJson("/api/playlists");
            if (res.last && res.last.tracks && res.last.tracks.length) {
                renderPlaylistResult(res.last);
            }
        } catch (err) {
            console.error(err);
        }
    }

    // =====================================================
    // TASTE MAP (2D / 3D / RADAR)
    // =====================================================
    async function loadTasteMap() {
        try {
            const data = await apiJson("/api/taste-map");
            const emptyEl = document.getElementById("mapEmptyState");
            const contentEl = document.getElementById("mapContent");
            if (!data.available) {
                emptyEl.style.display = "block";
                contentEl.style.display = "none";
                return;
            }
            emptyEl.style.display = "none";
            contentEl.style.display = "block";
            state.mapData = data;
            state.drillClusterId = null;

            state.mapMaxWeight = Math.max(0, ...data.tracks.map((t) => Number(t.recency_weight) || 0));

            const projMethod = (data.projection_method || "pca").toLowerCase();
            document.getElementById("mapProjectionBadge").textContent = projMethod.toUpperCase();
            document.getElementById("mapProjectionSelect").value = projMethod;

            syncMapDrillView();
        } catch (err) {
            showToast(err.message, "err");
        }
    }

    function getMapMinWeight() {
        const slider = document.getElementById("mapWeightSlider");
        const fraction = Number(slider.value || 0) / 1000;
        // Expand the low end: most listening weights are fractions or below 20.
        const weight = Number((0.1 * Math.expm1(fraction * Math.log1p(state.mapMaxWeight / 0.1))).toFixed(2));
        document.getElementById("mapWeightVal").textContent = String(weight);
        slider.setAttribute("aria-valuetext", `${weight} minimum weight`);
        return weight;
    }

    function computeMapMarkerStyle(item, color, q, minW, is3d) {
        const hasFilter = Boolean(q || minW > 0);
        const textMatch = !q || item.searchKey.includes(q);
        const weightMatch = item.weight >= minW;
        const active = textMatch && weightMatch;
        if (!hasFilter) {
            return {
                active: true,
                color,
                size: is3d ? item.baseSize3d : item.baseSize2d,
                opacity: item.baseOp,
                lineColor: "rgba(0, 0, 0, 0.25)",
                lineWidth: 0.5,
            };
        }
        if (active) {
            return {
                active: true,
                color,
                size: is3d ? Math.max(9, item.baseSize3d + 4) : Math.max(14, item.baseSize2d + 5),
                opacity: 1.0,
                lineColor: "rgba(255, 255, 255, 0.85)",
                lineWidth: 1.5,
            };
        }
        return {
            active: false,
            color: is3d ? "rgba(68, 70, 80, 0.14)" : "rgba(74, 76, 86, 0.12)",
            size: is3d ? 2.5 : 3,
            opacity: 0.14,
            lineColor: "rgba(0, 0, 0, 0)",
            lineWidth: 0,
        };
    }

    function getActivePlotEl() {
        const isDrilled = Boolean(getActiveDrillCluster()?.subclusters?.length);
        const drillEl = document.getElementById("tasteMapDrillPlot");
        if (isDrilled && drillEl && drillEl.dataset?.role === "drill") {
            return drillEl;
        }
        return document.getElementById("tasteMapPlot");
    }

    function setPlotLayerVisibility(isDrilled) {
        const overviewEl = document.getElementById("tasteMapPlot");
        const drillEl = document.getElementById("tasteMapDrillPlot");
        if (!drillEl || drillEl.dataset?.role !== "drill") return;
        overviewEl?.classList?.toggle("plot-hidden", isDrilled);
        drillEl.classList?.toggle("plot-active", isDrilled);
    }

    function applyTasteMapFilter() {
        const plotEl = getActivePlotEl();
        if (!state.mapData || typeof Plotly === "undefined" || !plotEl || !plotEl.data || !state.mapTraceIndex.length) {
            renderTasteMapPlot();
            return;
        }
        const q = document.getElementById("mapSearchBox").value.trim().toLowerCase();
        const minW = getMapMinWeight();
        const hasFilter = Boolean(q || minW > 0);

        let matches = 0;
        const traceIndices = [];
        const colorArrays = [];
        const sizeArrays = [];
        const opArrays = [];
        const lineColorArrays = [];
        const lineWidthArrays = [];

        state.mapTraceIndex.forEach((tr) => {
            const cArr = [];
            const sArr = [];
            const oArr = [];
            const lcArr = [];
            const lwArr = [];
            tr.items.forEach((item) => {
                const st = computeMapMarkerStyle(item, tr.color, q, minW, state.is3d);
                if (hasFilter && st.active) matches += 1;
                cArr.push(st.color);
                sArr.push(st.size);
                oArr.push(st.opacity);
                lcArr.push(st.lineColor);
                lwArr.push(st.lineWidth);
            });
            traceIndices.push(tr.traceIdx);
            colorArrays.push(cArr);
            sizeArrays.push(sArr);
            opArrays.push(oArr);
            lineColorArrays.push(lcArr);
            lineWidthArrays.push(lwArr);
        });

        document.getElementById("mapMatchCount").textContent = hasFilter ? `${matches} matching tracks` : "";
        Promise.resolve(
            Plotly.restyle(
                plotEl,
                {
                    "marker.color": colorArrays,
                    "marker.size": sizeArrays,
                    "marker.opacity": opArrays,
                    "marker.line.color": lineColorArrays,
                    "marker.line.width": lineWidthArrays,
                },
                traceIndices
            )
        ).then(() => {
            if (state.is3d) disableMesh3dPicking(plotEl);
        });
    }

    function updateTasteMapClouds() {
        const plotEl = getActivePlotEl();
        if (typeof Plotly === "undefined" || !plotEl?.data) return;
        const indices = plotEl.data.flatMap((trace, idx) => trace.meta?.layer === "cluster-cloud" ? [idx] : []);
        if (!indices.length) return;
        const visible = document.getElementById("chkShowClouds").checked;
        return Promise.resolve(Plotly.restyle(plotEl, { visible }, indices)).then(() => {
            if (state.is3d) disableMesh3dPicking(plotEl);
        });
    }

    function hexToRgba(hex, alpha) {
        const clean = String(hex || "").replace("#", "");
        if (clean.length !== 6) return `rgba(229, 160, 75, ${alpha})`;
        const r = parseInt(clean.slice(0, 2), 16);
        const g = parseInt(clean.slice(2, 4), 16);
        const b = parseInt(clean.slice(4, 6), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    function trimOutliers(points, maxSigma = 1.35) {
        if (!Array.isArray(points) || points.length < 5) return points || [];
        const n = points.length;
        const mx = points.reduce((s, p) => s + p.x, 0) / n;
        const my = points.reduce((s, p) => s + p.y, 0) / n;
        const mz = points.reduce((s, p) => s + (p.z || 0), 0) / n;
        const dists = points.map((p) => Math.hypot(p.x - mx, p.y - my, (p.z || 0) - mz));
        const meanD = dists.reduce((s, d) => s + d, 0) / n;
        const varD = dists.reduce((s, d) => s + (d - meanD) ** 2, 0) / n;
        const stdD = Math.sqrt(varD);
        if (stdD < 1e-9) return points;
        const limit = meanD + maxSigma * stdD;
        const kept = points.filter((_, i) => dists[i] <= limit);
        return kept.length >= 4 ? kept : points;
    }

    function disableMesh3dPicking(plotEl) {
        try {
            const scene = plotEl?._fullLayout?.scene?._scene;
            if (!scene) return;
            const noop = () => {};
            const nullPick = () => null;
            if (scene.traces && typeof scene.traces === "object") {
                Object.values(scene.traces).forEach((tr) => {
                    const isMesh = tr?.data?.type === "mesh3d" || tr?.type === "mesh3d";
                    if (isMesh && tr.mesh) {
                        tr.mesh.drawPick = noop;
                        tr.mesh.pick = nullPick;
                    }
                });
            }
            if (Array.isArray(scene.glplot?.objects)) {
                scene.glplot.objects.forEach((obj) => {
                    if (obj && (obj._trace?.data?.type === "mesh3d" || obj._trace?.type === "mesh3d")) {
                        obj.drawPick = noop;
                        obj.pick = nullPick;
                    }
                });
                scene.glplot.dirty = true;
            }
        } catch {
            // Non-fatal fallback if Plotly internals differ
        }
    }

    function computeConvexHull2D(points) {
        if (!Array.isArray(points) || points.length < 3) return points || [];
        const sorted = [...points].sort((a, b) => (a.x !== b.x ? a.x - b.x : a.y - b.y));
        const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
        const lower = [];
        for (const p of sorted) {
            while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
                lower.pop();
            }
            lower.push(p);
        }
        const upper = [];
        for (let i = sorted.length - 1; i >= 0; i--) {
            const p = sorted[i];
            while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
                upper.pop();
            }
            upper.push(p);
        }
        upper.pop();
        lower.pop();
        const hull = lower.concat(upper);
        if (hull.length >= 3) hull.push(hull[0]);
        return hull;
    }

    function getActiveDrillCluster() {
        if (!state.mapData || state.drillClusterId === null || state.drillClusterId === undefined) return null;
        return (state.mapData.clusters || []).find((c) => Number(c.cluster_id) === Number(state.drillClusterId)) || null;
    }

    function setMapLoading(visible, text = "Rendering map…") {
        const overlay = document.getElementById("mapLoadingOverlay");
        if (!overlay) return;
        const txt = document.getElementById("mapLoadingText");
        if (txt && text) txt.textContent = text;
        overlay.style.display = visible ? "flex" : "none";
    }

    function runWithMapLoader(text, workFn) {
        if (typeof requestAnimationFrame !== "function") {
            return Promise.resolve().then(workFn);
        }
        setMapLoading(true, text);
        return new Promise((resolve) => {
            requestAnimationFrame(() => {
                setTimeout(() => {
                    Promise.resolve(workFn())
                        .catch(() => {})
                        .finally(() => {
                            setMapLoading(false);
                            resolve();
                        });
                }, 0);
            });
        });
    }

    function sampleHullPoints3D(points, maxPts = 28) {
        if (!Array.isArray(points) || points.length <= maxPts) return points || [];
        // Keep the 6 axis extremes + farthest points from centroid so the 3D hull shape stays intact
        // while keeping Plotly's synchronous Delaunay alphahull fast (<2ms per cloud).
        const n = points.length;
        const mx = points.reduce((s, p) => s + p.x, 0) / n;
        const my = points.reduce((s, p) => s + p.y, 0) / n;
        const mz = points.reduce((s, p) => s + (p.z || 0), 0) / n;
        const ranked = points
            .map((p, idx) => ({ p, d: Math.hypot(p.x - mx, p.y - my, (p.z || 0) - mz), idx }))
            .sort((a, b) => b.d - a.d);
        const picked = [];
        const step = Math.max(1, Math.floor(ranked.length / maxPts));
        for (let i = 0; i < ranked.length && picked.length < maxPts; i += step) {
            picked.push(ranked[i].p);
        }
        return picked;
    }

    function syncMapDrillView(withLoader = false) {
        const data = state.mapData;
        if (!data) return;
        const drillBar = document.getElementById("mapDrillBar");
        const drillTitle = document.getElementById("mapDrillTitle");
        const drillMeta = document.getElementById("mapDrillMeta");
        const active = getActiveDrillCluster();

        const applyView = () => {
            if (active && Array.isArray(active.subclusters) && active.subclusters.length) {
                if (drillBar) drillBar.style.display = "flex";
                if (drillTitle) drillTitle.textContent = `C${active.cluster_id}: ${active.vibe_label}`;
                if (drillMeta) drillMeta.textContent = `${active.track_count} tracks · ${active.subclusters.length} sub-vibes`;
                renderClusterCards(active.subclusters, active);
                const p = renderTasteMapPlot();
                if (typeof setTimeout === "function") {
                    setTimeout(() => renderClusterRadar(active.subclusters.map((sc) => ({ ...sc, cluster_id: `S${sc.sub_cluster_id}` }))), 0);
                } else {
                    renderClusterRadar(active.subclusters.map((sc) => ({ ...sc, cluster_id: `S${sc.sub_cluster_id}` })));
                }
                return p;
            }
            state.drillClusterId = null;
            if (drillBar) drillBar.style.display = "none";
            renderClusterCards(data.clusters || [], null);
            const p = renderTasteMapPlot();
            if (typeof setTimeout === "function") {
                setTimeout(() => renderClusterRadar(data.clusters || []), 0);
            } else {
                renderClusterRadar(data.clusters || []);
            }
            return p;
        };

        if (withLoader && state.is3d && active) {
            return runWithMapLoader(`Zooming into ${active.vibe_label}…`, applyView);
        }
        return applyView();
    }

    function enterClusterDrill(cid) {
        const target = (state.mapData?.clusters || []).find((c) => Number(c.cluster_id) === Number(cid));
        if (!target || !Array.isArray(target.subclusters) || !target.subclusters.length) {
            showToast("This cluster does not have sub-vibes to zoom into.", "info");
            return;
        }
        state.drillClusterId = Number(cid);
        syncMapDrillView(true);
    }

    function exitClusterDrill() {
        if (state.drillClusterId === null) return;
        state.drillClusterId = null;
        syncMapDrillView(true);
    }

    function renderTasteMapPlot() {
        const data = state.mapData;
        if (!data) return;
        const activeDrill = getActiveDrillCluster();
        const isDrilled = Boolean(activeDrill && Array.isArray(activeDrill.subclusters) && activeDrill.subclusters.length);
        setPlotLayerVisibility(isDrilled);
        const plotEl = getActivePlotEl();
        if (typeof Plotly === "undefined") {
            document.getElementById("mapOfflineBanner").style.display = "block";
            return;
        }
        document.getElementById("mapOfflineBanner").style.display = "none";

        const q = document.getElementById("mapSearchBox").value.trim().toLowerCase();
        const minW = getMapMinWeight();
        const hasFilter = Boolean(q || minW > 0);
        const showEraPath = !isDrilled && document.getElementById("chkShowEraPath").checked;
        const showCentroids = document.getElementById("chkShowCentroids")?.checked !== false;
        const showClouds = document.getElementById("chkShowClouds")?.checked !== false;

        if (state.mapCacheSource !== data) {
            state.mapViewCache.clear();
            state.mapCacheSource = data;
            state.overviewRenderedKey = null;
            state.drillRenderedKey = null;
        }
        const cacheKey = `${state.is3d ? "3d" : "2d"}:${isDrilled ? activeDrill.cluster_id : "all"}:${showCentroids ? 1 : 0}:${showEraPath ? 1 : 0}`;
        let cached = state.mapViewCache.get(cacheKey);
        let matches = 0;

        // Instant 0ms swap: if this container already has the exact same view rendered in WebGL/DOM,
        // skip Plotly.react completely and only restyle clouds/filters if they changed.
        const useSeparateDrill = Boolean(document.getElementById("tasteMapDrillPlot")?.dataset?.role === "drill");
        const renderedKey = isDrilled ? state.drillRenderedKey : state.overviewRenderedKey;
        if (useSeparateDrill && cached && plotEl.data && renderedKey === cacheKey) {
            state.mapTraceIndex = cached.traceIndex;
            state.mapLegendClusterIds = cached.legendClusterIds;
            const filterKey = `${q}|${minW}`;
            if (hasFilter || cached.lastFilterKey !== "") {
                applyTasteMapFilter();
                cached.lastFilterKey = hasFilter ? filterKey : "";
            } else {
                document.getElementById("mapMatchCount").textContent = "";
            }
            const firstCloud = plotEl.data.find((tr) => tr?.meta?.layer === "cluster-cloud");
            if (firstCloud && firstCloud.visible !== showClouds) {
                updateTasteMapClouds();
            }
            return Promise.resolve();
        }

        if (!cached) {
            const traces = [];
            const centroids = [];
            const legendClusterIds = {};
            const traceIndex = [];

            const displayGroups = isDrilled
                ? activeDrill.subclusters.map((sc, idx) => ({
                      id: sc.sub_cluster_id,
                      prefix: "S",
                      label: sc.vibe_label,
                      share: sc.raw_percentage,
                      color: CLUSTER_COLORS[idx % CLUSTER_COLORS.length],
                      pts: (data.tracks || []).filter(
                          (t) => Number(t.cluster_id) === Number(activeDrill.cluster_id) && Number(t.sub_cluster_id) === Number(sc.sub_cluster_id)
                      ),
                  }))
                : (data.clusters || []).map((c, idx) => ({
                      id: c.cluster_id,
                      prefix: "C",
                      label: c.vibe_label,
                      share: c.weighted_share_pct,
                      color: CLUSTER_COLORS[idx % CLUSTER_COLORS.length],
                      pts: (data.tracks || []).filter((t) => Number(t.cluster_id) === Number(c.cluster_id)),
                  }));

            displayGroups.forEach((grp) => {
                if (grp.pts.length < 4) return;
                const rawCoords = grp.pts.map((t) => ({
                    x: state.is3d ? t.x3 : t.x,
                    y: state.is3d ? t.y3 : t.y,
                    z: t.z || 0,
                }));
                const trimmed = trimOutliers(rawCoords, 1.35);
                if (trimmed.length < 4) return;

                if (state.is3d) {
                    const hullPts = sampleHullPoints3D(trimmed, 28);
                    traces.push({
                        type: "mesh3d",
                        x: hullPts.map((p) => p.x),
                        y: hullPts.map((p) => p.y),
                        z: hullPts.map((p) => p.z),
                        alphahull: 0,
                        color: grp.color,
                        opacity: 0.08,
                        flatshading: true,
                        lighting: { ambient: 0.9, diffuse: 0.3, specular: 0.0 },
                        hoverinfo: "skip",
                        showlegend: false,
                        meta: { layer: "cluster-cloud" },
                        visible: showClouds,
                    });
                } else {
                    const hull = computeConvexHull2D(trimmed);
                    if (hull.length >= 4) {
                        traces.push({
                            type: "scatter",
                            mode: "lines",
                            x: hull.map((p) => p.x),
                            y: hull.map((p) => p.y),
                            fill: "toself",
                            fillcolor: hexToRgba(grp.color, 0.06),
                            line: { color: hexToRgba(grp.color, 0.22), width: 1 },
                            hoverinfo: "skip",
                            showlegend: false,
                            meta: { layer: "cluster-cloud" },
                            visible: showClouds,
                        });
                    }
                }
            });

            const appendTrackGroupTrace = (pts, color, legendName, clusterLabel, options = {}) => {
                const { collectCentroid = false, clusterId = -1, prefix = "C", sizeScale = 1.0, fixedOp = null } = options;
                const x = [];
                const y = [];
                const z = [];
                const colorsArr = [];
                const sizes = [];
                const ops = [];
                const lineColors = [];
                const lineWidths = [];
                const texts = [];
                const meta = [];
                const indexedItems = [];

                pts.forEach((t) => {
                    const raw2d = Math.min(20, Math.max(5, Math.round(4 + Math.pow(t.recency_weight || 1, 0.6) * 2)));
                    const baseSize2d = Math.max(4, Math.round(raw2d * sizeScale));
                    const baseSize3d = Math.max(2.5, Math.round(baseSize2d * 0.55));
                    const baseOp = fixedOp != null ? fixedOp : Math.min(1.0, Math.max(0.35, 0.35 + (t.recency_weight || 1) / 30.0));
                    const item = {
                        searchKey: `${t.name} ${t.artist} ${t.genres}`.toLowerCase(),
                        weight: Number(t.recency_weight || 0),
                        baseSize2d,
                        baseSize3d,
                        baseOp,
                    };
                    indexedItems.push(item);

                    const st = computeMapMarkerStyle(item, color, "", 0, state.is3d);
                    x.push(state.is3d ? t.x3 : t.x);
                    y.push(state.is3d ? t.y3 : t.y);
                    z.push(t.z);
                    colorsArr.push(st.color);
                    sizes.push(st.size);
                    ops.push(st.opacity);
                    lineColors.push(st.lineColor);
                    lineWidths.push(st.lineWidth);
                    meta.push({
                        url: t.spotify_url || "",
                        yt_url: t.youtube_url || "",
                        id: t.id || "",
                        vid: t.video_id || "",
                        provider: t.provider || "spotify",
                        name: t.name || "",
                        artist: t.artist || "",
                        label: `${t.name} — ${t.artist}`,
                    });

                    const afLine = t.bpm != null ? `<br>Audio: ${t.bpm} BPM | Energy ${t.energy} | Valence ${t.valence}` : "";
                    texts.push(
                        `<b>${plotlyText(t.name)}</b><br>${plotlyText(t.artist)}` +
                        `<br>Cluster: ${plotlyText(clusterLabel)}` +
                        `<br>Genres: ${plotlyText(t.genres || "None")}` +
                        afLine +
                        `<br>Weight: ${t.recency_weight}` +
                        (t.id ? "<br>Click to preview or open track" : "")
                    );
                });

                if (collectCentroid && pts.length) {
                    centroids.push({
                        cx: x.reduce((acc, v) => acc + v, 0) / x.length,
                        cy: y.reduce((acc, v) => acc + v, 0) / y.length,
                        cz: z.reduce((acc, v) => acc + v, 0) / z.length,
                        color,
                        label: `${prefix}${clusterId} · ${clusterLabel}`,
                    });
                }

                traceIndex.push({
                    traceIdx: traces.length,
                    color,
                    items: indexedItems,
                });

                const trace = {
                    x, y,
                    mode: "markers",
                    type: state.is3d ? "scatter3d" : "scattergl",
                    name: legendName,
                    text: texts,
                    customdata: meta,
                    hoverinfo: "text",
                    marker: { size: sizes, color: colorsArr, opacity: ops, line: { color: lineColors, width: lineWidths } },
                };
                if (state.is3d) trace.z = z;
                traces.push(trace);
            };

            displayGroups.forEach((grp) => {
                if (!isDrilled) {
                    legendClusterIds[traces.length] = grp.id;
                }
                appendTrackGroupTrace(
                    grp.pts,
                    grp.color,
                    `${grp.prefix}${grp.id}: ${grp.label} (${grp.share}%)`,
                    grp.label,
                    { collectCentroid: true, clusterId: grp.id, prefix: grp.prefix }
                );
            });

            const unclusteredPts = isDrilled ? [] : (data.tracks || []).filter((t) => t.cluster_id === -1);
            if (unclusteredPts.length) {
                appendTrackGroupTrace(
                    unclusteredPts,
                    "#7a7d88",
                    `Unclustered / Eclectic (${unclusteredPts.length})`,
                    "Unclustered / Eclectic",
                    { collectCentroid: false, sizeScale: 0.78, fixedOp: 0.42 }
                );
            }

            if (showCentroids && centroids.length) {
                const centroidTrace = {
                    x: centroids.map((ct) => ct.cx),
                    y: centroids.map((ct) => ct.cy),
                    mode: "markers+text",
                    type: state.is3d ? "scatter3d" : "scatter",
                    name: "Cluster Centers",
                    showlegend: false,
                    hoverinfo: "skip",
                    text: centroids.map((ct) => ct.label),
                    textposition: "top center",
                    textfont: {
                        color: "#ededed",
                        size: state.is3d ? 10 : 11,
                        family: "'IBM Plex Sans', sans-serif",
                    },
                    marker: {
                        size: state.is3d ? 5 : 7,
                        symbol: "diamond",
                        color: centroids.map((ct) => ct.color),
                        line: { color: "#ededed", width: 1 },
                    },
                };
                if (state.is3d) centroidTrace.z = centroids.map((ct) => ct.cz);
                traces.push(centroidTrace);
            }

            const eraTraj = Array.isArray(data.era_trajectory) ? data.era_trajectory : [];
            if (showEraPath && eraTraj.length) {
                if (state.is3d) {
                    traces.push({
                        x: eraTraj.map((e) => e.x3 ?? e.x),
                        y: eraTraj.map((e) => e.y3 ?? e.y),
                        z: eraTraj.map((e) => e.z ?? 0),
                        mode: "lines+markers+text",
                        type: "scatter3d",
                        name: "Era Evolution Path",
                        text: eraTraj.map((e) => String(e.year)),
                        hoverinfo: "text",
                        hovertext: eraTraj.map((e) => `<b>${escapeHtml(e.year)}</b>: ${escapeHtml(e.plays)} plays`),
                        line: { color: "#e5a04b", width: 4 },
                        marker: { size: 6, color: "#e5a04b" },
                    });
                } else {
                    traces.push({
                        x: eraTraj.map((e) => e.x),
                        y: eraTraj.map((e) => e.y),
                        mode: "lines+markers+text",
                        type: "scatter",
                        name: "Era Evolution Path",
                        text: eraTraj.map((e) => String(e.year)),
                        textposition: "top center",
                        hoverinfo: "text",
                        hovertext: eraTraj.map((e) => `<b>${escapeHtml(e.year)}</b>: ${escapeHtml(e.plays)} plays (${escapeHtml(e.unique_tracks)} tracks)`),
                        line: { color: "#e5a04b", width: 3, dash: "dot" },
                        marker: { size: 10, color: "#e5a04b", symbol: "circle-open", line: { width: 2, color: "#ededed" } },
                    });
                }
            }

            const projUpper = (data.projection_method || "pca").toUpperCase();
            const ev2 = data.explained_variance_2d || [0, 0];
            const ev3 = data.explained_variance_3d || [0, 0, 0];
            const axisLabel = (idx, evArr) => {
                if (projUpper === "PCA" && evArr[idx] > 0) {
                    return `PC${idx + 1} (${(evArr[idx] * 100).toFixed(1)}%)`;
                }
                return `${projUpper} ${idx + 1}`;
            };
            const uiRev = isDrilled ? `taste-map-drill-${activeDrill.cluster_id}` : "taste-map-v1";
            let xRange = undefined;
            let yRange = undefined;
            if (isDrilled && !state.is3d) {
                const drillPts = displayGroups.flatMap((g) => g.pts);
                if (drillPts.length) {
                    const xs = drillPts.map((t) => t.x);
                    const ys = drillPts.map((t) => t.y);
                    const minX = Math.min(...xs);
                    const maxX = Math.max(...xs);
                    const minY = Math.min(...ys);
                    const maxY = Math.max(...ys);
                    const padX = Math.max(0.05, (maxX - minX) * 0.14);
                    const padY = Math.max(0.05, (maxY - minY) * 0.14);
                    xRange = [minX - padX, maxX + padX];
                    yRange = [minY - padY, maxY + padY];
                }
            }
            const layout = state.is3d
                ? {
                      uirevision: uiRev,
                      paper_bgcolor: "#19191b",
                      font: { color: "#a1a1a6", family: "'IBM Plex Mono', monospace", size: 11 },
                      scene: {
                          bgcolor: "#141415",
                          xaxis: { title: axisLabel(0, ev3), gridcolor: "rgba(237,237,237,0.08)" },
                          yaxis: { title: axisLabel(1, ev3), gridcolor: "rgba(237,237,237,0.08)" },
                          zaxis: { title: axisLabel(2, ev3), gridcolor: "rgba(237,237,237,0.08)" },
                      },
                      margin: { l: 10, r: 10, t: 20, b: 30 },
                      legend: { orientation: "h", y: -0.05 },
                  }
                : {
                      uirevision: uiRev,
                      paper_bgcolor: "#19191b",
                      plot_bgcolor: "#141415",
                      font: { color: "#a1a1a6", family: "'IBM Plex Mono', monospace", size: 11 },
                      xaxis: { title: axisLabel(0, ev2), gridcolor: "rgba(237,237,237,0.08)", ...(xRange ? { range: xRange } : {}) },
                      yaxis: { title: axisLabel(1, ev2), gridcolor: "rgba(237,237,237,0.08)", ...(yRange ? { range: yRange } : {}) },
                      margin: { l: 50, r: 20, t: 20, b: 70 },
                      legend: { orientation: "h", y: -0.18 },
                  };

            cached = { traces, layout, traceIndex, legendClusterIds, lastFilterKey: "" };
            state.mapViewCache.set(cacheKey, cached);
        }

        // Sync cloud visibility and active search/weight filter onto cached traces
        cached.traces.forEach((tr) => {
            if (tr?.meta?.layer === "cluster-cloud") tr.visible = showClouds;
        });
        const filterKey = `${q}|${minW}`;
        if (hasFilter || cached.lastFilterKey !== "") {
            cached.traceIndex.forEach((entry) => {
                const tr = cached.traces[entry.traceIdx];
                if (!tr?.marker) return;
                const colorsArr = [];
                const sizes = [];
                const ops = [];
                const lineColors = [];
                const lineWidths = [];
                entry.items.forEach((item) => {
                    const st = computeMapMarkerStyle(item, entry.color, q, minW, state.is3d);
                    if (hasFilter && st.active) matches += 1;
                    colorsArr.push(st.color);
                    sizes.push(st.size);
                    ops.push(st.opacity);
                    lineColors.push(st.lineColor);
                    lineWidths.push(st.lineWidth);
                });
                tr.marker.color = colorsArr;
                tr.marker.size = sizes;
                tr.marker.opacity = ops;
                tr.marker.line = { color: lineColors, width: lineWidths };
            });
            cached.lastFilterKey = hasFilter ? filterKey : "";
        }

        state.mapTraceIndex = cached.traceIndex;
        state.mapLegendClusterIds = cached.legendClusterIds;
        document.getElementById("mapMatchCount").textContent = hasFilter ? `${matches} matching tracks` : "";

        const plotFn = typeof Plotly.react === "function" && plotEl.data ? Plotly.react : Plotly.newPlot;
        const plotPromise = Promise.resolve(plotFn(plotEl, cached.traces, cached.layout, { responsive: true })).then(() => {
            if (isDrilled) state.drillRenderedKey = cacheKey;
            else state.overviewRenderedKey = cacheKey;
            if (state.is3d) disableMesh3dPicking(plotEl);
        });
        if (!plotEl._tasteMapListenersBound) {
            plotEl._tasteMapListenersBound = true;
            plotEl.on("plotly_legendclick", (evt) => {
                if (state.drillClusterId === null && evt && typeof evt.curveNumber === "number" && state.mapLegendClusterIds[evt.curveNumber] !== undefined) {
                    enterClusterDrill(state.mapLegendClusterIds[evt.curveNumber]);
                    return false;
                }
                return true;
            });
            plotEl.on("plotly_click", (evt) => {
                const cd = evt?.points?.[0]?.customdata;
                if (!cd || typeof cd !== "object") return;
                const action = document.getElementById("mapClickAction").value;
                if (action === "preview" && (cd.id || cd.vid)) {
                    openPreviewTrack({ id: cd.id, video_id: cd.vid, provider: cd.provider, name: cd.name || cd.label, artist: cd.artist || "" }, cd.label);
                } else {
                    const targetUrl = safeHttpUrl(cd.url || cd.yt_url);
                    if (targetUrl) window.open(targetUrl, "_blank", "noopener,noreferrer");
                }
            });
        }
        return plotPromise;
    }

    function renderClusterRadar(clusters) {
        const el = document.getElementById("clusterRadarPlot");
        if (typeof Plotly === "undefined") return;
        const traces = clusters.map((c, idx) => {
            const t = c.telemetry || {};
            const normBpm = t.avg_bpm != null ? Math.min(1, Math.max(0, (t.avg_bpm - 60) / 120)) : 0.5;
            const vals = [
                t.avg_energy ?? 0.5,
                t.avg_valence ?? 0.5,
                t.avg_danceability ?? 0.5,
                t.avg_acousticness ?? 0.5,
                Number(normBpm.toFixed(2)),
                t.avg_energy ?? 0.5,
            ];
            return {
                type: "scatterpolar",
                r: vals,
                theta: ["Energy", "Valence", "Danceability", "Acousticness", "Tempo", "Energy"],
                fill: "toself",
                name: `C${c.cluster_id}`,
                line: { color: CLUSTER_COLORS[idx % CLUSTER_COLORS.length] },
                opacity: 0.7,
            };
        });
        const radarFn = typeof Plotly.react === "function" && el.data ? Plotly.react : Plotly.newPlot;
        radarFn(
            el,
            traces,
            {
                paper_bgcolor: "#19191b",
                font: { color: "#a1a1a6", family: "'IBM Plex Mono', monospace", size: 11 },
                polar: {
                    bgcolor: "#141415",
                    radialaxis: { visible: true, range: [0, 1], gridcolor: "rgba(237,237,237,0.08)" },
                    angularaxis: { gridcolor: "rgba(237,237,237,0.08)" },
                },
                margin: { l: 40, r: 40, t: 25, b: 25 },
                legend: { orientation: "h", y: -0.15 },
            },
            { responsive: true }
        );
    }

    function renderClusterCards(clusters, parentDrillCluster = null) {
        const grid = document.getElementById("clusterCardsGrid");
        if (parentDrillCluster) {
            grid.innerHTML = clusters
                .map((sc) => {
                    const t = sc.telemetry || {};
                    const metrics = t.avg_bpm != null ? `${t.avg_bpm} BPM · E ${t.avg_energy} · V ${t.avg_valence}` : "No telemetry";
                    const anchors = (sc.anchor_tracks || []).slice(0, 2).map((a) => `<li>${escapeHtml(a.name)} — ${escapeHtml(a.artist)}</li>`).join("");
                    return `
                    <div class="cluster-mini">
                        <div class="cluster-mini-title">S${escapeHtml(sc.sub_cluster_id)}: ${escapeHtml(sc.vibe_label)}</div>
                        <div class="cluster-mini-meta">${escapeHtml(sc.raw_percentage)}% of ${escapeHtml(parentDrillCluster.vibe_label)} · ${escapeHtml(sc.track_count)} tracks · ${escapeHtml(metrics)}</div>
                        <ul class="cluster-mini-anchors">${anchors}</ul>
                    </div>`;
                })
                .join("");
            return;
        }
        grid.innerHTML = clusters
            .map((c) => {
                const t = c.telemetry || {};
                const metrics = t.avg_bpm != null ? `${t.avg_bpm} BPM · E ${t.avg_energy} · V ${t.avg_valence}` : "No telemetry";
                const anchors = (c.anchor_tracks || []).slice(0, 2).map((a) => `<li>${escapeHtml(a.name)} — ${escapeHtml(a.artist)}</li>`).join("");
                const subCount = Array.isArray(c.subclusters) ? c.subclusters.length : 0;
                const zoomBtn = subCount
                    ? `<button class="btn btn-xs btn-primary drill-cluster-btn" data-cid="${escapeHtml(c.cluster_id)}" type="button">Zoom Sub-Vibes (${subCount})</button>`
                    : "";
                return `
                <div class="cluster-mini">
                    <div class="cluster-mini-title ${subCount ? "clickable drill-cluster-btn" : ""}" data-cid="${escapeHtml(c.cluster_id)}">C${escapeHtml(c.cluster_id)}: ${escapeHtml(c.vibe_label)}</div>
                    <div class="cluster-mini-meta">${escapeHtml(c.weighted_share_pct)}% Share · ${escapeHtml(c.track_count)} tracks · ${escapeHtml(metrics)}</div>
                    <ul class="cluster-mini-anchors">${anchors}</ul>
                    <div class="btn-row">
                        ${zoomBtn}
                        <button class="btn btn-xs expand-cluster-btn" data-cid="${escapeHtml(c.cluster_id)}" type="button">Studio</button>
                        <button class="btn btn-xs rename-cluster-btn" data-cid="${escapeHtml(c.cluster_id)}" data-label="${escapeHtml(c.vibe_label)}" type="button">Rename</button>
                    </div>
                </div>`;
            })
            .join("");
    }

    // =====================================================
    // PLAYLIST ARCHIVE, ANALYTICS & FORGOTTEN FAVORITES
    // =====================================================
    async function loadPlaylistsArchive() {
        try {
            const data = await apiJson("/api/playlists");
            const archived = data.archived || [];
            const emptyEl = document.getElementById("archiveEmpty");
            const wrapEl = document.getElementById("archiveTableWrap");
            const bodyEl = document.getElementById("archiveBody");
            if (!archived.length) {
                emptyEl.style.display = "block";
                wrapEl.style.display = "none";
                return;
            }
            emptyEl.style.display = "none";
            wrapEl.style.display = "block";
            bodyEl.innerHTML = archived
                .map(
                    (p) => `
                <tr>
                    <td><b>${escapeHtml(p.playlist_name)}</b></td>
                    <td class="muted">${escapeHtml((p.prompt || "").slice(0, 60))}</td>
                    <td>${escapeHtml(p.track_count)}</td>
                    <td>${p.avg_bpm != null ? escapeHtml(p.avg_bpm) : "-"}</td>
                    <td>${escapeHtml(p.flow_mode || "mmr")}</td>
                    <td>
                        <div class="btn-row">
                            <button class="btn btn-xs arch-load-btn" data-id="${escapeHtml(p.id)}" type="button">Open</button>
                            <button class="btn btn-xs btn-primary arch-exp-btn" data-id="${escapeHtml(p.id)}" type="button">Export</button>
                            <a class="btn btn-xs" href="/api/playlists/${encodeURIComponent(p.id)}/download?format=csv">CSV</a>
                            <button class="btn btn-xs btn-danger arch-del-btn" data-id="${escapeHtml(p.id)}" type="button">Del</button>
                        </div>
                    </td>
                </tr>`
                )
                .join("");
        } catch (err) {
            console.error(err);
        }
    }

    async function loadAnalytics() {
        try {
            const data = await apiJson("/api/analytics");
            const missEl = document.getElementById("extAnalyticsMissing");
            const contEl = document.getElementById("extAnalyticsContent");
            if (!data.available) {
                missEl.style.display = "block";
                contEl.style.display = "none";
                return;
            }
            missEl.style.display = "none";
            contEl.style.display = "block";
            state.analytics = data;

            const ov = data.overview || {};
            document.getElementById("analyticsOverviewCards").innerHTML = [
                { l: "Total Valid Plays", v: (ov.total_plays || 0).toLocaleString(), s: "Streams ≥ 30s" },
                { l: "Total Listening Time", v: `${(ov.total_hours || 0).toLocaleString()} hrs`, s: `${(ov.unique_tracks || 0).toLocaleString()} unique tracks` },
                { l: "Skip Rate", v: `${ov.avg_skip_rate_pct || 0}%`, s: `${(ov.total_skips || 0).toLocaleString()} skips filtered` },
                { l: "History Span", v: `${(ov.earliest_stream || "").slice(0, 4)} → ${(ov.latest_stream || "").slice(0, 4)}`, s: "UTC Streaming Archive" },
            ]
                .map(
                    (c) => `
                <div class="status-card">
                    <div class="status-label">${escapeHtml(c.l)}</div>
                    <div class="status-val">${escapeHtml(c.v)}</div>
                    <div class="status-sub">${escapeHtml(c.s)}</div>
                </div>`
                )
                .join("");

            if (typeof Plotly !== "undefined") {
                const yrs = data.yearly_evolution || [];
                Plotly.newPlot(
                    "yearlyChartPlot",
                    [
                        { x: yrs.map((r) => r.year), y: yrs.map((r) => r.hours), type: "bar", name: "Hours", marker: { color: "#85858b" } },
                        { x: yrs.map((r) => r.year), y: yrs.map((r) => r.avg_bpm), type: "scatter", mode: "lines+markers", name: "Avg BPM", yaxis: "y2", line: { color: "#e5a04b" } },
                    ],
                    {
                        paper_bgcolor: "#19191b",
                        plot_bgcolor: "#141415",
                        font: { color: "#a1a1a6", family: "'IBM Plex Mono', monospace", size: 11 },
                        yaxis: { title: "Hours", gridcolor: "rgba(237,237,237,0.08)" },
                        yaxis2: { title: "Avg BPM", overlaying: "y", side: "right" },
                        margin: { l: 45, r: 45, t: 20, b: 35 },
                        legend: { orientation: "h", y: -0.18 },
                    },
                    { responsive: true }
                );

                const hrs = data.hourly_distribution || [];
                Plotly.newPlot(
                    "hourlyChartPlot",
                    [{ x: hrs.map((h) => `${h.hour}:00`), y: hrs.map((h) => h.plays), type: "bar", marker: { color: "rgba(229, 160, 75, 0.72)" } }],
                    {
                        paper_bgcolor: "#19191b",
                        plot_bgcolor: "#141415",
                        font: { color: "#a1a1a6", family: "'IBM Plex Mono', monospace", size: 11 },
                        yaxis: { title: "Plays", gridcolor: "rgba(237,237,237,0.08)" },
                        margin: { l: 45, r: 15, t: 20, b: 45 },
                    },
                    { responsive: true }
                );
            }

            document.getElementById("topArtistsList").innerHTML = `<table class="data-table"><thead><tr><th>Artist</th><th>Hours</th><th>Plays</th></tr></thead><tbody>${
                (data.top_artists || [])
                    .map((a) => `<tr><td>${escapeHtml(a.artist)}</td><td>${escapeHtml(a.hours)}</td><td>${escapeHtml(a.plays)}</td></tr>`)
                    .join("")
            }</tbody></table>`;

            document.getElementById("topTracksList").innerHTML = `<table class="data-table"><thead><tr><th>Track</th><th>Artist</th><th>Plays</th><th></th></tr></thead><tbody>${
                (data.top_tracks || [])
                    .map((t) => {
                        const prev = t.id && SPOTIFY_ID_RE.test(String(t.id))
                            ? `<button class="btn btn-xs preview-btn" data-id="${escapeHtml(t.id)}" data-title="${escapeHtml(t.name)} — ${escapeHtml(t.artist)}" type="button">▶</button>`
                            : "";
                        return `<tr><td>${escapeHtml(t.name)}</td><td>${escapeHtml(t.artist)}</td><td>${escapeHtml(t.play_count)}</td><td>${prev}</td></tr>`;
                    })
                    .join("")
            }</tbody></table>`;

            document.getElementById("mostSkippedList").innerHTML = `<table class="data-table"><thead><tr><th>Track</th><th>Skip %</th><th>Skips</th></tr></thead><tbody>${
                (data.most_skipped || [])
                    .map((s) => `<tr><td>${escapeHtml(s.name)}</td><td>${escapeHtml(s.skip_rate_pct)}%</td><td>${escapeHtml(s.skip_count)}</td></tr>`)
                    .join("")
            }</tbody></table>`;
        } catch (err) {
            console.error(err);
        }
    }

    async function runForgottenFavorites(saveAsPlaylist) {
        try {
            const res = await apiJson("/api/analytics/forgotten-favorites", {
                method: "POST",
                body: {
                    min_plays: parseInt(document.getElementById("ffMinPlays").value || "8", 10),
                    dormant_days: parseInt(document.getElementById("ffDormantDays").value || "180", 10),
                    limit: parseInt(document.getElementById("ffLimit").value || "30", 10),
                    use_session_exclusions: true,
                    save_as_playlist: saveAsPlaylist,
                    flow_mode: "mmr",
                },
            });
            const tracks = res.tracks || [];
            const wrap = document.getElementById("forgottenTableWrap");
            const body = document.getElementById("forgottenBody");
            if (!tracks.length) {
                wrap.style.display = "none";
                showToast("No Forgotten Favorites matched those thresholds.", "info");
                return;
            }
            wrap.style.display = "block";
            body.innerHTML = tracks
                .map(
                    (t, idx) => `
                <tr>
                    <td>${idx + 1}</td>
                    <td>${escapeHtml(t.name)} ${t.id && SPOTIFY_ID_RE.test(String(t.id)) ? `<button class="btn btn-xs preview-btn" data-id="${escapeHtml(t.id)}" data-title="${escapeHtml(t.name)} — ${escapeHtml(t.artist)}" type="button">▶</button>` : ""}</td>
                    <td>${escapeHtml(t.artist)}</td>
                    <td>${escapeHtml(t.play_count)}</td>
                    <td>${escapeHtml(t.hours)}</td>
                    <td>${escapeHtml((t.last_played || "").slice(0, 10))}</td>
                    <td>${t.bpm != null ? escapeHtml(t.bpm) : "-"}</td>
                </tr>`
                )
                .join("");
            if (saveAsPlaylist && res.playlist) {
                renderPlaylistResult(res.playlist);
                loadPlaylistsArchive();
                switchTab("studio");
                showToast("Forgotten Favorites saved and opened in Studio!", "ok");
            }
        } catch (err) {
            showToast(err.message, "err");
        }
    }

    async function runReleaseRadar(saveAsPlaylist) {
        const sumEl = document.getElementById("radarSummary");
        const wrap = document.getElementById("radarTableWrap");
        const body = document.getElementById("radarBody");
        sumEl.textContent = "Scanning recent releases from your Top & Followed artists...";
        try {
            const res = await apiJson("/api/radar/releases", {
                method: "POST",
                body: {
                    days: parseInt(document.getElementById("radarDays").value || "45", 10),
                    max_artists: parseInt(document.getElementById("radarMaxArtists").value || "20", 10),
                    save_as_playlist: saveAsPlaylist,
                    flow_mode: "mmr",
                },
            });
            const tracks = res.tracks || [];
            const releases = res.releases || [];
            sumEl.textContent = `Found ${tracks.length} tracks across ${releases.length} recent release(s).`;
            if (!tracks.length) {
                wrap.style.display = "none";
                showToast("No recent releases matched in this lookback window.", "info");
                return;
            }
            wrap.style.display = "block";
            body.innerHTML = tracks
                .map((t, idx) => {
                    const hasId = Boolean(t.id && SPOTIFY_ID_RE.test(String(t.id)));
                    return `
                    <tr>
                        <td>${idx + 1}</td>
                        <td>${escapeHtml(t.name)}</td>
                        <td>${escapeHtml(t.artist)}</td>
                        <td>${escapeHtml(t.album || "")}</td>
                        <td>${escapeHtml(t.release_date || "")}</td>
                        <td>${t.bpm != null ? escapeHtml(t.bpm) : "-"}</td>
                        <td>
                            <div class="btn-row" style="gap:4px;">
                                ${hasId ? `<button class="btn btn-xs preview-btn" data-id="${escapeHtml(t.id)}" data-title="${escapeHtml(t.name)} — ${escapeHtml(t.artist)}" type="button">▶</button><button class="btn btn-xs save-lib-btn" data-id="${escapeHtml(t.id)}" data-title="${escapeHtml(t.name)}" type="button" title="Save to Spotify Library">♡</button>` : ""}
                            </div>
                        </td>
                    </tr>`;
                })
                .join("");
            if (saveAsPlaylist && res.playlist) {
                renderPlaylistResult(res.playlist);
                loadPlaylistsArchive();
                switchTab("studio");
                showToast("Release Radar opened in Studio!", "ok");
            }
        } catch (err) {
            sumEl.textContent = `Release Radar error: ${err.message}`;
            showToast(err.message, "err");
        }
    }
    // =====================================================
    function renderPenalties() {
        const wrap = document.getElementById("penaltyList");
        const entries = Object.entries(state.penalizedArtists);
        if (!entries.length) {
            wrap.innerHTML = `<span class="muted">No artist penalties configured.</span>`;
            return;
        }
        wrap.innerHTML = entries
            .map(
                ([name, mult]) =>
                    `<span class="chip chip-artist">${escapeHtml(name)} (×${escapeHtml(mult)}) <button type="button" class="rm-pen" data-art="${escapeHtml(name)}">×</button></span>`
            )
            .join("");
    }

    async function loadYtmAuthStatus() {
        try {
            const st = await apiJson("/api/ytmusic/auth-status");
            const badge = document.getElementById("ytmAuthStatusBadge");
            const det = document.getElementById("ytmAuthStatusDetails");
            if (!badge || !det) return;
            if (st.configured) {
                badge.className = "badge badge-ok";
                badge.textContent = st.method === "oauth" ? "Authenticated (Google OAuth)" : "Authenticated (Browser Headers)";
                det.textContent = st.has_oauth_client
                    ? "Credentials saved locally (gitignored). OAuth Client ID also configured."
                    : "Credentials saved locally (gitignored). Ready for Stage 1 YT Music sync & export.";
            } else if (st.has_oauth_client) {
                badge.className = "badge badge-info";
                badge.textContent = "OAuth Client Saved";
                det.textContent = "Google OAuth Client ID saved. Click 'Start Google OAuth Device Flow' to finish linking.";
            } else {
                badge.className = "badge badge-warn";
                badge.textContent = "Not Linked";
                det.textContent = "Paste Browser Headers from music.youtube.com or enter Google OAuth Client credentials below.";
            }
        } catch (err) {
            console.error(err);
        }
    }

    async function loadPreferences() {
        try {
            const res = await apiJson("/api/preferences");
            const p = res.preferences || {};
            document.getElementById("prefAvoidArtists").value = (p.avoid_artists || []).join(", ");
            document.getElementById("prefAvoidGenres").value = (p.avoid_genres || []).join(", ");
            document.getElementById("prefHalfLife").value = p.half_life_days ?? 120;
            document.getElementById("prefMaxLiked").value = p.max_liked_songs ?? 500;
            document.getElementById("prefDefaultSize").value = p.default_playlist_size ?? 30;
            document.getElementById("prefSkipPenalty").checked = p.skip_penalty_enabled !== false;
            document.getElementById("prefDefaultFast").checked = Boolean(p.default_fast_mode);
            document.getElementById("prefDefaultPublic").checked = Boolean(p.default_public_export);
            document.getElementById("prefSyncLikesSpotify").checked = Boolean(p.sync_likes_to_spotify);
            document.getElementById("prefCidMask").textContent = res.has_client_id ? res.spotify_client_id_masked : "Not set";
            state.penalizedArtists = { ...(p.penalized_artists || {}) };
            renderPenalties();
            await loadYtmAuthStatus();
            await loadLlmCatalog();
        } catch (err) {
            showToast(err.message, "err");
        }
    }

    function toggleOllamaBox() {
        const backend = document.getElementById("prefLlmBackend").value;
        const box = document.getElementById("ollamaConfigBox");
        if (box) box.style.display = (backend === "ollama" || backend === "openai_local") ? "block" : "none";
    }

    function renderOllamaSuggestions(tiers, detectedModels) {
        const chipsEl = document.getElementById("ollamaTierChips");
        const detEl = document.getElementById("ollamaDetectedModels");
        if (chipsEl) {
            chipsEl.innerHTML = `<span class="muted">Curated Tags:</span>` +
                (tiers || [])
                    .map(
                        (t) => `<button type="button" class="chip-btn ollama-pick-btn" data-model="${escapeHtml(t.model)}" title="${escapeHtml(t.label)}">${escapeHtml(t.model)} (${escapeHtml(t.label)})</button>`
                    )
                    .join("");
        }
        if (detEl) {
            if (Array.isArray(detectedModels) && detectedModels.length) {
                detEl.innerHTML = `<b>Running on local server:</b> ` + detectedModels.map((m) => `<code>${escapeHtml(m)}</code>`).join(" · ");
            } else {
                detEl.textContent = "Local Ollama server not detected or no models pulled yet (completely optional).";
            }
        }
    }

    function renderLlmCatalog(data) {
        state.llmCatalogData = data;
        const hw = data.hardware || {};
        const active = data.active || {};
        const catalog = data.catalog || [];
        state.selectedLlmModelKey = active.llm_model_key || "auto";

        const backendEl = document.getElementById("prefLlmBackend");
        const deviceEl = document.getElementById("prefLlmDevice");
        const urlEl = document.getElementById("prefOllamaUrl");
        const modelEl = document.getElementById("prefOllamaModel");
        if (backendEl) backendEl.value = active.llm_backend || "openvino";
        if (deviceEl) deviceEl.value = active.llm_device || "auto";
        if (urlEl && active.ollama_url) urlEl.value = active.ollama_url;
        if (modelEl && active.ollama_model) modelEl.value = active.ollama_model;
        toggleOllamaBox();

        const bannerEl = document.getElementById("llmHardwareBanner");
        if (bannerEl) {
            const nvTxt = hw.nvidia_gpu_name
                ? `${hw.nvidia_gpu_name}${hw.nvidia_vram_gb ? ` (${hw.nvidia_vram_gb} GB VRAM)` : ""}`
                : "None detected";
            const ramTxt = hw.ram_total_gb != null ? `${hw.ram_total_gb} GB RAM` : "RAM Unknown";
            bannerEl.innerHTML =
                `<b>Detected Hardware:</b> ${escapeHtml(hw.cpu_name || hw.silicon || "CPU")} (${escapeHtml(hw.cpu_cores || "?")} threads) · ` +
                `<b>${escapeHtml(ramTxt)}</b> · <b>NVIDIA GPU:</b> ${escapeHtml(nvTxt)} · ` +
                `<b>OpenVINO Devices:</b> ${escapeHtml((hw.openvino_devices || ["CPU"]).join(", "))}<br>` +
                `<span class="accent-text">${escapeHtml(hw.recommended_tier_label || "")}:</span> ${escapeHtml(hw.advice || "")}`;
        }

        const cardsEl = document.getElementById("llmTierCards");
        if (cardsEl) {
            const autoActive = state.selectedLlmModelKey === "auto";
            const autoCard = `
                <button type="button" class="tier-card ${autoActive ? "active" : ""}" data-model-key="auto">
                    <div class="tier-card-head">
                        <span class="tier-card-tier">AUTO-DETECT HARDWARE TIER</span>
                        ${autoActive ? `<span class="badge badge-ok">Selected</span>` : ""}
                    </div>
                    <div class="tier-card-title">Auto (Recommended for this PC)</div>
                    <div class="tier-card-desc">Automatically picks the right model per machine (${escapeHtml(hw.recommended_model_key || "qwen3-4b-ov")} here).</div>
                </button>`;

            const modelCards = catalog
                .map((m) => {
                    const isSelected = state.selectedLlmModelKey === m.key;
                    const badges = [];
                    if (m.recommended) badges.push(`<span class="badge badge-info">Recommended</span>`);
                    if (m.downloaded) badges.push(`<span class="badge badge-ok">Downloaded (${escapeHtml(m.size_on_disk_mb)} MB)</span>`);
                    if (!m.fits_ram) badges.push(`<span class="badge badge-warn">Needs ${escapeHtml(m.min_ram_gb)}GB+ RAM</span>`);
                    if (isSelected) badges.push(`<span class="badge badge-ok">Selected</span>`);
                    return `
                    <button type="button" class="tier-card ${isSelected ? "active" : ""}" data-model-key="${escapeHtml(m.key)}">
                        <div class="tier-card-head">
                            <span class="tier-card-tier">${escapeHtml(m.tier_label)}</span>
                            <div class="btn-row" style="gap:4px;">${badges.join("")}</div>
                        </div>
                        <div class="tier-card-title">${escapeHtml(m.name)} (${escapeHtml(m.size_gb)} GB)</div>
                        <div class="tier-card-desc">${escapeHtml(m.summary)}<br><span class="muted">Best on: ${escapeHtml(m.recommended_hw)}</span></div>
                    </button>`;
                })
                .join("");

            cardsEl.innerHTML = autoCard + modelCards;
        }

        renderOllamaSuggestions(data.ollama_tiers || [], (hw.ollama && hw.ollama.models) || []);
    }

    async function loadLlmCatalog(probeOllama = true) {
        try {
            const data = await apiJson(`/api/llm/models?probe_ollama=${probeOllama ? "1" : "0"}`);
            renderLlmCatalog(data);
        } catch (err) {
            console.error(err);
        }
    }

    async function loadDoctor() {
        try {
            const d = await apiJson("/api/doctor");
            const m = d.models || {};
            const rb = d.reccobeats_cache || {};

            document.getElementById("doctorSummaryCards").innerHTML = [
                { l: "Python Runtime", v: d.python_version, s: d.venv_exists ? ".venv present" : "Using system Python" },
                { l: "OpenVINO Silicon", v: (d.hardware?.openvino_devices || ["CPU"]).join(", "), s: d.hardware?.nvidia_gpu_name ? `${d.hardware.silicon} | ${d.hardware.nvidia_gpu_name}` : (d.hardware?.silicon || "CPU") },
                { l: "Local LLM Cache", v: m.llm_backend === "openvino" ? (m.llm_cached ? `${m.llm_size_mb} MB (${m.llm_downloaded_count || 1} cached)` : "Not Downloaded") : (m.llm_label || "Ollama"), s: `Active: ${m.llm_label || "Auto"}` },
                { l: "MiniLM Embed Cache", v: m.embed_cached ? `${m.embed_size_mb} MB` : "Not Exported", s: m.embed_cached ? "Dynamic OpenVINO IR ready" : "Exports on first Stage 2 run" },
            ]
                .map(
                    (c) => `
                <div class="status-card">
                    <div class="status-label">${escapeHtml(c.l)}</div>
                    <div class="status-val">${escapeHtml(c.v)}</div>
                    <div class="status-sub">${escapeHtml(c.s)}</div>
                </div>`
                )
                .join("");

            const pkgs = d.packages || {};
            document.getElementById("doctorPackages").innerHTML = Object.entries(pkgs)
                .map(([name, ok]) => `<div class="pkg-item"><span>${escapeHtml(name)}</span><span class="badge ${ok ? "badge-ok" : "badge-bad"}">${ok ? "OK" : "Missing"}</span></div>`)
                .join("");

            document.getElementById("doctorRbStats").innerHTML = `
                <p>Total Entries: <b>${escapeHtml(rb.total)}</b> | Real Telemetry: <b>${escapeHtml(rb.real)}</b> | Not Found: <b>${escapeHtml(rb.not_found)}</b> | Poisoned: <b>${escapeHtml(rb.poisoned)}</b> (${escapeHtml(rb.file_size_kb)} KB)</p>`;

            const logsEl = document.getElementById("logFilesList");
            const files = d.logs || [];
            logsEl.innerHTML = files.length
                ? files.map((f) => `<button class="log-file-btn" data-file="${escapeHtml(f.name)}" type="button">${escapeHtml(f.name)} (${escapeHtml(f.size_kb)} KB)</button>`).join("")
                : `<span class="muted">No persistent log files recorded yet.</span>`;
        } catch (err) {
            showToast(err.message, "err");
        }
    }

    function openConfirm(title, message, onConfirm) {
        const modal = document.getElementById("confirmModal");
        document.getElementById("confirmTitle").textContent = title;
        document.getElementById("confirmMessage").textContent = message;
        modal.style.display = "flex";
        const okBtn = document.getElementById("confirmOkBtn");
        const cancelBtn = document.getElementById("confirmCancelBtn");
        const cleanup = () => {
            modal.style.display = "none";
            okBtn.onclick = null;
            cancelBtn.onclick = null;
        };
        cancelBtn.onclick = cleanup;
        okBtn.onclick = async () => {
            cleanup();
            await onConfirm();
        };
    }

    // =====================================================
    // PRESETS, HISTORY UPLOAD, HYGIENE, AUTOMATION & FEEDBACK
    // =====================================================
    function renderPresets() {
        const wrap = document.getElementById("presetsChips");
        const entries = Object.entries(state.presets || {});
        if (!entries.length) {
            wrap.innerHTML = `<span class="muted">No saved presets yet.</span>`;
            return;
        }
        wrap.innerHTML = entries
            .map(
                ([name, cfg]) => `
                <span class="preset-chip">
                    <button type="button" class="preset-apply-btn" data-name="${escapeHtml(name)}">${escapeHtml(name)}</button>
                    ${cfg && cfg.builtin ? "" : `<button type="button" class="preset-del-btn" data-name="${escapeHtml(name)}" aria-label="Delete preset">×</button>`}
                </span>`
            )
            .join("");
    }

    async function loadPresets() {
        try {
            const res = await apiJson("/api/presets");
            state.presets = res.presets || {};
            renderPresets();
        } catch (err) {
            console.error(err);
        }
    }

    async function applyPresetByName(name) {
        const preset = (state.presets || {})[name];
        if (!preset) return;
        state.avoidArtists = Array.isArray(preset.avoid_artists) ? [...preset.avoid_artists] : [];
        state.avoidGenres = Array.isArray(preset.avoid_genres) ? [...preset.avoid_genres] : [];
        renderGuardrailsUI();

        const pText = preset.prompt_hint ?? preset.prompt;
        if (pText != null) document.getElementById("studioPrompt").value = String(pText);
        if (preset.size != null) {
            document.getElementById("studioSize").value = String(preset.size);
            document.getElementById("studioSizeVal").textContent = String(preset.size);
        }
        if (preset.flow_mode) document.getElementById("studioFlow").value = preset.flow_mode;
        if (preset.fast != null) document.getElementById("studioFastMode").checked = Boolean(preset.fast);
        if (preset.local_only != null) document.getElementById("studioLocalOnly").checked = Boolean(preset.local_only);
        if (preset.clean_only != null) document.getElementById("studioCleanOnly").checked = Boolean(preset.clean_only);
        if (preset.deep_cuts != null) document.getElementById("studioDeepCuts").checked = Boolean(preset.deep_cuts);

        try {
            await pushSessionGuardrails(false);
            showToast(`Applied mode "${name}"!`, "ok");
        } catch (err) {
            showToast(err.message, "err");
        }
    }

    async function uploadHistoryFiles(fileList) {
        const files = Array.from(fileList || []);
        if (!files.length) return;
        const statusEl = document.getElementById("historyUploadStatus");
        let savedTotal = 0;
        let skippedTotal = 0;
        const errors = [];

        statusEl.textContent = `Uploading ${files.length} file(s)...`;
        for (const f of files) {
            try {
                const res = await apiJson("/api/history/upload", {
                    method: "POST",
                    headers: { "X-Filename": encodeURIComponent(f.name) },
                    body: f,
                });
                savedTotal += (res.saved_files || []).length;
                skippedTotal += (res.skipped_files || []).length;
            } catch (err) {
                errors.push(`${f.name}: ${err.message}`);
            }
        }
        if (errors.length) {
            statusEl.textContent = `Uploaded ${savedTotal} JSON(s) with errors: ${errors.join("; ")}`;
            showToast(errors[0], "err");
        } else {
            statusEl.textContent = `Saved ${savedTotal} audio history JSON(s) (${skippedTotal} skipped). Click "4. Import Extended History" to index.`;
            showToast(`Uploaded ${savedTotal} Extended History file(s)!`, "ok");
        }
        refreshStatus();
    }

    async function fetchLibrarySuggestions(query) {
        try {
            const res = await apiJson(`/api/library/search?q=${encodeURIComponent(query || "")}&limit=15`);
            const dl = document.getElementById("localTrackDatalist");
            dl.innerHTML = (res.results || [])
                .map((r) => `<option value="${escapeHtml(r.name)} - ${escapeHtml(r.artist)}"></option>`)
                .join("");
        } catch (err) {
            console.error(err);
        }
    }

    async function loadHygiene() {
        const sumEl = document.getElementById("hygieneSummary");
        const boxEl = document.getElementById("hygieneContent");
        sumEl.textContent = "Scanning synced playlists and history...";
        try {
            const d = await apiJson("/api/hygiene");
            if (!d.available) {
                sumEl.textContent = "Run Stage 1 Spotify Sync first so playlists are available to audit.";
                boxEl.style.display = "none";
                return;
            }
            const s = d.summary || {};
            sumEl.textContent = `Scanned ${s.playlists_scanned || 0} playlists | Duplicate groups: ${s.duplicate_groups || 0} (${s.duplicate_extra_tracks || 0} extra tracks) | Chronically skipped (>=70%): ${s.chronically_skipped_count || 0} | Disliked in playlists: ${s.disliked_in_playlists_count || 0} | Audio telemetry coverage: ${s.telemetry_coverage_pct || 0}%`;
            boxEl.style.display = "grid";

            const dups = d.playlist_duplicates || [];
            const skips = d.chronically_skipped || [];
            const disliked = d.disliked_in_playlists || [];

            const dupHtml = dups.length
                ? dups
                      .map((g) => {
                          const canDedupe = g.playlist_id && SPOTIFY_ID_RE.test(String(g.playlist_id));
                          return `<div style="padding:6px 0; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; gap:8px;">
                              <div><b>${escapeHtml(g.track_name)}</b> — ${escapeHtml(g.artist)}<div class="muted">${escapeHtml(g.playlist_name)} (${escapeHtml(g.duplicate_count)} copies)</div></div>
                              ${canDedupe ? `<button class="btn btn-xs btn-danger dedupe-pl-btn" data-pid="${escapeHtml(g.playlist_id)}" data-pname="${escapeHtml(g.playlist_name)}" type="button">Dedupe playlist</button>` : ""}
                          </div>`;
                      })
                      .join("")
                : `<div class="muted">No duplicate fingerprints found in your playlists!</div>`;

            const skipHtml = skips.length
                ? `<table class="data-table"><thead><tr><th>Track</th><th>Playlist</th><th>Skip %</th></tr></thead><tbody>${skips
                      .slice(0, 20)
                      .map((r) => `<tr><td>${escapeHtml(r.name)} — ${escapeHtml(r.artist)}</td><td>${escapeHtml(r.playlist_name)}</td><td>${escapeHtml(r.skip_rate_pct)}%</td></tr>`)
                      .join("")}</tbody></table>`
                : `<div class="muted">No chronically skipped tracks lingering in playlists.</div>`;

            const disHtml = disliked.length
                ? disliked
                      .map((r) => `<div style="padding:4px 0; border-bottom:1px solid var(--border);">${escapeHtml(r.name)} — ${escapeHtml(r.artist)} <span class="muted">(${escapeHtml(r.playlist_name)})</span></div>`)
                      .join("")
                : `<div class="muted">No disliked tracks found in your synced playlists</div>`;

            boxEl.innerHTML = `
                <div class="card"><h3>Playlist Duplicates</h3><div class="mini-table-wrap mt-8">${dupHtml}</div></div>
                <div class="card"><h3>Chronically Skipped in Playlists</h3><div class="mini-table-wrap mt-8">${skipHtml}</div></div>
                <div class="card"><h3>Disliked Tracks in Playlists</h3><div class="mini-table-wrap mt-8">${disHtml}</div></div>
            `;
        } catch (err) {
            sumEl.textContent = `Audit failed: ${err.message}`;
            showToast(err.message, "err");
        }
    }

    async function loadFeedbackMemory() {
        try {
            const fb = await apiJson("/api/feedback");
            state.feedback = { liked: fb.liked || {}, disliked: fb.disliked || {} };

            const renderFbList = (elId, mapObj, ratingLabel) => {
                const el = document.getElementById(elId);
                const entries = Object.values(mapObj || {});
                if (!entries.length) {
                    el.innerHTML = `<span class="muted">No ${escapeHtml(ratingLabel)} tracks recorded yet.</span>`;
                    return;
                }
                el.innerHTML = entries
                    .map(
                        (t) => `<div style="display:flex; justify-content:space-between; align-items:center; padding:5px 0; border-bottom:1px solid var(--border); gap:8px;">
                            <span><b>${escapeHtml(t.name)}</b> — ${escapeHtml(t.artist)}</span>
                            <button class="btn btn-xs clear-fb-btn" data-id="${escapeHtml(t.id)}" data-name="${escapeHtml(t.name)}" data-artist="${escapeHtml(t.artist)}" type="button">Clear</button>
                        </div>`
                    )
                    .join("");
            };
            renderFbList("feedbackLikedList", state.feedback.liked, "liked");
            renderFbList("feedbackDislikedList", state.feedback.disliked, "disliked");
            if (state.currentPlaylist) renderPlaylistResult(state.currentPlaylist);
        } catch (err) {
            console.error(err);
        }
    }

    function readAutomationForm() {
        return {
            target_playlist_id: document.getElementById("autoTargetPlaylistId").value.trim(),
            prompt: document.getElementById("autoPrompt").value.trim(),
            size: parseInt(document.getElementById("autoSize").value || "30", 10),
            flow_mode: document.getElementById("autoFlowMode").value,
            day_of_week: document.getElementById("autoDay").value,
            time: document.getElementById("autoTime").value.trim() || "09:00",
            fast: document.getElementById("autoFast").checked,
            run_stage1_sync: document.getElementById("autoRefreshS1").checked,
            run_stage2_refresh: document.getElementById("autoRefreshS2").checked,
        };
    }

    function applyAutomationData(res) {
        const a = res.automation || {};
        const t = res.task || {};
        document.getElementById("autoTargetPlaylistId").value = a.target_playlist_id || "";
        document.getElementById("autoPrompt").value = a.prompt || "";
        document.getElementById("autoSize").value = String(a.size ?? 30);
        if (a.flow_mode) document.getElementById("autoFlowMode").value = a.flow_mode;
        if (a.day_of_week) document.getElementById("autoDay").value = a.day_of_week;
        document.getElementById("autoTime").value = a.time || "09:00";
        document.getElementById("autoFast").checked = a.fast !== false;
        document.getElementById("autoRefreshS1").checked = Boolean(a.run_stage1_sync);
        document.getElementById("autoRefreshS2").checked = Boolean(a.run_stage2_refresh);
        document.getElementById("autoStatusLine").textContent = a.last_run_at
            ? `Last Run: ${a.last_run_at} — ${a.last_status || ""}`
            : "Last Run: Never";
        document.getElementById("autoTaskLine").textContent = !t.supported
            ? "Scheduler: Windows Task Scheduler unavailable on this OS (CLI: python automation_runner.py --run-now)"
            : t.exists
              ? `Scheduler: Active (${t.task_name} · Next: ${t.next_run_time || "Pending"} · Status: ${t.status || "Ready"})`
              : "Scheduler: Not registered";
    }

    async function loadAutomation() {
        try {
            const res = await apiJson("/api/automation");
            applyAutomationData(res);
        } catch (err) {
            console.error(err);
        }
    }

    async function actAutomation(action) {
        try {
            if (action === "run_now") {
                document.getElementById("consoleDrawer").classList.remove("collapsed");
            }
            const res = await apiJson("/api/automation", {
                method: "POST",
                body: { action, automation: readAutomationForm() },
            });
            if (action === "run_now") {
                showToast("Started weekly playlist auto-refresh job...", "info");
                return;
            }
            applyAutomationData(res);
            if (res.task_result && res.task_result.error) {
                showToast(res.task_result.error, "err");
            } else {
                showToast(`Automation ${action.replace("_", " ")} complete!`, "ok");
            }
        } catch (err) {
            showToast(err.message, "err");
        }
    }

    // =====================================================
    // EVENT BINDINGS
    // =====================================================
    function collectEraPayload() {
        const out = {};
        const es = document.getElementById("dashEraStart").value.trim();
        const ee = document.getElementById("dashEraEnd").value.trim();
        if (es) out.era_start = es;
        if (ee) out.era_end = ee;
        out.projection = document.getElementById("mapProjectionSelect").value;
        return out;
    }

    function bindEvents() {
        document.querySelectorAll(".nav-btn").forEach((btn) => {
            btn.addEventListener("click", () => switchTab(btn.dataset.tab));
        });

        document.getElementById("guardPill").addEventListener("click", () => switchTab("dashboard"));
        document.getElementById("activeProviderSelect").addEventListener("change", async (e) => {
            const nextProv = e.target.value || "spotify";
            state.activeProvider = nextProv;
            try {
                await apiJson("/api/preferences", {
                    method: "POST",
                    body: { preferences: { active_provider: nextProv }, sync_session: false },
                });
                await refreshStatus();
                const labelMap = {
                    spotify: "Spotify Only",
                    ytmusic: "YouTube Music Only",
                    unified: "Unified (Spotify + YouTube Music)",
                };
                showToast(`Active music mode: ${labelMap[nextProv] || nextProv}. Rebuild Stage 2 if switching libraries.`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });
        document.getElementById("btnClosePreview").addEventListener("click", closePreviewDock);
        document.getElementById("toggleConsoleBtn").addEventListener("click", () => {
            document.getElementById("consoleDrawer").classList.toggle("collapsed");
        });
        document.getElementById("btnCollapseConsole").addEventListener("click", () => {
            document.getElementById("consoleDrawer").classList.toggle("collapsed");
        });
        document.getElementById("btnClearConsole").addEventListener("click", () => {
            document.getElementById("consoleOutput").textContent = "";
        });
        document.getElementById("btnCancelJob").addEventListener("click", async () => {
            try {
                await apiJson("/api/jobs/cancel", { method: "POST", body: {} });
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        // Dashboard actions
        document.getElementById("btnFullSync").addEventListener("click", () =>
            startJob("full_sync", {
                force_reembed: document.getElementById("chkForceReembed").checked,
                ...collectEraPayload(),
            })
        );
        document.getElementById("btnRunStage1").addEventListener("click", () => startJob("stage1"));
        document.getElementById("btnRunStage1Ytm").addEventListener("click", () => startJob("stage1_ytm"));
        document.getElementById("btnRunStage2").addEventListener("click", () =>
            startJob("stage2", {
                force_reembed: document.getElementById("chkForceReembed").checked,
                ...collectEraPayload(),
            })
        );
        document.getElementById("btnRunMap").addEventListener("click", () =>
            startJob("tastemap", { projection: document.getElementById("mapProjectionSelect").value })
        );
        document.getElementById("btnRefreshMap").addEventListener("click", () =>
            startJob("tastemap", { projection: document.getElementById("mapProjectionSelect").value })
        );
        document.getElementById("mapProjectionSelect").addEventListener("change", (e) => {
            const proj = e.target.value;
            if (!state.mapData || (state.mapData.projection_method || "pca").toLowerCase() !== proj) {
                startJob("tastemap", { projection: proj });
            }
        });
        document.getElementById("btnRunHistory").addEventListener("click", () => startJob("history"));

        // Drag-and-drop Extended History Upload
        const dz = document.getElementById("historyDropzone");
        const hfInp = document.getElementById("historyFileInput");
        dz.addEventListener("click", () => hfInp.click());
        dz.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                hfInp.click();
            }
        });
        dz.addEventListener("dragover", (e) => {
            e.preventDefault();
            dz.classList.add("dragover");
        });
        dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
        dz.addEventListener("drop", (e) => {
            e.preventDefault();
            dz.classList.remove("dragover");
            uploadHistoryFiles(e.dataTransfer?.files);
        });
        hfInp.addEventListener("change", () => {
            uploadHistoryFiles(hfInp.files);
            hfInp.value = "";
        });

        // Named Session Presets ("Modes")
        document.getElementById("presetsChips").addEventListener("click", async (e) => {
            const applyBtn = e.target.closest(".preset-apply-btn");
            const delBtn = e.target.closest(".preset-del-btn");
            if (applyBtn) {
                await applyPresetByName(applyBtn.dataset.name);
            } else if (delBtn) {
                try {
                    const res = await apiJson(`/api/presets/${encodeURIComponent(delBtn.dataset.name)}`, { method: "DELETE" });
                    state.presets = res.presets || {};
                    renderPresets();
                    showToast(`Removed mode "${delBtn.dataset.name}".`, "ok");
                } catch (err) {
                    showToast(err.message, "err");
                }
            }
        });
        document.getElementById("btnSavePreset").addEventListener("click", async () => {
            const name = document.getElementById("presetNameInput").value.trim();
            if (!name) {
                showToast("Enter a name for the preset mode.", "err");
                return;
            }
            const preset = {
                builtin: false,
                avoid_artists: [...state.avoidArtists],
                avoid_genres: [...state.avoidGenres],
                prompt_hint: document.getElementById("studioPrompt").value.trim(),
                size: parseInt(document.getElementById("studioSize").value || "30", 10),
                flow_mode: document.getElementById("studioFlow").value,
                fast: document.getElementById("studioFastMode").checked,
                local_only: document.getElementById("studioLocalOnly").checked,
                clean_only: document.getElementById("studioCleanOnly").checked,
                deep_cuts: document.getElementById("studioDeepCuts").checked,
            };
            try {
                const res = await apiJson("/api/presets", { method: "POST", body: { name, preset } });
                state.presets = res.presets || {};
                document.getElementById("presetNameInput").value = "";
                renderPresets();
                showToast(`Saved mode "${name}"!`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnCleanSystem").addEventListener("click", () => {
            openConfirm(
                "Clean System Caches",
                "Remove cached OAuth tokens, user profile JSON, vector matrices, and generated HTML map? Your Extended History archive is preserved.",
                async () => {
                    try {
                        const res = await apiJson("/api/system/clean", { method: "POST", body: {} });
                        showToast(`Removed ${res.removed.length} cached files.`, "ok");
                        refreshStatus(true);
                    } catch (err) {
                        showToast(err.message, "err");
                    }
                }
            );
        });

        // Guardrails chips
        const addChip = (kind, inputId) => {
            const inp = document.getElementById(inputId);
            const items = inp.value.split(",").map((s) => s.trim()).filter(Boolean);
            if (!items.length) return;
            const arr = kind === "artist" ? state.avoidArtists : state.avoidGenres;
            items.forEach((it) => {
                if (!arr.some((existing) => existing.toLowerCase() === it.toLowerCase())) arr.push(it);
            });
            inp.value = "";
            renderGuardrailsUI();
        };
        document.getElementById("btnAddAvoidArtist").addEventListener("click", () => addChip("artist", "dashAvoidArtistInput"));
        document.getElementById("btnAddAvoidGenre").addEventListener("click", () => addChip("genre", "dashAvoidGenreInput"));
        document.getElementById("dashAvoidArtistInput").addEventListener("keydown", (e) => {
            if (e.key === "Enter") { e.preventDefault(); addChip("artist", "dashAvoidArtistInput"); }
        });
        document.getElementById("dashAvoidGenreInput").addEventListener("keydown", (e) => {
            if (e.key === "Enter") { e.preventDefault(); addChip("genre", "dashAvoidGenreInput"); }
        });

        ["dashAvoidArtistChips", "dashAvoidGenreChips"].forEach((id) => {
            document.getElementById(id).addEventListener("click", (e) => {
                const btn = e.target.closest("button");
                if (!btn) return;
                const idx = parseInt(btn.dataset.idx, 10);
                if (btn.dataset.kind === "artist") state.avoidArtists.splice(idx, 1);
                else state.avoidGenres.splice(idx, 1);
                renderGuardrailsUI();
            });
        });

        document.getElementById("btnApplySessionGuardrails").addEventListener("click", () => pushSessionGuardrails(false));
        document.getElementById("btnSaveGuardrailsPrefs").addEventListener("click", () => pushSessionGuardrails(true));
        document.getElementById("btnRebuildWithGuardrails").addEventListener("click", async () => {
            await pushSessionGuardrails(false);
            await startJob("stage2", { force_reembed: false, ...collectEraPayload() });
        });

        // Studio Sub-modes & Generators
        document.querySelectorAll(".subtab-btn").forEach((b) => {
            b.addEventListener("click", () => switchStudioMode(b.dataset.studioMode));
        });
        document.querySelectorAll(".preset-btn").forEach((b) => {
            b.addEventListener("click", () => {
                document.getElementById("studioPrompt").value = b.dataset.prompt;
            });
        });
        document.getElementById("studioSize").addEventListener("input", (e) => {
            document.getElementById("studioSizeVal").textContent = e.target.value;
        });
        document.getElementById("bridgeSteps").addEventListener("input", (e) => {
            document.getElementById("bridgeStepsVal").textContent = e.target.value;
        });
        document.getElementById("blendRatio").addEventListener("input", (e) => {
            const friendPct = parseInt(e.target.value, 10);
            document.getElementById("blendRatioVal").textContent = `${100 - friendPct}% Me / ${friendPct}% Friend`;
        });
        document.getElementById("btnEditGuardrailsFromStudio").addEventListener("click", () => switchTab("dashboard"));

        document.getElementById("btnWarmMinilm").addEventListener("click", async () => {
            showToast("Warming MiniLM vector engine...", "info");
            try {
                await apiJson("/api/studio/warmup", { method: "POST", body: { load_llm: false } });
                await refreshStatus();
                showToast("Vector engine warm in memory!", "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });
        document.getElementById("btnWarmLlm").addEventListener("click", async () => {
            showToast("Warming MiniLM + Local LLM (may take a moment)...", "info");
            try {
                const warmSt = await apiJson("/api/studio/warmup", { method: "POST", body: { load_llm: true } });
                await refreshStatus(true);
                showToast(`${warmSt.llm_label || "Local LLM"} + Vector engine warm!`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });
        document.getElementById("btnUnloadLlm").addEventListener("click", async () => {
            try {
                await apiJson("/api/studio/unload-llm", { method: "POST", body: {} });
                await refreshStatus(true);
                showToast("Unloaded Local LLM from memory.", "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });
        document.getElementById("studioActiveModelPill").addEventListener("click", () => switchTab("preferences"));

        document.getElementById("btnGeneratePlaylist").addEventListener("click", async () => {
            const btn = document.getElementById("btnGeneratePlaylist");
            const clusterRaw = document.getElementById("studioCluster").value;
            const prompt = document.getElementById("studioPrompt").value.trim();
            if (!prompt && !clusterRaw) {
                showToast("Enter a prompt or select a taste cluster to expand.", "err");
                return;
            }
            btn.disabled = true;
            btn.textContent = "⏳ Generating Playlist...";
            try {
                const res = await apiJson("/api/studio/generate", {
                    method: "POST",
                    body: {
                        prompt,
                        fast: document.getElementById("studioFastMode").checked,
                        size: parseInt(document.getElementById("studioSize").value, 10),
                        local_only: document.getElementById("studioLocalOnly").checked,
                        cluster_id: clusterRaw === "" ? null : parseInt(clusterRaw, 10),
                        flow_mode: document.getElementById("studioFlow").value,
                        clean_only: document.getElementById("studioCleanOnly").checked,
                        deep_cuts: document.getElementById("studioDeepCuts").checked,
                    },
                });
                renderPlaylistResult(res.playlist, res.candidate_pool_size);
                await refreshStatus();
                showToast(`Generated "${res.playlist.playlist_name}"!`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            } finally {
                btn.disabled = false;
                btn.textContent = "Generate playlist";
            }
        });

        document.getElementById("btnContinuePlaying").addEventListener("click", async () => {
            const btn = document.getElementById("btnContinuePlaying");
            btn.disabled = true;
            btn.textContent = "Reading now playing...";
            try {
                const res = await apiJson("/api/studio/continue-playing", {
                    method: "POST",
                    body: {
                        size: parseInt(document.getElementById("studioSize").value || "25", 10),
                        fast: document.getElementById("studioFastMode").checked,
                        flow_mode: document.getElementById("studioFlow").value,
                        clean_only: document.getElementById("studioCleanOnly").checked,
                        deep_cuts: document.getElementById("studioDeepCuts").checked,
                    },
                });
                renderPlaylistResult(res.playlist, res.candidate_pool_size);
                await refreshStatus();
                showToast(`Created continuation mix: "${res.playlist.playlist_name}"!`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            } finally {
                btn.disabled = false;
                btn.textContent = "Continue playing now";
            }
        });

        ["bridgeStartInput", "bridgeEndInput"].forEach((id) => {
            document.getElementById(id).addEventListener("input", (e) => {
                clearTimeout(state.librarySearchTimer);
                const val = e.target.value;
                state.librarySearchTimer = setTimeout(() => fetchLibrarySuggestions(val), 220);
            });
        });

        document.getElementById("btnGenerateBridge").addEventListener("click", async () => {
            const startQ = document.getElementById("bridgeStartInput").value.trim();
            const endQ = document.getElementById("bridgeEndInput").value.trim();
            if (!startQ || !endQ) {
                showToast("Enter both Start and Destination tracks or vibes.", "err");
                return;
            }
            const btn = document.getElementById("btnGenerateBridge");
            btn.disabled = true;
            btn.textContent = "Building vibe bridge...";
            try {
                const res = await apiJson("/api/studio/vibe-bridge", {
                    method: "POST",
                    body: {
                        start_query: startQ,
                        end_query: endQ,
                        steps: parseInt(document.getElementById("bridgeSteps").value || "20", 10),
                        clean_only: document.getElementById("bridgeCleanOnly").checked,
                    },
                });
                renderPlaylistResult(res.playlist);
                await refreshStatus();
                showToast(`Built Vibe Bridge: "${res.playlist.playlist_name}"!`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            } finally {
                btn.disabled = false;
                btn.textContent = "Build vibe bridge";
            }
        });

        document.getElementById("blendFriendFile").addEventListener("change", (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = () => {
                try {
                    const parsed = JSON.parse(String(reader.result || "{}"));
                    if (!parsed.master_taste_centroid) {
                        throw new Error("Invalid Taste Card JSON (missing master_taste_centroid).");
                    }
                    state.blendFriendCard = parsed;
                    const clCount = (parsed.clusters || []).length;
                    document.getElementById("blendFriendStatus").textContent = `Loaded "${file.name}" (${clCount} clusters)`;
                    showToast("Friend Taste Card loaded!", "ok");
                } catch (err) {
                    state.blendFriendCard = null;
                    document.getElementById("blendFriendStatus").textContent = err.message;
                    showToast(err.message, "err");
                }
            };
            reader.readAsText(file);
        });

        document.getElementById("btnGenerateBlend").addEventListener("click", async () => {
            if (!state.blendFriendCard) {
                showToast("Upload a friend's music_taste_card.json first.", "err");
                return;
            }
            const btn = document.getElementById("btnGenerateBlend");
            btn.disabled = true;
            btn.textContent = "Blending tastes...";
            try {
                const ratio = parseInt(document.getElementById("blendRatio").value || "50", 10) / 100.0;
                const res = await apiJson("/api/studio/taste-blend", {
                    method: "POST",
                    body: {
                        friend_card: state.blendFriendCard,
                        blend_ratio: ratio,
                        prompt: document.getElementById("blendPrompt").value.trim(),
                        size: parseInt(document.getElementById("blendSize").value || "30", 10),
                        fast: document.getElementById("studioFastMode").checked,
                        flow_mode: document.getElementById("studioFlow").value,
                        clean_only: document.getElementById("studioCleanOnly").checked,
                    },
                });
                const banner = document.getElementById("blendCompatBanner");
                const sharedHtml = (res.shared_vibes || [])
                    .map((s) => `${escapeHtml(s.my_cluster)} × ${escapeHtml(s.friend_cluster)} (${escapeHtml(s.similarity_pct)}%)`)
                    .join(" · ");
                banner.innerHTML = `<b>Taste Compatibility: ${escapeHtml(res.compatibility_pct)}%</b>${sharedHtml ? `<div class="muted mt-8">Closest Shared Vibes: ${sharedHtml}</div>` : ""}`;
                banner.style.display = "block";
                renderPlaylistResult(res.playlist, res.candidate_pool_size);
                await refreshStatus();
                showToast(`Taste Blend (${res.compatibility_pct}% match) ready!`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            } finally {
                btn.disabled = false;
                btn.textContent = "Blend tastes and generate";
            }
        });

        // Inline Edit, Refinement, Drag Reorder & Track Row Actions
        document.getElementById("btnSavePlaylistMeta").addEventListener("click", async () => {
            try {
                const res = await apiJson("/api/studio/edit", {
                    method: "POST",
                    body: {
                        playlist_name: document.getElementById("resTitleInput").value.trim(),
                        playlist_description: document.getElementById("resDescInput").value.trim(),
                    },
                });
                renderPlaylistResult(res.playlist);
                showToast("Saved playlist title & description.", "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        const runRefine = async () => {
            const inp = document.getElementById("refineInput");
            const instr = inp.value.trim();
            if (!instr) return;
            const btn = document.getElementById("btnRefinePlaylist");
            btn.disabled = true;
            try {
                const res = await apiJson("/api/studio/refine", {
                    method: "POST",
                    body: { instruction: instr, fast: document.getElementById("studioFastMode").checked },
                });
                inp.value = "";
                renderPlaylistResult(res.playlist);
                showToast(`Refined playlist (${(res.playlist?.tracks || []).length} tracks).`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            } finally {
                btn.disabled = false;
            }
        };
        document.getElementById("btnRefinePlaylist").addEventListener("click", runRefine);
        document.getElementById("refineInput").addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                runRefine();
            }
        });

        const studioBody = document.getElementById("studioTracksBody");
        studioBody.addEventListener("click", async (e) => {
            const prevBtn = e.target.closest(".preview-btn");
            if (prevBtn) {
                openPreviewTrack(
                    {
                        spotify_id: prevBtn.dataset.spId || "",
                        video_id: prevBtn.dataset.vid || "",
                        provider: prevBtn.dataset.provider || "",
                        name: prevBtn.dataset.title || "",
                    },
                    prevBtn.dataset.title || ""
                );
                return;
            }

            const saveLibBtn = e.target.closest(".save-lib-btn");
            if (saveLibBtn) {
                await handleSaveLibraryButton(saveLibBtn);
                return;
            }

            const likeBtn = e.target.closest(".like-btn");
            const dislikeBtn = e.target.closest(".dislike-btn");
            if (likeBtn || dislikeBtn) {
                const idx = parseInt((likeBtn || dislikeBtn).dataset.idx, 10);
                const tr = (state.currentPlaylist?.tracks || [])[idx];
                if (!tr) return;
                const desired = likeBtn ? "like" : "dislike";
                const currentlyActive = likeBtn
                    ? Boolean(state.feedback.liked[tr.id])
                    : Boolean(state.feedback.disliked[tr.id]);
                try {
                    const res = await apiJson("/api/feedback", {
                        method: "POST",
                        body: { track: tr, rating: currentlyActive ? "clear" : desired },
                    });
                    state.feedback = res.feedback || { liked: {}, disliked: {} };
                    renderPlaylistResult(state.currentPlaylist);
                    const syncNotes = [];
                    if (res.spotify_synced) syncNotes.push("saved to Spotify");
                    if (res.ytmusic_synced) syncNotes.push("liked on YT Music");
                    const syncTxt = syncNotes.length ? ` (+ ${syncNotes.join(" & ")}!)` : " (boosted in Studio).";
                    showToast(
                        currentlyActive
                            ? `Cleared feedback for "${tr.name}".`
                            : desired === "like"
                              ? `Liked "${tr.name}"${syncTxt}`
                              : `Disliked "${tr.name}" (blocked from future mixes).`,
                        "ok"
                    );
                } catch (err) {
                    showToast(err.message, "err");
                }
                return;
            }

            const removeBtn = e.target.closest(".remove-btn");
            if (removeBtn) {
                const idx = parseInt(removeBtn.dataset.idx, 10);
                try {
                    const res = await apiJson("/api/studio/edit", {
                        method: "POST",
                        body: { remove_indices: [idx] },
                    });
                    renderPlaylistResult(res.playlist);
                    showToast("Removed track from playlist.", "ok");
                } catch (err) {
                    showToast(err.message, "err");
                }
                return;
            }

            const swapBtn = e.target.closest(".swap-btn");
            if (!swapBtn) return;
            const idx = parseInt(swapBtn.dataset.idx, 10);
            swapBtn.disabled = true;
            try {
                const res = await apiJson("/api/studio/swap", { method: "POST", body: { drop_index: idx } });
                renderPlaylistResult(res.playlist);
                showToast(`Swapped in "${res.replacement.name}"`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            } finally {
                swapBtn.disabled = false;
            }
        });

        studioBody.addEventListener("dragstart", (e) => {
            const tr = e.target.closest("tr[data-idx]");
            if (!tr) return;
            state.dragSrcIdx = parseInt(tr.dataset.idx, 10);
            tr.classList.add("dragging");
            if (e.dataTransfer) e.dataTransfer.effectAllowed = "move";
        });
        studioBody.addEventListener("dragover", (e) => {
            e.preventDefault();
            const tr = e.target.closest("tr[data-idx]");
            if (tr) tr.classList.add("drag-over");
        });
        studioBody.addEventListener("dragleave", (e) => {
            const tr = e.target.closest("tr[data-idx]");
            if (tr) tr.classList.remove("drag-over");
        });
        studioBody.addEventListener("dragend", () => {
            state.dragSrcIdx = null;
            studioBody.querySelectorAll("tr").forEach((r) => r.classList.remove("dragging", "drag-over"));
        });
        studioBody.addEventListener("drop", async (e) => {
            e.preventDefault();
            const targetTr = e.target.closest("tr[data-idx]");
            studioBody.querySelectorAll("tr").forEach((r) => r.classList.remove("dragging", "drag-over"));
            if (!targetTr || state.dragSrcIdx == null) return;
            const dstIdx = parseInt(targetTr.dataset.idx, 10);
            const srcIdx = state.dragSrcIdx;
            state.dragSrcIdx = null;
            const total = (state.currentPlaylist?.tracks || []).length;
            if (Number.isNaN(dstIdx) || srcIdx === dstIdx || total < 2) return;

            const order = Array.from({ length: total }, (_, i) => i);
            const [moved] = order.splice(srcIdx, 1);
            order.splice(dstIdx, 0, moved);
            try {
                const res = await apiJson("/api/studio/edit", {
                    method: "POST",
                    body: { order_indices: order },
                });
                renderPlaylistResult(res.playlist);
                showToast("Track order updated.", "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnResequence").addEventListener("click", async () => {
            const flow = document.getElementById("resequenceSelect").value;
            try {
                const res = await apiJson("/api/studio/resequence", { method: "POST", body: { flow_mode: flow } });
                renderPlaylistResult(res.playlist);
                showToast(`Re-ordered playlist (${flow}).`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnLoadSpotifyPlaylists").addEventListener("click", async () => {
            try {
                const res = await apiJson("/api/spotify/playlists");
                const sel = document.getElementById("exportTargetSelect");
                const cur = sel.value;
                sel.innerHTML = `<option value="">Create New Playlist</option>` +
                    (res.playlists || [])
                        .map((p) => `<option value="${escapeHtml(p.id)}">Update: ${escapeHtml(p.name)} (${escapeHtml(p.tracks_total)} tracks)</option>`)
                        .join("");
                if (cur && Array.from(sel.options).some((o) => o.value === cur)) sel.value = cur;
                showToast(`Loaded ${(res.playlists || []).length} owned Spotify playlists.`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnExportSpotify").addEventListener("click", async () => {
            const pid = state.currentPlaylist?.id || "last";
            const isPub = document.getElementById("chkExportPublic").checked;
            const targetPid = document.getElementById("exportTargetSelect").value || null;
            try {
                const res = await apiJson(`/api/playlists/${encodeURIComponent(pid)}/export-spotify`, {
                    method: "POST",
                    body: { public: isPub, target_playlist_id: targetPid },
                });
                showToast(targetPid ? "Updated existing Spotify playlist!" : "Exported new playlist to Spotify!", "ok");
                const safeSpUrl = safeHttpUrl(res.spotify_url);
                if (safeSpUrl) window.open(safeSpUrl, "_blank", "noopener,noreferrer");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnLoadYtmPlaylists").addEventListener("click", async () => {
            try {
                const res = await apiJson("/api/ytmusic/playlists");
                const sel = document.getElementById("ytmTargetSelect");
                const cur = sel.value;
                sel.innerHTML = `<option value="">Create New YTM Playlist</option>` +
                    (res.playlists || [])
                        .map((p) => `<option value="${escapeHtml(p.id)}">Update: ${escapeHtml(p.name)} (${escapeHtml(p.tracks_total)} tracks)</option>`)
                        .join("");
                if (cur && Array.from(sel.options).some((o) => o.value === cur)) sel.value = cur;
                showToast(`Loaded ${(res.playlists || []).length} YouTube Music playlists.`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnExportYtm").addEventListener("click", async () => {
            const pid = state.currentPlaylist?.id || "last";
            const targetPid = document.getElementById("ytmTargetSelect").value || null;
            const isPub = document.getElementById("chkExportPublic").checked;
            try {
                const res = await apiJson(`/api/playlists/${encodeURIComponent(pid)}/export-ytmusic`, {
                    method: "POST",
                    body: { privacy: isPub ? "PUBLIC" : "PRIVATE", target_playlist_id: targetPid },
                });
                renderTransferReport({ ...res, target_provider: "ytmusic" }, "YT Music Export");
                showToast(`Exported ${res.matched_count || 0}/${res.total_tracks || 0} tracks to YouTube Music!`, "ok");
                const safeYtmUrl = safeHttpUrl(res.url);
                if (safeYtmUrl) window.open(safeYtmUrl, "_blank", "noopener,noreferrer");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnTransferPlaylist").addEventListener("click", async () => {
            const pid = state.currentPlaylist?.id || "last";
            const tracks = state.currentPlaylist?.tracks || [];
            const ytmCount = tracks.filter((t) => String(t.provider || "").toLowerCase() === "ytmusic").length;
            const targetProv = state.activeProvider === "ytmusic" || ytmCount > tracks.length / 2 ? "spotify" : "ytmusic";
            const isPub = document.getElementById("chkExportPublic").checked;
            const targetPid = targetProv === "spotify"
                ? (document.getElementById("exportTargetSelect").value || null)
                : (document.getElementById("ytmTargetSelect").value || null);
            try {
                const res = await apiJson(`/api/playlists/${encodeURIComponent(pid)}/transfer`, {
                    method: "POST",
                    body: {
                        target_provider: targetProv,
                        public: isPub,
                        privacy: isPub ? "PUBLIC" : "PRIVATE",
                        target_playlist_id: targetPid,
                    },
                });
                renderTransferReport(res, `Transfer to ${targetProv === "spotify" ? "Spotify" : "YT Music"}`);
                const safeTrUrl = safeHttpUrl(res.url);
                if (safeTrUrl) {
                    showToast(`Transferred ${res.matched_count}/${res.total_tracks} tracks to ${targetProv === "spotify" ? "Spotify" : "YouTube Music"}!`, "ok");
                    window.open(safeTrUrl, "_blank", "noopener,noreferrer");
                } else {
                    showToast(res.error || "Transfer completed with 0 matched tracks.", "err");
                }
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnPlaySpotify").addEventListener("click", async () => {
            const pid = state.currentPlaylist?.id || "last";
            try {
                const res = await apiJson("/api/studio/play", { method: "POST", body: { playlist_id: pid } });
                showToast(`Started playing ${res.count} track(s) on your active Spotify device!`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnQueueSpotify").addEventListener("click", async () => {
            const pid = state.currentPlaylist?.id || "last";
            try {
                const res = await apiJson("/api/studio/queue", { method: "POST", body: { playlist_id: pid, max_items: 25 } });
                if (res.queued > 0) {
                    showToast(`Queued ${res.queued} track(s) to your Spotify playback queue!`, "ok");
                } else {
                    showToast((res.errors || [])[0] || "Open Spotify on a device to accept queue items.", "err");
                }
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        // Taste Map
        const scheduleMapFilter = () => {
            clearTimeout(state.mapFilterTimer);
            state.mapFilterTimer = setTimeout(applyTasteMapFilter, 80);
        };
        document.getElementById("mapSearchBox").addEventListener("input", scheduleMapFilter);
        document.getElementById("mapWeightSlider").addEventListener("input", scheduleMapFilter);
        document.getElementById("chkShowEraPath").addEventListener("change", renderTasteMapPlot);
        document.getElementById("chkShowCentroids").addEventListener("change", renderTasteMapPlot);
        document.getElementById("chkShowClouds").addEventListener("change", updateTasteMapClouds);
        document.getElementById("btnToggle3D").addEventListener("click", () => {
            state.is3d = !state.is3d;
            document.getElementById("btnToggle3D").textContent = state.is3d ? "Switch to 2D" : "Switch to 3D";
            runWithMapLoader(
                state.is3d ? "Switching to 3D view…" : "Switching to 2D view…",
                () => renderTasteMapPlot()
            );
        });
        document.getElementById("btnResetMapFilter").addEventListener("click", () => {
            document.getElementById("mapSearchBox").value = "";
            document.getElementById("mapWeightSlider").value = "0";
            applyTasteMapFilter();
        });
        document.getElementById("btnMapDrillBack")?.addEventListener("click", () => {
            exitClusterDrill();
        });
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape" && state.drillClusterId !== null) {
                exitClusterDrill();
            }
        });
        document.getElementById("clusterCardsGrid").addEventListener("click", (e) => {
            const drillBtn = e.target.closest(".drill-cluster-btn");
            if (drillBtn) {
                enterClusterDrill(drillBtn.dataset.cid);
                return;
            }
            const renameBtn = e.target.closest(".rename-cluster-btn");
            if (renameBtn) {
                const cid = renameBtn.dataset.cid;
                const oldLbl = renameBtn.dataset.label || "";
                openPromptModal(
                    `Rename Cluster C${cid}`,
                    "Enter a custom label (or clear it blank to restore auto-generated vibe title):",
                    oldLbl,
                    async (newLabel) => {
                        try {
                            await apiJson("/api/clusters/rename", {
                                method: "POST",
                                body: { cluster_id: parseInt(cid, 10), label: newLabel },
                            });
                            await refreshStatus();
                            await loadTasteMap();
                            showToast(`Updated Cluster C${cid} label.`, "ok");
                        } catch (err) {
                            showToast(err.message, "err");
                        }
                    }
                );
                return;
            }
            const btn = e.target.closest(".expand-cluster-btn");
            if (!btn) return;
            const cid = btn.dataset.cid;
            switchTab("studio");
            switchStudioMode("prompt");
            document.getElementById("studioCluster").value = cid;
            document.getElementById("studioPrompt").value = `Discover fresh tracks matching Cluster C${cid}`;
        });

        // History, Archive & Hygiene
        document.getElementById("btnReloadAnalytics").addEventListener("click", () => {
            loadPlaylistsArchive();
            loadAnalytics();
        });
        document.getElementById("btnScanHygiene").addEventListener("click", loadHygiene);
        document.getElementById("tab-analytics").addEventListener("click", async (e) => {
            const prevBtn = e.target.closest(".preview-btn");
            if (prevBtn) {
                openPreviewTrack(prevBtn.dataset.id, prevBtn.dataset.title);
                return;
            }
            const saveLibBtn = e.target.closest(".save-lib-btn");
            if (saveLibBtn) {
                await handleSaveLibraryButton(saveLibBtn);
                return;
            }
            const dedupeBtn = e.target.closest(".dedupe-pl-btn");
            if (dedupeBtn) {
                const pid = dedupeBtn.dataset.pid;
                const pname = dedupeBtn.dataset.pname || pid;
                openConfirm(
                    "Remove Duplicate Tracks on Spotify",
                    `Remove duplicate copies from Spotify playlist "${pname}" while preserving track order?`,
                    async () => {
                        try {
                            const res = await apiJson("/api/hygiene/dedupe-playlist", {
                                method: "POST",
                                body: { playlist_id: pid },
                            });
                            showToast(`Removed ${res.removed_count} duplicate(s) (${res.kept_count} unique tracks kept).`, "ok");
                            loadHygiene();
                        } catch (err) {
                            showToast(err.message, "err");
                        }
                    }
                );
            }
        });
        document.getElementById("archiveBody").addEventListener("click", async (e) => {
            const loadBtn = e.target.closest(".arch-load-btn");
            const expBtn = e.target.closest(".arch-exp-btn");
            const delBtn = e.target.closest(".arch-del-btn");
            if (loadBtn) {
                try {
                    const pl = await apiJson(`/api/playlists/${encodeURIComponent(loadBtn.dataset.id)}`);
                    renderPlaylistResult(pl);
                    switchTab("studio");
                } catch (err) {
                    showToast(err.message, "err");
                }
            } else if (expBtn) {
                try {
                    const res = await apiJson(`/api/playlists/${encodeURIComponent(expBtn.dataset.id)}/export-spotify`, {
                        method: "POST",
                        body: { public: false },
                    });
                    showToast("Exported archived playlist to Spotify!", "ok");
                    const safeUrl = safeHttpUrl(res.spotify_url);
                    if (safeUrl) window.open(safeUrl, "_blank", "noopener,noreferrer");
                } catch (err) {
                    showToast(err.message, "err");
                }
            } else if (delBtn) {
                try {
                    await apiJson(`/api/playlists/${encodeURIComponent(delBtn.dataset.id)}`, { method: "DELETE" });
                    loadPlaylistsArchive();
                    showToast("Archived playlist deleted.", "ok");
                } catch (err) {
                    showToast(err.message, "err");
                }
            }
        });

        document.getElementById("btnPreviewForgotten").addEventListener("click", () => runForgottenFavorites(false));
        document.getElementById("btnSaveForgotten").addEventListener("click", () => runForgottenFavorites(true));
        document.getElementById("btnScanRadar").addEventListener("click", () => runReleaseRadar(false));
        document.getElementById("btnSaveRadar").addEventListener("click", () => runReleaseRadar(true));

        // Preferences, Automation & Feedback Memory
        document.getElementById("btnAddPenalty").addEventListener("click", () => {
            const name = document.getElementById("penArtistName").value.trim();
            const mult = parseFloat(document.getElementById("penArtistMult").value || "0.05");
            if (!name) return;
            state.penalizedArtists[name] = Math.max(0, Math.min(2, mult));
            document.getElementById("penArtistName").value = "";
            renderPenalties();
        });
        document.getElementById("penaltyList").addEventListener("click", (e) => {
            const btn = e.target.closest(".rm-pen");
            if (!btn) return;
            delete state.penalizedArtists[btn.dataset.art];
            renderPenalties();
        });
        document.getElementById("btnSavePreferences").addEventListener("click", async () => {
            try {
                const payload = {
                    preferences: {
                        avoid_artists: document.getElementById("prefAvoidArtists").value,
                        avoid_genres: document.getElementById("prefAvoidGenres").value,
                        half_life_days: parseFloat(document.getElementById("prefHalfLife").value || "120"),
                        max_liked_songs: parseInt(document.getElementById("prefMaxLiked").value || "500", 10),
                        default_playlist_size: parseInt(document.getElementById("prefDefaultSize").value || "30", 10),
                        skip_penalty_enabled: document.getElementById("prefSkipPenalty").checked,
                        default_fast_mode: document.getElementById("prefDefaultFast").checked,
                        default_public_export: document.getElementById("prefDefaultPublic").checked,
                        sync_likes_to_spotify: document.getElementById("prefSyncLikesSpotify").checked,
                        penalized_artists: state.penalizedArtists,
                    },
                    spotify_client_id: document.getElementById("prefSpotifyCid").value.trim(),
                    sync_session: true,
                };
                const saved = await apiJson("/api/preferences", { method: "POST", body: payload });
                applyStudioDefaultsFromPrefs(saved.preferences);
                document.getElementById("prefSpotifyCid").value = "";
                await refreshStatus();
                await loadPreferences();
                showToast("Preferences saved & synced with active session!", "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("prefLlmBackend").addEventListener("change", toggleOllamaBox);
        document.getElementById("llmTierCards").addEventListener("click", (e) => {
            const card = e.target.closest(".tier-card[data-model-key]");
            if (!card) return;
            state.selectedLlmModelKey = card.dataset.modelKey || "auto";
            document.querySelectorAll("#llmTierCards .tier-card").forEach((el) => {
                el.classList.toggle("active", el.dataset.modelKey === state.selectedLlmModelKey);
            });
            document.getElementById("prefLlmBackend").value = "openvino";
            toggleOllamaBox();
        });
        document.getElementById("ollamaTierChips").addEventListener("click", (e) => {
            const chip = e.target.closest(".ollama-pick-btn");
            if (!chip) return;
            document.getElementById("prefOllamaModel").value = chip.dataset.model || "qwen3:8b";
        });
        document.getElementById("btnProbeOllama").addEventListener("click", async () => {
            const url = document.getElementById("prefOllamaUrl").value.trim();
            try {
                const res = await apiJson("/api/llm/probe-ollama", { method: "POST", body: { ollama_url: url } });
                renderOllamaSuggestions((state.llmCatalogData && state.llmCatalogData.ollama_tiers) || [], res.models || []);
                if (res.running) {
                    showToast(`Detected local Ollama (${(res.models || []).length} model tags).`, "ok");
                } else {
                    showToast("Ollama server not reachable at that URL.", "err");
                }
            } catch (err) {
                showToast(err.message, "err");
            }
        });
        document.getElementById("btnSaveLlmConfig").addEventListener("click", async () => {
            try {
                const payload = {
                    preferences: {
                        llm_backend: document.getElementById("prefLlmBackend").value,
                        llm_model_key: state.selectedLlmModelKey || "auto",
                        llm_device: document.getElementById("prefLlmDevice").value,
                        ollama_url: document.getElementById("prefOllamaUrl").value.trim(),
                        ollama_model: document.getElementById("prefOllamaModel").value.trim(),
                    },
                };
                await apiJson("/api/preferences", { method: "POST", body: payload });
                await loadLlmCatalog();
                await refreshStatus();
                showToast("Saved Local AI Model & Hardware Tier settings!", "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnSaveYtmHeaders").addEventListener("click", async () => {
            const rawHeaders = document.getElementById("ytmHeadersInput").value.trim();
            if (!rawHeaders) {
                showToast("Paste your raw request headers from music.youtube.com first.", "err");
                return;
            }
            try {
                const res = await apiJson("/api/ytmusic/auth/browser", { method: "POST", body: { headers: rawHeaders } });
                document.getElementById("ytmHeadersInput").value = "";
                await loadYtmAuthStatus();
                await refreshStatus(true);
                if (res.warning) {
                    showToast(res.warning, "err");
                } else {
                    showToast("Saved YouTube Music browser headers!", "ok");
                }
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnStartYtmOauth").addEventListener("click", async () => {
            const cid = document.getElementById("ytmOauthClientId").value.trim();
            const csec = document.getElementById("ytmOauthClientSecret").value.trim();
            try {
                const res = await apiJson("/api/ytmusic/auth/oauth/start", {
                    method: "POST",
                    body: { client_id: cid, client_secret: csec },
                });
                state.ytmOauthDeviceCode = res.device_code || null;
                const vUrl = safeHttpUrl(res.verification_url) || "https://www.google.com/device";
                const uCode = res.user_code || "";
                const box = document.getElementById("ytmOauthPromptBox");
                const inst = document.getElementById("ytmOauthInstructions");
                inst.innerHTML = `1. Visit <a class="track-link" href="${escapeHtml(vUrl)}" target="_blank" rel="noopener noreferrer">${escapeHtml(vUrl)}</a><br>2. Enter code: <b>${escapeHtml(uCode)}</b><br>3. Approve access, then click <b>I Authorized — Complete Login</b> below.`;
                box.style.display = "block";
                showToast("Google OAuth device code ready!", "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnCompleteYtmOauth").addEventListener("click", async () => {
            if (!state.ytmOauthDeviceCode) {
                showToast("Click 'Start Google OAuth Device Flow' first.", "err");
                return;
            }
            const cid = document.getElementById("ytmOauthClientId").value.trim();
            const csec = document.getElementById("ytmOauthClientSecret").value.trim();
            try {
                await apiJson("/api/ytmusic/auth/oauth/complete", {
                    method: "POST",
                    body: { device_code: state.ytmOauthDeviceCode, client_id: cid, client_secret: csec },
                });
                state.ytmOauthDeviceCode = null;
                document.getElementById("ytmOauthPromptBox").style.display = "none";
                await loadYtmAuthStatus();
                await refreshStatus();
                showToast("YouTube Music Google OAuth linked successfully!", "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        document.getElementById("btnAutoSave").addEventListener("click", () => actAutomation("save"));
        document.getElementById("btnAutoInstall").addEventListener("click", () => actAutomation("install_task"));
        document.getElementById("btnAutoRemove").addEventListener("click", () => actAutomation("remove_task"));
        document.getElementById("btnAutoRunNow").addEventListener("click", () => actAutomation("run_now"));

        document.getElementById("btnRefreshFeedback").addEventListener("click", loadFeedbackMemory);
        document.getElementById("tab-preferences").addEventListener("click", async (e) => {
            const clr = e.target.closest(".clear-fb-btn");
            if (!clr) return;
            try {
                await apiJson("/api/feedback", {
                    method: "POST",
                    body: {
                        track: { id: clr.dataset.id, name: clr.dataset.name, artist: clr.dataset.artist },
                        rating: "clear",
                    },
                });
                loadFeedbackMemory();
                showToast(`Cleared feedback for "${clr.dataset.name}".`, "ok");
            } catch (err) {
                showToast(err.message, "err");
            }
        });

        // Doctor
        document.getElementById("btnRunDoctor").addEventListener("click", loadDoctor);
        document.getElementById("btnPurgePoisoned").addEventListener("click", async () => {
            try {
                const res = await apiJson("/api/doctor/purge-reccobeats", { method: "POST", body: { include_not_found: false } });
                showToast(`Purged ${res.removed} poisoned entries.`, "ok");
                loadDoctor();
            } catch (err) {
                showToast(err.message, "err");
            }
        });
        document.getElementById("btnPurgeNotFound").addEventListener("click", async () => {
            try {
                const res = await apiJson("/api/doctor/purge-reccobeats", { method: "POST", body: { include_not_found: true } });
                showToast(`Cleared ${res.removed} entries for retry.`, "ok");
                loadDoctor();
            } catch (err) {
                showToast(err.message, "err");
            }
        });
        document.getElementById("logFilesList").addEventListener("click", async (e) => {
            const btn = e.target.closest(".log-file-btn");
            if (!btn) return;
            try {
                const res = await apiJson(`/api/jobs/logs?file=${encodeURIComponent(btn.dataset.file)}`);
                document.getElementById("logFileViewer").textContent = res.content || "(empty)";
            } catch (err) {
                showToast(err.message, "err");
            }
        });
    }

    function applyStudioDefaultsFromPrefs(p) {
        if (!p) return;
        const defSize = p.default_playlist_size ?? 30;
        document.getElementById("studioSize").value = String(defSize);
        document.getElementById("studioSizeVal").textContent = String(defSize);
        document.getElementById("studioFastMode").checked = Boolean(p.default_fast_mode);
        document.getElementById("chkExportPublic").checked = Boolean(p.default_public_export);
        if (p.active_provider) {
            state.activeProvider = p.active_provider;
            const provSel = document.getElementById("activeProviderSelect");
            if (provSel) provSel.value = p.active_provider;
        }
    }

    async function initStudioDefaults() {
        try {
            const res = await apiJson("/api/preferences");
            applyStudioDefaultsFromPrefs(res.preferences);
        } catch (err) {
            console.error(err);
        }
    }

    window.addEventListener("plotly-ready", () => {
        if (state.mapData) {
            renderTasteMapPlot();
            renderClusterRadar(state.mapData.clusters || []);
        }
        if (state.analytics) {
            loadAnalytics();
        }
    });

    // Boot
    bindEvents();
    connectJobStream();
    refreshStatus(true);
    initStudioDefaults();
    loadLlmCatalog(false);
    loadPresets();
    loadFeedbackMemory();
    loadLastPlaylistIntoStudio();
    document.addEventListener("visibilitychange", () => {
        if (!document.hidden) refreshStatus(true);
    });
    setInterval(() => refreshStatus(false), 5000);
})();
