const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function previewApp(fetchResult, { apiReady = true } = {}) {
    const elements = new Map();
    const scripts = [];
    const players = [];
    const warnings = [];
    const timers = new Set();
    const element = () => ({
        style: {}, dataset: {}, src: "", textContent: "", paused: true,
        pause() { this.paused = true; },
        load() {},
        play() { assert.fail("Opening a preview must not start playback"); },
        removeAttribute(name) { this[name] = ""; },
        replaceChildren(...children) { this.children = children; },
    });
    const document = {
        head: { appendChild(script) { scripts.push(script); } },
        createElement: element,
        getElementById(id) {
            if (!elements.has(id)) {
                elements.set(id, element());
            }
            return elements.get(id);
        },
    };
    const yt = {
        Player: class {
            constructor(mount, options) {
                this.options = options;
                this.destroyed = false;
                players.push(this);
            }
            destroy() { this.destroyed = true; }
            playVideo() { assert.fail("Opening YouTube must not start playback"); }
        },
    };
    const window = { location: { origin: "http://127.0.0.1:8765" }, addEventListener() {} };
    if (apiReady) window.YT = yt;
    const requests = [];
    const context = vm.createContext({
        document,
        window,
        console: { warn: (...args) => warnings.push(args) },
        setTimeout: (fn) => { timers.add(fn); return fn; },
        clearTimeout: (fn) => timers.delete(fn),
        URLSearchParams,
        fetch: async (url) => {
            requests.push(url);
            return {
                ok: true,
                headers: { get: () => "application/json" },
                json: async () => typeof fetchResult === "function" ? fetchResult() : fetchResult,
            };
        },
    });
    const source = fs.readFileSync(path.join(__dirname, "../web_ui/app.js"), "utf8");
    const boot = source.indexOf("    // Boot");
    assert.ok(boot > 0);
    // Exercise the real UI functions without starting background requests or timers.
    vm.runInContext(source.slice(0, boot) +
        "globalThis.app = { openPreviewTrack, closePreviewDock, plotlyText, escapeHtml };})();", context);
    return { ...context.app, elements, requests, players, scripts, warnings, timers, window, yt };
}

const track = {
    id: "yt:l_foa0DjG5k", provider: "ytmusic",
    name: "The Devil Went Down to Georgia", artist: "The Charlie Daniels Band",
};

test("YouTube opens first, paused, without requesting an available audio preview", async () => {
    const app = previewApp({ found: true, preview_url: "https://example.com/preview.mp3", album: "Full Moon" });
    await app.openPreviewTrack(track);
    assert.equal(app.requests.length, 0);
    assert.equal(app.players.length, 1);
    assert.equal(app.players[0].options.videoId, "l_foa0DjG5k");
    assert.equal(app.players[0].options.playerVars.autoplay, 0);
    assert.equal(app.players[0].options.playerVars.origin, "http://127.0.0.1:8765");
    assert.equal(app.elements.get("previewYoutubeWrap").style.display, "block");
    assert.equal(app.elements.get("previewAudioWrap").style.display, "none");
    app.players[0].options.events.onReady();
    assert.equal(app.elements.get("previewEmbedNotice").style.display, "none");
});

test("YouTube error triggers paused audio fallback and retains the actual diagnostic", async () => {
    const app = previewApp({ found: true, preview_url: "https://example.com/preview.mp3", album: "Full Moon" });
    await app.openPreviewTrack(track);
    await app.players[0].options.events.onError({ data: 150 });
    assert.equal(app.requests.length, 1);
    const query = new URL(app.requests[0], "http://localhost").searchParams;
    assert.equal(query.get("title"), track.name);
    assert.equal(query.get("artist"), track.artist);
    assert.equal(app.elements.get("previewAudio").src, "https://example.com/preview.mp3");
    assert.equal(app.elements.get("previewAudio").paused, true);
    assert.equal(app.elements.get("previewIframe").src, "about:blank");
    assert.equal(app.elements.get("previewAudioMeta").textContent, "30s audio preview · Full Moon");
    assert.equal(app.players[0].destroyed, true);
    assert.equal(app.elements.get("previewYoutubeWrap").style.display, "none");
    assert.equal(app.elements.get("previewEmbedNotice").dataset.youtubeError, "150");
    assert.match(app.elements.get("previewEmbedNotice").textContent, /150: This upload does not allow embedded playback/);
    assert.equal(app.warnings.length, 1);
});

test("missing or failed fallback audio keeps the error and external link without retry loops", async () => {
    for (const response of [{ found: false }, { found: true, preview_url: "https://example.com/preview.mp3" }]) {
        const app = previewApp(response);
        await app.openPreviewTrack(track);
        await app.players[0].options.events.onError({ data: 100 });
        if (response.found) app.elements.get("previewAudio").onerror();
        assert.equal(app.elements.get("previewExternalLink").href, "https://music.youtube.com/watch?v=l_foa0DjG5k");
        assert.equal(app.elements.get("previewEmbedNotice").style.display, "block");
        assert.equal(app.elements.get("previewAudioWrap").style.display, "none");
        assert.match(app.elements.get("previewEmbedNotice").textContent, /100:.*removed or is private.*No audio preview available/);
        assert.equal(app.players.length, 1);
        assert.equal(app.requests.length, 1);
    }
});

test("closing the dock prevents a late fallback response from reopening or playing it", async () => {
    let resolve;
    const response = new Promise((done) => { resolve = done; });
    const app = previewApp(() => response);
    await app.openPreviewTrack(track);
    const opening = app.players[0].options.events.onError({ data: 150 });
    app.closePreviewDock();
    resolve({ found: true, preview_url: "https://example.com/late.mp3" });
    await opening;
    assert.equal(app.elements.get("previewDock").style.display, "none");
    assert.equal(app.elements.get("previewAudio").src, "");
    assert.equal(app.elements.get("previewIframe").src, "about:blank");
    assert.equal(app.players[0].destroyed, true);
});

test("selecting another track prevents a stale response from replacing its preview", async () => {
    let resolve;
    let calls = 0;
    const response = new Promise((done) => { resolve = done; });
    const app = previewApp(() => ++calls === 1 ? response : { found: false });
    await app.openPreviewTrack(track);
    const first = app.players[0].options.events.onError({ data: 150 });
    await app.openPreviewTrack({ ...track, id: "yt:M-WYsKB9jOI", name: "Sindhu Sesh" });
    resolve({ found: true, preview_url: "https://example.com/stale.mp3" });
    await first;
    assert.equal(app.players[1].options.videoId, "M-WYsKB9jOI");
    assert.equal(app.players[1].destroyed, false);
    assert.equal(app.elements.get("previewAudio").src, "");
    assert.equal(app.elements.get("previewEmbedNotice").dataset.youtubeError, undefined);
});

test("error 153 is reported as client identification failure, not an embedding prohibition", async () => {
    const app = previewApp({ found: false });
    await app.openPreviewTrack(track);
    await app.players[0].options.events.onError({ data: 153 });
    assert.match(app.elements.get("previewEmbedNotice").textContent, /153:.*missing referrer/);
    assert.doesNotMatch(app.elements.get("previewEmbedNotice").textContent, /does not allow/);
});

test("late errors from a closed player cannot affect a new selection", async () => {
    const app = previewApp({ found: false });
    await app.openPreviewTrack(track);
    await app.openPreviewTrack({ ...track, id: "yt:M-WYsKB9jOI", name: "Sindhu Sesh" });
    await app.players[0].options.events.onError({ data: 150 });
    assert.equal(app.requests.length, 0);
    assert.equal(app.players[1].destroyed, false);
});

test("YouTube API load failures and timeouts fall back without inventing a player error code", async () => {
    for (const timeout of [false, true]) {
        const app = previewApp({ found: false }, { apiReady: false });
        const opening = app.openPreviewTrack(track);
        assert.equal(app.scripts[0].src, "https://www.youtube.com/iframe_api");
        if (timeout) [...app.timers][0]();
        else app.scripts[0].onerror();
        await opening;
        assert.match(app.elements.get("previewEmbedNotice").textContent, /YouTube player could not load/);
        assert.equal(app.elements.get("previewEmbedNotice").dataset.youtubeError, undefined);
        assert.equal(app.requests.length, 1);
        assert.equal(app.timers.size, 0);
    }
});

test("closing while the YouTube API loads prevents a player being created later", async () => {
    const app = previewApp({ found: false }, { apiReady: false });
    const opening = app.openPreviewTrack(track);
    app.closePreviewDock();
    app.window.YT = app.yt;
    app.window.onYouTubeIframeAPIReady();
    await opening;
    assert.equal(app.players.length, 0);
    assert.equal(app.requests.length, 0);
    assert.equal(app.elements.get("previewDock").style.display, "none");
    assert.equal(app.timers.size, 0);
});

test("Spotify previews also open without autoplay", async () => {
    const app = previewApp({ found: false });
    await app.openPreviewTrack("4cOdK2wGLETKBW3PvgPWqT", "Song");
    assert.match(app.elements.get("previewIframe").src, /open.spotify.com\/embed\/track\//);
    assert.doesNotMatch(app.elements.get("previewIframe").src, /autoplay=1/);
    assert.equal(app.requests.length, 0);
});

test("Plotly titles retain quotes, wrap long text, and escape injected markup", () => {
    const app = previewApp({ found: false });
    const title = 'KSI "Thick of It" PARODY I\'ve Got a Little Di.. ~ Rucka Rucka Ali';
    const formatted = app.plotlyText(title);
    assert.ok(formatted.includes('"Thick of It"'));
    assert.ok(formatted.includes("I've"));
    assert.ok(formatted.includes("<br>"));
    assert.doesNotMatch(formatted, /&quot;|&#39;/);
    assert.equal(app.plotlyText('<img src=x onerror="alert(1)"> & more'), '&lt;img src=x onerror="alert(1)"&gt; &amp; more');
    assert.equal(app.escapeHtml('" onclick="bad'), "&quot; onclick=&quot;bad");
    assert.ok(app.plotlyText("x".repeat(110)).split("<br>").every((line) => line.length <= 48));
});
