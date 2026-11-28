const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function mapApp({ is3d = false, clouds = true, drilled = false, pointsPerGroup = 4 } = {}) {
    const elements = new Map();
    const calls = { newPlot: 0, restyle: [] };
    const document = {
        getElementById(id) {
            if (!elements.has(id)) elements.set(id, {
                value: "", checked: true, style: {}, on() {},
                setAttribute(name, value) { this[name] = value; },
            });
            return elements.get(id);
        },
    };
    const Plotly = {
        newPlot(plot, data, layout) {
            calls.newPlot++;
            plot.data = data;
            plot.layout = layout;
        },
        restyle(plot, update, indices) {
            calls.restyle.push({ update, indices });
            indices.forEach((idx, pos) => {
                for (const [key, values] of Object.entries(update)) {
                    if (key.startsWith("marker.")) plot.data[idx].marker[key.slice(7)] = values[pos];
                    else plot.data[idx][key] = values;
                }
            });
        },
    };
    const context = vm.createContext({ document, Plotly, window: { addEventListener() {} } });
    const source = fs.readFileSync(path.join(__dirname, "../web_ui/app.js"), "utf8");
    const boot = source.indexOf("    // Boot");
    assert.ok(boot > 0);
    vm.runInContext(source.slice(0, boot) +
        "globalThis.app = { state, renderTasteMapPlot, updateTasteMapClouds, getMapMinWeight, applyTasteMapFilter };})();", context);
    const app = context.app;
    app.state.is3d = is3d;
    app.state.mapMaxWeight = 100;
    const clusters = [0, 1].map((cid) => ({ cluster_id: cid, vibe_label: `Group ${cid}`, weighted_share_pct: 50 }));
    const tracks = [0, 1].flatMap((group) =>
        [[0, 0, 0], [1, 0, 0], [1, 1, 1], [0, 1, 0]].slice(0, pointsPerGroup).map(([x, y, z], idx) => ({
            id: `${group}-${idx}`, name: `Track ${group}-${idx}`, artist: "Artist", genres: "jazz",
            cluster_id: drilled ? 0 : group, sub_cluster_id: group, recency_weight: 1,
            x: x + group * 3, y, x3: x + group * 3, y3: y, z,
        }))
    );
    if (drilled) {
        clusters[0].subclusters = [0, 1].map((cid) => ({ sub_cluster_id: cid, vibe_label: `Sub ${cid}`, raw_percentage: 50 }));
        app.state.drillClusterId = 0;
    }
    app.state.mapData = { clusters, tracks, era_trajectory: [{ year: 2025, x: 0, y: 0 }, { year: 2026, x: 1, y: 1 }] };
    document.getElementById("chkShowClouds").checked = clouds;
    return { ...app, elements, calls, plot: document.getElementById("tasteMapPlot") };
}

for (const is3d of [false, true]) {
    test(`${is3d ? "3D" : "2D"} clouds toggle without recreating the plot or changing tracks and view`, async () => {
        const app = mapApp({ is3d });
        app.renderTasteMapPlot();
        const plotData = app.plot.data;
        const layout = app.plot.layout;
        if (is3d) layout.scene.camera = { eye: { x: 2, y: -1, z: 0.5 } };
        else layout.xaxis.range = [0.2, 0.8];
        const savedView = JSON.stringify(layout);
        const tracks = plotData.filter((trace) => trace.customdata);
        const trackSnapshot = JSON.stringify(tracks);
        const cloudIndices = plotData.flatMap((trace, idx) => trace.meta?.layer === "cluster-cloud" ? [idx] : []);
        assert.equal(cloudIndices.length, 2);

        for (const visible of [false, true]) {
            app.elements.get("chkShowClouds").checked = visible;
            await app.updateTasteMapClouds();
            assert.equal(app.calls.newPlot, 1);
            assert.equal(app.plot.data, plotData);
            assert.equal(app.plot.layout, layout);
            assert.equal(JSON.stringify(layout), savedView);
            assert.equal(JSON.stringify(tracks), trackSnapshot);
            assert.deepEqual(Array.from(app.calls.restyle.at(-1).indices), Array.from(cloudIndices));
            assert.ok(cloudIndices.every((idx) => plotData[idx].visible === visible));
        }
    });

    test(`${is3d ? "3D" : "2D"} clouds initially hidden can be enabled without rebuilding`, async () => {
        const app = mapApp({ is3d, clouds: false, drilled: true });
        app.renderTasteMapPlot();
        const clouds = app.plot.data.filter((trace) => trace.meta?.layer === "cluster-cloud");
        assert.equal(clouds.length, 2);
        assert.ok(clouds.every((trace) => trace.visible === false));
        app.elements.get("chkShowClouds").checked = true;
        await app.updateTasteMapClouds();
        assert.ok(clouds.every((trace) => trace.visible === true));
        assert.equal(app.calls.newPlot, 1);
    });
}

test("a map without clouds leaves all track layers alone", async () => {
    const app = mapApp({ pointsPerGroup: 3 });
    app.renderTasteMapPlot();
    await app.updateTasteMapClouds();
    assert.equal(app.calls.restyle.length, 0);
    assert.equal(app.calls.newPlot, 1);
    assert.ok(app.plot.data.some((trace) => trace.customdata));
});

test("toggling before the map is loaded does not create a plot", async () => {
    const app = mapApp();
    await app.updateTasteMapClouds();
    assert.equal(app.calls.newPlot, 0);
    assert.equal(app.calls.restyle.length, 0);
});

test("3D clouds remain excluded from track picking after becoming visible", async () => {
    const app = mapApp({ is3d: true, clouds: false });
    app.renderTasteMapPlot();
    await Promise.resolve();
    const mesh = { drawPick() {}, pick: () => "cloud" };
    app.plot._fullLayout = { scene: { _scene: { traces: { cloud: { data: { type: "mesh3d" }, mesh } } } } };
    app.elements.get("chkShowClouds").checked = true;
    await app.updateTasteMapClouds();
    assert.equal(mesh.pick(), null);
});

test("weight slider gives most of its travel to low weights and retains the full range", () => {
    const app = mapApp();
    app.renderTasteMapPlot();
    const input = app.elements.get("mapWeightSlider");
    app.state.mapMaxWeight = 97.13;
    const weights = [0, 250, 500, 750, 1000].map((position) => {
        input.value = String(position);
        return app.getMapMinWeight();
    });
    assert.equal(weights[0], 0);
    assert.ok(weights[1] < 1);
    assert.ok(weights[3] < 20);
    assert.equal(weights[4], 97.13);
    assert.ok(weights.every((weight, idx) => idx === 0 || weight > weights[idx - 1]));
    assert.equal(app.elements.get("mapWeightVal").textContent, "97.13");
    assert.equal(input["aria-valuetext"], "97.13 minimum weight");
    app.state.mapMaxWeight = 0;
    assert.equal(app.getMapMinWeight(), 0);
});

test("fractional weight filtering uses the displayed threshold and does not recreate the map", () => {
    const app = mapApp();
    const weights = [0.15, 0.21, 0.33, 0.9, 1, 5, 20, 100];
    app.state.mapData.tracks.forEach((track, idx) => { track.recency_weight = weights[idx]; });
    app.renderTasteMapPlot();
    const slider = app.elements.get("mapWeightSlider");
    slider.value = String(Math.round(1000 * Math.log1p(0.21 / 0.1) / Math.log1p(100 / 0.1)));
    const callsBefore = app.calls.newPlot;
    app.applyTasteMapFilter();
    assert.equal(app.elements.get("mapWeightVal").textContent, "0.21");
    assert.equal(app.elements.get("mapMatchCount").textContent, "7 matching tracks");
    assert.equal(app.calls.newPlot, callsBefore);
    const active = app.plot.data.filter((trace) => trace.customdata)
        .flatMap((trace) => trace.marker.opacity).filter((opacity) => opacity === 1);
    assert.equal(active.length, 7);
    slider.value = "0";
    app.applyTasteMapFilter();
    assert.equal(app.elements.get("mapWeightVal").textContent, "0");
    assert.equal(app.elements.get("mapMatchCount").textContent, "");
});

test("drill-in and back-out reuses cached view traces via Plotly.react", () => {
    const app = mapApp({ drilled: false });
    app.state.mapData.clusters[0].subclusters = [0, 1].map((cid) => ({ sub_cluster_id: cid, vibe_label: `Sub ${cid}`, raw_percentage: 50 }));
    let reactCalls = 0;
    app.plot._reactTracker = 0;
    // Initial overview render
    app.renderTasteMapPlot();
    const overviewTraces = app.plot.data;
    assert.equal(app.state.mapViewCache.size, 1);

    // Drill into cluster 0
    app.state.drillClusterId = 0;
    app.renderTasteMapPlot();
    const drilledTraces = app.plot.data;
    assert.equal(app.state.mapViewCache.size, 2);
    assert.notEqual(drilledTraces, overviewTraces);

    // Return to overview -> exact cached traces reused without rebuilding
    app.state.drillClusterId = null;
    app.renderTasteMapPlot();
    assert.equal(app.plot.data, overviewTraces);
    assert.equal(app.state.mapViewCache.size, 2);
});
