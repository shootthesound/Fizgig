import json
import os
import socket
import threading
import webbrowser

from http.server import HTTPServer, SimpleHTTPRequestHandler

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT


class GalleryLikenessMixin:
    def get_samples_dir(self):
        """Get the samples directory from output dir"""
        output_dir = self.settings.get("LORA_OUTPUT_DIR", "")
        if output_dir:
            return os.path.join(output_dir, "sample")
        # Fallback to local samples folder
        return os.path.join(_REPO_ROOT, "output_loras", "sample")

    def find_free_port(self):
        """Find an available port for the HTTP server"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    def start_gallery_server(self):
        """Start HTTP server to serve samples directory (avoids CORS issues)"""
        if self.gallery_server is not None:
            return  # Already running

        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)
        # LoRA checkpoints live in the output dir (parent of sample/); serve them via /loras/.
        output_dir = self.settings.get("LORA_OUTPUT_DIR", "") or os.path.dirname(samples_dir)

        # Find free port
        self.gallery_server_port = self.find_free_port()

        # Create handler that serves images from samples/ and checkpoints from /loras/ (output dir).
        app = self   # for the likeness endpoints (never touch Tk vars from handler threads)

        class SamplesHandler(SimpleHTTPRequestHandler):
            def __init__(handler_self, *args, **kwargs):
                super().__init__(*args, directory=samples_dir, **kwargs)

            def translate_path(handler_self, path):
                # /loras/<file> -> the checkpoint in the output dir (basename-only, no traversal).
                # /dataset/<file> -> a training image (for the likeness baseline picker).
                clean = path.split('?', 1)[0].split('#', 1)[0]
                if clean.startswith('/loras/'):
                    import posixpath, urllib.parse
                    fname = posixpath.basename(urllib.parse.unquote(clean[len('/loras/'):]))
                    return os.path.join(output_dir, fname)
                if clean.startswith('/dataset/'):
                    import posixpath, urllib.parse
                    fname = posixpath.basename(urllib.parse.unquote(clean[len('/dataset/'):]))
                    return os.path.join(getattr(app, "_gal_dataset_dir", "") or "", fname)
                return super().translate_path(path)

            def do_POST(handler_self):
                # /set_baselines {"baselines": [3 names]} -> start CPU likeness scoring;
                # empty list clears it. Everything else is a 404.
                clean = handler_self.path.split('?', 1)[0].split('#', 1)[0]
                if clean != '/set_baselines':
                    handler_self.send_error(404)
                    return
                try:
                    ln = int(handler_self.headers.get('Content-Length') or 0)
                    data = json.loads(handler_self.rfile.read(ln) or b'{}')
                    ok, msg = app._gallery_set_baselines(data.get('baselines') or [])
                except Exception as e:
                    ok, msg = False, str(e)
                body = json.dumps({"ok": ok, "msg": msg}).encode("utf-8")
                handler_self.send_response(200 if ok else 400)
                handler_self.send_header('Content-Type', 'application/json')
                handler_self.send_header('Content-Length', str(len(body)))
                handler_self.end_headers()
                handler_self.wfile.write(body)

            def log_message(handler_self, format, *args):
                pass  # Suppress logging

        try:
            self.gallery_server = HTTPServer(('127.0.0.1', self.gallery_server_port), SamplesHandler)

            # Run server in background thread
            def serve_forever():
                self.gallery_server.serve_forever()

            self.gallery_server_thread = threading.Thread(target=serve_forever, daemon=True)
            self.gallery_server_thread.start()

        except Exception as e:
            print(f"Failed to start gallery server: {e}")
            self.gallery_server = None
            self.gallery_server_port = None

    def stop_gallery_server(self):
        """Stop the HTTP server"""
        if self.gallery_server is not None:
            self.gallery_server.shutdown()
            self.gallery_server = None
            self.gallery_server_port = None

    def open_samples_gallery(self):
        """Open the samples gallery HTML viewer in browser via HTTP server"""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)

        gallery_path = os.path.join(samples_dir, "gallery.html")

        # Snapshot the dataset folder for the likeness picker/scorer — the HTTP handler and
        # the scoring worker run on background threads and must never touch Tk variables.
        self._gal_dataset_dir = (self.image_folder_var.get().strip()
                                 if hasattr(self, "image_folder_var") else "")

        # Opening the gallery claims the samples dir for THIS session — any other running
        # Fizgig instance's watcher/scorer stands down instead of fighting over the sidecars.
        self._gallery_claim(samples_dir)

        # Always regenerate the template so template changes (e.g. the per-epoch download link) are
        # picked up — otherwise a stale gallery.html from an earlier run keeps the old JS forever.
        # The file is purely generated (static template + embedded data filled by update_gallery_html),
        # so overwriting it loses nothing.
        self.create_gallery_html(gallery_path)

        # Generate/update the gallery HTML with current files
        self.update_gallery_html()

        # If a previous session picked likeness baselines, resume scoring automatically.
        self._gallery_resume_likeness()

        # Start HTTP server if not running
        self.start_gallery_server()

        if self.gallery_server_port:
            # Open via HTTP (avoids CORS issues)
            webbrowser.open(f"http://127.0.0.1:{self.gallery_server_port}/gallery.html")
        else:
            # Fallback to file:// if server failed
            webbrowser.open(f"file://{os.path.abspath(gallery_path)}")

    def open_samples_folder(self):
        """Open the samples folder in file explorer"""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)
        self._open_in_file_manager(samples_dir)

    def create_gallery_html(self, gallery_path):
        """Create the gallery HTML file if it doesn't exist"""
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample Gallery - Fizgig</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1B2A38; color: #ECF0F1; min-height: 100vh; }
        header { background-color: #2C3E50; padding: 20px; border-bottom: 2px solid #2980B9; position: sticky; top: 0; z-index: 100; }
        header h1 { color: #ECF0F1; font-size: 24px; margin-bottom: 15px; display: flex; align-items: center; gap: 15px; }
        .live-indicator { width: 12px; height: 12px; background-color: #27AE60; border-radius: 50%; animation: pulse 2s infinite; }
        .live-indicator.paused { background-color: #95A5A6; animation: none; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .controls { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
        .controls label { display: flex; align-items: center; gap: 8px; }
        .controls select { padding: 5px 8px; border: 1px solid #2980B9; border-radius: 4px; background-color: #1B2A38; color: #ECF0F1; }
        .controls button { padding: 8px 16px; background-color: #2980B9; color: #ECF0F1; border: none; border-radius: 4px; cursor: pointer; }
        .controls button:hover { background-color: #3498DB; }
        .status { color: #95A5A6; font-size: 14px; }
        main { padding: 20px; }
        #gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }
        .gallery-item { background-color: #2C3E50; border-radius: 8px; overflow: hidden; cursor: pointer; transition: transform 0.2s, box-shadow 0.2s; }
        .gallery-item:hover { transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); }
        .gallery-item.new { animation: highlight 2s ease-out; }
        @keyframes highlight { 0%, 30% { box-shadow: 0 0 30px #27AE60; } 100% { box-shadow: none; } }
        .image-container { position: relative; }
        /* contain, not cover: a widescreen preview letterboxes instead of losing its sides —
           the grid is for JUDGING samples, and a cropped frame lies about the composition */
        .gallery-item img { width: 100%; height: 280px; object-fit: contain; display: block; background-color: #1B2A38; }
        .badge { position: absolute; top: 10px; left: 10px; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
        .epoch-badge { background-color: #27AE60; color: white; }
        .clip-badge { background-color: #8E44AD; color: white; right: 8px; left: auto; }
        /* below the clip badge with clear air — the two overlapped at 34px (Peter) */
        .sound-badge { background-color: #16A085; color: white; right: 8px; left: auto; top: 40px; }
        #lightbox-vid { display: none; max-width: 90vw; max-height: 72vh; border-radius: 4px; }
        #lb-scrub-wrap { display: none; width: min(80vw, 640px); margin-top: 10px; text-align: center; }
        #lb-scrub-wrap.active { display: block; }
        #lb-scrub { width: 100%; }
        #lb-scrub-label { color: #95A5A6; font-size: 12px; margin-top: 2px; }
        .new-badge { position: absolute; top: 10px; right: 10px; background-color: #E74C3C; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
        .image-info { padding: 12px; }
        .lora-name { color: #9B59B6; font-weight: 600; font-size: 14px; margin-bottom: 6px; }
        .meta-row { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; }
        .meta-item { font-size: 13px; color: #BDC3C7; }
        .lora-download { display: inline-block; margin-top: 8px; padding: 5px 10px; font-size: 12px; font-weight: 600;
                         color: #fff; background: #9B59B6; border-radius: 6px; text-decoration: none; }
        .lora-download:hover { background: #8E44AD; }
        .final-lora-btn { padding: 6px 12px; font-size: 13px; font-weight: 600; color: #fff; background: #27AE60;
                          border-radius: 6px; text-decoration: none; }
        .final-lora-btn:hover { background: #219150; }
        .meta-item.seed { color: #3498DB; font-family: monospace; }
        .meta-item.time { color: #95A5A6; }
        .no-images { grid-column: 1 / -1; text-align: center; padding: 60px 20px; color: #95A5A6; }
        .no-images h2 { margin-bottom: 15px; color: #ECF0F1; }
        .stats { background-color: #2C3E50; padding: 8px 15px; border-radius: 4px; font-size: 14px; }
        #lightbox { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.95); z-index: 1000; justify-content: center; align-items: center; flex-direction: column; }
        #lightbox.active { display: flex; }
        #lightbox img { max-width: 90%; max-height: 80%; object-fit: contain; }
        #lightbox .close-btn { position: absolute; top: 20px; right: 30px; font-size: 40px; color: #ECF0F1; cursor: pointer; }
        #lightbox .close-btn:hover { color: #E74C3C; }
        #lightbox .nav-btn { position: absolute; top: 50%; transform: translateY(-50%); font-size: 50px; color: #ECF0F1; cursor: pointer; padding: 20px; user-select: none; }
        #lightbox .nav-btn:hover { color: #2980B9; }
        #lightbox .prev-btn { left: 20px; }
        #lightbox .next-btn { right: 20px; }
        #lightbox .image-details { margin-top: 15px; text-align: center; }
        #lightbox .image-name { color: #ECF0F1; font-size: 16px; }
        #lightbox .image-meta { color: #95A5A6; font-size: 14px; margin-top: 5px; }
        .lik-badge { position: absolute; bottom: 10px; left: 10px; padding: 4px 10px; border-radius: 4px;
                     font-weight: bold; font-size: 13px; color: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
        .lik-good { background-color: #27AE60; } .lik-mid { background-color: #E67E22; }
        .lik-bad { background-color: #C0392B; } .lik-na { background-color: #5D6D7E; }
        #likeness-panel { display: none; background-color: #22303F; border-bottom: 1px solid #2C3E50; padding: 12px 20px; }
        #likeness-panel h3 { font-size: 15px; margin-bottom: 8px; color: #ECF0F1; }
        #lik-chart { background-color: #1B2A38; border-radius: 6px; width: 100%; height: 150px; display: block; }
        #likeness-panel .lik-note { color: #95A5A6; font-size: 12px; margin-top: 6px; }
        #basepicker { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                      background-color: rgba(0,0,0,0.93); z-index: 1100; overflow-y: auto; padding: 24px 30px; }
        #basepicker.active { display: block; }
        #bp-bar { position: sticky; top: -24px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
                  background-color: rgba(0,0,0,0.93); padding: 12px 0; z-index: 1; }
        #bp-bar h2 { font-size: 20px; margin-right: 8px; }
        #bp-bar button { padding: 8px 16px; background-color: #2980B9; color: #ECF0F1; border: none; border-radius: 4px; cursor: pointer; }
        #bp-bar button:disabled { background-color: #5D6D7E; cursor: default; }
        .bp-sub { color: #95A5A6; margin: 6px 0 14px 0; font-size: 13px; }
        #bp-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 10px; }
        .bp-item { border-radius: 6px; overflow: hidden; cursor: pointer; outline: 3px solid transparent; position: relative; }
        .bp-item img { width: 100%; height: 140px; object-fit: cover; display: block; background-color: #1B2A38; }
        .bp-item.selected { outline-color: #27AE60; }
        .bp-item .bp-num { position: absolute; top: 6px; left: 6px; background-color: #27AE60; color: #fff; font-weight: bold;
                           border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; }
        #runviz { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                  background-color: rgba(0,0,0,0.95); z-index: 1150; overflow-y: auto; padding: 20px 30px; }
        #runviz.active { display: flex; flex-direction: column; align-items: center; }
        .rv-bar { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; justify-content: center; padding: 6px 0; }
        .rv-bar h2 { font-size: 20px; }
        #runviz button, #runviz select { padding: 6px 12px; background-color: #2980B9; color: #ECF0F1;
                                         border: none; border-radius: 4px; cursor: pointer; }
        #runviz select { background-color: #1B2A38; border: 1px solid #2980B9; }
        #runviz label { display: flex; align-items: center; gap: 6px; font-size: 13px; }
        #rv-stage { position: relative; margin: 10px 0; }
        #rv-img { max-width: min(85vw, 900px); max-height: 62vh; display: block; background-color: #1B2A38; border-radius: 6px; }
        #rv-epoch { position: absolute; bottom: 12px; left: 12px; background-color: rgba(0,0,0,0.65); color: #fff;
                    padding: 5px 12px; border-radius: 4px; font-weight: bold; font-size: 15px; }
        #rv-slider { width: min(85vw, 900px); }
        #rv-note { color: #95A5A6; font-size: 12px; margin-top: 10px; max-width: 720px; text-align: center; }
        #rv-status { color: #95A5A6; font-size: 13px; }
    </style>
</head>
<body>
    <header>
        <h1><span class="live-indicator" id="live-dot"></span> Fizgig Sample Gallery</h1>
        <div class="controls">
            <label>Show: <select id="run-select">
                <option value="all">All samples</option>
                <option value="current">Current run only</option>
            </select></label>
            <label>Sort: <select id="sort-select">
                <option value="newest">Newest First</option>
                <option value="oldest">Oldest First</option>
                <option value="epoch-desc">Epoch (High-Low)</option>
                <option value="epoch-asc">Epoch (Low-High)</option>
            </select></label>
            <label>Refresh: <select id="refresh-select">
                <option value="3">3 sec</option>
                <option value="5">5 sec</option>
                <option value="10" selected>10 sec</option>
                <option value="30">30 sec</option>
                <option value="0">Off</option>
            </select></label>
            <button onclick="loadImages()">Refresh Now</button>
            <button onclick="openBaselinePicker()">🎯 Likeness scoring…</button>
            <button onclick="openRunViz()">🎞 Training Run Visualiser</button>
            <a id="final-lora-btn" class="final-lora-btn" href="#" download style="display:none">⬇ Download Final LoRA</a>
            <span class="stats" id="stats">0 images</span>
            <span class="status" id="status">Ready</span>
            <span class="status" id="lik-status"></span>
        </div>
    </header>
    <div id="likeness-panel">
        <h3>Likeness vs baselines — <span id="lik-run"></span></h3>
        <canvas id="lik-chart" width="940" height="150"></canvas>
        <div class="lik-note">Average ArcFace similarity of each epoch's samples to your 3 baseline photos
            (scoreable faces only — no-face samples are skipped). Same person typically lands 30–70%.
            This measures identity likeness ONLY: overbake / plastic skin still needs your eyes.</div>
    </div>
    <main>
        <div id="gallery">
            <div class="no-images">
                <h2>Loading...</h2>
            </div>
        </div>
    </main>
    <div id="lightbox">
        <span class="close-btn" onclick="closeLightbox()">&times;</span>
        <span class="nav-btn prev-btn" onclick="navigateLightbox(-1)">&#10094;</span>
        <img id="lightbox-img" src="" alt="">
        <video id="lightbox-vid" controls preload="metadata"></video>
        <span class="nav-btn next-btn" onclick="navigateLightbox(1)">&#10095;</span>
        <div id="lb-scrub-wrap">
            <input type="range" id="lb-scrub" min="0" max="0" value="0"
                   oninput="lbScrub(parseInt(this.value))">
            <div id="lb-scrub-label"></div>
        </div>
        <div id="lb-audio-wrap" style="display:none; margin-top:8px; text-align:center;">
            <audio id="lb-audio" controls preload="none"
                   style="width:min(80vw,640px);"></audio>
        </div>
        <div class="image-details">
            <div class="image-name" id="lightbox-name"></div>
            <div class="image-meta" id="lightbox-meta"></div>
        </div>
    </div>
    <div id="basepicker">
        <div id="bp-bar">
            <h2>🎯 Pick 3 baseline images</h2>
            <button id="bp-start" onclick="submitBaselines()" disabled>Start scoring</button>
            <button onclick="clearBaselines()">Clear scoring</button>
            <button onclick="closeBaselinePicker()">Cancel</button>
            <span class="status" id="bp-status"></span>
        </div>
        <div class="bp-sub">Choose the 3 training images that best nail the look you want — every sample is scored
            against all three and averaged, so no single photo's angle/lighting biases the result.
            Scoring runs on CPU with zero impact on training speed.<br>
            Listing: <span id="bp-folder" style="color:#3498DB">…</span> (the Start-tab training folder,
            snapshotted when the gallery was opened — reopen the gallery after changing it)</div>
        <div id="bp-grid"></div>
    </div>
    <div id="runviz">
        <div class="rv-bar">
            <h2>🎞 Training Run Visualiser</h2>
            <label>Sample slot: <select id="rv-slot" onchange="rvBuild()"></select></label>
            <button id="rv-play" onclick="rvTogglePlay()">▶ Play</button>
            <label>Speed: <select id="rv-speed">
                <option value="600">Slow</option>
                <option value="350" selected>Normal</option>
                <option value="180">Fast</option>
            </select></label>
            <button onclick="closeRunViz()">✕ Close</button>
        </div>
        <div class="rv-bar">
            <label><input type="checkbox" id="rv-pingpong" checked> Loop (ping-pong)</label>
            <label><input type="checkbox" id="rv-ticker" checked> Epoch ticker</label>
            <label><input type="checkbox" id="rv-tag" checked> Fizgig tag</label>
            <button onclick="rvExport()">⬇ Export clip (WebM)</button>
            <button onclick="rvSaveFrame()">⬇ Save frame (PNG)</button>
            <span id="rv-status"></span>
        </div>
        <div id="rv-stage">
            <img id="rv-img" src="" alt="">
            <div id="rv-epoch"></div>
        </div>
        <input type="range" id="rv-slider" min="0" max="0" value="0" oninput="rvShow(parseInt(this.value))">
        <div id="rv-note">Scrubbing this run's epochs, one sample slot at a time. Like it? The <b>LoRA Royale</b> tab
            in Fizgig does much more — checkpoint-vs-checkpoint battles, seed &amp; prompt travel, likeness scoring,
            and full MP4/GIF export with the same ticker and tag options.</div>
    </div>
    <!-- EMBEDDED_FILES_START -->
    <script id="files-data" type="application/json">[]</script>
    <!-- EMBEDDED_FILES_END -->
    <script>
        let images = [];
        let currentLightboxIndex = 0;
        let refreshTimer = null;
        let likeness = null;      // {baselines, status, scores} from likeness.json
        let bpSelected = [];      // baseline picker selection (max 3)

        document.getElementById('sort-select').value = localStorage.getItem('fizgig-sort') || 'newest';
        document.getElementById('refresh-select').value = localStorage.getItem('fizgig-refresh') || '10';
        document.getElementById('run-select').value = localStorage.getItem('fizgig-run') || 'all';

        document.getElementById('sort-select').addEventListener('change', (e) => {
            localStorage.setItem('fizgig-sort', e.target.value);
            renderGallery();
        });

        document.getElementById('run-select').addEventListener('change', (e) => {
            localStorage.setItem('fizgig-run', e.target.value);
            renderGallery();
        });

        function currentRunName() {
            // Current run = the LoRA name of the newest sample. Output folders are commonly
            // reused across trains, so old runs' samples share the folder — filter them out
            // by default rather than mixing subjects in one grid.
            let newest = null;
            images.forEach(im => { if (!newest || im.timestamp > newest.timestamp) newest = im; });
            return newest ? newest.loraName : null;
        }

        document.getElementById('refresh-select').addEventListener('change', (e) => {
            localStorage.setItem('fizgig-refresh', e.target.value);
            setupTimer();
        });

        function setupTimer() {
            if (refreshTimer) clearInterval(refreshTimer);
            const sec = parseInt(document.getElementById('refresh-select').value);
            const dot = document.getElementById('live-dot');
            if (sec > 0) {
                refreshTimer = setInterval(loadImages, sec * 1000);
                dot.classList.remove('paused');
            } else {
                dot.classList.add('paused');
            }
        }

        function parseFilename(filename) {
            // Seed segment is OPTIONAL: samples generated with a random/unspecified seed are named
            // without a trailing _<seed> (the trainer omits it when seed is None). Requiring it made
            // those files fall back to epoch 0 / no timestamp — breaking order + epoch labels.
            const match = filename.match(/^(.+)_e(\\d{6})_(\\d{2})_(\\d{14})(?:_(\\d+))?\\.png$/i);
            if (match) {
                const ts = match[4];
                return {
                    filename,
                    loraName: match[1],
                    epoch: parseInt(match[2]),
                    idx: parseInt(match[3]),
                    timestamp: ts,
                    seed: match[5] || '',
                    time: `${ts.slice(8,10)}:${ts.slice(10,12)}:${ts.slice(12,14)}`
                };
            }
            return { filename, loraName: 'Unknown', epoch: 0, idx: 0, timestamp: '', seed: '', time: '' };
        }

        async function loadImages() {
            document.getElementById('status').textContent = 'Loading...';

            // Try fetch first (works with HTTP server), fallback to embedded
            try {
                const resp = await fetch('files.json?t=' + Date.now());
                if (resp.ok) {
                    const files = await resp.json();
                    images = files.map(f => parseFilename(f));
                    // Attach a per-epoch LoRA download link where a checkpoint exists (loras.json
                    // is an epoch -> filename map written by the trainer; served via /loras/).
                    try {
                        const lr = await fetch('loras.json?t=' + Date.now());
                        if (lr.ok) {
                            const lm = await lr.json();
                            images.forEach(im => { const ck = lm[String(im.epoch)]; if (ck) im.lora = 'loras/' + encodeURIComponent(ck); });
                            // Final LoRA ({name}.safetensors) -> header button (reserved "final" key).
                            const fb = document.getElementById('final-lora-btn');
                            if (fb) {
                                if (lm.final) { fb.href = 'loras/' + encodeURIComponent(lm.final); fb.style.display = 'inline-block'; }
                                else { fb.style.display = 'none'; }
                            }
                        }
                    } catch (e) {}
                    // Clip scrub data (MiniMax clip previews): filename -> frame list.
                    try {
                        const cj = await fetch('clips.json?t=' + Date.now());
                        if (cj.ok) { const cm = await cj.json(); images.forEach(im => { if (cm[im.filename]) im.clip = cm[im.filename]; }); }
                    } catch (e) {}
                    // Sample sound (previews with audio): filename -> wav. Never autoplays.
                    try {
                        const sj = await fetch('sounds.json?t=' + Date.now());
                        if (sj.ok) { const sm = await sj.json(); images.forEach(im => { if (sm[im.filename]) im.sound = sm[im.filename]; }); }
                    } catch (e) {}
                    // Playable clips (frames + sound muxed): filename -> mp4. Never autoplays.
                    try {
                        const vj = await fetch('videos.json?t=' + Date.now());
                        if (vj.ok) { const vm = await vj.json(); images.forEach(im => { if (vm[im.filename]) im.video = vm[im.filename]; }); }
                    } catch (e) {}
                    await loadLikeness();
                    renderGallery();
                    renderLikenessChart();
                    document.getElementById('stats').textContent = `${images.length} image${images.length !== 1 ? 's' : ''}`;
                    document.getElementById('status').textContent = `Updated: ${new Date().toLocaleTimeString()}`;
                    return;
                }
            } catch (e) {
                // Fetch failed, try embedded data
            }

            // Fallback to embedded data
            const filesData = document.getElementById('files-data');
            if (filesData) {
                try {
                    const files = JSON.parse(filesData.textContent);
                    images = files.map(f => parseFilename(f));
                    renderGallery();
                    document.getElementById('stats').textContent = `${images.length} image${images.length !== 1 ? 's' : ''}`;
                    document.getElementById('status').textContent = `Loaded: ${new Date().toLocaleTimeString()}`;
                } catch (e) {
                    document.getElementById('status').textContent = 'Error loading files';
                }
            }
        }

        function renderGallery() {
            const gallery = document.getElementById('gallery');
            const sortBy = document.getElementById('sort-select').value;

            if (images.length === 0) {
                gallery.innerHTML = `<div class="no-images"><h2>No samples yet</h2><p>Start training with sample generation enabled.</p></div>`;
                return;
            }

            // Optional opt-in filter — default shows everything (output folders are often
            // shared across runs, and comparing runs side by side is a feature).
            let sorted = [...images];
            if (document.getElementById('run-select').value === 'current') {
                const run = currentRunName();
                if (run) sorted = sorted.filter(im => im.loraName === run);
            }
            switch (sortBy) {
                case 'newest': sorted.sort((a, b) => b.timestamp.localeCompare(a.timestamp)); break;
                case 'oldest': sorted.sort((a, b) => a.timestamp.localeCompare(b.timestamp)); break;
                case 'epoch-desc': sorted.sort((a, b) => b.epoch - a.epoch || b.timestamp.localeCompare(a.timestamp)); break;
                case 'epoch-asc': sorted.sort((a, b) => a.epoch - b.epoch || a.timestamp.localeCompare(b.timestamp)); break;
            }

            gallery.innerHTML = sorted.map(img => `
                <div class="gallery-item" onclick="openLightbox('${img.filename}')">
                    <div class="image-container">
                        <img src="${img.filename}" alt="${img.filename}" loading="lazy">
                        <span class="badge epoch-badge">Epoch ${img.epoch}</span>
                        ${likBadge(img)}
                        ${img.video ? `<span class="badge clip-badge">🎬 video</span>` : ''}
                        ${!img.video && img.clip ? `<span class="badge clip-badge">🎞 scrub</span>` : ''}
                        ${!img.video && img.sound ? `<span class="badge sound-badge">🔊 sound</span>` : ''}
                    </div>
                    <div class="image-info">
                        <div class="lora-name">${img.loraName}</div>
                        <div class="meta-row">
                            <span class="meta-item seed">Seed: ${img.seed || '—'}</span>
                            <span class="meta-item time">${img.time}</span>
                        </div>
                        ${img.lora ? `<a class="lora-download" href="${img.lora}" download onclick="event.stopPropagation()">⬇ Download LoRA (epoch ${img.epoch})</a>` : ''}
                    </div>
                </div>`).join('');
        }

        // ---------- Likeness scoring (CPU ArcFace vs 3 baselines, scored by Fizgig) ----------

        async function loadLikeness() {
            try {
                const r = await fetch('likeness.json?t=' + Date.now());
                if (r.ok) likeness = await r.json();
            } catch (e) {}
            const active = likeness && likeness.baselines && likeness.baselines.length === 3;
            document.getElementById('lik-status').textContent = active ? ('🎯 ' + (likeness.status || '')) : '';
        }

        function likBadge(img) {
            if (!likeness || !likeness.baselines || likeness.baselines.length !== 3) return '';
            const s = likeness.scores ? likeness.scores[img.filename] : undefined;
            if (s === undefined) return `<span class="lik-badge lik-na">…</span>`;
            if (s === null) return `<span class="lik-badge lik-na">no face</span>`;
            const cls = s >= 0.45 ? 'lik-good' : (s >= 0.30 ? 'lik-mid' : 'lik-bad');
            return `<span class="lik-badge ${cls}">${Math.round(s * 100)}%</span>`;
        }

        function renderLikenessChart() {
            const panel = document.getElementById('likeness-panel');
            const active = likeness && likeness.baselines && likeness.baselines.length === 3 && likeness.scores;
            if (!active || images.length === 0) { panel.style.display = 'none'; return; }
            // Current run = the LoRA name of the newest sample (old runs' samples share the
            // folder but must not pollute the trend).
            let newest = null;
            images.forEach(im => { if (!newest || im.timestamp > newest.timestamp) newest = im; });
            const byEpoch = {};
            images.forEach(im => {
                if (im.loraName !== newest.loraName) return;
                const s = likeness.scores[im.filename];
                if (typeof s === 'number') (byEpoch[im.epoch] = byEpoch[im.epoch] || []).push(s);
            });
            const epochs = Object.keys(byEpoch).map(Number).sort((a, b) => a - b);
            if (!epochs.length) { panel.style.display = 'none'; return; }
            const avgs = epochs.map(e => byEpoch[e].reduce((a, b) => a + b, 0) / byEpoch[e].length);
            let bestI = 0;
            avgs.forEach((v, i) => { if (v > avgs[bestI]) bestI = i; });
            document.getElementById('lik-run').textContent =
                `${newest.loraName} — best so far: epoch ${epochs[bestI]} (${Math.round(avgs[bestI] * 100)}%)`;
            panel.style.display = 'block';
            const cv = document.getElementById('lik-chart');
            // Size the backing store to the rendered width so the chart spans the page like the
            // thumbnail grid does (a fixed-width canvas stretched by CSS goes blurry instead).
            const cssW = cv.clientWidth || 940;
            if (cv.width !== cssW) cv.width = cssW;
            const ctx = cv.getContext('2d');
            const W = cv.width, H = cv.height, padL = 42, padR = 12, padT = 12, padB = 22;
            ctx.clearRect(0, 0, W, H);
            const ymax = Math.max(0.7, Math.max(...avgs) + 0.05);
            const x = i => epochs.length === 1 ? (padL + (W - padL - padR) / 2)
                                               : padL + (W - padL - padR) * i / (epochs.length - 1);
            const y = v => padT + (H - padT - padB) * (1 - v / ymax);
            ctx.font = '11px Segoe UI';
            ctx.lineWidth = 1;
            [0.30, 0.45].forEach(g => {   // the badge colour bands, for orientation
                ctx.strokeStyle = '#34495E'; ctx.fillStyle = '#7F8C8D';
                ctx.beginPath(); ctx.moveTo(padL, y(g)); ctx.lineTo(W - padR, y(g)); ctx.stroke();
                ctx.fillText(Math.round(g * 100) + '%', 8, y(g) + 4);
            });
            ctx.strokeStyle = '#3498DB'; ctx.lineWidth = 2; ctx.beginPath();
            avgs.forEach((v, i) => { i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v)); });
            ctx.stroke();
            const labelEvery = Math.max(1, Math.ceil(epochs.length / 40));
            avgs.forEach((v, i) => {
                ctx.fillStyle = i === bestI ? '#27AE60' : '#3498DB';
                ctx.beginPath(); ctx.arc(x(i), y(v), i === bestI ? 5 : 3.5, 0, Math.PI * 2); ctx.fill();
                if (i % labelEvery === 0 || i === bestI) {
                    ctx.fillStyle = '#95A5A6';
                    ctx.fillText(epochs[i], x(i) - 6, H - 6);
                }
            });
        }

        async function openBaselinePicker() {
            const bp = document.getElementById('basepicker');
            bp.classList.add('active');
            document.body.style.overflow = 'hidden';
            const grid = document.getElementById('bp-grid');
            grid.innerHTML = '<div style="color:#95A5A6">Loading dataset…</div>';
            let names = [];
            try {
                const r = await fetch('dataset.json?t=' + Date.now());
                if (r.ok) {
                    const d = await r.json();
                    names = Array.isArray(d) ? d : (d.images || []);
                    document.getElementById('bp-folder').textContent = (d && d.folder) || 'unknown';
                }
            } catch (e) {}
            if (!names.length) {
                grid.innerHTML = '<div style="color:#E74C3C">No dataset images found — set the training ' +
                                 'image folder on the Start tab, then reopen the gallery from Fizgig.</div>';
                return;
            }
            bpSelected = (likeness && likeness.baselines && likeness.baselines.length === 3)
                         ? [...likeness.baselines] : [];
            grid.innerHTML = names.map(n => `
                <div class="bp-item" data-name="${n}" onclick="toggleBaseline(this)">
                    <img src="dataset/${encodeURIComponent(n)}" loading="lazy">
                </div>`).join('');
            refreshBpMarks();
        }

        function toggleBaseline(el) {
            const n = el.dataset.name;
            const i = bpSelected.indexOf(n);
            if (i >= 0) bpSelected.splice(i, 1);
            else { if (bpSelected.length >= 3) bpSelected.shift(); bpSelected.push(n); }
            refreshBpMarks();
        }

        function refreshBpMarks() {
            document.querySelectorAll('.bp-item').forEach(el => {
                const i = bpSelected.indexOf(el.dataset.name);
                el.classList.toggle('selected', i >= 0);
                let num = el.querySelector('.bp-num');
                if (i >= 0) {
                    if (!num) { num = document.createElement('div'); num.className = 'bp-num'; el.appendChild(num); }
                    num.textContent = i + 1;
                } else if (num) num.remove();
            });
            document.getElementById('bp-start').disabled = bpSelected.length !== 3;
            document.getElementById('bp-status').textContent = `${bpSelected.length}/3 selected`;
        }

        async function submitBaselines() {
            await postBaselines(bpSelected, true);
        }

        async function clearBaselines() {
            await postBaselines([], true);
            likeness = null;
            renderGallery();
            renderLikenessChart();
            document.getElementById('lik-status').textContent = '';
        }

        async function postBaselines(names, closeOnOk) {
            const st = document.getElementById('bp-status');
            st.textContent = 'Sending…';
            try {
                const r = await fetch('set_baselines', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ baselines: names })
                });
                const res = await r.json();
                if (res.ok) { if (closeOnOk) closeBaselinePicker(); loadImages(); }
                else st.textContent = '⚠ ' + res.msg;
            } catch (e) {
                st.textContent = '⚠ Fizgig not reachable — reopen the gallery from the app.';
            }
        }

        function closeBaselinePicker() {
            document.getElementById('basepicker').classList.remove('active');
            document.body.style.overflow = '';
        }

        // ---------- Training Run Visualiser (epoch carousel, Royale-style) ----------

        let rvFrames = [];      // [{epoch, filename, sim}] ascending epochs, one sample slot
        let rvIdx = 0;
        let rvTimer = null;
        let rvDir = 1;          // ping-pong direction while playing

        function rvCurrentRunImages() {
            const run = currentRunName();
            return run ? images.filter(im => im.loraName === run) : [];
        }

        function openRunViz() {
            const runImgs = rvCurrentRunImages();
            if (!runImgs.length) { alert('No samples yet — start a run with previews enabled.'); return; }
            const slots = [...new Set(runImgs.map(im => im.idx))].sort((a, b) => a - b);
            const sel = document.getElementById('rv-slot');
            const keep = sel.value;
            sel.innerHTML = slots.map(s => `<option value="${s}">${s}</option>`).join('');
            if (slots.map(String).includes(keep)) sel.value = keep;
            document.getElementById('runviz').classList.add('active');
            document.body.style.overflow = 'hidden';
            rvBuild();
        }

        function rvBuild() {
            const slot = parseInt(document.getElementById('rv-slot').value || '0');
            const byEpoch = {};
            rvCurrentRunImages().forEach(im => {
                if (im.idx !== slot) return;
                // Same epoch rendered twice (e.g. after a resume) -> keep the newest.
                if (!byEpoch[im.epoch] || im.timestamp > byEpoch[im.epoch].timestamp) byEpoch[im.epoch] = im;
            });
            rvFrames = Object.keys(byEpoch).map(Number).sort((a, b) => a - b).map(e => ({
                epoch: e,
                filename: byEpoch[e].filename,
                sim: (likeness && likeness.scores) ? likeness.scores[byEpoch[e].filename] : undefined,
            }));
            const slider = document.getElementById('rv-slider');
            slider.max = Math.max(0, rvFrames.length - 1);
            rvShow(rvFrames.length - 1);   // land on the newest epoch
        }

        function rvShow(i) {
            if (!rvFrames.length) return;
            rvIdx = Math.max(0, Math.min(i, rvFrames.length - 1));
            const fr = rvFrames[rvIdx];
            document.getElementById('rv-img').src = fr.filename;
            document.getElementById('rv-slider').value = rvIdx;
            let label = `Epoch ${fr.epoch}`;
            if (typeof fr.sim === 'number') label += `  ·  likeness ${Math.round(fr.sim * 100)}%`;
            document.getElementById('rv-epoch').textContent = label;
        }

        function rvTogglePlay() {
            if (rvTimer) { rvStop(); return; }
            if (rvFrames.length < 2) return;
            rvDir = 1;
            document.getElementById('rv-play').textContent = '⏸ Pause';
            const tick = () => {
                let next = rvIdx + rvDir;
                if (document.getElementById('rv-pingpong').checked) {
                    if (next >= rvFrames.length || next < 0) { rvDir = -rvDir; next = rvIdx + rvDir; }
                } else if (next >= rvFrames.length) next = 0;
                rvShow(next);
                rvTimer = setTimeout(tick, parseInt(document.getElementById('rv-speed').value));
            };
            rvTimer = setTimeout(tick, parseInt(document.getElementById('rv-speed').value));
        }

        function rvStop() {
            if (rvTimer) clearTimeout(rvTimer);
            rvTimer = null;
            document.getElementById('rv-play').textContent = '▶ Play';
        }

        function closeRunViz() {
            rvStop();
            document.getElementById('runviz').classList.remove('active');
            document.body.style.overflow = '';
        }

        function rvDrawFrame(ctx, imgEl, fr, W, H) {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, W, H);
            const sc = Math.min(W / imgEl.naturalWidth, H / imgEl.naturalHeight);
            const dw = imgEl.naturalWidth * sc, dh = imgEl.naturalHeight * sc;
            ctx.drawImage(imgEl, (W - dw) / 2, (H - dh) / 2, dw, dh);
            if (document.getElementById('rv-ticker').checked) {
                const label = 'Epoch ' + fr.epoch;
                ctx.font = 'bold 26px Segoe UI';
                const tw = ctx.measureText(label).width;
                ctx.fillStyle = 'rgba(0,0,0,0.65)';
                ctx.fillRect(16, H - 56, tw + 24, 40);
                ctx.fillStyle = '#fff';
                ctx.fillText(label, 28, H - 28);
            }
            if (document.getElementById('rv-tag').checked) {
                // Scale with the frame — a fixed small px size vanished next to the epoch ticker
                // (and shrank to nothing on full-resolution saved frames).
                const fs = Math.max(28, Math.round(H * 0.055));
                ctx.font = `bold ${fs}px Segoe UI`;
                const tag = 'Fizgig';
                const tw = ctx.measureText(tag).width;
                ctx.fillStyle = 'rgba(255,255,255,0.7)';
                ctx.fillText(tag, W - tw - 20, H - 20);
            }
        }

        function rvPreload() {
            return Promise.all(rvFrames.map(fr => new Promise(res => {
                const im = new Image();
                im.onload = () => res({ fr, im });
                im.onerror = () => res(null);
                im.src = fr.filename;
            }))).then(list => list.filter(Boolean));
        }

        async function rvExport() {
            if (rvFrames.length < 2) { alert('Need at least 2 epochs to export a clip.'); return; }
            const st = document.getElementById('rv-status');
            st.textContent = 'Preparing frames…';
            rvStop();
            const loaded = await rvPreload();
            if (loaded.length < 2) { st.textContent = 'Could not load frames.'; return; }
            let seq = [...loaded];
            if (document.getElementById('rv-pingpong').checked) {
                seq = seq.concat([...loaded].reverse().slice(1, -1));
            }
            const first = loaded[0].im;
            const W = Math.min(1024, first.naturalWidth), H = Math.round(W * first.naturalHeight / first.naturalWidth);
            const cv = document.createElement('canvas');
            cv.width = W; cv.height = H;
            const ctx = cv.getContext('2d');
            const stream = cv.captureStream(30);
            let mime = 'video/webm;codecs=vp9';
            if (!MediaRecorder.isTypeSupported(mime)) mime = 'video/webm';
            const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8_000_000 });
            const chunks = [];
            rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
            const doneRec = new Promise(res => { rec.onstop = res; });
            const stepMs = parseInt(document.getElementById('rv-speed').value);
            rec.start();
            for (let i = 0; i < seq.length; i++) {
                rvDrawFrame(ctx, seq[i].im, seq[i].fr, W, H);
                st.textContent = `Recording… ${i + 1}/${seq.length}`;
                await new Promise(r => setTimeout(r, stepMs));
            }
            await new Promise(r => setTimeout(r, 200));   // tail so the last frame lands
            rec.stop();
            await doneRec;
            const blob = new Blob(chunks, { type: 'video/webm' });
            const a = document.createElement('a');
            const run = currentRunName() || 'run';
            a.href = URL.createObjectURL(blob);
            a.download = `${run}_training_run.webm`;
            a.click();
            setTimeout(() => URL.revokeObjectURL(a.href), 5000);
            st.textContent = `Saved ${a.download} (${seq.length} frames). MP4/GIF export lives in LoRA Royale.`;
        }

        function rvSaveFrame() {
            if (!rvFrames.length) return;
            const fr = rvFrames[rvIdx];
            const im = new Image();
            im.onload = () => {
                const cv = document.createElement('canvas');
                cv.width = im.naturalWidth; cv.height = im.naturalHeight;
                rvDrawFrame(cv.getContext('2d'), im, fr, cv.width, cv.height);
                cv.toBlob(blob => {
                    const a = document.createElement('a');
                    const run = currentRunName() || 'run';
                    a.href = URL.createObjectURL(blob);
                    a.download = `${run}_epoch${fr.epoch}.png`;
                    a.click();
                    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
                }, 'image/png');
            };
            im.src = fr.filename;
        }

        // ---------- Lightbox ----------

        function openLightbox(filename) {
            const idx = images.findIndex(img => img.filename === filename);
            if (idx >= 0) { currentLightboxIndex = idx; showLightbox(images[idx]); }
        }

        let lbClip = null;              // active clip frame list, or null for a plain still

        function lbScrub(i) {
            if (!lbClip) return;
            i = Math.max(0, Math.min(lbClip.length - 1, i));
            document.getElementById('lightbox-img').src = lbClip[i];
            document.getElementById('lb-scrub-label').textContent =
                `frame ${i + 1} / ${lbClip.length} — drag to scrub (clips never autoplay)`;
        }

        function showLightbox(img) {
            const wrap = document.getElementById('lb-scrub-wrap');
            const slider = document.getElementById('lb-scrub');
            const aw = document.getElementById('lb-audio-wrap');
            const au = document.getElementById('lb-audio');
            const vid = document.getElementById('lightbox-vid');
            const imEl = document.getElementById('lightbox-img');
            au.pause();
            vid.pause();
            // A sample with a muxed mp4 plays as a REAL clip — controls, never autoplay —
            // replacing both the scrub slider and the separate audio player.
            if (img.video) {
                vid.src = img.video;
                vid.style.display = 'block';
                imEl.style.display = 'none';
                aw.style.display = 'none';
                au.removeAttribute('src');
                wrap.classList.remove('active');
                lbClip = null;
                document.getElementById('lightbox-name').textContent = img.filename;
                document.getElementById('lightbox-meta').textContent = `${img.loraName} | Epoch ${img.epoch} | Seed: ${img.seed} | ${img.time}`;
                document.getElementById('lightbox').classList.add('active');
                document.body.style.overflow = 'hidden';
                return;
            }
            vid.removeAttribute('src');
            vid.style.display = 'none';
            imEl.style.display = '';
            // The sample's generated sound, when it has one (wav without an mp4 — e.g. the
            // mux failed). A play CONTROL, never autoplay — scrubbing stays silent.
            if (img.sound) { au.src = img.sound; aw.style.display = 'block'; }
            else { au.removeAttribute('src'); aw.style.display = 'none'; }
            lbClip = img.clip || null;
            if (lbClip) {
                // Preload on OPEN, not up front — a 60-epoch gallery would otherwise pull
                // thousands of frames nobody asked for.
                lbClip.forEach(f => { const im = new Image(); im.src = f; });
                slider.max = lbClip.length - 1;
                const mid = Math.floor(lbClip.length / 2);
                slider.value = mid;
                wrap.classList.add('active');
                lbScrub(mid);
            } else {
                wrap.classList.remove('active');
                document.getElementById('lightbox-img').src = img.filename;
            }
            document.getElementById('lightbox-name').textContent = img.filename;
            document.getElementById('lightbox-meta').textContent = `${img.loraName} | Epoch ${img.epoch} | Seed: ${img.seed} | ${img.time}`;
            document.getElementById('lightbox').classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function closeLightbox() {
            document.getElementById('lb-audio').pause();
            document.getElementById('lightbox-vid').pause();
            document.getElementById('lightbox').classList.remove('active');
            document.body.style.overflow = '';
        }

        function navigateLightbox(dir) {
            if (images.length === 0) return;
            currentLightboxIndex = (currentLightboxIndex + dir + images.length) % images.length;
            showLightbox(images[currentLightboxIndex]);
        }

        document.addEventListener('keydown', (e) => {
            // Arrows on the clip scrub slider step the slider natively; without this guard the
            // same keydown bubbles here and ALSO jumps to the next image — two actions per key.
            if (e.target && e.target.id === 'lb-scrub') return;
            if (document.getElementById('runviz').classList.contains('active')) {
                if (e.key === 'Escape') closeRunViz();
                if (e.key === 'ArrowLeft') { rvStop(); rvShow(rvIdx - 1); }
                if (e.key === 'ArrowRight') { rvStop(); rvShow(rvIdx + 1); }
                if (e.key === ' ') { e.preventDefault(); rvTogglePlay(); }
                return;
            }
            if (e.key === 'Escape' && document.getElementById('basepicker').classList.contains('active')) {
                closeBaselinePicker();
                return;
            }
            if (!document.getElementById('lightbox').classList.contains('active')) return;
            if (e.key === 'Escape') closeLightbox();
            if (e.key === 'ArrowLeft') navigateLightbox(-1);
            if (e.key === 'ArrowRight') navigateLightbox(1);
        });

        document.getElementById('lightbox').addEventListener('click', (e) => {
            if (e.target.id === 'lightbox') closeLightbox();
        });

        let chartResizeTimer = null;
        window.addEventListener('resize', () => {
            if (chartResizeTimer) clearTimeout(chartResizeTimer);
            chartResizeTimer = setTimeout(renderLikenessChart, 150);
        });

        setupTimer();
        loadImages();
    </script>
</body>
</html>'''

        with open(gallery_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
