"""Fizgig as a desktop app window around the web UI (`fizgig.py`), through pywebview: Windows'
WebView2, GTK/WebKit or Qt on Linux, WebKit on macOS - no browser, no address bar.

It uses the web UI already running on the saved port when there is one (the desktop app's, say), else starts its own
on this computer and stops it when the window closes. Its own server logs the window in with a one-time key, so a web
password guards the network, not this window. Size and place are remembered (prefs.json "app_window"). Without a
pywebview engine on the machine it opens the web UI in the default browser instead and says how to get the window.

The window can also show a pod (the Pods tab sends it there): on any page but this computer's it adds a bar with
"Switch to this computer", whatever version of Fizgig the pod runs, and names the pod in the title; back here, a bar
offers "Switch to the pod" (✕ hides it). A pod's page can ask the window to come back here, and read this computer's
training folders to send one up (local_bridge: read-only, those folders only). Closed on a pod, it opens on that pod
next time if it's up.
"""
import json
import os
import secrets
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

from fizgig.web.port import free_from, is_fizgig, listening, saved_port

from fizgig.user_data import user_file  # noqa: E402  (Fizgig's own files: user_data/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from fizgig import user_data  # noqa: E402
PREFS_FILE = user_data.PREFS_FILE


def _prefs():
    try:
        with open(PREFS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _sane_geom(geom):
    """Ignore minimized / off-screen coordinates (Windows reports -32000 / DPI when closed from the taskbar)."""
    if not isinstance(geom, dict):
        return {}
    x, y = geom.get("x"), geom.get("y")
    if (isinstance(x, (int, float)) and x < -10000) or (isinstance(y, (int, float)) and y < -10000):
        return {}
    return geom


def _remember(geom):
    """The window's size and place, written into prefs.json beside what is there (the desktop app may have it open)."""
    if not _sane_geom(geom):
        return
    p = _prefs()
    p["app_window"] = geom
    tmp = PREFS_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2)
        os.replace(tmp, PREFS_FILE)
    except OSError:
        pass


def _remember_last(url):
    p = _prefs()
    p["app_last_url"] = url
    tmp = PREFS_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2)
        os.replace(tmp, PREFS_FILE)
    except OSError:
        pass


def _icon(profile):
    """Fizgig's icon for the window: icon.png, made an .ico for Windows (which takes nothing else)."""
    png = os.path.join(BASE_DIR, "src", "fizgig", "web", "static", "icon.png")
    if not os.path.isfile(png):
        return None
    if os.name != "nt":
        return png
    ico = os.path.join(profile, "fizgig.ico")
    try:
        if not os.path.isfile(ico) or os.path.getmtime(ico) < os.path.getmtime(png):
            from PIL import Image
            os.makedirs(profile, exist_ok=True)
            Image.open(png).save(ico, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
        return ico
    except Exception:
        return None


# the bar the window adds to a page (so a pod on older Fizgig gets it too): on a pod, where you are and the way back
# here; back here, the pod you left and the way back to it (with a ✕ to put it away)
_BAR_JS = r"""
(function (text, button, title, call, dismiss) {
  var old = document.getElementById('fizgigAppBar');
  if (old) old.remove();
  // room above the page for the bar (and a build note under it), so neither covers the page's own top line
  var sp = document.getElementById('fizgigAppBarSpace');
  if (!sp) {
    sp = document.createElement('div');
    sp.id = 'fizgigAppBarSpace';
    sp.style.cssText = 'flex:none;height:54px';
    document.body.insertBefore(sp, document.body.firstChild);
  }
  var b = document.createElement('div');
  b.id = 'fizgigAppBar';
  // the top middle: clear of the page's own top-right buttons (a full-screen picture's close)
  b.style.cssText = 'position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:2147483647;display:flex;' +
    'align-items:center;gap:10px;' +
    'max-width:60vw;padding:5px 6px 5px 14px;border-radius:999px;background:var(--app-bar, #5B21B6);color:var(--text-primary, #F0F4F8);' +
    'font:13px Segoe UI,system-ui,sans-serif;box-shadow:0 2px 10px color-mix(in srgb, var(--shadow, #000) 45%, transparent)';
  var t = document.createElement('span');
  t.textContent = text;
  t.style.cssText = 'flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
  var h = document.createElement('button');
  h.textContent = button;
  h.title = title;
  h.style.cssText = 'background:var(--text-primary, #F0F4F8);color:var(--bg-surface, #1A2029);border:0;border-radius:999px;padding:4px 12px;cursor:pointer;' +
    'font:600 13px Segoe UI,system-ui,sans-serif';
  h.onclick = function () { if (window.pywebview && window.pywebview.api) window.pywebview.api[call](); };
  b.appendChild(t); b.appendChild(h);
  if (dismiss) {
    var x = document.createElement('button');
    x.textContent = '✕';
    x.title = 'Hide this';
    x.style.cssText = 'background:none;color:var(--text-primary, #F0F4F8);border:0;cursor:pointer;font:13px Segoe UI,system-ui,sans-serif;padding:2px 6px';
    x.onclick = function () { b.remove(); sp.remove(); if (window.pywebview && window.pywebview.api) window.pywebview.api.forget_pod(); };
    b.appendChild(x);
  }
  document.body.appendChild(b);
})(__ARGS__);
"""


def _bar(text, button, title, call, dismiss=False):
    return _BAR_JS.replace("__ARGS__", ", ".join(json.dumps(v) for v in (text, button, title, call, dismiss)))


def _git(*args):
    try:
        r = subprocess.run(["git", "-C", BASE_DIR, *args], capture_output=True, text=True, timeout=5,
                           creationflags=0x08000000 if os.name == "nt" else 0)
        return r.returncode, r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 1, ""


_BUILD = []


def _build():
    """This computer's build: the commit it runs ("" when it isn't a git checkout - then nothing is compared)."""
    if not _BUILD:
        code, out = _git("rev-parse", "HEAD")
        _BUILD.append(out if code == 0 else "")
    return _BUILD[0]


def _update_how():
    if os.name == "nt":
        return "close Fizgig and run update_fizgig.bat (update_fizgig_rocm.bat on AMD), then open it again"
    return "close Fizgig, run venv/bin/python update_fizgig.py in the Fizgig folder, then open it again"


def build_verdict(pod_build, pod=True, name=""):
    """The window and a pod (or your other computer) on different builds. An OLDER one opens as it is, with a note:
    its page only asks the window for what it already knew (the window keeps those names), and updating it would mean
    stopping a run that may be hours in. A NEWER one, or one this computer has never seen, may ask for what this window
    doesn't have yet: its page is covered with what to do, and Open anyway."""
    mine = _build()
    if not mine:
        return {"ok": True}
    theirs = str(pod_build or "").strip()
    if theirs == mine:
        return {"ok": True}
    other = "the pod" if pod else (name or "your other computer")      # your computer by its name
    title = ("This computer and the pod run different builds of Fizgig" if pod else
             "Your two computers run different builds of Fizgig")
    mine_s, theirs_s = mine[:7], theirs[:7]
    older = bool(theirs) and _git("merge-base", "--is-ancestor", theirs, mine)[0] == 0
    if not older:
        # the pod's build is newer than this computer's, one this computer has never seen, or off another line
        said = f"runs build {theirs_s}" if theirs else "doesn't say which build it runs"
        msg = (f"{other[:1].upper() + other[1:]} {said}, this computer build {mine_s}. Update Fizgig on this computer - "
               f"{_update_how()} - and the two match. Opened anyway, most of it works; something newer than this "
               f"computer's Fizgig may not.")
        return {"ok": False, "title": title, "message": msg}
    # older there: it works as it is, and a run on it carries on
    note = (f"{other[:1].upper() + other[1:]} runs an older Fizgig ({theirs_s or 'an older build'}). It works as it is; to update it, " +
            ("stop and start the pod when its run is done (a pod takes the newest Fizgig every time it starts)." if pod
             else "run its updater when it's free."))
    return {"ok": True, "older": True, "note": note}


# a page in the window that runs a different build from this computer: covered, with what to do and the way back
_BLOCK_JS = r"""
(function (title, message) {
  if (document.getElementById('fizgigBuildBlock')) return;
  var o = document.createElement('div');
  o.id = 'fizgigBuildBlock';
  o.style.cssText = 'position:fixed;inset:0;z-index:2147483646;background:color-mix(in srgb, var(--overlay, #0A0E14) 92%, transparent);display:flex;' +
    'align-items:center;justify-content:center;padding:16px;font:14px Segoe UI,system-ui,sans-serif;color:var(--text-primary, #F0F4F8)';
  var c = document.createElement('div');
  c.style.cssText = 'max-width:560px;background:var(--bg-surface, #1A2029);border:2px solid var(--warning, #F59E0B);border-radius:12px;padding:22px 24px';
  var h = document.createElement('div');
  h.textContent = title;
  h.style.cssText = 'font-size:18px;font-weight:700;color:var(--warning, #F59E0B);margin-bottom:10px';
  var p = document.createElement('div');
  p.textContent = message;
  p.style.cssText = 'line-height:1.5;margin-bottom:18px';
  var b = document.createElement('button');
  b.textContent = 'Switch to this computer';
  b.style.cssText = 'background:var(--text-primary, #F0F4F8);color:var(--bg-surface, #1A2029);border:0;border-radius:999px;padding:6px 16px;cursor:pointer;' +
    'font:600 14px Segoe UI,system-ui,sans-serif';
  b.onclick = function () { if (window.pywebview && window.pywebview.api) window.pywebview.api.home(); };
  var a = document.createElement('button');
  a.textContent = 'Open anyway';
  a.title = "Use it as it is: most of it works, something newer than this computer's Fizgig may not";
  a.style.cssText = 'margin-left:10px;background:none;color:var(--text-primary, #F0F4F8);border:1px solid var(--text-muted, #6B7280);border-radius:999px;' +
    'padding:6px 16px;cursor:pointer;font:600 14px Segoe UI,system-ui,sans-serif';
  a.onclick = function () { o.remove(); };
  c.appendChild(h); c.appendChild(p); c.appendChild(b); c.appendChild(a); o.appendChild(c);
  document.body.appendChild(o);
})(__ARGS__);
"""


# an older build there: a note under the bar, not a wall (✕ hides it)
_NOTE_JS = r"""
(function (text) {
  if (document.getElementById('fizgigBuildNote')) return;
  var n = document.createElement('div');
  n.id = 'fizgigBuildNote';
  n.style.cssText = 'position:fixed;top:52px;left:50%;transform:translateX(-50%);z-index:2147483646;display:flex;' +
    'align-items:center;gap:8px;max-width:70vw;padding:6px 8px 6px 14px;border-radius:10px;background:var(--app-note-bg, #3A2A08);' +
    'border:1px solid var(--warning, #F59E0B);color:var(--warning-text, #FCD34D);font:12.5px Segoe UI,system-ui,sans-serif;box-shadow:0 2px 10px color-mix(in srgb, var(--shadow, #000) 45%, transparent)';
  var t = document.createElement('span');
  t.textContent = text;
  var x = document.createElement('button');
  x.textContent = '✕';
  x.title = 'Hide this';
  x.style.cssText = 'background:none;color:var(--warning-text, #FCD34D);border:0;cursor:pointer;font:13px Segoe UI,system-ui,sans-serif';
  var sp = document.getElementById('fizgigAppBarSpace');
  x.onclick = function () { n.remove(); if (sp) sp.style.height = '54px'; };
  n.appendChild(t); n.appendChild(x);
  document.body.appendChild(n);
  if (sp) sp.style.height = (n.offsetHeight + 66) + 'px';     // the bar and this note both clear of the page
})(__ARGS__);
"""

# on a pod's page: its build (its About, once logged in), checked against this computer's
# a pod of your account, opened in the window: its login filled in with the password from its RunPod settings, once.
# Whatever happens the login box stays usable - if that password doesn't work, it says to type it.
_LOGIN_JS = r"""
(function (pw) {
  var tries = 0;
  function go() {
    var m = document.getElementById('loginModal'), i = document.getElementById('loginPassword'),
        b = document.getElementById('loginButton');
    if (!m || !i || !b) { if (++tries < 40) setTimeout(go, 250); return; }   // the page is still coming
    if (!m.classList.contains('active')) { if (++tries < 40) setTimeout(go, 250); return; }   // logged in already
    i.value = pw;
    var r = document.getElementById('loginRemember'); if (r) r.checked = true;
    b.click();
    setTimeout(function () {
      if (m.classList.contains('active')) {
        i.value = '';
        var e = document.getElementById('loginError');
        if (e) e.textContent = "Fizgig couldn't log you in to this pod by itself (its password isn't the one in its " +
          "RunPod settings). Type its password here.";
      }
    }, 4000);
  }
  go();
})(__PW__);
"""


_CHECK_JS = r"""
(function () {
  if (window.__fizgigBuildCheck) return;
  window.__fizgigBuildCheck = true;
  var tries = 0;
  function ask() {
    var t = document.querySelector('meta[name="fizgig-token"]');            // its API answers its own page only
    fetch('/api/about', {credentials: 'same-origin', headers: {'X-Fizgig-Token': t ? t.content : ''}}).then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    }).then(function (d) {
      window.pywebview.api.build_check(d.commit || '', d.pod_version || null);
    }).catch(function () { if (++tries < 400) setTimeout(ask, 3000); });   // the login page first, then it answers
  }
  ask();
})();
"""


# a pod on another build that can switch (pod_version.py): asked to run this computer's, with a cover while it does
# (its web UI starts again: a few seconds, a minute or two when the version needs other packages), then the page
# again. A pod that can't now (busy, or the version isn't on GitHub) gets the usual build note or cover, with why.
_MATCH_JS = r"""
(function (mine, by) {
  if (document.getElementById('fizgigMatch')) return;
  var tok = function () { var t = document.querySelector('meta[name="fizgig-token"]'); return t ? t.content : ''; };
  var o = document.createElement('div');
  o.id = 'fizgigMatch';
  o.style.cssText = 'position:fixed;inset:0;z-index:2147483646;background:color-mix(in srgb, var(--overlay, #0A0E14) 92%, transparent);display:flex;' +
    'align-items:center;justify-content:center;padding:16px;font:14px Segoe UI,system-ui,sans-serif;color:var(--text-primary, #F0F4F8)';
  var c = document.createElement('div');
  c.style.cssText = 'max-width:520px;background:var(--bg-surface, #1A2029);border:1px solid var(--border, #2A3341);border-radius:12px;padding:22px 24px;text-align:center';
  var h = document.createElement('div');
  h.textContent = "Putting this pod on this computer's Fizgig";
  h.style.cssText = 'font-size:17px;font-weight:700;margin-bottom:8px';
  var p = document.createElement('div');
  p.textContent = 'So the two work together (moving a run, helpers, this window). It takes a few seconds, a minute or two when the version needs other packages.';
  p.style.cssText = 'line-height:1.5;color:var(--text-secondary, #93A3B5)';
  c.appendChild(h); c.appendChild(p); o.appendChild(c);
  var shown = false, t0 = Date.now();
  var show = function () { if (!shown) { shown = true; document.body.appendChild(o); } };
  var give = function (why) { o.remove(); window.pywebview.api.build_show(why || ''); };
  var wait = function () {
    if (Date.now() - t0 > 10 * 60 * 1000) return give("It didn't come back on this computer's Fizgig within 10 minutes.");
    fetch('/api/about', {credentials: 'same-origin', headers: {'X-Fizgig-Token': tok()}}).then(function (r) {
      if (r.status === 403) { location.reload(); return null; }   // started again: a new page token, a new page
      if (!r.ok) throw new Error(r.status); return r.json();
    }).then(function (d) {
      if (!d) return;
      if (d.commit === mine) return location.reload();
      var v = d.pod_version || {};
      if (v.state === 'failed') return give(v.error);
      setTimeout(wait, 3000);
    }).catch(function () { setTimeout(wait, 3000); });     // its web UI is starting again
  };
  fetch('/api/pod/match', {method: 'POST', credentials: 'same-origin',
    headers: {'Content-Type': 'application/json', 'X-Fizgig-Token': tok()},
    body: JSON.stringify({commit: mine, by: by})}).then(function (r) { return r.json(); }).then(function (d) {
    if (d.same) return;
    if (d.switching) { show(); setTimeout(wait, 4000); return; }
    give(d.busy || d.error || '');
  }).catch(function () { give(''); });
})(__ARGS__);
"""


def _where(url):
    """'pod <id>' for a pod's address, else its host:port."""
    from urllib.parse import urlparse
    u = urlparse(url or "")
    host = u.hostname or ""
    if host.endswith(".proxy.runpod.net"):
        return "pod " + host.split("-")[0]
    return u.netloc or url


def _pod_name(url):
    """The name the Pods tab saved it under, else where it is. Another computer of yours: its name alone."""
    for p in _prefs().get("app_pods") or []:
        if isinstance(p, dict) and p.get("url") == url and p.get("name"):
            return p["name"] if p.get("lan_id") else f"{p['name']} ({_where(url)})"
    return _where(url)


def _is_pod(url):
    from urllib.parse import urlparse
    return (urlparse(url or "").hostname or "").endswith(".proxy.runpod.net")


def _is_local(url, port):
    from urllib.parse import urlparse
    u = urlparse(url or "")
    return u.hostname in ("127.0.0.1", "localhost") and (u.port or 80) == port


def _last_pod(port):
    """The pod the window was closed on, if Fizgig on it answers now (a stopped pod: this computer instead)."""
    url = str(_prefs().get("app_last_url") or "")
    if not url or _is_local(url, port):
        return ""
    try:
        from fizgig.web.pods import probe
        return url if probe(url, timeout=5) == "ready" else ""
    except Exception:
        return ""


def _start_server(port, key):
    """The web UI as `--web` serves it (this computer only, unless a password is saved: then your other devices too),
    the window's key in its environment."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", FIZGIG_APP_KEY=key,
               FIZGIG_APP_PARENT=str(os.getpid()))            # the server stops if this window's process goes
    kw = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
          if os.name == "nt" else {"start_new_session": True})
    exe = sys.executable
    # the window runs on pythonw.exe (no console window of its own); the server on python.exe with a hidden console,
    # which every run, caption job and tool under it shares - under pythonw each console tool they start (torch.compile's
    # Triton builds, a compiler check) opened its own window, flashing up and away
    con = os.path.join(os.path.dirname(exe), "python.exe")
    if os.name == "nt" and os.path.basename(exe).lower() == "pythonw.exe" and os.path.isfile(con):
        exe = con
    return subprocess.Popen([exe, os.path.join(BASE_DIR, "fizgig.py"), "--web", "--port",
                             str(port)], cwd=BASE_DIR, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, **kw)


def _stop(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _unhook_requests():
    """pywebview's WebView2 window passes every request the page makes through Python on the window's UI thread
    (for its request_sent / response_received events, which Fizgig doesn't use). A clip streams as many byte-range
    requests, so playing one froze the window - the pointer included - while it loaded. The hook is taken off as
    soon as WebView2 is ready."""
    if os.name != "nt":
        return
    try:
        from webview.platforms import edgechromium as ec
    except Exception:
        return
    ready = ec.EdgeChrome.on_webview_ready
    if getattr(ready, "_fizgig", False):
        return

    def on_ready(self, sender, args):
        ready(self, sender, args)
        try:
            core = sender.CoreWebView2
            core.WebResourceRequested -= self.on_web_resource_request
            core.WebResourceResponseReceived -= self.on_web_resource_response
            core.RemoveWebResourceRequestedFilter("*", ec.CoreWebView2WebResourceContext.All)
        except Exception:
            pass
    on_ready._fizgig = True
    ec.EdgeChrome.on_webview_ready = on_ready


_MUTEX = []


def _one_window() -> bool:
    """Windows: one Fizgig window per Fizgig folder. Launched again while it's open, that window comes forward and
    this launch ends (False). One still closing gets 15 s to go; one that never appeared (its WebView2 stuck, say,
    on a profile the window before it hadn't let go of yet) is ended, so it can't keep a server - and an old
    password - running unseen. True: carry on and open."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        import hashlib
        from ctypes import wintypes
        k, u = ctypes.windll.kernel32, ctypes.windll.user32
        k.CreateMutexW.restype = wintypes.HANDLE
        h = k.CreateMutexW(None, False, "Local\\FizgigAppWindow-" + hashlib.sha1(BASE_DIR.lower().encode()).hexdigest()[:12])
        if not h:
            return True
        _MUTEX.append(h)
        pid_file = user_file("app_window.pid")

        def mine():
            try:
                with open(pid_file, "w") as f:
                    f.write(str(os.getpid()))
            except OSError:
                pass
            return True
        if k.WaitForSingleObject(h, 0) in (0, 0x80):            # nobody else (0x80: one that crashed)
            return mine()
        try:
            other = int(open(pid_file).read().strip() or 0)
        except (OSError, ValueError):
            other = 0
        found = []
        proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def each(hwnd, _):
            pid = wintypes.DWORD()
            u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            n = u.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(hwnd, buf, n + 1)
            if u.IsWindowVisible(hwnd) and pid.value == other and (buf.value == "Fizgig" or buf.value.startswith("Fizgig — ")):
                found.append(hwnd)
            return True
        u.EnumWindows(proc(each), 0)
        if found:                                              # it's open: that one, not a second
            if u.IsIconic(found[0]):
                u.ShowWindow(found[0], 9)
            u.SetForegroundWindow(found[0])
            return False
        if k.WaitForSingleObject(h, 15000) in (0, 0x80):        # it was closing
            return mine()
        if other:                                              # no window after all that: stuck, so ended
            subprocess.run(["taskkill", "/PID", str(other), "/T", "/F"], capture_output=True,
                           creationflags=0x08000000)
            k.WaitForSingleObject(h, 5000)
        return mine()
    except Exception:
        return True


WEBVIEW2_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"


def _log(msg):
    """The app window's problems, to user_data/app_window.log: the Windows launcher runs without a console, so a
    window that fails would otherwise say nothing anywhere."""
    try:
        with open(user_file("app_window.log"), "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + str(msg).rstrip() + "\n")
    except OSError:
        pass
    try:
        print(msg, flush=True)
    except (OSError, ValueError):     # the terminal that started it has closed (launch_fizgig.sh runs it in the background)
        pass


def _log_pywebview():
    """pywebview's own errors (a WebView2 that fails to start says why there) into the same log."""
    import logging
    try:
        h = logging.FileHandler(user_file("app_window.log"), encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s pywebview %(levelname)s: %(message)s"))
        logging.getLogger("pywebview").addHandler(h)
    except OSError:
        pass


def _windows_engine_problem():
    """Why the window can't show the web UI on this Windows machine, or "". pywebview falls back to the Internet
    Explorer engine when it finds no WebView2 runtime (or .NET under 4.6.2), and the web UI can't run there: the
    window stayed white (#189)."""
    if os.name != "nt":
        return ""
    try:
        from webview.platforms import winforms
    except Exception as e:
        return f"the window's Windows layer did not load ({type(e).__name__}: {e})"
    if not getattr(winforms, "is_chromium", True):
        return ("Microsoft Edge WebView2 Runtime isn't installed (or .NET Framework is older than 4.6.2), so the "
                f"window can only use Internet Explorer's engine, which can't show Fizgig. Install WebView2 from "
                f"{WEBVIEW2_URL} (the Evergreen Bootstrapper) to get the window back")
    return ""


def main():
    if not _one_window():
        return
    user_data.migrate()                       # Fizgig 7's loose files into user_data, before the prefs are read
    port = saved_port(_prefs())
    own, key = None, ""
    if not is_fizgig(port):
        if listening(port):                      # another program has it: the next free one
            port = free_from(port + 1) or port
        key = secrets.token_urlsafe(24)
        own = _start_server(port, key)
        for _ in range(240):                     # the server's first start imports a lot: up to a minute
            if is_fizgig(port) or own.poll() is not None:
                break
            time.sleep(0.25)
        if not is_fizgig(port):
            _stop(own)
            sys.exit("Fizgig's web UI did not start. Run `python fizgig.py --web` to see why.")
    url = f"http://127.0.0.1:{port}/" + (f"?app_key={key}" if key else "")
    home = f"http://127.0.0.1:{port}/"
    fallback = [""]                       # why the window gave way to the browser, when it did
    try:
        import webview
        _log_pywebview()
        if os.name == "nt":            # the taskbar shows Fizgig's own button and icon, not pythonw's (as the old app)
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("fizgig.lora.studio")
            except Exception:
                pass
        why = _windows_engine_problem()
        if why:
            raise RuntimeError(why)
        _unhook_requests()
        # clips decode on the CPU in this window: the first clip played froze the window (pointer included) while
        # WebView2 brought up the GPU's video decoder, on more than one machine; Fizgig's previews are small clips
        # the CPU decodes with ease. (pywebview's own flag is kept: this variable stands in for its arguments.)
        os.environ.setdefault("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
                              "--disable-features=ElasticOverscroll --disable-accelerated-video-decode")

        class Api:
            """What a page in the window - a pod's page included - can ask of it: to come back to this computer, and
            to read this computer's training folders (read-only, those folders only: local_bridge) so a pod's page
            can send one up with its own upload."""
            def ready(self):
                """The page's script ran (app.js calls this as the bridge arrives): the window shows the web UI."""
                page_ready.set()

            def home(self):
                # after this call has returned to the page: pywebview hands the page a reply, and the page must
                # still be there for it
                threading.Timer(0.05, lambda: window.load_url(home)).start()

            def to_pod(self):
                if left[0]:
                    threading.Timer(0.05, lambda: window.load_url(left[0])).start()

            def forget_pod(self):
                left[0] = ""

            def build(self):
                return _build()

            def build_check(self, pod_build, pod_version=None):
                try:
                    url = window.get_current_url() or ""
                except Exception:
                    url = ""
                pv = pod_version if isinstance(pod_version, dict) else {}
                pod = ".proxy.runpod.net" in url or not url or bool(pv)
                v = build_verdict(pod_build, pod=pod, name="" if pod else _pod_name(url.split("#")[0]))
                shown[0] = v
                if (not v["ok"] or v.get("older")) and pv.get("allowed") and _build():
                    import socket                 # a pod that can switch: to this computer's build
                    window.evaluate_js(_MATCH_JS.replace("__ARGS__", json.dumps(_build()) + ", " +
                                                         json.dumps(socket.gethostname())))
                    return {**v, "matching": True}
                return self.build_show("")

            def build_show(self, why):
                """The build note or cover for the page as it is (why: what stopped the pod switching)."""
                v = shown[0] or {"ok": True}
                why = str(why or "").strip()
                if not v["ok"]:
                    window.evaluate_js(_BLOCK_JS.replace("__ARGS__", json.dumps(v["title"]) + ", " +
                                                         json.dumps(v["message"] + (f" (It can't switch to this "
                                                                    f"computer's: {why})" if why else ""))))
                elif v.get("note"):
                    window.evaluate_js(_NOTE_JS.replace("__ARGS__", json.dumps(v["note"] + (
                        f" It can't switch to this computer's now: {why}" if why else ""))))
                return v

            def local_datasets(self):
                from fizgig.web import local_bridge
                return local_bridge.local_datasets()

            def local_files(self, folder, subfolders=False):
                from fizgig.web import local_bridge
                return local_bridge.local_files(folder, bool(subfolders))

            def read_local(self, folder, name, offset, n):
                from fizgig.web import local_bridge
                return local_bridge.read_local(folder, name, offset, n)

        geom = _sane_geom(_prefs().get("app_window"))
        first = _last_pod(port) or url
        window = webview.create_window(
            "Fizgig", first, width=int(geom.get("width") or 1440), height=int(geom.get("height") or 920),
            x=geom.get("x"), y=geom.get("y"), min_size=(380, 600), text_select=True, confirm_close=False,
            js_api=Api())
        if first != url:
            # its own server's one-time login, for when it comes back here: the key's cookie first, then the pod
            home = url

        left = [""]                       # the pod this window was last on, for Switch to the pod
        shown = [None]                    # the last page's build verdict (build_show tells it after a switch fails)

        zoom_keys_done = [False]

        def _install_zoom_keys():
            """Add Ctrl+= / Ctrl+- / Ctrl+0: WebKitGTK ships no zoom shortcuts of its own."""
            try:
                from webview.platforms import gtk as _g
                inst = _g.BrowserView.instances.get(window.uid)
                if inst is None:
                    return                                 # the window isn't up yet: retry on the next load
            except Exception:
                zoom_keys_done[0] = True                   # no GTK here: not worth retrying on every load
                return
            from gi.repository import Gtk, Gdk, GLib
            # loaded() runs off the GTK main thread, and add_accel_group isn't thread-safe: post to the main loop
            def _install(*_a):
                try:
                    wv, level = inst.webview, [1.0]

                    def _step(delta, *_x):
                        level[0] = max(0.5, min(3.0, round(level[0] + delta, 2)))
                        wv.set_zoom_level(level[0])

                    def _reset(*_x):
                        level[0] = 1.0
                        wv.set_zoom_level(1.0)

                    ag = Gtk.AccelGroup()
                    inst.window.add_accel_group(ag)
                    ag.connect(Gdk.KEY_plus, Gdk.ModifierType.CONTROL_MASK, 0,
                               lambda *_x: _step(0.1))
                    ag.connect(Gdk.KEY_equal, Gdk.ModifierType.CONTROL_MASK, 0,
                               lambda *_x: _step(0.1))
                    ag.connect(Gdk.KEY_minus, Gdk.ModifierType.CONTROL_MASK, 0,
                               lambda *_x: _step(-0.1))
                    ag.connect(Gdk.KEY_0, Gdk.ModifierType.CONTROL_MASK, 0, _reset)
                    for kc, d in ((Gdk.KEY_KP_Add, 0.1), (Gdk.KEY_KP_Subtract, -0.1)):
                        ag.connect(kc, Gdk.ModifierType.CONTROL_MASK, 0, lambda *_x, d=d: _step(d))
                    _log("[app] zoom keys installed: Ctrl+= / Ctrl+- / Ctrl+0, numpad too")
                except Exception as e:
                    _log(f"[app] zoom keys not installed: {e}")
                return False                               # an idle callback: run once
            GLib.idle_add(_install)
            zoom_keys_done[0] = True

        def loaded():
            try:
                if not zoom_keys_done[0]:
                    _install_zoom_keys()
                cur = window.get_current_url() or ""
                if _is_local(cur, port):
                    window.set_title("Fizgig")
                    if left[0]:
                        name = _pod_name(left[0])
                        if _is_pod(left[0]):
                            bar = (f"☁ {name} is still running", "Switch to the pod", "Back to the pod's Fizgig")
                        else:
                            bar = (f"🖥 {name} is still open", f"Switch to {name}", f"Back to Fizgig on {name}")
                        window.evaluate_js(_bar(*bar, "to_pod", dismiss=True))
                    return
                left[0] = cur.split("#")[0]
                name = _pod_name(cur)
                window.set_title(f"Fizgig — {name}")
                window.evaluate_js(_bar(f"☁ You're on {name}" if _is_pod(cur) else f"🖥 You're on {name}",
                                        "Switch to this computer", "Back to Fizgig on this computer ("
                                        + ("the pod" if _is_pod(cur) else name) + " keeps running)", "home"))
                window.evaluate_js(_CHECK_JS)
                if _is_pod(cur) and left[0] not in tried:
                    tried.add(left[0])                    # once a window session: a refused one is left to you
                    threading.Thread(target=_auto_login, args=(left[0],), daemon=True).start()
            except Exception:
                pass

        tried = set()

        def _auto_login(pod):
            try:
                from fizgig.web import pod_stop, pods
                pw = pods.login_password(pod_stop.api_key(_prefs()), pod)
                if pw and (window.get_current_url() or "").startswith(pod.rstrip("/")):
                    window.evaluate_js(_LOGIN_JS.replace("__PW__", json.dumps(pw)))
            except Exception:
                pass                                      # the login box as ever
        window.events.loaded += loaded

        def keep():
            # window.width & get_current_url deadlock here: the GTK backend answers such asks through the main loop,
            # and this handler runs inside it (the delete-event); the native objects allow the same asks, so ask them
            inst = None
            try:
                from webview.platforms import gtk as _g
                inst = _g.BrowserView.instances.get(window.uid)
            except Exception:
                inst = None
            try:
                if inst is not None:
                    w, h = inst.window.get_size()
                    x, y = inst.window.get_position()
                    _remember({"width": w, "height": h, "x": x, "y": y})
                else:
                    _remember({"width": window.width, "height": window.height, "x": window.x, "y": window.y})
            except Exception:
                pass
            try:
                cur = inst.get_current_url() if inst is not None else window.get_current_url()
                cur = cur or ""
                _remember_last("" if _is_local(cur, port) else cur.split("#")[0])
            except Exception:
                pass
        window.events.closing += keep

        def closing():
            """Quitting while a run trains: it would train on with no server to stop it - so say so, and stop it."""
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/api/app/busy", data=b"{}", method="POST",
                                             headers={"X-Fizgig-App-Key": key, "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=3) as r:
                    busy = json.loads(r.read() or b"{}")
            except Exception:
                return True                               # no answer: nothing of ours to stop
            if not busy.get("training"):
                return True
            name = busy.get("lora") or "A run"
            if not window.create_confirmation_dialog(
                    "Quit Fizgig?", f"{name} is training. Quitting stops it: the epochs it has saved stay, and you "
                    f"can pick it up again from its last saved state.\n\nPause it first instead to resume it "
                    f"exactly. Quit and stop the run?"):
                return False                              # stay open
            try:
                req = urllib.request.Request(f"http://127.0.0.1:{port}/api/app/stop_run", data=b"{}", method="POST",
                                             headers={"X-Fizgig-App-Key": key, "Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=30).read()
            except Exception:
                pass
            return True
        window.events.closing += closing
        try:
            webview.settings["ALLOW_DOWNLOADS"] = True       # Runs & outputs' Download saves where you choose
            webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True   # Buy me a coffee, RunPod, the docs
        except Exception:
            pass
        page_ready = threading.Event()
        started = threading.Event()       # webview.start came back (or failed: no engine) - the watchdog stands down

        def watchdog():                    # a window that never appears ends, its server with it, not left unseen
            for _ in range(120):
                if window.events.shown.wait(1):
                    break
                if started.is_set():       # no window at all: the browser took over and the server must stay up
                    return
            else:
                _stop(own)
                os._exit(1)
            # a window that appears but never runs the page (a white window: a broken WebView2, its sandbox on a
            # junctioned profile folder, a blocked GPU process) gives way to the browser rather than staying white.
            # Only for this computer's own page: a pod's may be an older Fizgig that doesn't report in
            if first == url and not page_ready.wait(30):
                fallback[0] = ("the window opened but the page never ran in it (WebView2 may be failing to draw, or "
                               "be blocked by security software)")
                try:
                    window.destroy()
                except Exception:
                    pass
        threading.Thread(target=watchdog, daemon=True).start()
        profile = user_file("app_profile")
        try:
            webview.start(private_mode=False, storage_path=profile, icon=_icon(profile))
        finally:
            started.set()
        if fallback[0]:
            raise RuntimeError(fallback[0])
    except Exception as e:
        # no engine here (a Linux machine without GTK/WebKit or Qt, Windows without WebView2) or a window that never
        # showed the page: the browser, and how to get the window
        how = (f"For the window, install Microsoft Edge WebView2 Runtime: {WEBVIEW2_URL}" if os.name == "nt" else
               "For the window on Linux, in Fizgig's venv: `venv/bin/pip install \"pywebview[qt]\"`, or WebKitGTK: "
               "your distro's gir1.2-webkit2-4.1 and libgirepository dev packages, then `venv/bin/pip install "
               "PyGObject` (the distro's python3-gi isn't seen inside the venv).")
        _log(f"[app] no app window ({e}) - opening the web UI in your browser instead. {how}")
        webbrowser.open(url)
        if own is not None:
            try:
                own.wait()                               # keep the server up until it is stopped (Ctrl+C)
            except KeyboardInterrupt:
                pass
    finally:
        _stop(own)


if __name__ == "__main__":
    main()
