"""Download a training folder onto this machine (a pod), from a link (to-do #13, @girzai #308): a Hugging Face dataset
(the repo, or a sub-folder of it; a private one with HF_TOKEN), a Google Drive share link to a file (a zip), or a direct
link to a zip / tar. Archives are unpacked and removed. Picks up where it left off when run again on the same folder.
    python -m fizgig.scripts.fetch_dataset LINK DEST_FOLDER
Prints one line per stage; the web server shows progress from the folder's size (web/pod_dataset.py)."""
import os
import re
import shutil
import sys
import tarfile
import zipfile

ARCHIVES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


def parse(link):
    """(kind, target) for a link: ("hf", (repo, sub)) | ("drive", file_id) | ("url", url). ValueError otherwise."""
    s = str(link or "").strip()
    m = re.match(r"^(?:https?://)?(?:www\.)?huggingface\.co/datasets/([^/\s]+/[^/\s?#]+)(?:/(?:tree|resolve)/[^/\s]+/?"
                 r"([^?#\s]*))?", s)
    if m:
        return "hf", (m.group(1), (m.group(2) or "").strip("/"))
    m = re.match(r"^hf:([^/\s]+/[^/\s]+)(?:/(.*))?$", s)
    if m:
        return "hf", (m.group(1), (m.group(2) or "").strip("/"))
    if re.match(r"^(?:https?://)?drive\.google\.com/", s):
        if "/folders/" in s:
            raise ValueError("A Google Drive folder can't be downloaded in one go: zip the folder in Drive, share the "
                             "zip with \"Anyone with the link\", and paste that link.")
        m = re.search(r"/d/([A-Za-z0-9_-]{10,})", s) or re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", s)
        if not m:
            raise ValueError("That Google Drive link has no file id in it.")
        return "drive", m.group(1)
    if re.match(r"^https?://\S+$", s):
        return "url", s
    raise ValueError("Paste a Hugging Face dataset link, a Google Drive share link, or a direct link to a zip or tar.")


def unpack(folder):
    """Unpack every archive in the folder into it, then delete the archive. A single top folder inside is lifted out."""
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if not name.lower().endswith(ARCHIVES) or not os.path.isfile(p):
            continue
        print(f"unpacking {name}", flush=True)
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(p) as z:
                z.extractall(folder)
        else:
            with tarfile.open(p) as t:
                t.extractall(folder, filter="data")
        os.remove(p)
    items = [n for n in os.listdir(folder) if not n.startswith(".")]
    if len(items) == 1 and os.path.isdir(os.path.join(folder, items[0])):    # zip of one folder: its contents
        inner = os.path.join(folder, items[0])
        for n in os.listdir(inner):
            shutil.move(os.path.join(inner, n), os.path.join(folder, n))
        os.rmdir(inner)


def _stream(resp, path):
    with open(path, "ab") as f:
        for chunk in resp.iter_content(1 << 20):
            if chunk:
                f.write(chunk)


def http_file(url, folder, session=None, name=None):
    """Download one file into the folder, resuming a .part left by an earlier try (HTTP Range)."""
    import requests
    s = session or requests.Session()
    if not name:
        name = os.path.basename(url.split("?")[0]) or "download"
    part = os.path.join(folder, name + ".part")
    have = os.path.getsize(part) if os.path.exists(part) else 0
    r = s.get(url, stream=True, timeout=60, headers={"Range": f"bytes={have}-"} if have else {})
    if r.status_code == 416:                       # nothing left to fetch
        r.close()
    else:
        r.raise_for_status()
        cd = r.headers.get("content-disposition", "")
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
        if m and not have:
            name = os.path.basename(m.group(1))
            part = os.path.join(folder, name + ".part")
        if have and r.status_code != 206:          # the server ignored the range: start again
            open(part, "wb").close()
        _stream(r, part)
    os.replace(part, os.path.join(folder, name))
    return name


def drive_file(file_id, folder):
    """A Google Drive file shared with "Anyone with the link", past the large-file confirmation page."""
    import requests
    s = requests.Session()
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    r = s.get(url, stream=True, timeout=60)
    ctype = r.headers.get("content-type", "")
    r.close()
    if "text/html" in ctype:
        raise RuntimeError("Google Drive didn't hand over the file: share it with \"Anyone with the link\", or, for a "
                           "popular file, try again later (Drive's download quota).")
    return http_file(url, folder, session=s, name=f"drive_{file_id}.zip")


def hf_dataset(repo, sub, folder):
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=folder,
                      allow_patterns=[f"{sub}/**"] if sub else None, token=os.environ.get("HF_TOKEN") or None)
    if sub:                                        # the sub-folder's contents become the folder
        inner = os.path.join(folder, sub)
        for n in os.listdir(inner):
            shutil.move(os.path.join(inner, n), os.path.join(folder, n))
        shutil.rmtree(os.path.join(folder, sub.split("/")[0]), ignore_errors=True)
    shutil.rmtree(os.path.join(folder, ".cache"), ignore_errors=True)


def main():
    link, folder = sys.argv[1], sys.argv[2]
    os.makedirs(folder, exist_ok=True)
    kind, target = parse(link)
    print(f"downloading ({kind})", flush=True)
    if kind == "hf":
        hf_dataset(target[0], target[1], folder)
    elif kind == "drive":
        drive_file(target, folder)
    else:
        http_file(target, folder)
    unpack(folder)
    print("DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                         # one plain line for the page
        print(f"ERROR {e}", flush=True)
        sys.exit(1)
