"""Public-model downloads that survive an expired Hugging Face login.

huggingface_hub sends the saved login (`hf auth login`, or HF_TOKEN) with every request, even for a public repo.
Once that login expires (a browser "OAuth" login lasts about a month) the Hub answers 401, so a repo anyone could
download fails with "Repository Not Found ... token has expired". download() and snapshot() try the normal way
first, and on a 401 sent with a token they try once more without it (token=False). Gated and private repos still
need a working login: if the anonymous try fails too, LoginExpired says so in one short line.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

NOTE = ("Your Hugging Face login has expired; downloaded without it. "
        "Log in again with `hf auth login` if you need private models.")
EXPIRED = ("Your Hugging Face login has expired or isn't valid. Run `hf auth login` "
           "(or `hf auth logout` to download without a login) and try again.")

_noted = False


class LoginExpired(RuntimeError):
    """The saved Hugging Face login was refused and the file can't be fetched without one."""

    def __init__(self, msg=EXPIRED):
        super().__init__(msg)


def _chain(e):
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        yield e
        e = e.__cause__ or e.__context__


def _status(e):
    for x in _chain(e):
        r = getattr(x, "response", None)
        code = getattr(r, "status_code", None)
        if code:
            return code
    return None


def _token_text(e):
    """True when the error text itself says the token was the problem."""
    t = " ".join(str(x) for x in _chain(e)).lower()
    return any(k in t for k in ("token has expired", "token is expired", "invalid credentials",
                                "invalid token", "invalid user token", '"exp" claim'))


def _sent_token(kw):
    tok = kw.get("token")
    if tok is False:
        return False
    if isinstance(tok, str) and tok:
        return True
    try:
        from huggingface_hub import get_token
        return bool(get_token())
    except Exception:
        return True                # can't tell: one extra anonymous try is harmless


def is_login_error(e, kw=None):
    """A 401 (or a token-shaped complaint) on a request that carried a login."""
    if isinstance(e, LoginExpired):
        return True
    if kw is not None and not _sent_token(kw):
        return False
    return _status(e) == 401 or _token_text(e)


def _with_retry(fn, kw, log):
    global _noted
    try:
        return fn(**kw)
    except Exception as e:
        if not is_login_error(e, kw):
            raise
        first = e
    try:
        got = fn(**{**kw, "token": False})
    except Exception as e2:
        if _status(e2) in (401, 403) or "gated" in type(e2).__name__.lower():
            raise LoginExpired() from first
        raise
    if not _noted:
        _noted = True
        (log or logger.warning)(NOTE)
    return got


def download(log=None, **kw):
    """hf_hub_download(**kw), tried once more without the login if the login is refused."""
    from huggingface_hub import hf_hub_download
    return _with_retry(hf_hub_download, kw, log)


def snapshot(log=None, **kw):
    """snapshot_download(**kw), tried once more without the login if the login is refused."""
    from huggingface_hub import snapshot_download
    return _with_retry(snapshot_download, kw, log)


def friendly_error(e):
    """The text a download card shows: one short line for a refused login, else the usual 'Name: message'."""
    if isinstance(e, LoginExpired) or _token_text(e):
        return EXPIRED
    return f"{type(e).__name__}: {e}"
