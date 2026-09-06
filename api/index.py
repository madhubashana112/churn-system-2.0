"""Vercel serverless entrypoint.

The application is a single ASGI callable, which is exactly what Vercel's
Python runtime expects a function to export. Two adaptations live here:

- the repo root is put on sys.path (the runtime starts us from api/);
- Vercel hands the function the *rewritten* path (/api/index.py) with the
  original URL nowhere in the headers, so vercel.json forwards the original
  path as a query parameter and the middleware below restores it before the
  app routes on it.
"""

import os
import sys
from urllib.parse import parse_qsl, urlencode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from churn_platform.main import app  # noqa: E402

_PATH_PARAM = "__vercel_path"


class _VercelPathRestore:
    """Undo Vercel's rewrite so the app sees the URL the browser requested."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            pairs = parse_qsl(scope.get("query_string", b"").decode("latin1"), keep_blank_values=True)
            original = [v for k, v in pairs if k == _PATH_PARAM]
            if original:
                path = original[0] or "/"
                rest = urlencode([(k, v) for k, v in pairs if k != _PATH_PARAM]).encode()
                scope = dict(scope, path=path, raw_path=path.encode(), query_string=rest)
        await self.app(scope, receive, send)


app = _VercelPathRestore(app)

__all__ = ["app"]
