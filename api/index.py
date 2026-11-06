# Ponto de entrada da Vercel (funções Python em api/). O vercel.json reescreve toda rota
# para /api/index?__path=<rota original>; o adaptador abaixo restaura a rota antes de
# chegar ao FastAPI. A aplicação real fica em src/futurespace/api/app.py.
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from futurespace.api.app import app as _app


class RestoreRewrittenPath:
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            query = parse_qsl(scope.get("query_string", b"").decode(), keep_blank_values=True)
            original = next((v for k, v in query if k == "__path"), None)
            if original is not None:
                path = "/" + original.lstrip("/")
                rest = urlencode([(k, v) for k, v in query if k != "__path"])
                scope = {**scope, "path": path, "raw_path": path.encode(), "query_string": rest.encode()}
        await self.inner(scope, receive, send)


app = RestoreRewrittenPath(_app)
