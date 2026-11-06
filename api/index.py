# Ponto de entrada da Vercel (funções Python em api/). Toda rota é redirecionada para cá
# pelo vercel.json; a aplicação real fica em src/futurespace/api/app.py.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from futurespace.api.app import app

__all__ = ["app"]
