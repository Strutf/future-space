import json
import re
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "src" / "futurespace" / "api" / "web"


def load(lang: str) -> dict:
    return json.loads((WEB / "locales" / f"{lang}.json").read_text(encoding="utf-8"))


def keys(tree: dict, prefix: str = "") -> set[str]:
    out = set()
    for k, v in tree.items():
        out |= keys(v, f"{prefix}{k}.") if isinstance(v, dict) else {prefix + k}
    return out


LANGS = ("pt", "en", "es", "fr")


def test_locales_have_the_same_keys():
    reference = keys(load("pt"))
    for lang in LANGS[1:]:
        other = keys(load(lang))
        assert reference - other == set(), f"faltam em {lang}.json"
        assert other - reference == set(), f"sobram em {lang}.json"


def test_locale_values_keep_the_same_placeholders():
    pt = load("pt")

    def flat(tree, prefix=""):
        for k, v in tree.items():
            if isinstance(v, dict):
                yield from flat(v, f"{prefix}{k}.")
            else:
                yield prefix + k, v

    pattern = r"\{\w+\}|\{\{url:\w+\}\}"
    for lang in LANGS[1:]:
        other = dict(flat(load(lang)))
        for key, value in flat(pt):
            assert sorted(re.findall(pattern, value)) == sorted(re.findall(pattern, other[key])), (lang, key)


def test_every_js_text_used_in_pages_exists():
    used = set()
    for page in (WEB / "pages").glob("*.html"):
        used |= set(re.findall(r'\bt\("(\w+)"', page.read_text(encoding="utf-8")))
        used |= set(re.findall(r'"(th_\w+)"', page.read_text(encoding="utf-8")))
    for lang in LANGS:
        missing = used - set(load(lang)["js"])
        assert not missing, (lang, missing)
