import html
import re
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse

try:
    import markdown as markdown_lib
except ImportError:
    markdown_lib = None

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DOCS_HTML_PATH = FRONTEND_DIR / "docs.html"
PRIMARY_DOC_PATHS = (
    ("index", PROJECT_ROOT / "docs" / "index.md"),
    ("readme", PROJECT_ROOT / "README.md"),
    ("development", PROJECT_ROOT / "docs" / "development.md"),
    ("api-reference", PROJECT_ROOT / "docs" / "api-reference.md"),
    ("testing", PROJECT_ROOT / "docs" / "testing.md"),
)

def _extract_doc_title(markdown_text: str, fallback: str) -> str:
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback

def _collect_docs_registry() -> tuple[list[dict[str, str]], dict[str, Path], dict[Path, str]]:
    pages: list[dict[str, str]] = []
    slug_to_path: dict[str, Path] = {}
    path_to_slug: dict[Path, str] = {}
    seen_paths: set[Path] = set()

    def register(slug: str, path: Path) -> None:
        resolved = path.resolve()
        if not path.exists() or slug in slug_to_path or resolved in seen_paths:
            return

        markdown_text = path.read_text(encoding="utf-8")
        slug_to_path[slug] = path
        path_to_slug[resolved] = slug
        seen_paths.add(resolved)
        pages.append(
            {
                "slug": slug,
                "title": _extract_doc_title(
                    markdown_text,
                    slug.replace("-", " ").title(),
                ),
                "source_path": path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    for slug, path in PRIMARY_DOC_PATHS:
        register(slug, path)

    for path in sorted((PROJECT_ROOT / "docs").glob("*.md")):
        register(path.stem, path)

    return pages, slug_to_path, path_to_slug

_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

def _rewrite_markdown_links(
    markdown_text: str,
    source_path: Path,
    path_to_slug: dict[Path, str],
) -> str:
    def replace_link(match: re.Match[str]) -> str:
        label, target = match.groups()

        if (
            "://" in target
            or target.startswith("#")
            or target.startswith("mailto:")
        ):
            return match.group(0)

        path_part, sep, anchor = target.partition("#")
        if not path_part.endswith(".md"):
            return match.group(0)

        resolved_target = (source_path.parent / path_part).resolve()
        slug = path_to_slug.get(resolved_target)
        if slug is None:
            return match.group(0)

        doc_href = "/docs" if slug == "index" else f"/docs/{slug}"
        if sep:
            doc_href = f"{doc_href}#{anchor}"
        return f"[{label}]({doc_href})"

    return _MARKDOWN_LINK_PATTERN.sub(replace_link, markdown_text)

def _render_markdown(markdown_text: str) -> str:
    if markdown_lib is None:
        return f"<pre>{html.escape(markdown_text)}</pre>"

    return markdown_lib.markdown(
        markdown_text,
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "sane_lists",
        ],
    )

@router.get("/api/docs/pages")
async def list_docs_pages():
    pages, slug_to_path, path_to_slug = _collect_docs_registry()
    
    enriched_pages = []
    for page in pages:
        slug = page["slug"]
        source_path = slug_to_path[slug]
        markdown_text = source_path.read_text(encoding="utf-8")
        rewritten_markdown = _rewrite_markdown_links(
            markdown_text,
            source_path,
            path_to_slug,
        )
        page_copy = dict(page)
        page_copy["markdown"] = rewritten_markdown
        enriched_pages.append(page_copy)
        
    default_slug = "index" if any(page["slug"] == "index" for page in pages) else ""
    if not default_slug and pages:
        default_slug = pages[0]["slug"]
    return {"pages": enriched_pages, "default_slug": default_slug}

@router.get("/docs", include_in_schema=False)
@router.get("/docs/", include_in_schema=False)
@router.get("/docs/{slug}", include_in_schema=False)
async def docs_page(slug: str | None = None):
    del slug
    return FileResponse(DOCS_HTML_PATH)
