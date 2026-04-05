const docsNav = document.getElementById("docs-nav");
const docsSource = document.getElementById("docs-source");
const docsContent = document.getElementById("docs-content");

const docsState = {
    pages: [],
    defaultSlug: "index",
    currentSlug: null,
};

function slugFromLocation() {
    const cleanPath = window.location.pathname.replace(/\/+$/, "");
    if (cleanPath === "/docs" || cleanPath === "") {
        return null;
    }

    const parts = cleanPath.split("/");
    return parts[2] || null;
}

function docsUrl(slug) {
    return slug === "index" ? "/docs" : `/docs/${slug}`;
}

function renderNav() {
    docsNav.innerHTML = "";

    for (const page of docsState.pages) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "docs-nav-item";
        button.dataset.slug = page.slug;
        button.innerHTML = `
            <span class="docs-nav-item-title">${page.title}</span>
            <span class="docs-nav-item-path">${page.source_path}</span>
        `;
        button.addEventListener("click", () => loadPage(page.slug));
        docsNav.appendChild(button);
    }
}

function markActiveSlug(slug) {
    for (const item of docsNav.querySelectorAll(".docs-nav-item")) {
        item.classList.toggle("is-active", item.dataset.slug === slug);
    }
}

function wireContentLinks() {
    for (const anchor of docsContent.querySelectorAll("a")) {
        const href = anchor.getAttribute("href");
        if (!href) {
            continue;
        }

        if (href.startsWith("/docs")) {
            anchor.addEventListener("click", (event) => {
                const target = anchor.getAttribute("href");
                if (!target) {
                    return;
                }

                const url = new URL(target, window.location.origin);
                if (!url.pathname.startsWith("/docs")) {
                    return;
                }

                event.preventDefault();
                const nextSlug = url.pathname === "/docs" ? "index" : url.pathname.split("/")[2];
                loadPage(nextSlug || "index", { hash: url.hash });
            });
            continue;
        }

        if (/^https?:\/\//.test(href)) {
            anchor.target = "_blank";
            anchor.rel = "noreferrer";
        }
    }
}

function renderMarkdown(md) {
    if (!md) return "";
    
    const mathBlocks = [];
    
    // Shield display math block $$...$$
    md = md.replace(/\$\$(.*?)\$\$/gs, (match) => {
        mathBlocks.push(match);
        return `%%%MATH_BLOCK_${mathBlocks.length - 1}%%%`;
    });
    
    // Shield inline math $...$
    md = md.replace(/\$(.*?)\$/g, (match) => {
        mathBlocks.push(match);
        return `%%%MATH_INLINE_${mathBlocks.length - 1}%%%`;
    });

    let html = window.marked ? marked.parse(md) : `<pre>${md}</pre>`;

    // Restore math blocks
    html = html.replace(/%%%MATH_BLOCK_(\d+)%%%/g, (match, p1) => {
        return mathBlocks[parseInt(p1, 10)];
    });
    html = html.replace(/%%%MATH_INLINE_(\d+)%%%/g, (match, p1) => {
        return mathBlocks[parseInt(p1, 10)];
    });

    return html;
}

async function loadPage(slug, options = {}) {
    const requestedSlug = slug || docsState.defaultSlug || "index";
    docsContent.innerHTML = '<div class="docs-loading">Loading page...</div>';

    try {
        const page = docsState.pages.find(p => p.slug === requestedSlug);
        if (!page) {
            throw new Error(`Failed to load ${requestedSlug}`);
        }

        docsState.currentSlug = page.slug;
        if (docsSource) docsSource.textContent = page.source_path;
        docsContent.innerHTML = renderMarkdown(page.markdown);
        markActiveSlug(page.slug);
        wireContentLinks();

        if (window.MathJax && window.MathJax.typesetPromise) {
            MathJax.typesetClear([docsContent]);
            await MathJax.typesetPromise([docsContent]);
        }

        const targetUrl = docsUrl(page.slug);
        if (window.location.pathname !== targetUrl) {
            window.history.pushState({ slug: page.slug }, "", targetUrl);
        }

        if (options.hash) {
            window.location.hash = options.hash;
        } else {
            window.scrollTo({ top: 0, behavior: "auto" });
        }
    } catch (error) {
        docsContent.innerHTML = `
            <div class="docs-error">
                Unable to load this docs page right now.
            </div>
        `;
    }
}

async function bootDocs() {
    try {
        const response = await fetch("/api/docs/pages");
        if (!response.ok) {
            throw new Error("Failed to load docs index");
        }

        const payload = await response.json();
        docsState.pages = payload.pages || [];
        docsState.defaultSlug = payload.default_slug || "index";
        renderNav();

        const initialSlug = slugFromLocation() || docsState.defaultSlug;
        await loadPage(initialSlug);
    } catch (error) {
        docsNav.innerHTML = '<div class="docs-error">Unable to load docs navigation.</div>';
        docsContent.innerHTML = '<div class="docs-error">Unable to load docs page.</div>';
    }
}

window.addEventListener("popstate", () => {
    const nextSlug = slugFromLocation() || docsState.defaultSlug;
    loadPage(nextSlug);
});

bootDocs();
