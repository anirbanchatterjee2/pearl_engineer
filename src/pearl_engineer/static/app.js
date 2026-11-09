/* Pearl Engineer — diagram library UI.
   Vanilla JS, no build step. All text coming from the API is escaped before it
   is inserted into the DOM. */

(() => {
  "use strict";

  const FALLBACK_CATEGORIES = [
    {
      id: "mechanical",
      label: "Mechanical",
      description: "Machine parts, assemblies, dimensions and tolerances.",
    },
    {
      id: "civil",
      label: "Civil Engineering",
      description: "Structural, site, utility and infrastructure drawings.",
    },
    {
      id: "computer_flow",
      label: "Computer Flow",
      description: "Software architecture, data-flow and process diagrams.",
    },
  ];

  const CATEGORY_COLOR = {
    mechanical: "var(--mechanical)",
    civil: "var(--civil)",
    computer_flow: "var(--computer)",
  };

  const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
  const ALLOWED_EXTENSIONS = /\.(png|jpe?g)$/i;

  const state = {
    categories: FALLBACK_CATEGORIES.slice(),
    category: "",
    lastQuery: "",
    results: [],
    lastUploaded: null,
    file: null,
    uploadCategory: "computer_flow",
    uploading: false,
    previewUrl: null,
    toastTimer: null,
    tenderSources: [],
    tenders: [],
    tenderSummary: null,
    tenderSourcesLoaded: false,
    crawling: false,
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const els = {};

  document.addEventListener("DOMContentLoaded", init);

  function init() {
    cacheElements();
    bindTabs();
    bindSearch();
    bindTenders();
    bindUpload();
    bindDialog();
    bindKeyboardShortcuts();

    restoreAdminToken();
    loadCategories();
    loadHealth();
    loadLibrary();
    loadTenderSources();
    loadTenders();
  }

  function cacheElements() {
    els.tabs = $$(".tab");
    els.views = $$(".view");
    els.status = $("#status");
    els.statusText = $("#status-text");
    els.footerRecords = $("#footer-records");

    els.searchForm = $("#search-form");
    els.searchInput = $("#search-input");
    els.searchButton = $("#search-button");
    els.nResults = $("#n-results");
    els.chips = $("#category-chips");
    els.suggestions = $("#suggestions");
    els.results = $("#results");
    els.resultsHeading = $("#results-heading");
    els.resultsMeta = $("#results-meta");
    els.resetSearch = $("#reset-search");

    els.uploadForm = $("#upload-form");
    els.dropzone = $("#dropzone");
    els.dropzoneEmpty = $("#dropzone-empty");
    els.dropzonePreview = $("#dropzone-preview");
    els.fileInput = $("#file-input");
    els.browseBtn = $("#browse-btn");
    els.clearFile = $("#clear-file");
    els.previewImg = $("#preview-img");
    els.previewName = $("#preview-name");
    els.previewSize = $("#preview-size");
    els.uploadCategories = $("#upload-categories");
    els.uploadTitle = $("#upload-title");
    els.adminToken = $("#admin-token");
    els.uploadSubmit = $("#upload-submit");
    els.uploadSuccess = $("#upload-success");
    els.successCopy = $("#success-copy");
    els.successImage = $("#success-image");
    els.searchNew = $("#search-new");
    els.uploadAnother = $("#upload-another");

    els.dialog = $("#detail-dialog");
    els.detailClose = $("#detail-close");
    els.detailImage = $("#detail-image");
    els.detailCategory = $("#detail-category");
    els.detailTitle = $("#detail-title");
    els.detailFile = $("#detail-file");
    els.detailSimilarityWrap = $("#detail-similarity-wrap");
    els.detailSimilarityBar = $("#detail-similarity-bar");
    els.detailSimilarity = $("#detail-similarity");
    els.detailDescription = $("#detail-description");
    els.detailDownload = $("#detail-download");
    els.detailOpen = $("#detail-open");

    els.tenderForm = $("#tender-form");
    els.tenderQuery = $("#tender-query");
    els.tenderSource = $("#tender-source");
    els.tenderStatus = $("#tender-status");
    els.tenderDemo = $("#tender-demo");
    els.tenderCrawlButton = $("#tender-crawl-button");
    els.tenderRefreshButton = $("#tender-refresh-button");
    els.tenderFilterButton = $("#tender-filter-button");
    els.gistHeadline = $("#gist-headline");
    els.gistBody = $("#gist-body");
    els.gistStats = $("#gist-stats");
    els.gistUpdated = $("#gist-updated");
    els.gistErrors = $("#gist-errors");
    els.tenderCount = $("#tender-count");
    els.tenderPortalCount = $("#tender-portal-count");
    els.tendersGrid = $("#tenders-grid");
    els.tenderDialog = $("#tender-dialog");
    els.tenderClose = $("#tender-close");
    els.tenderBadge = $("#tender-badge");
    els.tenderTitle = $("#tender-title");
    els.tenderMeta = $("#tender-meta");
    els.tenderFacts = $("#tender-facts");
    els.tenderDescription = $("#tender-description");
    els.tenderOpen = $("#tender-open");

    els.toast = $("#toast");
  }

  /* ── Tabs ─────────────────────────────────────────────────────────── */

  function bindTabs() {
    els.tabs.forEach((tab) => {
      tab.addEventListener("click", () => switchView(tab.dataset.view));
    });
  }

  function switchView(view) {
    els.tabs.forEach((tab) => {
      const active = tab.dataset.view === view;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    els.views.forEach((section) => {
      const active = section.id === `view-${view}`;
      section.classList.toggle("is-active", active);
      section.hidden = !active;
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  /* ── API ──────────────────────────────────────────────────────────── */

  async function api(path, options = {}) {
    const { headers, ...rest } = options;
    let response;
    try {
      response = await fetch(path, {
        ...rest,
        headers: { Accept: "application/json", ...(headers || {}) },
      });
    } catch (error) {
      throw new Error("Cannot reach the API. Is the Pearl server running?");
    }

    if (!response.ok) {
      let body = null;
      try {
        body = await response.json();
      } catch {
        /* not JSON */
      }
      throw new Error(errorMessage(body, response.statusText || `HTTP ${response.status}`));
    }
    if (response.status === 204) return null;
    return response.json();
  }

  function errorMessage(body, fallback) {
    if (body && typeof body.detail === "string") return body.detail;
    if (body && Array.isArray(body.detail)) {
      return body.detail
        .map((item) => item.msg || JSON.stringify(item))
        .join("; ");
    }
    return fallback;
  }

  /* ── Health & categories ──────────────────────────────────────────── */

  async function loadHealth() {
    try {
      const health = await api("/health");
      const records = Number(health.records) || 0;
      const label = `${records} diagram${records === 1 ? "" : "s"} indexed`;
      setStatus(health.status === "ok" ? "ok" : "warn", label);
      els.footerRecords.textContent = `${records} record${records === 1 ? "" : "s"}`;
    } catch {
      setStatus("error", "API offline");
      els.footerRecords.textContent = "API offline";
    }
  }

  function setStatus(kind, text) {
    els.status.classList.remove("is-ok", "is-warn", "is-error");
    els.status.classList.add(`is-${kind}`);
    els.statusText.textContent = text;
  }

  async function loadCategories() {
    try {
      const categories = await api("/categories");
      if (Array.isArray(categories) && categories.length) {
        state.categories = categories;
      }
    } catch {
      /* keep fallback categories */
    }
    if (!state.categories.some((category) => category.id === state.uploadCategory)) {
      state.uploadCategory = state.categories[0]?.id || "computer_flow";
    }
    renderCategoryChips();
    renderUploadCategories();
  }

  function renderCategoryChips() {
    const chips = [
      `<button class="chip is-active" type="button" data-category="" aria-pressed="true">All diagrams</button>`,
      ...state.categories.map(
        (category) =>
          `<button class="chip" type="button" data-category="${escapeAttr(category.id)}" aria-pressed="false">${escapeHtml(category.label)}</button>`,
      ),
    ];
    els.chips.innerHTML = chips.join("");
    updateChipState();
  }

  function updateChipState() {
    $$(".chip", els.chips).forEach((chip) => {
      const active = (chip.dataset.category || "") === state.category;
      chip.classList.toggle("is-active", active);
      chip.setAttribute("aria-pressed", String(active));
    });
  }

  function renderUploadCategories() {
    els.uploadCategories.innerHTML = state.categories
      .map((category) => {
        const checked = category.id === state.uploadCategory;
        return `
          <label class="category-option" style="--cat-color: ${CATEGORY_COLOR[category.id] || "var(--faint)"}">
            <input type="radio" name="upload-category" value="${escapeAttr(category.id)}" ${checked ? "checked" : ""} />
            <span class="cat-dot" aria-hidden="true"></span>
            <strong>${escapeHtml(category.label)}</strong>
            <small>${escapeHtml(category.description)}</small>
          </label>`;
      })
      .join("");
  }

  /* ── Search ───────────────────────────────────────────────────────── */

  function bindSearch() {
    els.searchForm.addEventListener("submit", (event) => {
      event.preventDefault();
      runSearch(els.searchInput.value);
    });

    els.chips.addEventListener("click", (event) => {
      const chip = event.target.closest(".chip");
      if (!chip) return;
      state.category = chip.dataset.category || "";
      updateChipState();
      if (state.lastQuery) {
        runSearch(state.lastQuery, state.category);
      } else {
        loadLibrary(state.category);
      }
    });

    els.suggestions.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-query]");
      if (!button) return;
      els.searchInput.value = button.dataset.query;
      runSearch(button.dataset.query);
    });

    els.nResults.addEventListener("change", () => {
      if (state.lastQuery) runSearch(state.lastQuery, state.category);
    });

    els.resetSearch.addEventListener("click", () => {
      state.lastQuery = "";
      state.category = "";
      els.searchInput.value = "";
      els.resetSearch.hidden = true;
      updateChipState();
      loadLibrary();
    });

    els.results.addEventListener("click", (event) => {
      const card = event.target.closest("[data-index]");
      if (!card) return;
      const action = event.target.closest("[data-action]")?.dataset.action;
      const item = state.results[Number(card.dataset.index)];
      if (!item) return;
      if (action === "download") return;
      if (action === "open" || !action) openDetail(item);
    });

    els.results.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const media = event.target.closest(".card-media");
      if (!media) return;
      event.preventDefault();
      const card = media.closest("[data-index]");
      const item = card ? state.results[Number(card.dataset.index)] : null;
      if (item) openDetail(item);
    });
  }

  async function runSearch(query, category = state.category) {
    const trimmed = String(query || "").trim();
    if (!trimmed) {
      els.searchInput.focus();
      showToast("Type a description of the diagram you are looking for.", true);
      return;
    }

    state.lastQuery = trimmed;
    state.category = category || "";
    updateChipState();
    els.resetSearch.hidden = false;
    setSearchLoading(true);
    els.resultsHeading.textContent = "Searching…";
    els.resultsMeta.textContent = "Matching your description against the diagram index.";
    renderSkeletons(6);

    try {
      const payload = {
        query: trimmed,
        n_results: Number(els.nResults.value) || 12,
      };
      if (state.category) payload.category = state.category;

      const data = await api("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      state.results = Array.isArray(data.results) ? data.results : [];
      const count = state.results.length;
      const meta = [`${count} match${count === 1 ? "" : "es"}`];
      if (data.cache && data.cache.hit) meta.push(`served from ${data.cache.kind} cache`);
      if (state.category) meta.push(labelForCategory(state.category));
      renderResults(state.results, {
        title: `Results for “${trimmed}”`,
        meta: meta.join(" · "),
        query: trimmed,
      });
    } catch (error) {
      renderError("Search failed", error.message, () => runSearch(trimmed, state.category));
    } finally {
      setSearchLoading(false);
    }
  }

  function setSearchLoading(loading) {
    els.searchButton.disabled = loading;
    els.searchButton.classList.toggle("is-loading", loading);
    $(".btn-label", els.searchButton).textContent = loading ? "Searching…" : "Search";
  }

  async function loadLibrary(category = state.category) {
    state.category = category || "";
    updateChipState();
    els.resultsHeading.textContent = category ? `${labelForCategory(category)} diagrams` : "Library";
    els.resultsMeta.textContent = "Loading indexed diagrams…";
    renderSkeletons(6);

    try {
      const params = new URLSearchParams({ limit: "24" });
      if (state.category) params.set("category", state.category);
      const records = await api(`/diagrams?${params.toString()}`);
      state.results = Array.isArray(records) ? records : [];
      const count = state.results.length;
      renderResults(state.results, {
        title: state.category ? `${labelForCategory(state.category)} diagrams` : "Library",
        meta: `${count} diagram${count === 1 ? "" : "s"}${state.category ? ` in ${labelForCategory(state.category)}` : " indexed"}`,
      });
    } catch (error) {
      renderError("Could not load the library", error.message, () => loadLibrary(category));
    }
  }

  /* ── Tenders: gravity-filter advertisements ─────────────────────────── */

  function bindTenders() {
    els.tenderForm.addEventListener("submit", (event) => {
      event.preventDefault();
      loadTenders();
    });
    els.tenderCrawlButton.addEventListener("click", () => crawlTenders());
    els.tenderRefreshButton.addEventListener("click", () => loadTenders());
    els.tenderSource.addEventListener("change", () => loadTenders());
    els.tenderStatus.addEventListener("change", () => loadTenders());
    els.tendersGrid.addEventListener("click", (event) => {
      const card = event.target.closest("[data-tender-index]");
      if (!card) return;
      const item = state.tenders[Number(card.dataset.tenderIndex)];
      if (item) openTender(item);
    });
    els.tenderClose.addEventListener("click", () => els.tenderDialog.close());
    els.tenderDialog.addEventListener("click", (event) => {
      if (event.target === els.tenderDialog) els.tenderDialog.close();
    });
  }

  async function loadTenderSources() {
    try {
      const sources = await api("/tenders/sources");
      if (Array.isArray(sources) && sources.length) {
        state.tenderSources = sources;
        els.tenderSource.innerHTML =
          `<option value="">All portals</option>` +
          sources
            .map(
              (s) =>
                `<option value="${escapeAttr(s.id)}">${escapeHtml(s.name)}</option>`,
            )
            .join("");
        state.tenderSourcesLoaded = true;
      }
    } catch {
      /* keep the single "All portals" option */
    }
  }

  function tenderParams() {
    const params = new URLSearchParams({ limit: "100" });
    const source = els.tenderSource.value || "";
    const status = els.tenderStatus.value || "";
    const query = (els.tenderQuery.value || "").trim();
    if (source) params.set("source", source);
    if (status) params.set("status", status);
    if (query) params.set("q", query);
    return params;
  }

  function renderSkeletonsTenders(count) {
    els.tendersGrid.innerHTML = Array.from(
      { length: count },
      () => `<div class="skeleton" aria-hidden="true"></div>`,
    ).join("");
  }

  async function loadTenders() {
    renderSkeletonsTenders(6);
    els.tenderCount.textContent = "Loading gravity-filter advertisements…";
    try {
      const [records, summary] = await Promise.all([
        api(`/tenders?${tenderParams().toString()}`),
        api("/tenders/summary")
          .catch(() => null),
      ]);
      state.tenders = Array.isArray(records) ? records : [];
      state.tenderSummary = summary;
      renderGist(summary);
      renderTenders(state.tenders);
      if (state.tenderSources.length) {
        els.tenderPortalCount.textContent =
          `Crawling ${state.tenderSources.length} portals · ` +
          `${summary?.total ?? state.tenders.length} found`;
      } else {
        els.tenderPortalCount.textContent =
          `${summary?.total ?? state.tenders.length} tender(s) in store`;
      }
      loadHealth();
    } catch (error) {
      els.tenderCount.textContent = "Could not load tenders.";
      els.tendersGrid.innerHTML = `
        <div class="state-card is-error">
          <h3>Could not load tenders</h3>
          <p>${escapeHtml(error.message || "Something went wrong.")}</p>
          <button class="btn btn-ghost" type="button" id="tender-retry">Try again</button>
        </div>`;
      $("#tender-retry", els.tendersGrid)?.addEventListener("click", loadTenders);
    }
  }

  function renderGist(summary) {
    if (!summary) {
      els.gistHeadline.textContent = "Gist unavailable";
      els.gistBody.textContent = "The gist summary could not be loaded.";
      els.gistStats.innerHTML = "";
      return;
    }
    els.gistHeadline.textContent = summary.headline || "Tender gist";
    els.gistBody.textContent = summary.gist || "";
    els.gistUpdated.textContent = summary.generated_at
      ? `Generated ${formatWhen(summary.generated_at)}`
      : "—";
    const stats = [
      `<div class="gist-stat"><strong>${Number(summary.total) || 0}</strong><span>Found</span></div>`,
      `<div class="gist-stat"><strong>${Number(summary.open) || 0}</strong><span>Open</span></div>`,
      `<div class="gist-stat"><strong>${Number(summary.closed) || 0}</strong><span>Closed</span></div>`,
      `<div class="gist-stat"><strong>${Object.keys(summary.by_source || {}).length}</strong><span>Portals</span></div>`,
    ];
    const buyers = summary.top_buyers || [];
    const keywords = summary.top_keywords || [];
    if (buyers.length) {
      stats.push(
        `<div class="gist-stat gist-stat-wide"><strong>${escapeHtml(buyers.slice(0, 3).join(", "))}</strong><span>Top buyers</span></div>`,
      );
    }
    if (keywords.length) {
      stats.push(
        `<div class="gist-stat gist-stat-wide"><strong>${escapeHtml(keywords.slice(0, 4).join(" · "))}</strong><span>Requirements</span></div>`,
      );
    }
    els.gistStats.innerHTML = stats.join("");
  }

  function renderTenders(items) {
    if (!items.length) {
      els.tenderCount.textContent = "No gravity-filter tenders match these filters.";
      els.tendersGrid.innerHTML = `
        <div class="state-card">
          <span class="state-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <circle cx="11" cy="11" r="6.5" stroke="currentColor" stroke-width="1.6"/>
              <path d="m16 16 4.5 4.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
            </svg>
          </span>
          <h3>No advertisements yet</h3>
          <p>Run a crawl to search GeM, CPPP/eProcure, DRDO, MES and MoD for
             gravity-filter tenders, or try demo data to preview the page.</p>
          <button class="btn btn-primary" type="button" id="tender-empty-crawl">Crawl demo data</button>
        </div>`;
      $("#tender-empty-crawl", els.tendersGrid)?.addEventListener("click", () => {
        els.tenderDemo.checked = true;
        crawlTenders();
      });
      return;
    }
    els.tenderCount.textContent =
      `${items.length} advertisement${items.length === 1 ? "" : "s"} for gravity filters`;
    els.tendersGrid.innerHTML = items.map((item, index) => renderTenderCard(item, index)).join("");
  }

  function renderTenderCard(item, index) {
    const title = item.title || "Untitled tender";
    const snippet = truncate(item.snippet || item.description || "No description captured.", 260);
    const status = item.status || "unknown";
    const due = item.closing_date ? `Closes ${item.closing_date}` : "Date TBA";
    const relevance =
      typeof item.relevance === "number" ? Math.round(Math.max(0, Math.min(1, item.relevance)) * 100) : null;
    const url = item.url || "#";
    return `
      <article class="card tender-card" data-tender-index="${index}">
        <div class="card-body tender-card-body">
          <div class="tender-badges">
            <span class="card-badge tender-badge-static" data-cat="tender">${escapeHtml(item.source_name || item.source_id || "Tender")}</span>
            <span class="status-pill status-${escapeAttr(status)}">${escapeHtml(status)}</span>
            <span class="muted tender-due">${escapeHtml(due)}</span>
          </div>
          <h3 class="card-title tender-title" title="${escapeAttr(title)}">${escapeHtml(title)}</h3>
          <p class="card-description">${escapeHtml(snippet)}</p>
          <div class="tender-facts">
            <span><strong>Buyer</strong>${escapeHtml(item.buyer || "—")}</span>
            <span><strong>Tender no.</strong>${escapeHtml(item.tender_no || "—")}</span>
            <span><strong>Value</strong>${escapeHtml(item.value || "—")}</span>
            <span><strong>Qty</strong>${escapeHtml(item.quantity || "—")}</span>
            <span><strong>Location</strong>${escapeHtml(item.location || "—")}</span>
          </div>
          ${
            relevance === null
              ? ""
              : `<div class="similarity" title="Relevance"><span>Relevance</span><div class="meter"><span style="width:${relevance}%"></span></div><strong>${relevance}%</strong></div>`
          }
          <div class="card-foot">
            <span class="card-file" title="${escapeAttr(item.url || "")}">${escapeHtml(item.tender_no || item.source_id || "tender")}</span>
            <div class="card-actions">
              <button class="icon-btn" type="button" data-action="open" aria-label="View tender details">
                <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" stroke="currentColor" stroke-width="1.6"/>
                  <circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.6"/>
                </svg>
              </button>
              <a class="icon-btn" href="${escapeAttr(url)}" target="_blank" rel="noopener" aria-label="Open on portal">
                <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M14 4h6v6m0-6L10 14M19 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1h6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </a>
            </div>
          </div>
        </div>
      </article>`;
  }

  async function crawlTenders() {
    if (state.crawling) return;
    state.crawling = true;
    const demo = Boolean(els.tenderDemo.checked);
    setCrawlLoading(true);
    try {
      const body = { demo };
      const source = els.tenderSource.value || "";
      if (source && !demo) body.source_ids = [source];
      const token = sessionStorage.getItem("pearl-admin-token") || "";
      const result = await api("/tenders/crawl", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { "X-Admin-Token": token } : {}),
        },
        body: JSON.stringify(body),
      });
      if (result.summary) {
        renderGist(result.summary);
        state.tenderSummary = result.summary;
      }
      const message = demo
        ? `Demo data loaded — ${result.found} sample tenders.`
        : `Crawl complete: ${result.new} new, ${result.updated} updated, ${result.found} found.`;
      showToast(message);
      if (Array.isArray(result.errors) && result.errors.length && !demo) {
        els.gistErrors.hidden = false;
        els.gistErrors.textContent =
          `Portal notes: ${result.errors.slice(0, 3).join(" · ")}`;
      } else {
        els.gistErrors.hidden = true;
      }
      await loadTenders();
    } catch (error) {
      showToast(error.message, true);
      if (/admin token/i.test(error.message)) {
        showToast("Live crawl needs the admin token — check Upload → Server access.", true);
      }
    } finally {
      state.crawling = false;
      setCrawlLoading(false);
    }
  }

  function setCrawlLoading(loading) {
    els.tenderCrawlButton.disabled = loading;
    els.tenderCrawlButton.classList.toggle("is-loading", loading);
    $(".btn-label", els.tenderCrawlButton).textContent = loading ? "Crawling…" : "Crawl portals";
  }

  function openTender(item) {
    els.tenderBadge.textContent = item.source_name || "Tender";
    els.tenderBadge.dataset.cat = "tender";
    els.tenderTitle.textContent = item.title || "Untitled tender";
    els.tenderMeta.textContent = [
      item.tender_no || "no number",
      item.buyer || "buyer n/a",
      item.closing_date ? `closes ${item.closing_date}` : "date TBA",
    ].join(" · ");
    const facts = [
      ["Status", item.status || "unknown"],
      ["Published", item.publish_date || "—"],
      ["Closing", item.closing_date || "—"],
      ["Value", item.value || "—"],
      ["Quantity", item.quantity || "—"],
      ["Location", item.location || "—"],
      ["Keywords", (item.keywords || []).join(", ") || "gravity filter"],
    ];
    els.tenderFacts.innerHTML = facts
      .map(
        ([k, v]) =>
          `<span><strong>${escapeHtml(k)}</strong>${escapeHtml(v)}</span>`,
      )
      .join("");
    els.tenderDescription.textContent =
      item.description || item.snippet || "No description captured.";
    els.tenderOpen.href = item.url || "#";
    els.tenderOpen.hidden = !item.url;
    if (typeof els.tenderDialog.showModal === "function") {
      els.tenderDialog.showModal();
    } else {
      els.tenderDialog.setAttribute("open", "");
    }
  }

  function formatWhen(value) {
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return String(value);
    return parsed.toLocaleString();
  }

  /* ── Result rendering ─────────────────────────────────────────────── */

  function renderSkeletons(count) {
    els.results.innerHTML = Array.from(
      { length: count },
      () => `<div class="skeleton" aria-hidden="true"></div>`,
    ).join("");
  }

  function renderResults(items, { title, meta, query = "" } = {}) {
    els.resultsHeading.textContent = title || "Library";
    els.resultsMeta.textContent = meta || "";

    if (!items.length) {
      renderEmpty(query);
      return;
    }

    els.results.innerHTML = items.map((item, index) => renderCard(item, index)).join("");
    $$(".card-media img", els.results).forEach((image) => {
      image.addEventListener("error", () => {
        image.closest(".card-media")?.classList.add("is-broken");
      });
    });
  }

  function renderCard(item, index) {
    const similarity =
      typeof item.similarity === "number"
        ? Math.max(0, Math.min(1, item.similarity))
        : null;
    const percent = similarity === null ? null : Math.round(similarity * 100);
    const category = item.category || "uncategorized";
    const categoryLabel = item.category_label || labelForCategory(category);
    const title = item.title || item.image_file || "Untitled diagram";
    const description = truncate(item.description || "No description is stored for this diagram.", 240);
    const imageUrl = item.image_url || "";
    const downloadUrl = item.download_url || "#";

    return `
      <article class="card" data-index="${index}">
        <div class="card-media" data-action="open" role="button" tabindex="0" aria-label="Preview ${escapeAttr(title)}">
          ${imageUrl ? `<img src="${escapeAttr(imageUrl)}" alt="${escapeAttr(title)}" loading="lazy" />` : ""}
          <span class="card-badge" data-cat="${escapeAttr(category)}">${escapeHtml(categoryLabel)}</span>
        </div>
        <div class="card-body">
          <h3 class="card-title" title="${escapeAttr(title)}">${escapeHtml(title)}</h3>
          <p class="card-description">${escapeHtml(description)}</p>
          ${
            percent === null
              ? ""
              : `<div class="similarity" title="Semantic similarity"><span>Match</span><div class="meter"><span style="width:${percent}%"></span></div><strong>${percent}%</strong></div>`
          }
          <div class="card-foot">
            <span class="card-file" title="${escapeAttr(item.image_file || "")}">${escapeHtml(item.image_file || "—")}</span>
            <div class="card-actions">
              <button class="icon-btn" type="button" data-action="open" aria-label="View details">
                <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z" stroke="currentColor" stroke-width="1.6"/>
                  <circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.6"/>
                </svg>
              </button>
              <a class="icon-btn" href="${escapeAttr(downloadUrl)}" download="${escapeAttr(item.image_file || "")}" data-action="download" aria-label="Download ${escapeAttr(item.image_file || "diagram")}">
                <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
                  <path d="M12 3v12m0 0 4.5-4.5M12 15l-4.5-4.5M4 20h16" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
              </a>
            </div>
          </div>
        </div>
      </article>`;
  }

  function renderEmpty(query) {
    const filtered = state.category ? ` in ${labelForCategory(state.category)}` : "";
    const title = query ? "No diagrams match that description" : "The library is empty";
    const message = query
      ? `Nothing${filtered} matched “${query}”. Try a broader description or a different diagram type.`
      : `Upload a mechanical, civil or computer-flow diagram to start building the library${filtered}.`;

    els.results.innerHTML = `
      <div class="state-card">
        <span class="state-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <circle cx="11" cy="11" r="6.5" stroke="currentColor" stroke-width="1.6"/>
            <path d="m16 16 4.5 4.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
            <path d="M8.5 11h5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
          </svg>
        </span>
        <h3>${escapeHtml(title)}</h3>
        <p>${escapeHtml(message)}</p>
        <button class="btn btn-primary" type="button" data-action="go-upload">Upload a diagram</button>
      </div>`;

    $("[data-action='go-upload']", els.results)?.addEventListener("click", () => {
      switchView("upload");
    });
  }

  function renderError(title, message, retry) {
    els.resultsHeading.textContent = title;
    els.resultsMeta.textContent = "";
    els.results.innerHTML = `
      <div class="state-card is-error">
        <span class="state-icon" aria-hidden="true">
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M12 8v5m0 3.5v.1M10.3 4.2 3.4 16.5A1.7 1.7 0 0 0 4.9 19h14.2a1.7 1.7 0 0 0 1.5-2.5L13.7 4.2a1.9 1.9 0 0 0-3.4 0Z" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </span>
        <h3>${escapeHtml(title)}</h3>
        <p>${escapeHtml(message || "Something went wrong.")}</p>
        <button class="btn btn-ghost" type="button" id="retry-button">Try again</button>
      </div>`;
    $("#retry-button", els.results)?.addEventListener("click", retry);
  }

  /* ── Detail dialog ────────────────────────────────────────────────── */

  function bindDialog() {
    els.detailClose.addEventListener("click", () => els.dialog.close());
    els.dialog.addEventListener("click", (event) => {
      if (event.target === els.dialog) els.dialog.close();
    });
    els.detailImage.addEventListener("error", () => {
      els.detailImage.style.display = "none";
    });
  }

  function openDetail(item) {
    const category = item.category || "uncategorized";
    const title = item.title || item.image_file || "Untitled diagram";
    const similarity =
      typeof item.similarity === "number"
        ? Math.max(0, Math.min(1, item.similarity))
        : null;

    els.detailImage.style.display = "";
    els.detailImage.src = item.image_url || "";
    els.detailImage.alt = title;
    els.detailCategory.textContent = item.category_label || labelForCategory(category);
    els.detailCategory.dataset.cat = category;
    els.detailTitle.textContent = title;
    els.detailFile.textContent = item.image_file || "";
    els.detailDescription.textContent = item.description || "No description is stored for this diagram.";
    els.detailDownload.href = item.download_url || "#";
    els.detailDownload.setAttribute("download", item.image_file || "");
    els.detailOpen.href = item.image_url || "#";

    if (similarity === null) {
      els.detailSimilarityWrap.hidden = true;
    } else {
      els.detailSimilarityWrap.hidden = false;
      const percent = Math.round(similarity * 100);
      els.detailSimilarity.textContent = `${percent}%`;
      requestAnimationFrame(() => {
        els.detailSimilarityBar.style.width = `${percent}%`;
      });
    }

    if (typeof els.dialog.showModal === "function") {
      els.dialog.showModal();
    } else {
      els.dialog.setAttribute("open", "");
    }
  }

  /* ── Upload ───────────────────────────────────────────────────────── */

  function bindUpload() {
    els.browseBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      els.fileInput.click();
    });

    els.clearFile.addEventListener("click", (event) => {
      event.stopPropagation();
      resetFile();
    });

    els.dropzone.addEventListener("click", (event) => {
      if (event.target.closest("button")) return;
      els.fileInput.click();
    });

    els.dropzone.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      if (event.target.closest("button")) return;
      event.preventDefault();
      els.fileInput.click();
    });

    ["dragenter", "dragover"].forEach((type) => {
      els.dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        event.stopPropagation();
        els.dropzone.classList.add("is-dragging");
      });
    });

    ["dragleave", "drop"].forEach((type) => {
      els.dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        event.stopPropagation();
        els.dropzone.classList.remove("is-dragging");
      });
    });

    els.dropzone.addEventListener("drop", (event) => {
      const file = event.dataTransfer?.files?.[0];
      if (file) handleFile(file);
    });

    els.fileInput.addEventListener("change", () => {
      const file = els.fileInput.files?.[0];
      if (file) handleFile(file);
    });

    els.uploadCategories.addEventListener("change", (event) => {
      const input = event.target.closest("input[name='upload-category']");
      if (input) state.uploadCategory = input.value;
    });

    els.uploadForm.addEventListener("submit", (event) => {
      event.preventDefault();
      uploadDiagram();
    });

    els.searchNew.addEventListener("click", () => {
      const diagram = state.lastUploaded;
      if (!diagram) return;
      switchView("search");
      const query = diagram.title || diagram.image_file || "";
      els.searchInput.value = query;
      state.category = diagram.category || "";
      updateChipState();
      runSearch(query, state.category);
    });

    els.uploadAnother.addEventListener("click", () => {
      els.uploadSuccess.hidden = true;
      els.uploadForm.hidden = false;
      resetFile();
      els.uploadTitle.value = "";
      els.uploadTitle.focus();
    });

    els.adminToken.addEventListener("change", () => {
      const token = els.adminToken.value.trim();
      if (token) sessionStorage.setItem("pearl-admin-token", token);
      else sessionStorage.removeItem("pearl-admin-token");
    });
  }

  function handleFile(file) {
    if (!ALLOWED_EXTENSIONS.test(file.name) && !/^image\/(png|jpeg)$/.test(file.type)) {
      showToast("Please choose a PNG or JPEG image.", true);
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      showToast(`That file is ${formatBytes(file.size)}. The limit is 20 MB.`, true);
      return;
    }

    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = file;
    state.previewUrl = URL.createObjectURL(file);

    els.previewImg.src = state.previewUrl;
    els.previewName.textContent = file.name;
    els.previewSize.textContent = formatBytes(file.size);
    els.dropzoneEmpty.hidden = true;
    els.dropzonePreview.hidden = false;
    els.uploadSubmit.disabled = false;
  }

  function resetFile() {
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
    state.file = null;
    els.fileInput.value = "";
    els.previewImg.removeAttribute("src");
    els.dropzonePreview.hidden = true;
    els.dropzoneEmpty.hidden = false;
    els.uploadSubmit.disabled = true;
  }

  async function uploadDiagram() {
    if (state.uploading) return;
    if (!state.file) {
      showToast("Choose a diagram image first.", true);
      els.fileInput.click();
      return;
    }

    state.uploading = true;
    els.uploadSubmit.disabled = true;
    els.uploadSubmit.classList.add("is-loading");
    $(".btn-label", els.uploadSubmit).textContent = "Uploading & indexing…";

    const form = new FormData();
    form.append("file", state.file, state.file.name);
    form.append("category", state.uploadCategory);
    const title = els.uploadTitle.value.trim();
    if (title) form.append("title", title);

    const token = els.adminToken.value.trim();
    if (token) sessionStorage.setItem("pearl-admin-token", token);

    try {
      const data = await api("/diagrams", {
        method: "POST",
        headers: token ? { "X-Admin-Token": token } : {},
        body: form,
      });
      state.lastUploaded = data.diagram;
      showUploadSuccess(data.diagram);
      showToast("Diagram uploaded and indexed.");
      loadHealth();
    } catch (error) {
      showToast(error.message, true);
    } finally {
      state.uploading = false;
      els.uploadSubmit.classList.remove("is-loading");
      $(".btn-label", els.uploadSubmit).textContent = "Upload & index";
      els.uploadSubmit.disabled = !state.file;
    }
  }

  function showUploadSuccess(diagram) {
    els.uploadForm.hidden = true;
    els.uploadSuccess.hidden = false;
    els.successCopy.textContent = `${diagram.image_file} is now searchable in ${diagram.category_label}.`;
    if (diagram.image_url) {
      els.successImage.src = diagram.image_url;
      els.successImage.alt = diagram.title || diagram.image_file;
      els.successImage.hidden = false;
    } else {
      els.successImage.hidden = true;
    }
    els.searchNew.focus();
  }

  function restoreAdminToken() {
    const token = sessionStorage.getItem("pearl-admin-token");
    if (token) els.adminToken.value = token;
  }

  /* ── Keyboard shortcuts & toast ───────────────────────────────────── */

  function bindKeyboardShortcuts() {
    document.addEventListener("keydown", (event) => {
      const tag = (event.target.tagName || "").toLowerCase();
      const typing = tag === "input" || tag === "textarea" || tag === "select";
      if (event.key === "/" && !typing) {
        event.preventDefault();
        switchView("search");
        els.searchInput.focus();
      }
      if (event.key.toLowerCase() === "u" && !typing && !event.metaKey && !event.ctrlKey) {
        switchView("upload");
      }
      if (event.key.toLowerCase() === "t" && !typing && !event.metaKey && !event.ctrlKey) {
        switchView("tenders");
      }
    });
  }

  function showToast(message, isError = false) {
    els.toast.textContent = message;
    els.toast.classList.toggle("is-error", Boolean(isError));
    els.toast.classList.add("is-visible");
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => {
      els.toast.classList.remove("is-visible");
    }, 4200);
  }

  /* ── Helpers ──────────────────────────────────────────────────────── */

  function labelForCategory(id) {
    if (!id || id === "uncategorized") return "Uncategorized";
    const category = state.categories.find((item) => item.id === id);
    return category ? category.label : humanize(id);
  }

  function humanize(value) {
    return String(value || "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function truncate(text, length) {
    const value = String(text || "");
    if (value.length <= length) return value;
    return `${value.slice(0, length - 1).trimEnd()}…`;
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 KB";
    if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  const HTML_ESCAPES = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (character) => HTML_ESCAPES[character]);
  }

  function escapeAttr(value) {
    return escapeHtml(value);
  }
})();
