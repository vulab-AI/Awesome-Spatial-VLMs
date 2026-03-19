(() => {
  // ====== CONFIG ======
  const CSV_URL = "./dev/awesome_papers.csv";
  const MAIN_COL = "Main_Section";
  const METH_COL = "Subsection";
  const YEAR_COL = "Year";

  // ====== DOM ======
  const tableBody = document.getElementById("tableBody");
  const searchInput = document.getElementById("searchInput");
  const methodologyFilter = document.getElementById("methodologyFilter");
  const mainSectionFilter = document.getElementById("mainSectionFilter");
  const yearFilter = document.getElementById("yearFilter");

  const firstPageBtn = document.getElementById("firstPage");
  const prevPageBtn = document.getElementById("prevPage");
  const nextPageBtn = document.getElementById("nextPage");
  const lastPageBtn = document.getElementById("lastPage");
  const recordsPerPageSelect = document.getElementById("recordsPerPage");

  const startRecordEl = document.getElementById("startRecord");
  const endRecordEl = document.getElementById("endRecord");
  const totalRecordsEl = document.getElementById("totalRecords");
  const currentPageEl = document.getElementById("currentPage");
  const totalPagesEl = document.getElementById("totalPages");

  // ====== STATE ======
  let allRows = [];
  let filteredRows = [];
  let currentPage = 1;
  let recordsPerPage = Number(recordsPerPageSelect?.value || 25);
  let sortState = { colIndex: null, order: "asc" };

  function syncTotalCount () {
    if (tableBody) tableBody.dataset.totalCount = String(allRows.length);
  }

  // 9 columns (must match thead order)
  const COLS = [
    MAIN_COL,         // 0
    METH_COL,         // 1
    "Venue",          // 2
    "Published_Date", // 3
    "Title",          // 4
    "Institution",    // 5
    "Paper_URL",      // 6
    "Code_URL",       // 7
    "Checkpoint_URL", // 8
  ];

  const norm = (s) => String(s ?? "").trim();
  const normL = (s) => norm(s).toLowerCase();

  function escapeHtml (s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function isEmptyValue (v) {
    const t = norm(v);
    return t === "" || t.toLowerCase() === "nan";
  }

  function uniqueSorted (values) {
    return [...new Set(values.map(norm).filter(v => v && v.toLowerCase() !== "nan"))]
      .sort((a, b) => a.localeCompare(b));
  }

  // Year descending (2026, 2025, ...)
  function uniqueYearDesc (values) {
    const ys = [...new Set(values.map(v => norm(v)).filter(v => v && v.toLowerCase() !== "nan"))];
    ys.sort((a, b) => (Number(b) || 0) - (Number(a) || 0));
    return ys;
  }

  function buildSearchText (row) {
    // search across visible columns (you can add YEAR_COL too if you want)
    return (COLS.concat([YEAR_COL])).map(k => normL(row[k])).join(" ");
  }

  function setOptions (selectEl, items, keepFirst = true) {
    if (!selectEl) return;
    const first = keepFirst ? selectEl.querySelector("option:first-child") : null;
    selectEl.innerHTML = "";
    if (first) selectEl.appendChild(first);

    for (const it of items) {
      const opt = document.createElement("option");
      opt.value = it;
      opt.textContent = it;
      selectEl.appendChild(opt);
    }
  }

  // ====== RENDER FILTER OPTIONS (linked) ======
  function renderFilterOptionsLinked () {
    // Use current selections to compute valid option sets (cascading/linked filters)
    const mainSel = norm(mainSectionFilter?.value || "");
    const methSel = norm(methodologyFilter?.value || "");
    const yearSel = norm(yearFilter?.value || "");

    const candidates = allRows.filter(r => {
      if (mainSel && norm(r[MAIN_COL]) !== mainSel) return false;
      if (methSel && norm(r[METH_COL]) !== methSel) return false;
      if (yearSel && norm(r[YEAR_COL]) !== yearSel) return false;
      return true;
    });

    // Build options from candidates but for each dropdown,
    // allow it to show options under the other filters (excluding itself).
    const candForMain = allRows.filter(r => {
      if (methSel && norm(r[METH_COL]) !== methSel) return false;
      if (yearSel && norm(r[YEAR_COL]) !== yearSel) return false;
      return true;
    });
    const candForMeth = allRows.filter(r => {
      if (mainSel && norm(r[MAIN_COL]) !== mainSel) return false;
      if (yearSel && norm(r[YEAR_COL]) !== yearSel) return false;
      return true;
    });
    const candForYear = allRows.filter(r => {
      if (mainSel && norm(r[MAIN_COL]) !== mainSel) return false;
      if (methSel && norm(r[METH_COL]) !== methSel) return false;
      return true;
    });

    const mainOptions = uniqueSorted(candForMain.map(r => r[MAIN_COL]));
    const methOptions = uniqueSorted(candForMeth.map(r => r[METH_COL]));
    const yearOptions = uniqueYearDesc(candForYear.map(r => r[YEAR_COL]));

    // Keep current value if still valid
    const prevMain = mainSel, prevMeth = methSel, prevYear = yearSel;

    setOptions(mainSectionFilter, mainOptions, true);
    setOptions(methodologyFilter, methOptions, true);
    setOptions(yearFilter, yearOptions, true);

    if (prevMain && mainOptions.includes(prevMain)) mainSectionFilter.value = prevMain;
    if (prevMeth && methOptions.includes(prevMeth)) methodologyFilter.value = prevMeth;
    if (prevYear && yearOptions.includes(prevYear)) yearFilter.value = prevYear;

    // candidates not directly used; applyFilters() uses full rules
    void candidates;
  }

  // ====== RENDER TABLE ======
  function linkOrEmpty (url, label) {
    if (isEmptyValue(url)) return "";
    const u = escapeHtml(url);
    return `<a href="${u}" target="_blank" rel="noreferrer">${label}</a>`;
  }

  function renderRow (r) {
    const tr = document.createElement("tr");

    const main = norm(r[MAIN_COL]);
    const meth = norm(r[METH_COL]);
    const venue = norm(r["Venue"]);
    const pub = norm(r["Published_Date"]);
    const title = norm(r["Title"]);
    const inst = norm(r["Institution"]);

    const paper = norm(r["Paper_URL"]);
    const code = norm(r["Code_URL"]);
    const ckpt = norm(r["Checkpoint_URL"]);

    tr.innerHTML = `
      <td>${escapeHtml(main)}</td>
      <td>${escapeHtml(meth)}</td>
      <td>${escapeHtml(venue)}</td>
      <td>${escapeHtml(pub)}</td>
      <td>${escapeHtml(title)}</td>
      <td>${escapeHtml(inst)}</td>
      <td>${linkOrEmpty(paper, "Open ↗")}</td>
      <td>${linkOrEmpty(code, "Open ↗")}</td>
      <td>${linkOrEmpty(ckpt, "Open ↗")}</td>
    `;
    return tr;
  }

  function renderTable () {
    const total = filteredRows.length;
    const totalPages = Math.max(1, Math.ceil(total / recordsPerPage));
    currentPage = Math.min(currentPage, totalPages);

    const startIdx = (currentPage - 1) * recordsPerPage;
    const endIdx = Math.min(startIdx + recordsPerPage, total);
    const pageRows = filteredRows.slice(startIdx, endIdx);

    startRecordEl && (startRecordEl.textContent = total === 0 ? "0" : String(startIdx + 1));
    endRecordEl && (endRecordEl.textContent = String(endIdx));
    totalRecordsEl && (totalRecordsEl.textContent = String(total));
    currentPageEl && (currentPageEl.textContent = String(currentPage));
    totalPagesEl && (totalPagesEl.textContent = String(totalPages));

    tableBody.innerHTML = "";
    if (pageRows.length === 0) {
      tableBody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:24px;">No results</td></tr>`;
      return;
    }
    for (const r of pageRows) tableBody.appendChild(renderRow(r));
  }

  // ====== FILTER + SEARCH ======
  function applyFilters ({ refreshOptions = true } = {}) {
    const q = normL(searchInput?.value || "");
    const tokens = q ? q.split(/\s+/).filter(Boolean) : [];

    const mainSel = norm(mainSectionFilter?.value || "");
    const methSel = norm(methodologyFilter?.value || "");
    const yearSel = norm(yearFilter?.value || "");

    filteredRows = allRows.filter(r => {
      if (mainSel && norm(r[MAIN_COL]) !== mainSel) return false;
      if (methSel && norm(r[METH_COL]) !== methSel) return false;
      if (yearSel && norm(r[YEAR_COL]) !== yearSel) return false;

      if (tokens.length > 0) {
        const text = buildSearchText(r);
        if (!tokens.every(t => text.includes(t))) return false;
      }
      return true;
    });

    currentPage = 1;

    if (sortState.colIndex != null) sortData(sortState.colIndex, sortState.order);
    if (refreshOptions) renderFilterOptionsLinked();
    renderTable();
  }

  // ====== SORT ======
  function sortData (colIndex, order) {
    const key = COLS[colIndex];
    filteredRows.sort((a, b) => {
      const sa = normL(a[key]);
      const sb = normL(b[key]);
      const cmp = sa.localeCompare(sb);
      return order === "asc" ? cmp : -cmp;
    });
  }

  function bindSortHeaders () {
    const headers = document.querySelectorAll("#surveyTable th.sortable");
    headers.forEach(th => {
      th.addEventListener("click", () => {
        const colIndex = Number(th.getAttribute("data-column"));
        if (Number.isNaN(colIndex)) return;

        const newOrder =
          (sortState.colIndex === colIndex && sortState.order === "asc") ? "desc" : "asc";
        sortState = { colIndex, order: newOrder };

        headers.forEach(h => h.removeAttribute("data-order"));
        th.setAttribute("data-order", newOrder);

        sortData(colIndex, newOrder);
        renderTable();
      });
    });
  }

  // ====== PAGINATION ======
  function bindPagination () {
    recordsPerPageSelect?.addEventListener("change", () => {
      recordsPerPage = Number(recordsPerPageSelect.value || 25);
      currentPage = 1;
      renderTable();
    });

    firstPageBtn?.addEventListener("click", () => { currentPage = 1; renderTable(); });
    prevPageBtn?.addEventListener("click", () => { currentPage = Math.max(1, currentPage - 1); renderTable(); });
    nextPageBtn?.addEventListener("click", () => {
      const tp = Math.max(1, Math.ceil(filteredRows.length / recordsPerPage));
      currentPage = Math.min(tp, currentPage + 1);
      renderTable();
    });
    lastPageBtn?.addEventListener("click", () => {
      const tp = Math.max(1, Math.ceil(filteredRows.length / recordsPerPage));
      currentPage = tp;
      renderTable();
    });
  }

  // ====== CLEAR ======
  window.clearAllFilters = function () {
    if (searchInput) searchInput.value = "";
    if (mainSectionFilter) mainSectionFilter.value = "";
    if (methodologyFilter) methodologyFilter.value = "";
    if (yearFilter) yearFilter.value = "";
    applyFilters({ refreshOptions: true });
  };

  // ====== LOAD CSV ======
  function loadCsv () {
    if (!window.Papa) {
      console.error("PapaParse not loaded");
      return;
    }

    Papa.parse(CSV_URL, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        allRows = (results.data || []).filter(r => Object.keys(r).length > 0);
        filteredRows = allRows.slice();
        syncTotalCount();

        // Init dropdowns once
        renderFilterOptionsLinked();

        bindSortHeaders();
        bindPagination();

        searchInput?.addEventListener("input", () => applyFilters({ refreshOptions: false }));

        mainSectionFilter?.addEventListener("change", () => applyFilters({ refreshOptions: true }));
        methodologyFilter?.addEventListener("change", () => applyFilters({ refreshOptions: true }));
        yearFilter?.addEventListener("change", () => applyFilters({ refreshOptions: true }));

        renderTable();
      },
      error: (err) => {
        console.error("CSV load error:", err);
        if (tableBody) tableBody.dataset.totalCount = "0";
        tableBody.innerHTML = `<tr><td colspan="9" style="text-align:center;color:#c00;">Failed to load CSV</td></tr>`;
      }
    });
  }

  loadCsv();
})();
