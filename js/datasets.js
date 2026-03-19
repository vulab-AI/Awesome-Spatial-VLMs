/**
 * datasets.js
 * Table columns (strict):
 * Dataset | Venue | Cognitive Level | Fundamental Task | Size | Image Source | Modality | Link
 *
 * Multi-values:
 * - Cognitive Level: "Perception; Understanding"
 * - Fundamental Task: "A, B, C"
 * - Modality: "RGB, Point Cloud"
 * We treat each separated value as atomic attribute (pill rendering + filter matching)
 */

(() => {
  const DATASETS_CSV_PATH = "./dev/spatial_datasets.csv"; // ✅ 修改为你的真实路径

  // ---- DOM ----
  const table = document.getElementById("datasetsTable");
  const tbody = document.getElementById("datasetTableBody");

  // 4 controls
  const searchInput = document.getElementById("datasetSearchInput");
  const cognitiveFilter = document.getElementById("datasetCognitiveFilter");
  const taskFilter = document.getElementById("datasetTaskFilter");
  const modalityFilter = document.getElementById("datasetModalityFilter");

  // Pagination
  const firstBtn = document.getElementById("datasetFirstPage");
  const prevBtn = document.getElementById("datasetPrevPage");
  const nextBtn = document.getElementById("datasetNextPage");
  const lastBtn = document.getElementById("datasetLastPage");
  const perPageSelect = document.getElementById("datasetRecordsPerPage");

  const startRecordEl = document.getElementById("datasetStartRecord");
  const endRecordEl = document.getElementById("datasetEndRecord");
  const totalRecordsEl = document.getElementById("datasetTotalRecords");
  const currentPageEl = document.getElementById("datasetCurrentPage");
  const totalPagesEl = document.getElementById("datasetTotalPages");

  // ---- State ----
  let allRows = [];
  let filteredRows = [];
  let currentPage = 1;
  let recordsPerPage = parseInt(perPageSelect?.value || "10", 10);

  let sortCol = null;       // 0..6 sortable, (Link column not sortable)
  let sortOrder = "asc";    // asc/desc

  function syncTotalCount() {
    if (tbody) tbody.dataset.totalCount = String(allRows.length);
  }

  // ---- Utils ----
  const safe = (v) => (v == null ? "" : String(v).trim());

  // multi-value splitter:
  // - supports comma "," and semicolon ";"
  // - handles things like "ScanNet , 3RScan , ARKitScenes"
  function splitMultiValue(str) {
    return safe(str)
      .split(/[;,]/g)
      .map(s => s.trim())
      .filter(Boolean);
  }

  function normalizeRow(raw) {
    return {
      dataset: safe(raw["Dataset"]),
      venue: safe(raw["Venue"]),
      cognitiveRaw: safe(raw["Cognitive Level"]),
      taskRaw: safe(raw["Fundamental Task"]),
      size: safe(raw["Size"]),
      sourceRaw: safe(raw["Image Source"]),
      modalityRaw: safe(raw["Modality"]),
      link: safe(raw["Link"]),

      // atomic arrays for filtering/rendering
      cognitive: splitMultiValue(raw["Cognitive Level"]),
      tasks: splitMultiValue(raw["Fundamental Task"]),
      modalities: splitMultiValue(raw["Modality"]),
    };
  }

  function pillHTML(text, type = "generic") {
    const t = safe(text);
    if (!t) return "";

    // cognitive color mapping
    if (type === "cognitive") {
      const key = t.toLowerCase();
      if (key.includes("perception")) return `<span class="pill pill-cog-perception">${t}</span>`;
      if (key.includes("understanding")) return `<span class="pill pill-cog-understanding">${t}</span>`;
      if (key.includes("extrapolation")) return `<span class="pill pill-cog-extrapolation">${t}</span>`;
      return `<span class="pill">${t}</span>`;
    }

    return `<span class="pill">${t}</span>`;
  }

  function renderPills(values, type) {
    if (!values || values.length === 0) return "-";

    const pills = values.map((v, idx) => {
      const isLast = idx === values.length - 1;
      const pill = pillHTML(v, type);
      const sep = isLast ? "" : `<span class="pill-sep">; </span>`;
      return pill + sep;
    }).join("");

    return `<div class="pill-wrap">${pills}</div>`;
  }
  function matchesSearch(row, query) {
    if (!query) return true;
    const q = query.toLowerCase();

    const blob = [
      row.dataset,
      row.venue,
      row.cognitiveRaw,
      row.taskRaw,
      row.size,
      row.sourceRaw,
      row.modalityRaw
    ].join(" ").toLowerCase();

    return blob.includes(q);
  }

  function matchesFilters(row) {
    const cogVal = safe(cognitiveFilter?.value);
    const taskVal = safe(taskFilter?.value);
    const modVal = safe(modalityFilter?.value);

    // cognitive: atomic contain
    if (cogVal && !row.cognitive.includes(cogVal)) return false;
    // task: atomic contain
    if (taskVal && !row.tasks.includes(taskVal)) return false;
    // modality: atomic contain
    if (modVal && !row.modalities.includes(modVal)) return false;

    return true;
  }

  // ---- Sorting ----
  function sortFilteredRows() {
    // 0 Dataset
    // 1 Venue
    // 2 Cognitive Level (raw string)
    // 3 Fundamental Task (raw string)
    // 4 Size
    // 5 Image Source (raw string)
    // 6 Modality (raw string)
    const keyMap = ["dataset", "venue", "cognitiveRaw", "taskRaw", "size", "sourceRaw", "modalityRaw"];
    const key = keyMap[sortCol];

    filteredRows.sort((a, b) => {
      const av = safe(a[key]);
      const bv = safe(b[key]);
      const comp = av.localeCompare(bv, undefined, { numeric: true, sensitivity: "base" });
      return sortOrder === "asc" ? comp : -comp;
    });
  }

  // ---- Render ----
  function renderPage() {
    if (!tbody) return;
    tbody.innerHTML = "";

    if (filteredRows.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:32px;">No datasets found.</td></tr>`;
      updatePaginationUI();
      return;
    }

    const start = (currentPage - 1) * recordsPerPage;
    const end = Math.min(start + recordsPerPage, filteredRows.length);
    const pageRows = filteredRows.slice(start, end);

    for (const row of pageRows) {
      const tr = document.createElement("tr");
      const linkHTML = row.link
        ? `<a href="${row.link}" target="_blank" rel="noopener noreferrer">Open ↗</a>`
        : "-";

      tr.innerHTML = `
        <td>${row.dataset || "-"}</td>
        <td>${row.venue || "-"}</td>
        <td>${renderPills(row.cognitive, "cognitive")}</td>
        <td>${renderPills(row.tasks, "task")}</td>
        <td>${row.size || "-"}</td>
        <td style="white-space: normal; word-break: break-word; line-height: 1.4;">
          ${row.sourceRaw || "-"}
        </td>
        <td>${renderPills(row.modalities, "modality")}</td>
        <td>${linkHTML}</td>
      `;
      tbody.appendChild(tr);
    }

    updatePaginationUI();
  }

  function updatePaginationUI() {
    const total = filteredRows.length;
    const totalPages = Math.max(1, Math.ceil(total / recordsPerPage));

    const start = total === 0 ? 0 : (currentPage - 1) * recordsPerPage + 1;
    const end = total === 0 ? 0 : Math.min(currentPage * recordsPerPage, total);

    if (startRecordEl) startRecordEl.textContent = String(start);
    if (endRecordEl) endRecordEl.textContent = String(end);
    if (totalRecordsEl) totalRecordsEl.textContent = String(total);

    if (currentPageEl) currentPageEl.textContent = String(currentPage);
    if (totalPagesEl) totalPagesEl.textContent = String(totalPages);

    const isFirst = currentPage <= 1;
    const isLast = currentPage >= totalPages;

    if (firstBtn) firstBtn.disabled = isFirst;
    if (prevBtn) prevBtn.disabled = isFirst;
    if (nextBtn) nextBtn.disabled = isLast;
    if (lastBtn) lastBtn.disabled = isLast;
  }

  function applySearchAndFilters() {
    const q = safe(searchInput?.value);

    filteredRows = allRows.filter(row => matchesSearch(row, q) && matchesFilters(row));

    if (sortCol !== null) sortFilteredRows();

    // page fix
    const totalPages = Math.max(1, Math.ceil(filteredRows.length / recordsPerPage));
    currentPage = Math.min(Math.max(1, currentPage), totalPages);

    renderPage();
  }

  // ---- Fill dropdown options (atomic) ----
  function fillFilterOptions() {
    const cogSet = new Set();
    const taskSet = new Set();
    const modSet = new Set();

    for (const row of allRows) {
      row.cognitive.forEach(v => cogSet.add(v));
      row.tasks.forEach(v => taskSet.add(v));
      row.modalities.forEach(v => modSet.add(v));
    }

    function setOptions(selectEl, values, placeholderText) {
      if (!selectEl) return;
      const keep = selectEl.value;

      selectEl.innerHTML = "";
      const opt0 = document.createElement("option");
      opt0.value = "";
      opt0.textContent = placeholderText;
      selectEl.appendChild(opt0);

      values.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        selectEl.appendChild(opt);
      });

      selectEl.value = keep;
    }

    setOptions(cognitiveFilter, Array.from(cogSet).sort(), "All Levels");
    setOptions(taskFilter, Array.from(taskSet).sort(), "All Tasks");
    setOptions(modalityFilter, Array.from(modSet).sort(), "All Modalities");
  }

  // ---- Bind Events ----
  function initSorting() {
    if (!table) return;
    const ths = table.querySelectorAll("thead th.sortable");

    ths.forEach(th => {
      th.addEventListener("click", () => {
        const col = parseInt(th.getAttribute("data-column"), 10);
        if (Number.isNaN(col)) return;

        if (sortCol === col) {
          sortOrder = sortOrder === "asc" ? "desc" : "asc";
        } else {
          sortCol = col;
          sortOrder = "asc";
        }

        ths.forEach(x => x.removeAttribute("data-order"));
        th.setAttribute("data-order", sortOrder);

        applySearchAndFilters();
      });
    });
  }

  function initPagination() {
    if (firstBtn) firstBtn.addEventListener("click", () => { currentPage = 1; renderPage(); });
    if (prevBtn) prevBtn.addEventListener("click", () => { currentPage = Math.max(1, currentPage - 1); renderPage(); });

    if (nextBtn) nextBtn.addEventListener("click", () => {
      const totalPages = Math.max(1, Math.ceil(filteredRows.length / recordsPerPage));
      currentPage = Math.min(totalPages, currentPage + 1);
      renderPage();
    });

    if (lastBtn) lastBtn.addEventListener("click", () => {
      currentPage = Math.max(1, Math.ceil(filteredRows.length / recordsPerPage));
      renderPage();
    });

    if (perPageSelect) {
      perPageSelect.addEventListener("change", () => {
        recordsPerPage = parseInt(perPageSelect.value, 10) || 10;
        currentPage = 1;
        renderPage();
      });
    }
  }

  function initFilters() {
    if (searchInput) {
      searchInput.addEventListener("input", () => {
        currentPage = 1;
        applySearchAndFilters();
      });
    }

    [cognitiveFilter, taskFilter, modalityFilter].forEach(sel => {
      if (!sel) return;
      sel.addEventListener("change", () => {
        currentPage = 1;
        applySearchAndFilters();
      });
    });
  }

  window.clearDatasetFilters = function clearDatasetFilters() {
    if (searchInput) searchInput.value = "";
    if (cognitiveFilter) cognitiveFilter.value = "";
    if (taskFilter) taskFilter.value = "";
    if (modalityFilter) modalityFilter.value = "";
    currentPage = 1;
    applySearchAndFilters();
  };

  // ---- Load CSV ----
  function loadDatasetsCSV() {
    if (!tbody) return;

    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:32px;">Loading datasets...</td></tr>`;

    Papa.parse(DATASETS_CSV_PATH, {
      download: true,
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        try {
          const rows = (results.data || [])
            .map(normalizeRow)
            .filter(r => r.dataset);

          allRows = rows;
          filteredRows = [...allRows];
          syncTotalCount();

          fillFilterOptions();

          // default sort by Dataset asc
          sortCol = 0;
          sortOrder = "asc";
          sortFilteredRows();

          currentPage = 1;
          renderPage();
        } catch (e) {
          console.error("datasets.js parse error:", e);
          tbody.dataset.totalCount = "0";
          tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:32px;color:#dc2626;">
            Failed to load datasets. Check console.
          </td></tr>`;
        }
      },
      error: (err) => {
        console.error("PapaParse load error:", err);
        tbody.dataset.totalCount = "0";
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:32px;color:#dc2626;">
          Failed to load datasets.csv
        </td></tr>`;
      }
    });
  }

  // ---- Init ----
  document.addEventListener("DOMContentLoaded", () => {
    initSorting();
    initPagination();
    initFilters();
    loadDatasetsCSV();
  });
})();
