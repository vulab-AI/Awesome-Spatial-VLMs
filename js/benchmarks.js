/* ============================
   Benchmarks Table (stable)
   Filters:
   - Search
   - Cognitive Level
   - Fundamental Task   ✅ replaced Venue
   - Modality
   ============================ */

const BENCHMARKS_CSV = "./dev/spatial_benchmarks.csv"; // ✅改成你的 benchmark csv 路径

let benchmarkRaw = [];
let benchmarkFiltered = [];

let benchmarkSort = { col: null, order: "asc" };

let benchmarkPage = 1;
let benchmarkPerPage = 10;

function syncBenchmarkTotalCount() {
  const tbody = document.getElementById("benchmarkTableBody");
  if (tbody) tbody.dataset.totalCount = String(benchmarkRaw.length);
}

function safeStr(v) {
  if (v === null || v === undefined) return "";
  return String(v).trim();
}

function splitBySemicolon(value) {
  return safeStr(value)
    .split(";")
    .map(s => s.trim())
    .filter(Boolean);
}

function splitByComma(value) {
  return safeStr(value)
    .split(",")
    .map(s => s.trim())
    .filter(Boolean);
}

function renderAtomicText(list) {
  if (!list || list.length === 0) return "";
  return list.join("，"); // ✅ 你要求 atomic 之间用中文逗号分开
}

function renderLinkCell(url) {
  const link = safeStr(url);
  if (!link) return "";
  return `<a href="${link}" target="_blank" rel="noopener noreferrer">Open ↗</a>`;
}

function normalizeBenchmarkRow(row) {
  const name = safeStr(row["Benchmark"] || row["Dataset"]);

  const venue = safeStr(row["Venue"]);
  const size = safeStr(row["Size"]);
  const link = safeStr(row["Link"]);

  const cognitiveList = splitBySemicolon(row["Cognitive Level"]); // ✅ ;
  const taskList = splitByComma(row["Fundamental Task"]);         // ✅ ,
  const imageSourceList = splitByComma(row["Image Source"]);      // ✅ ,
  const modalityList = splitByComma(row["Modality"]);             // ✅ ,

  return {
    Benchmark: name,
    Venue: venue,
    CognitiveLevelList: cognitiveList,
    FundamentalTaskList: taskList,
    Size: size,
    ImageSourceList: imageSourceList,
    ModalityList: modalityList,
    Link: link,
  };
}

function matchAtomicFilter(list, selected) {
  const sel = safeStr(selected);
  if (!sel) return true;
  return list.some(x => x.toLowerCase() === sel.toLowerCase());
}

function matchSearch(row, keyword) {
  const q = safeStr(keyword).toLowerCase();
  if (!q) return true;

  const hay = [
    row.Benchmark,
    row.Venue,
    row.Size,
    row.Link,
    row.CognitiveLevelList.join(" "),
    row.FundamentalTaskList.join(" "),
    row.ImageSourceList.join(" "),
    row.ModalityList.join(" "),
  ].join(" ").toLowerCase();

  return hay.includes(q);
}

function fillSelectOptions(selectId, values) {
  const sel = document.getElementById(selectId);
  if (!sel) return;

  const first = sel.querySelector("option[value='']");
  sel.innerHTML = "";
  if (first) sel.appendChild(first);

  const uniq = Array.from(new Set(values.filter(Boolean))).sort((a, b) => a.localeCompare(b));
  for (const v of uniq) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  }
}

function renderBenchmarkTable() {
  const tbody = document.getElementById("benchmarkTableBody");
  if (!tbody) return;

  const start = (benchmarkPage - 1) * benchmarkPerPage;
  const end = start + benchmarkPerPage;
  const pageRows = benchmarkFiltered.slice(start, end);

  if (benchmarkFiltered.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="8" style="text-align:center;padding:28px;color:#57606a;">
          No benchmarks found.
        </td>
      </tr>
    `;
    updateBenchmarkPagination();
    return;
  }

  tbody.innerHTML = pageRows.map(r => `
    <tr>
      <td>${safeStr(r.Benchmark)}</td>
      <td>${safeStr(r.Venue)}</td>
      <td>${renderAtomicText(r.CognitiveLevelList)}</td>
      <td>${renderAtomicText(r.FundamentalTaskList)}</td>
      <td>${safeStr(r.Size)}</td>
      <td>${renderAtomicText(r.ImageSourceList)}</td>
      <td>${renderAtomicText(r.ModalityList)}</td>
      <td style="white-space:nowrap;text-align:center;">${renderLinkCell(r.Link)}</td>
    </tr>
  `).join("");

  updateBenchmarkPagination();
}

function updateBenchmarkPagination() {
  const total = benchmarkFiltered.length;
  const totalPages = Math.max(1, Math.ceil(total / benchmarkPerPage));

  const startRecord = total === 0 ? 0 : (benchmarkPage - 1) * benchmarkPerPage + 1;
  const endRecord = Math.min(total, benchmarkPage * benchmarkPerPage);

  document.getElementById("benchmarkStartRecord").textContent = startRecord;
  document.getElementById("benchmarkEndRecord").textContent = endRecord;
  document.getElementById("benchmarkTotalRecords").textContent = total;
  document.getElementById("benchmarkCurrentPage").textContent = benchmarkPage;
  document.getElementById("benchmarkTotalPages").textContent = totalPages;

  document.getElementById("benchmarkFirstPage").disabled = benchmarkPage <= 1;
  document.getElementById("benchmarkPrevPage").disabled = benchmarkPage <= 1;
  document.getElementById("benchmarkNextPage").disabled = benchmarkPage >= totalPages;
  document.getElementById("benchmarkLastPage").disabled = benchmarkPage >= totalPages;
}

function applyBenchmarkFilters() {
  const searchVal = safeStr(document.getElementById("benchmarkSearchInput")?.value);
  const cognitiveVal = safeStr(document.getElementById("cognitiveFilter")?.value);
  const taskVal = safeStr(document.getElementById("benchmarkTaskFilter")?.value); // ✅ new
  const modalityVal = safeStr(document.getElementById("modalityFilter")?.value);

  benchmarkFiltered = benchmarkRaw.filter(r => {
    const okSearch = matchSearch(r, searchVal);
    const okCog = matchAtomicFilter(r.CognitiveLevelList, cognitiveVal);
    const okTask = matchAtomicFilter(r.FundamentalTaskList, taskVal); // ✅ new
    const okMod = matchAtomicFilter(r.ModalityList, modalityVal);
    return okSearch && okCog && okTask && okMod;
  });

  benchmarkPage = 1;
  renderBenchmarkTable();
}

function sortBenchmarkByColumn(colIndex) {
  const map = {
    0: "Benchmark",
    1: "Venue",
    2: "Cognitive",
    3: "Task",
    4: "Size",
    5: "Source",
    6: "Modality",
  };

  const key = map[colIndex];
  if (!key) return;

  if (benchmarkSort.col === colIndex) {
    benchmarkSort.order = benchmarkSort.order === "asc" ? "desc" : "asc";
  } else {
    benchmarkSort.col = colIndex;
    benchmarkSort.order = "asc";
  }

  const dir = benchmarkSort.order === "asc" ? 1 : -1;

  benchmarkFiltered.sort((a, b) => {
    const getVal = (r) => {
      if (key === "Benchmark") return safeStr(r.Benchmark);
      if (key === "Venue") return safeStr(r.Venue);
      if (key === "Size") return safeStr(r.Size);
      if (key === "Cognitive") return r.CognitiveLevelList.join("，");
      if (key === "Task") return r.FundamentalTaskList.join("，");
      if (key === "Source") return r.ImageSourceList.join("，");
      if (key === "Modality") return r.ModalityList.join("，");
      return "";
    };
    return getVal(a).localeCompare(getVal(b)) * dir;
  });

  renderBenchmarkTable();
}

function clearBenchmarkFilters() {
  const ids = ["benchmarkSearchInput", "cognitiveFilter", "benchmarkTaskFilter", "modalityFilter"]; // ✅ updated
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = "";
  });
  applyBenchmarkFilters();
}

function initBenchmarks() {
  const tbody = document.getElementById("benchmarkTableBody");
  if (tbody) {
    tbody.innerHTML = `
      <tr><td colspan="8" style="text-align:center;padding:28px;">Loading benchmarks...</td></tr>
    `;
  }

  Papa.parse(BENCHMARKS_CSV, {
    download: true,
    header: true,
    skipEmptyLines: true,
    complete: (results) => {
      const rows = results.data || [];

      benchmarkRaw = rows
        .map(normalizeBenchmarkRow)
        .filter(r => r.Benchmark);
      syncBenchmarkTotalCount();

      console.log("[Benchmarks] loaded:", benchmarkRaw.length, "rows");

      // ✅ dropdown values: Fundamental Task + Modality
      const taskList = benchmarkRaw.flatMap(r => r.FundamentalTaskList).filter(Boolean);
      const modalityList = benchmarkRaw.flatMap(r => r.ModalityList).filter(Boolean);

      fillSelectOptions("benchmarkTaskFilter", taskList);
      fillSelectOptions("modalityFilter", modalityList);

      benchmarkFiltered = [...benchmarkRaw];

      const perSel = document.getElementById("benchmarkRecordsPerPage");
      if (perSel) benchmarkPerPage = parseInt(perSel.value || "10", 10);

      renderBenchmarkTable();
    },
    error: (err) => {
      console.error("[Benchmarks] CSV load failed:", err);
      if (tbody) tbody.dataset.totalCount = "0";
      if (tbody) {
        tbody.innerHTML = `
          <tr>
            <td colspan="8" style="text-align:center;padding:28px;color:#dc2626;">
              Failed to load benchmarks CSV: ${safeStr(err?.message || err)}
            </td>
          </tr>
        `;
      }
    }
  });

  // bind filters
  document.getElementById("benchmarkSearchInput")?.addEventListener("input", applyBenchmarkFilters);
  document.getElementById("cognitiveFilter")?.addEventListener("change", applyBenchmarkFilters);
  document.getElementById("benchmarkTaskFilter")?.addEventListener("change", applyBenchmarkFilters); // ✅ new
  document.getElementById("modalityFilter")?.addEventListener("change", applyBenchmarkFilters);

  // pagination
  document.getElementById("benchmarkFirstPage")?.addEventListener("click", () => {
    benchmarkPage = 1; renderBenchmarkTable();
  });
  document.getElementById("benchmarkPrevPage")?.addEventListener("click", () => {
    benchmarkPage = Math.max(1, benchmarkPage - 1); renderBenchmarkTable();
  });
  document.getElementById("benchmarkNextPage")?.addEventListener("click", () => {
    const totalPages = Math.max(1, Math.ceil(benchmarkFiltered.length / benchmarkPerPage));
    benchmarkPage = Math.min(totalPages, benchmarkPage + 1); renderBenchmarkTable();
  });
  document.getElementById("benchmarkLastPage")?.addEventListener("click", () => {
    benchmarkPage = Math.max(1, Math.ceil(benchmarkFiltered.length / benchmarkPerPage));
    renderBenchmarkTable();
  });

  document.getElementById("benchmarkRecordsPerPage")?.addEventListener("change", (e) => {
    benchmarkPerPage = parseInt(e.target.value || "10", 10);
    benchmarkPage = 1;
    renderBenchmarkTable();
  });

  // sorting
  document.querySelectorAll("#benchmarkTable thead th.sortable")?.forEach(th => {
    th.addEventListener("click", () => {
      const col = parseInt(th.getAttribute("data-column"), 10);
      sortBenchmarkByColumn(col);
    });
  });
}

document.addEventListener("DOMContentLoaded", initBenchmarks);
