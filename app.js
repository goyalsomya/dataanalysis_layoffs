// Global State
let currentPage = 1;
let totalPages = 1;
let currentSortBy = 'total_laid_off';
let currentSortOrder = 'desc';

// Chart Instances
let chartTrends = null;
let chartCompanies = null;
let chartIndustry = null;
let chartCountry = null;
let chartStage = null;

document.addEventListener('DOMContentLoaded', () => {
  initFilters();
  loadAllDashboardData();
  loadEDA();
  setupEventListeners();
});

// Setup UI Event Listeners
function setupEventListeners() {
  document.getElementById('filter-company').addEventListener('change', (e) => {
    const co = e.target.value;
    const compChartSel = document.getElementById('comp-chart-select');
    if (compChartSel) compChartSel.value = co;
    onFilterChange();
  });

  const compChartSel = document.getElementById('comp-chart-select');
  if (compChartSel) {
    compChartSel.addEventListener('change', (e) => {
      const co = e.target.value;
      const sbComp = document.getElementById('filter-company');
      if (sbComp) sbComp.value = co;
      onFilterChange();
    });
  }
  
  // Year filter synchronization
  document.getElementById('filter-year').addEventListener('change', (e) => {
    const yr = e.target.value;
    const trendYr = document.getElementById('trend-year-select');
    if (trendYr) trendYr.value = yr;
    onFilterChange();
  });

  const trendYrSelect = document.getElementById('trend-year-select');
  if (trendYrSelect) {
    trendYrSelect.addEventListener('change', (e) => {
      const yr = e.target.value;
      const sbYr = document.getElementById('filter-year');
      if (sbYr) sbYr.value = yr;
      onFilterChange();
    });
  }

  const trendGroupSelect = document.getElementById('trend-groupby-select');
  if (trendGroupSelect) {
    trendGroupSelect.addEventListener('change', () => {
      loadTrendsChart();
    });
  }

  document.getElementById('filter-industry').addEventListener('change', onFilterChange);
  document.getElementById('filter-country').addEventListener('change', onFilterChange);
  document.getElementById('filter-stage').addEventListener('change', onFilterChange);

  // Column Filters Listeners
  ['col-filter-industry', 'col-filter-stage', 'col-filter-country', 'col-filter-year'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', onFilterChange);
  });

  let colSearchTimeout;
  ['col-filter-company', 'col-filter-laidoff'].forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', () => {
        clearTimeout(colSearchTimeout);
        colSearchTimeout = setTimeout(() => {
          currentPage = 1;
          loadAllDashboardData();
        }, 300);
      });
    }
  });

  let searchTimeout;
  document.getElementById('filter-search').addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      currentPage = 1;
      loadAllDashboardData();
    }, 300);
  });

  document.getElementById('btn-reset').addEventListener('click', () => {
    document.getElementById('filter-company').value = 'All';
    document.getElementById('filter-year').value = '0';
    document.getElementById('filter-industry').value = 'All';
    document.getElementById('filter-country').value = 'All';
    document.getElementById('filter-stage').value = 'All';
    document.getElementById('filter-search').value = '';

    const c1 = document.getElementById('col-filter-company'); if (c1) c1.value = '';
    const c2 = document.getElementById('col-filter-industry'); if (c2) c2.value = 'All';
    const c3 = document.getElementById('col-filter-laidoff'); if (c3) c3.value = '';
    const c4 = document.getElementById('col-filter-stage'); if (c4) c4.value = 'All';
    const c5 = document.getElementById('col-filter-country'); if (c5) c5.value = 'All';
    const c6 = document.getElementById('col-filter-year'); if (c6) c6.value = '0';

    currentPage = 1;
    loadAllDashboardData();
  });

  document.getElementById('btn-export').addEventListener('click', () => {
    const query = getFilterQueryString();
    window.open(`/api/export?${query}`, '_blank');
  });

  document.getElementById('btn-run-predict').addEventListener('click', runMLPrediction);

  document.getElementById('btn-prev-page').addEventListener('click', () => {
    if (currentPage > 1) {
      currentPage--;
      loadTableData();
    }
  });

  document.getElementById('btn-next-page').addEventListener('click', () => {
    if (currentPage < totalPages) {
      currentPage++;
      loadTableData();
    }
  });
}

function onFilterChange() {
  currentPage = 1;
  loadAllDashboardData();
}

function getFilterQueryString() {
  const company = document.getElementById('filter-company')?.value || 'All';
  const year = document.getElementById('filter-year')?.value || '0';
  const industry = document.getElementById('filter-industry')?.value || 'All';
  const country = document.getElementById('filter-country')?.value || 'All';
  const stage = document.getElementById('filter-stage')?.value || 'All';
  const search = document.getElementById('filter-search')?.value || '';
  
  // Table column filters
  const colCompany = document.getElementById('col-filter-company')?.value || '';
  const colIndustry = document.getElementById('col-filter-industry')?.value || 'All';
  const colLaidoff = document.getElementById('col-filter-laidoff')?.value || '';
  const colStage = document.getElementById('col-filter-stage')?.value || 'All';
  const colCountry = document.getElementById('col-filter-country')?.value || 'All';
  const colYear = document.getElementById('col-filter-year')?.value || '0';

  const params = new URLSearchParams();
  
  const finalComp = (company !== 'All') ? company : '';
  const finalSearch = colCompany || search || finalComp;
  if (finalSearch) params.append('search', finalSearch);

  const finalYear = (colYear !== '0') ? colYear : year;
  if (finalYear !== '0') params.append('year', finalYear);

  const finalInd = (colIndustry !== 'All') ? colIndustry : industry;
  if (finalInd !== 'All') params.append('industry', finalInd);

  const finalCountry = (colCountry !== 'All') ? colCountry : country;
  if (finalCountry !== 'All') params.append('country', finalCountry);

  const finalStage = (colStage !== 'All') ? colStage : stage;
  if (finalStage !== 'All') params.append('stage', finalStage);

  if (colLaidoff && Number(colLaidoff) > 0) {
    params.append('min_laidoff', colLaidoff);
  }

  return params.toString();
}

// Initialize Dropdowns from API
async function initFilters() {
  try {
    const res = await fetch('/api/filters');
    const data = await res.json();

    // Populate Company filter dropdowns
    const compSelect = document.getElementById('filter-company');
    const compChartSelect = document.getElementById('comp-chart-select');
    data.companies.forEach(co => {
      if (co !== 'All') {
        const opt = document.createElement('option');
        opt.value = co;
        opt.textContent = co;
        compSelect.appendChild(opt);

        if (compChartSelect) {
          const opt2 = document.createElement('option');
          opt2.value = co;
          opt2.textContent = co;
          compChartSelect.appendChild(opt2);
        }
      }
    });

    // Populate Year filter dropdowns
    const yrSelect = document.getElementById('filter-year');
    const colYrSelect = document.getElementById('col-filter-year');
    data.years.forEach(yr => {
      if (yr !== 0) {
        const opt = document.createElement('option');
        opt.value = yr;
        opt.textContent = yr;
        yrSelect.appendChild(opt);

        if (colYrSelect) {
          const opt2 = document.createElement('option');
          opt2.value = yr;
          opt2.textContent = yr;
          colYrSelect.appendChild(opt2);
        }
      }
    });

    const indSelect = document.getElementById('filter-industry');
    const colIndSelect = document.getElementById('col-filter-industry');
    const predInd = document.getElementById('pred-industry');
    data.industries.forEach(ind => {
      if (ind !== 'All') {
        const opt = document.createElement('option');
        opt.value = ind;
        opt.textContent = ind;
        indSelect.appendChild(opt);

        if (colIndSelect) {
          const optCol = document.createElement('option');
          optCol.value = ind;
          optCol.textContent = ind;
          colIndSelect.appendChild(optCol);
        }

        const optPred = document.createElement('option');
        optPred.value = ind;
        optPred.textContent = ind;
        predInd.appendChild(optPred);
      }
    });

    const cntSelect = document.getElementById('filter-country');
    const colCntSelect = document.getElementById('col-filter-country');
    const predCnt = document.getElementById('pred-country');
    data.countries.forEach(cnt => {
      if (cnt !== 'All') {
        const opt = document.createElement('option');
        opt.value = cnt;
        opt.textContent = cnt;
        cntSelect.appendChild(opt);

        if (colCntSelect) {
          const optCol = document.createElement('option');
          optCol.value = cnt;
          optCol.textContent = cnt;
          colCntSelect.appendChild(optCol);
        }

        const optPred = document.createElement('option');
        optPred.value = cnt;
        optPred.textContent = cnt;
        predCnt.appendChild(optPred);
      }
    });

    const stgSelect = document.getElementById('filter-stage');
    const colStgSelect = document.getElementById('col-filter-stage');
    const predStg = document.getElementById('pred-stage');
    data.stages.forEach(stg => {
      if (stg !== 'All') {
        const opt = document.createElement('option');
        opt.value = stg;
        opt.textContent = stg;
        stgSelect.appendChild(opt);

        if (colStgSelect) {
          const optCol = document.createElement('option');
          optCol.value = stg;
          optCol.textContent = stg;
          colStgSelect.appendChild(optCol);
        }

        const optPred = document.createElement('option');
        optPred.value = stg;
        optPred.textContent = stg;
        predStg.appendChild(optPred);
      }
    });
  } catch (err) {
    console.error("Failed to load filters:", err);
  }
}

// Main Data Fetch Coordinator
function loadAllDashboardData() {
  loadKPIs();
  loadInsights();
  loadTrendsChart();
  loadTopCompaniesChart();
  loadIndustryChart();
  loadCountryChart();
  loadStageChart();
  loadTableData();
}

// Fetch KPIs
async function loadKPIs() {
  try {
    const query = getFilterQueryString();
    const res = await fetch(`/api/kpis?${query}`);
    const data = await res.json();

    document.getElementById('kpi-total-laidoff').textContent = data.total_laid_off.toLocaleString();
    document.getElementById('kpi-companies').textContent = data.total_companies.toLocaleString();
    document.getElementById('kpi-countries').textContent = data.total_countries;
    document.getElementById('kpi-shutdowns').textContent = data.shutdown_count;
    document.getElementById('kpi-avg-layoff').textContent = data.avg_layoff_per_event.toLocaleString();
    document.getElementById('kpi-date-range').textContent = `${data.date_range.start} to ${data.date_range.end}`;
  } catch (err) {
    console.error("Failed to load KPIs:", err);
  }
}

// Fetch Automated Insights
async function loadInsights() {
  try {
    const res = await fetch('/api/insights');
    const data = await res.json();
    const container = document.getElementById('insights-list');
    container.innerHTML = '';
    
    data.insights.forEach(item => {
      const div = document.createElement('div');
      div.className = 'insight-item';
      div.innerHTML = item.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      container.appendChild(div);
    });
  } catch (err) {
    console.error("Failed to load insights:", err);
  }
}

// Fetch Exploratory Data Analysis (EDA) Statistics
async function loadEDA() {
  try {
    const res = await fetch('/api/eda');
    const data = await res.json();

    const tbody = document.getElementById('eda-tbody');
    tbody.innerHTML = '';

    const statsMap = data.descriptive_stats;
    for (const [col, s] of Object.entries(statsMap)) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight: 700; color: #fff;">${col}</td>
        <td>${s.count.toLocaleString()}</td>
        <td>${s.null_count.toLocaleString()} (${s.null_pct}%)</td>
        <td>${s.mean.toLocaleString()}</td>
        <td>${s.std.toLocaleString()}</td>
        <td>${s.min.toLocaleString()}</td>
        <td>${s.q25.toLocaleString()}</td>
        <td style="font-weight: 700; color: var(--primary);">${s.median.toLocaleString()}</td>
        <td>${s.q75.toLocaleString()}</td>
        <td>${s.max.toLocaleString()}</td>
        <td>${s.skewness}</td>
      `;
      tbody.appendChild(tr);
    }
  } catch (err) {
    console.error("Failed to load EDA:", err);
  }
}

// ML Prediction Handler
async function runMLPrediction() {
  const ind = document.getElementById('pred-industry').value;
  const stg = document.getElementById('pred-stage').value;
  const cnt = document.getElementById('pred-country').value;
  const funds = document.getElementById('pred-funds').value || 500;

  try {
    const url = `/api/predict?industry=${encodeURIComponent(ind)}&stage=${encodeURIComponent(stg)}&country=${encodeURIComponent(cnt)}&funds_raised=${funds}&year=2026`;
    const res = await fetch(url);
    const data = await res.json();

    document.getElementById('pred-result-number').textContent = data.predicted_layoffs.toLocaleString();
    const badge = document.getElementById('pred-result-badge');
    badge.textContent = data.risk_level;
    badge.style.backgroundColor = `${data.risk_color}25`;
    badge.style.borderColor = data.risk_color;
    badge.style.color = data.risk_color;

    document.getElementById('pred-result-sub').textContent = `Model Confidence (R²): ${(data.model_r2 * 100).toFixed(1)}%`;
  } catch (err) {
    console.error("Prediction failed:", err);
  }
}

// Chart 1: Trends Line/Area Chart
async function loadTrendsChart() {
  const query = getFilterQueryString();
  const groupBy = document.getElementById('trend-groupby-select')?.value || 'month';
  const res = await fetch(`/api/trends?${query}&group_by=${groupBy}`);
  const data = await res.json();

  const labels = data.map(d => d.label);
  const values = data.map(d => d.total_laid_off);

  const ctx = document.getElementById('chart-trends').getContext('2d');
  if (chartTrends) chartTrends.destroy();

  chartTrends = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Total Laid Off',
        data: values,
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56, 189, 248, 0.15)',
        fill: true,
        tension: 0.35,
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
      }
    }
  });
}

// Chart 2: Top Companies Bar Chart
async function loadTopCompaniesChart() {
  const query = getFilterQueryString();
  const res = await fetch(`/api/top-companies?${query}&limit=10`);
  const data = await res.json();

  const labels = data.map(d => d.company);
  const values = data.map(d => d.total_laid_off);

  const ctx = document.getElementById('chart-companies').getContext('2d');
  if (chartCompanies) chartCompanies.destroy();

  chartCompanies = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Headcount Laid Off',
        data: values,
        backgroundColor: '#f43f5e',
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
      }
    }
  });
}

// Chart 3: Industry Horizontal Bar Chart
async function loadIndustryChart() {
  const query = getFilterQueryString();
  const res = await fetch(`/api/by-industry?${query}&limit=10`);
  const data = await res.json();

  const labels = data.map(d => d.industry);
  const values = data.map(d => d.total_laid_off);

  const ctx = document.getElementById('chart-industry').getContext('2d');
  if (chartIndustry) chartIndustry.destroy();

  chartIndustry = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Total Laid Off',
        data: values,
        backgroundColor: '#a855f7',
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
        y: { grid: { display: false }, ticks: { color: '#94a3b8' } }
      }
    }
  });
}

// Chart 4: Country Doughnut Chart
async function loadCountryChart() {
  const query = getFilterQueryString();
  const res = await fetch(`/api/by-country?${query}&limit=8`);
  const data = await res.json();

  const labels = data.map(d => d.country);
  const values = data.map(d => d.total_laid_off);

  const ctx = document.getElementById('chart-country').getContext('2d');
  if (chartCountry) chartCountry.destroy();

  chartCountry = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: values,
        backgroundColor: [
          '#38bdf8', '#a855f7', '#f43f5e', '#f59e0b', '#10b981', '#ec4899', '#6366f1', '#84cc16'
        ],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: 'right', labels: { color: '#cbd5e1', boxWidth: 12 } }
      }
    }
  });
}

// Chart 5: Funding Stage Bar Chart
async function loadStageChart() {
  const query = getFilterQueryString();
  const res = await fetch(`/api/by-stage?${query}`);
  const data = await res.json();

  const labels = data.map(d => d.stage);
  const values = data.map(d => d.total_laid_off);

  const ctx = document.getElementById('chart-stage').getContext('2d');
  if (chartStage) chartStage.destroy();

  chartStage = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Total Laid Off',
        data: values,
        backgroundColor: '#10b981',
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
      }
    }
  });
}

// Fetch Paginated Table
async function loadTableData() {
  const query = getFilterQueryString();
  const url = `/api/layoffs?${query}&page=${currentPage}&page_size=12&sort_by=${currentSortBy}&sort_order=${currentSortOrder}`;
  
  try {
    const res = await fetch(url);
    const data = await res.json();

    totalPages = data.total_pages;
    const tbody = document.getElementById('table-body');
    tbody.innerHTML = '';

    if (data.data.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-muted);">No records found matching filters.</td></tr>';
    } else {
      data.data.forEach(row => {
        const tr = document.createElement('tr');
        
        const laidoffVal = row.total_laid_off ? Number(row.total_laid_off).toLocaleString() : 'N/A';
        const pctVal = row.percentage_laid_off ? `${(Number(row.percentage_laid_off) * 100).toFixed(0)}%` : 'N/A';

        tr.innerHTML = `
          <td style="font-weight: 600; color: #fff;">${escapeHtml(row.company)}</td>
          <td>${escapeHtml(row.industry)}</td>
          <td style="font-weight: 700; color: var(--accent-red);">${laidoffVal}</td>
          <td>${pctVal}</td>
          <td><span class="badge-stage">${escapeHtml(row.stage)}</span></td>
          <td><span class="badge-country">${escapeHtml(row.country)}</span></td>
          <td>${row.date_str}</td>
        `;
        tbody.appendChild(tr);
      });
    }

    document.getElementById('table-info').textContent = `Showing page ${data.page} of ${data.total_pages} (${data.total_records} total records)`;
    document.getElementById('page-indicator').textContent = `Page ${data.page} of ${data.total_pages}`;
    document.getElementById('btn-prev-page').disabled = (data.page <= 1);
    document.getElementById('btn-next-page').disabled = (data.page >= data.total_pages);

  } catch (err) {
    console.error("Failed to load table data:", err);
  }
}

function sortTable(column) {
  if (currentSortBy === column) {
    currentSortOrder = (currentSortOrder === 'desc') ? 'asc' : 'desc';
  } else {
    currentSortBy = column;
    currentSortOrder = 'desc';
  }
  currentPage = 1;
  loadTableData();
}

function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
