import { useEffect, useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Activity,
  ArrowDownToLine,
  BarChart3,
  Check,
  ChevronDown,
  CircleHelp,
  Clipboard,
  Dices,
  FileSpreadsheet,
  FlaskConical,
  RefreshCw,
  Sigma,
  Sparkles,
  Table2,
} from 'lucide-react'
import './App.css'
import {
  analyzeData,
  calculateDistribution,
  calculateTest,
  type AnalysisResult,
  type DistributionResult,
  type StatisticalTestResult,
} from './api'

type InputMode = 'csv' | 'manual' | 'random'
type DistributionType = 'normal' | 'binomial' | 'poisson'
type TestType = 'ttest' | 'chi_square'

const initialCsv = `study_hours,exam_score
1.2,48
1.8,52
2.1,56
2.7,59
3.2,64
3.8,67
4.1,70
4.8,74
5.1,78
5.7,82
6.2,86
6.8,89
7.2,94
7.8,96
8.3,98`

const initialManualX = '1.2, 1.8, 2.1, 2.7, 3.2, 3.8, 4.1, 4.8, 5.1, 5.7, 6.2, 6.8, 7.2, 7.8, 8.3'
const initialManualY = '48, 52, 56, 59, 64, 67, 70, 74, 78, 82, 86, 89, 94, 96, 98'

function parseDelimited(text: string): Record<string, number[]> {
  const firstLine = text.split(/\r?\n/, 1)[0] ?? ''
  const delimiter = firstLine.includes('\t') ? '\t' : firstLine.includes(';') ? ';' : ','
  const records: string[][] = []
  let row: string[] = []
  let cell = ''
  let quoted = false

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index]
    if (character === '"' && text[index + 1] === '"' && quoted) {
      cell += '"'
      index += 1
    } else if (character === '"') {
      quoted = !quoted
    } else if (character === delimiter && !quoted) {
      row.push(cell.trim())
      cell = ''
    } else if ((character === '\n' || character === '\r') && !quoted) {
      if (character === '\r' && text[index + 1] === '\n') index += 1
      row.push(cell.trim())
      if (row.some((value) => value !== '')) records.push(row)
      row = []
      cell = ''
    } else {
      cell += character
    }
  }
  row.push(cell.trim())
  if (row.some((value) => value !== '')) records.push(row)

  if (records.length < 2) throw new Error('Add a header and at least one data row.')
  const headers = records[0].map((header, index) => header || `variable_${index + 1}`)
  if (new Set(headers).size !== headers.length) throw new Error('Each column must have a unique name.')

  const columns = Object.fromEntries(headers.map((header) => [header, [] as number[]]))
  let skippedRows = 0
  for (const record of records.slice(1)) {
    if (record.length !== headers.length) {
      skippedRows += 1
      continue
    }
    const values = record.map((value) => Number(value.replace(',', '.')))
    if (values.some((value) => !Number.isFinite(value))) {
      skippedRows += 1
      continue
    }
    headers.forEach((header, index) => columns[header].push(values[index]))
  }
  if (Object.values(columns).some((values) => values.length < 2)) {
    throw new Error('At least two complete numeric rows are required.')
  }
  if (skippedRows > 0) {
    throw new Error(`${skippedRows} row(s) have missing, non-numeric, or mismatched values. Fix the data and try again.`)
  }
  return columns
}

function parseManualValues(text: string): number[] {
  const values = text.split(/[\s,;]+/).filter(Boolean).map((value) => Number(value.replace(',', '.')))
  if (values.length < 2 || values.some((value) => !Number.isFinite(value))) {
    throw new Error('Enter at least two valid numeric values in each column.')
  }
  return values
}

function formatNumber(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? '—'
    : new Intl.NumberFormat('en-US', { maximumFractionDigits: digits }).format(value)
}

function App() {
  const [mode, setMode] = useState<InputMode>('csv')
  const [csv, setCsv] = useState(initialCsv)
  const [manualX, setManualX] = useState(initialManualX)
  const [manualY, setManualY] = useState(initialManualY)
  const [xName, setXName] = useState('study_hours')
  const [yName, setYName] = useState('exam_score')
  const [randomRows, setRandomRows] = useState(50)
  const [randomColumns, setRandomColumns] = useState(3)
  const [columns, setColumns] = useState<Record<string, number[]>>({})
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [target, setTarget] = useState('')
  const [predictors, setPredictors] = useState<string[]>([])
  const [selectedColumn, setSelectedColumn] = useState('')
  const [scatterColumn, setScatterColumn] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'regression' | 'distribution' | 'tests'>('overview')
  const [distributionType, setDistributionType] = useState<DistributionType>('normal')
  const [distributionParams, setDistributionParams] = useState({ mean: '0', std: '1', n: '20', p: '0.5', lambda: '5' })
  const [distribution, setDistribution] = useState<DistributionResult | null>(null)
  const [testType, setTestType] = useState<TestType>('ttest')
  const [testValuesA, setTestValuesA] = useState('12, 14, 15, 11, 13, 16')
  const [testValuesB, setTestValuesB] = useState('10, 11, 12, 13, 9, 12')
  const [expectedCounts, setExpectedCounts] = useState('')
  const [testResult, setTestResult] = useState<StatisticalTestResult | null>(null)

  const columnNames = Object.keys(columns)
  const activeColumn = selectedColumn || columnNames[0] || ''
  const activeScatterColumn = scatterColumn || predictors[0] || columnNames.find((column) => column !== target) || ''
  const regression = analysis?.regression ?? null
  const statistics = activeColumn ? analysis?.statistics[activeColumn] : undefined
  const histogram = activeColumn ? analysis?.histograms[activeColumn] : undefined

  const scatterData = useMemo(() => {
    if (!analysis || !regression || !target || !activeScatterColumn) return null
    const values = columns[activeScatterColumn] ?? []
    const coefficient = regression.coefficients[activeScatterColumn]
    if (coefficient === undefined) return null
    const otherPredictorMean = regression.predictors
      .filter((name) => name !== activeScatterColumn)
      .reduce((sum, name) => {
        const predictorValues = columns[name] ?? []
        const mean = predictorValues.reduce((total, value) => total + value, 0) / Math.max(predictorValues.length, 1)
        return sum + (regression.coefficients[name] ?? 0) * mean
      }, 0)
    const min = Math.min(...values)
    const max = Math.max(...values)
    const line = [min, max].map((x) => ({
      x,
      y: regression.intercept + coefficient * x + otherPredictorMean,
    }))
    return { points: values.map((x, index) => ({ x, y: columns[target][index] })), line }
  }, [analysis, regression, target, activeScatterColumn, columns])

  async function runAnalysis(nextColumns = columns) {
    const names = Object.keys(nextColumns)
    if (names.length === 0) {
      setError('Add data before running the analysis.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const nextTarget = names.includes(target) ? target : names[names.length - 1]
      const chosenPredictors = predictors.filter((name) => names.includes(name) && name !== nextTarget)
      const nextPredictors = chosenPredictors.length
        ? chosenPredictors
        : names.filter((name) => name !== nextTarget).slice(0, 1)
      const result = await analyzeData(nextColumns, nextTarget, nextPredictors)
      setColumns(nextColumns)
      setAnalysis(result)
      setTarget(nextTarget)
      setPredictors(nextPredictors)
      setSelectedColumn((current) => names.includes(current) ? current : names[0])
      setScatterColumn((current) => names.includes(current) && current !== nextTarget ? current : names.find((name) => name !== nextTarget) ?? '')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The analysis could not be calculated.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const initial = parseDelimited(initialCsv)
    void runAnalysis(initial)
    // Load the sample dataset once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function readInput(): Record<string, number[]> {
    if (mode === 'csv') return parseDelimited(csv)
    if (mode === 'manual') {
      const x = parseManualValues(manualX)
      const y = parseManualValues(manualY)
      if (x.length !== y.length) throw new Error('The X and Y columns must contain the same number of values.')
      const names = [xName.trim() || 'X', yName.trim() || 'Y']
      if (names[0] === names[1]) throw new Error('Choose different names for the X and Y columns.')
      return { [names[0]]: x, [names[1]]: y }
    }
    const rows = Math.min(500, Math.max(5, randomRows))
    const generated: Record<string, number[]> = {}
    for (let index = 1; index <= Math.min(6, Math.max(2, randomColumns)); index += 1) {
      generated[`variable_${index}`] = Array.from({ length: rows }, () => Number((Math.random() * 100).toFixed(2)))
    }
    const first = Object.values(generated)[0]
    const response = first.map((value) => Number((10 + value * 0.65 + (Math.random() - 0.5) * 20).toFixed(2)))
    generated.outcome = response
    return generated
  }

  async function handleAnalyze() {
    try {
      const nextColumns = readInput()
      setTarget('')
      setPredictors([])
      await runAnalysis(nextColumns)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The input data is invalid.')
    }
  }

  function updateTarget(nextTarget: string) {
    setTarget(nextTarget)
    const nextPredictors = columnNames.filter((name) => name !== nextTarget).slice(0, Math.max(1, predictors.length))
    setPredictors(nextPredictors)
    void analyzeData(columns, nextTarget, nextPredictors)
      .then(setAnalysis)
      .catch((cause: unknown) => setError(cause instanceof Error ? cause.message : 'The regression could not be calculated.'))
  }

  function togglePredictor(name: string) {
    const nextPredictors = predictors.includes(name)
      ? predictors.filter((item) => item !== name)
      : [...predictors, name]
    setPredictors(nextPredictors)
    void analyzeData(columns, target, nextPredictors)
      .then(setAnalysis)
      .catch((cause: unknown) => setError(cause instanceof Error ? cause.message : 'The regression could not be calculated.'))
  }

  async function runDistribution() {
    try {
      setError('')
      setDistribution(await calculateDistribution(distributionType, distributionParams))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The distribution could not be calculated.')
    }
  }

  async function runTest() {
    try {
      setError('')
      const valuesA = parseManualValues(testValuesA)
      const valuesB = testType === 'ttest' ? parseManualValues(testValuesB) : undefined
      const expected = testType === 'chi_square' && expectedCounts.trim()
        ? parseManualValues(expectedCounts)
        : undefined
      setTestResult(await calculateTest(testType, valuesA, valuesB, expected))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The test could not be calculated.')
    }
  }

  async function copySummary() {
    if (!statistics || !activeColumn) return
    const text = [
      `Analysis: ${activeColumn}`,
      `Value count: ${statistics.count}`,
      `Mean: ${formatNumber(statistics.mean)}`,
      `Median: ${formatNumber(statistics.median)}`,
      `Standard deviation: ${formatNumber(statistics.std_dev)}`,
      `R²: ${formatNumber(regression?.r_squared)}`,
    ].join('\n')
    await navigator.clipboard.writeText(text)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1600)
  }

  function downloadCsv() {
    const header = columnNames.join(',')
    const rowCount = columns[columnNames[0]]?.length ?? 0
    const lines = Array.from({ length: rowCount }, (_, rowIndex) =>
      columnNames.map((name) => columns[name][rowIndex]).join(','),
    )
    const blob = new Blob([[header, ...lines].join('\n')], { type: 'text/csv;charset=utf-8' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = 'statistical-analysis.csv'
    link.click()
    URL.revokeObjectURL(link.href)
  }

  const tabs = [
    { id: 'overview' as const, label: 'Overview', icon: BarChart3 },
    { id: 'regression' as const, label: 'Regression', icon: Activity },
    { id: 'distribution' as const, label: 'Distributions', icon: Sigma },
    { id: 'tests' as const, label: 'Statistical tests', icon: FlaskConical },
  ]

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="StatLab home">
          <span className="brand-mark"><Activity size={19} strokeWidth={2.4} /></span>
          <span>stat<span className="brand-accent">lab</span></span>
        </a>
        <div className="topbar-right">
          <span className="engine-status"><span /> Python engine connected</span>
          <button className="icon-button help-button" aria-label="Help" title="Analyses are calculated by the Python backend">
            <CircleHelp size={18} />
          </button>
          <div className="avatar">A</div>
        </div>
      </header>

      <main id="top" className="workspace">
        <aside className="sidebar">
          <div className="sidebar-label">ANALYSIS</div>
          <button className="sidebar-link active"><BarChart3 size={17} /> New analysis</button>
          <button className="sidebar-link" onClick={() => setActiveTab('regression')}><Activity size={17} /> Regression</button>
          <button className="sidebar-link" onClick={() => setActiveTab('distribution')}><Sigma size={17} /> Distributions</button>
          <button className="sidebar-link" onClick={() => setActiveTab('tests')}><FlaskConical size={17} /> Statistical tests</button>
          <div className="sidebar-divider" />
          <div className="sidebar-label">YOUR DATA</div>
          <div className="dataset-info">
            <span className="dataset-icon"><Table2 size={16} /></span>
            <span className="dataset-copy"><strong>Active dataset</strong><small>{analysis?.rows ?? 0} rows · {columnNames.length} columns</small></span>
            <ChevronDown size={14} />
          </div>
          <div className="sidebar-note">
            <Sparkles size={17} />
            <strong>Insight without limits</strong>
            <p>Bring your data and discover the stories behind it.</p>
          </div>
          <div className="sidebar-footer">StatLab <span>v1.0</span></div>
        </aside>

        <section className="main-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow"><span className="eyebrow-dot" /> ANALYSIS WORKSPACE</div>
              <h1>Data analysis</h1>
              <p className="page-subtitle">Understand your data. Discover what matters.</p>
            </div>
            <div className="heading-actions">
              <button className="button button-secondary" onClick={copySummary} disabled={!statistics}>
                {copied ? <Check size={16} /> : <Clipboard size={16} />}
                {copied ? 'Copied' : 'Copy summary'}
              </button>
              <button className="button button-secondary" onClick={downloadCsv} disabled={columnNames.length === 0}>
                <ArrowDownToLine size={16} /> Export CSV
              </button>
            </div>
          </div>

          <section className="data-card">
            <div className="section-title-row">
              <div className="section-heading">
                <span className="section-icon"><FileSpreadsheet size={17} /></span>
                <div><h2>Data source</h2><p>Choose how to add your data</p></div>
              </div>
              <div className="source-tabs" role="tablist" aria-label="Data input method">
                <button className={mode === 'csv' ? 'source-tab selected' : 'source-tab'} onClick={() => setMode('csv')}><Clipboard size={15} /> Paste CSV</button>
                <button className={mode === 'manual' ? 'source-tab selected' : 'source-tab'} onClick={() => setMode('manual')}><Table2 size={15} /> Manual entry</button>
                <button className={mode === 'random' ? 'source-tab selected' : 'source-tab'} onClick={() => setMode('random')}><Dices size={15} /> Generate</button>
              </div>
            </div>

            {mode === 'csv' && (
              <div className="csv-input-wrap">
                <textarea value={csv} onChange={(event) => setCsv(event.target.value)} aria-label="CSV data" spellCheck={false} />
                <div className="input-hint"><span><CircleHelp size={13} /> The first row must contain column names</span><span>CSV · TSV · ; delimiter</span></div>
              </div>
            )}
            {mode === 'manual' && (
              <div className="manual-grid">
                <label className="field-label">Column names
                  <div className="dual-field"><input value={xName} onChange={(event) => setXName(event.target.value)} aria-label="X column name" /><input value={yName} onChange={(event) => setYName(event.target.value)} aria-label="Y column name" /></div>
                </label>
                <label className="field-label">X values
                  <textarea value={manualX} onChange={(event) => setManualX(event.target.value)} aria-label="X values" />
                </label>
                <label className="field-label">Y values
                  <textarea value={manualY} onChange={(event) => setManualY(event.target.value)} aria-label="Y values" />
                </label>
                <p className="input-hint manual-hint">Separate values with commas, spaces, or new lines. Use Paste CSV for more columns.</p>
              </div>
            )}
            {mode === 'random' && (
              <div className="random-panel">
                <div className="random-field"><label className="field-label" htmlFor="random-rows">Number of rows</label><input id="random-rows" type="number" min="5" max="500" value={randomRows} onChange={(event) => setRandomRows(Number(event.target.value))} /></div>
                <div className="random-field"><label className="field-label" htmlFor="random-columns">Generated predictors</label><input id="random-columns" type="number" min="1" max="5" value={randomColumns} onChange={(event) => setRandomColumns(Number(event.target.value))} /></div>
                <p>Random data with an outcome variable, generated locally in your browser.</p>
              </div>
            )}
            {error && <div className="error-banner" role="alert">{error}</div>}
            <div className="data-card-footer">
              <span className="data-meta">{analysis ? <><span className="live-dot" /> {analysis.rows} observations · {columnNames.length} variables</> : 'Prepare your data for analysis'}</span>
              <button className="button button-primary" onClick={handleAnalyze} disabled={loading}>
                {loading ? <RefreshCw className="spin" size={16} /> : <Sparkles size={16} />}
                {loading ? 'Analyzing…' : 'Analyze data'}
              </button>
            </div>
          </section>

          <nav className="analysis-tabs" aria-label="Analysis sections">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button key={id} className={activeTab === id ? 'analysis-tab active' : 'analysis-tab'} onClick={() => setActiveTab(id)}>
                <Icon size={16} /> {label}
              </button>
            ))}
          </nav>

          {!analysis && !loading && <div className="empty-state">Add a dataset to view the analysis.</div>}
          {activeTab === 'overview' && analysis && (
            <>
              <div className="section-title-row results-title">
                <div><h2>Descriptive statistics</h2><p>An overview of your data distribution.</p></div>
                <label className="select-wrap compact-select"><span>Variable</span><select value={activeColumn} onChange={(event) => setSelectedColumn(event.target.value)}>{columnNames.map((name) => <option key={name}>{name}</option>)}</select><ChevronDown size={14} /></label>
              </div>
              <div className="metrics-grid">
                <MetricCard label="MEAN" value={formatNumber(statistics?.mean)} note="average value" color="purple" icon={<Sigma size={16} />} />
                <MetricCard label="MEDIAN" value={formatNumber(statistics?.median)} note="50th percentile" color="blue" icon={<Activity size={16} />} />
                <MetricCard label="STD. DEV." value={formatNumber(statistics?.std_dev)} note="value dispersion" color="orange" icon={<Activity size={16} />} />
                <MetricCard label="OUTLIERS" value={String(statistics?.outliers.length ?? 0)} note="detected using IQR" color="pink" icon={<Sparkles size={16} />} />
              </div>
              <div className="chart-grid">
                <section className="panel chart-panel">
                  <div className="panel-heading"><div><h3>Value distribution</h3><p>Histogram · {activeColumn}</p></div><button className="icon-button" aria-label="Histogram information" title="Observation counts grouped into bins"><CircleHelp size={16} /></button></div>
                  {histogram && <div className="histogram-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={histogram.bins} margin={{ top: 12, right: 12, bottom: 2, left: -16 }}><CartesianGrid vertical={false} stroke="#edf0f5" /><XAxis dataKey="label" tick={{ fill: '#9299a8', fontSize: 10 }} axisLine={false} tickLine={false} interval="preserveStartEnd" /><YAxis allowDecimals={false} tick={{ fill: '#9299a8', fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip cursor={{ fill: '#f5f4ff' }} contentStyle={{ border: '1px solid #e8e8f0', borderRadius: 10, fontSize: 12 }} /><Bar dataKey="count" name="Frequency" fill="#7668ee" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div>}
                  <div className="chart-footer"><span>Range</span><strong>{formatNumber(statistics?.min)} — {formatNumber(statistics?.max)}</strong></div>
                </section>
                <section className="panel chart-panel">
                  <div className="panel-heading"><div><h3>Box plot</h3><p>Quartiles and extreme values</p></div><span className="chart-badge">IQR</span></div>
                  {statistics && <BoxPlot stats={statistics} />}
                  <div className="quartile-row"><div><small>Q1</small><strong>{formatNumber(statistics?.q1)}</strong></div><div><small>MEDIAN</small><strong>{formatNumber(statistics?.median)}</strong></div><div><small>Q3</small><strong>{formatNumber(statistics?.q3)}</strong></div></div>
                </section>
              </div>
              <section className="panel details-panel">
                <div className="panel-heading"><div><h3>Statistical summary</h3><p>Measures of central tendency and dispersion</p></div></div>
                <div className="stats-table">
                  {[
                    ['Observation count', statistics?.count],
                    ['Mode', statistics?.mode?.join(', ') || 'No unique mode'],
                    ['Variance (sample)', formatNumber(statistics?.variance)],
                    ['Minimum', formatNumber(statistics?.min)],
                    ['Maximum', formatNumber(statistics?.max)],
                    ['IQR', formatNumber(statistics?.iqr)],
                    ['Outlier thresholds', `${formatNumber(statistics?.lower_fence)} — ${formatNumber(statistics?.upper_fence)}`],
                  ].map(([label, value]) => <div className="stats-row" key={String(label)}><span>{label}</span><strong>{value}</strong></div>)}
                </div>
                <div className="outlier-note"><Sparkles size={15} /><span>{statistics?.outliers.length ? `${statistics.outliers.length} value(s) fall outside the IQR range.` : 'No outliers were detected using the 1.5 × IQR rule.'}</span></div>
              </section>
              <CorrelationPanel columns={columnNames} correlations={analysis.correlations} spearman={analysis.spearman_correlations} />
            </>
          )}
          {activeTab === 'regression' && analysis && (
            <div className="tab-content">
              <div className="section-title-row results-title"><div><h2>Linear regression</h2><p>Model relationships between variables and estimate outcomes.</p></div><span className="chart-badge">OLS</span></div>
              <section className="panel regression-controls">
                <label className="select-wrap"><span>Outcome variable (Y)</span><select value={target} onChange={(event) => updateTarget(event.target.value)}>{columnNames.map((name) => <option key={name}>{name}</option>)}</select><ChevronDown size={14} /></label>
                <div className="predictor-picker"><span className="field-label">Predictors (X)</span><div className="predictor-list">{columnNames.filter((name) => name !== target).map((name) => <label key={name} className={predictors.includes(name) ? 'predictor-chip checked' : 'predictor-chip'}><input type="checkbox" checked={predictors.includes(name)} onChange={() => togglePredictor(name)} /><span>{name}</span></label>)}</div></div>
              </section>
              {regression && (
                <>
                  <div className="metrics-grid regression-metrics"><MetricCard label="R²" value={formatNumber(regression.r_squared, 4)} note="coefficient of determination" color="purple" icon={<Activity size={16} />} /><MetricCard label="ADJUSTED R²" value={formatNumber(regression.adjusted_r_squared, 4)} note="adjusted for predictors" color="blue" icon={<Sigma size={16} />} /><MetricCard label="STD. ERROR" value={formatNumber(regression.residual_std_error)} note="of residuals" color="orange" icon={<FlaskConical size={16} />} /><MetricCard label="OBSERVATIONS" value={String(regression.observations)} note="included in the model" color="pink" icon={<Table2 size={16} />} /></div>
                  <section className="panel chart-panel regression-chart-panel"><div className="panel-heading"><div><h3>Observed values and regression line</h3><p>Estimated at the mean values of other predictors</p></div><label className="select-wrap chart-select"><select value={activeScatterColumn} onChange={(event) => setScatterColumn(event.target.value)}>{predictors.map((name) => <option key={name}>{name}</option>)}</select><ChevronDown size={14} /></label></div>
                    {scatterData && <div className="regression-chart"><ResponsiveContainer width="100%" height="100%"><ComposedChart margin={{ top: 12, right: 16, bottom: 4, left: 0 }}><CartesianGrid stroke="#edf0f5" /><XAxis type="number" dataKey="x" name={activeScatterColumn} tick={{ fill: '#9299a8', fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis type="number" dataKey="y" name={target} tick={{ fill: '#9299a8', fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ border: '1px solid #e8e8f0', borderRadius: 10, fontSize: 12 }} /><Scatter name="Observed" data={scatterData.points} fill="#9187f5" fillOpacity={0.76} /><Line name="Regression" data={scatterData.line} dataKey="y" type="linear" stroke="#5145cd" strokeWidth={2.5} dot={false} activeDot={false} /></ComposedChart></ResponsiveContainer></div>}
                  </section>
                  <section className="panel details-panel"><div className="panel-heading"><div><h3>Model coefficients</h3><p>Estimated equation and the effect of each predictor</p></div></div><div className="coefficient-table"><div className="coefficient-row coefficient-header"><span>Term</span><span>Coefficient</span><span>Interpretation</span></div><div className="coefficient-row"><strong>Intercept</strong><b>{formatNumber(regression.intercept)}</b><span>Estimated value when predictors are 0</span></div>{regression.predictors.map((name) => <div className="coefficient-row" key={name}><strong>{name}</strong><b>{formatNumber(regression.coefficients[name])}</b><span>Estimated change per +1 unit</span></div>)}</div><div className="equation-box"><span>MODEL</span><code>{target} = {formatNumber(regression.intercept)}{regression.predictors.map((name) => ` ${regression.coefficients[name] >= 0 ? '+' : '−'} ${formatNumber(Math.abs(regression.coefficients[name]))} × ${name}`).join('')}</code></div></section>
                  <section className="panel details-panel"><div className="panel-heading"><div><h3>Residuals</h3><p>Difference between observed and predicted values</p></div></div>                  <div className="residual-chart"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 8, right: 12, bottom: 4, left: 0 }}><CartesianGrid stroke="#edf0f5" /><XAxis type="number" dataKey="predicted" name="Predicted" tick={{ fill: '#9299a8', fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis type="number" dataKey="residual" name="Residual" tick={{ fill: '#9299a8', fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ border: '1px solid #e8e8f0', borderRadius: 10, fontSize: 12 }} /><ReferenceLine y={0} stroke="#a6a9b4" strokeDasharray="4 4" /><Scatter name="Residuals" data={regression.residuals} fill="#39a992" /></ScatterChart></ResponsiveContainer></div></section>
                </>
              )}
              {!regression && <div className="empty-state">Select at least one predictor and provide enough observations to calculate a regression.</div>}
            </div>
          )}
          {activeTab === 'distribution' && (
            <div className="tab-content">
              <div className="section-title-row results-title"><div><h2>Probability distributions</h2><p>Explore the probability/density function and cumulative distribution.</p></div></div>
              <section className="panel distribution-controls"><div className="distribution-type-tabs">{(['normal', 'binomial', 'poisson'] as DistributionType[]).map((type) => <button key={type} className={distributionType === type ? 'distribution-type active' : 'distribution-type'} onClick={() => { setDistributionType(type); setDistribution(null) }}>{type === 'normal' ? 'Normal' : type === 'binomial' ? 'Binomial' : 'Poisson'}</button>)}</div>
                <div className="parameter-grid">{distributionType === 'normal' ? <><ParameterField label="Mean (μ)" value={distributionParams.mean} onChange={(value) => setDistributionParams({ ...distributionParams, mean: value })} /><ParameterField label="Standard deviation (σ)" value={distributionParams.std} onChange={(value) => setDistributionParams({ ...distributionParams, std: value })} /></> : distributionType === 'binomial' ? <><ParameterField label="Number of trials (n)" value={distributionParams.n} onChange={(value) => setDistributionParams({ ...distributionParams, n: value })} /><ParameterField label="Probability (p)" value={distributionParams.p} onChange={(value) => setDistributionParams({ ...distributionParams, p: value })} /></> : <ParameterField label="Rate (λ)" value={distributionParams.lambda} onChange={(value) => setDistributionParams({ ...distributionParams, lambda: value })} />}</div>
                <button className="button button-primary" onClick={runDistribution}><Sparkles size={16} /> Calculate distribution</button>
              </section>
              {distribution && <section className="panel chart-panel distribution-chart-panel"><div className="panel-heading"><div><h3>{distribution.label}</h3><p>Probability function and cumulative distribution</p></div></div><div className="distribution-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={distribution.points} margin={{ top: 12, right: 18, bottom: 4, left: 0 }}><CartesianGrid stroke="#edf0f5" /><XAxis dataKey="x" tick={{ fill: '#9299a8', fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis domain={[0, 1]} tick={{ fill: '#9299a8', fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ border: '1px solid #e8e8f0', borderRadius: 10, fontSize: 12 }} /><Line type="monotone" dataKey="density" name={distribution.type === 'normal' ? 'Density (PDF)' : 'Probability (PMF)'} stroke="#7668ee" strokeWidth={2.5} dot={false} /><Line type="monotone" dataKey="cdf" name="Cumulative (CDF)" stroke="#37a88e" strokeWidth={2} dot={false} strokeDasharray="5 4" /></LineChart></ResponsiveContainer></div><div className="chart-legend"><span><i className="legend-purple" /> {distribution.type === 'normal' ? 'Density (PDF)' : 'Probability (PMF)'}</span><span><i className="legend-green" /> Cumulative (CDF)</span></div></section>}
              {!distribution && <div className="distribution-help"><CircleHelp size={17} /><span>Enter distribution parameters and calculate to view the PDF/PMF and CDF.</span></div>}
            </div>
          )}
          {activeTab === 'tests' && (
            <div className="tab-content">
              <div className="section-title-row results-title"><div><h2>Statistical tests</h2><p>Test hypotheses and compare data with expected distributions.</p></div></div>
              <section className="panel test-controls"><label className="field-label">Test type<select className="standard-select" value={testType} onChange={(event) => { setTestType(event.target.value as TestType); setTestResult(null) }}><option value="ttest">Welch's t-test (two samples)</option><option value="chi_square">Chi-square (goodness of fit)</option></select></label>
                {testType === 'ttest' ? <div className="test-input-grid"><label className="field-label">Sample A<textarea value={testValuesA} onChange={(event) => setTestValuesA(event.target.value)} /></label><label className="field-label">Sample B<textarea value={testValuesB} onChange={(event) => setTestValuesB(event.target.value)} /></label></div> : <div className="test-input-grid"><label className="field-label">Observed counts<textarea value={testValuesA} onChange={(event) => setTestValuesA(event.target.value)} /></label><label className="field-label">Expected counts (optional)<textarea value={expectedCounts} onChange={(event) => setExpectedCounts(event.target.value)} placeholder="Leave blank to assume equal frequencies" /></label></div>}
                <p className="test-note">{testType === 'ttest' ? "Two-sided Welch's test for the difference between two independent sample means." : 'Chi-square goodness-of-fit test. Equal expected frequencies are assumed if none are provided.'}</p><button className="button button-primary" onClick={runTest}><FlaskConical size={16} /> Run test</button>
              </section>
              {testResult && <section className="panel test-result"><div className="result-icon"><Check size={18} /></div><div className="result-heading"><h3>{testResult.label}</h3><p>{testResult.interpretation}</p></div><div className="test-result-grid"><ResultValue label="STATISTIC" value={formatNumber(testResult.statistic, 4)} /><ResultValue label="P-VALUE" value={formatNumber(testResult.p_value, 5)} /><ResultValue label="DEGREES OF FREEDOM" value={formatNumber(testResult.degrees_of_freedom, 2)} /></div><div className={testResult.significant ? 'significance significant' : 'significance'}>{testResult.significant ? 'Statistically significant at α = 0.05' : 'Not statistically significant at α = 0.05'}</div></section>}
            </div>
          )}
          <footer className="page-footer"><span>StatLab <span className="footer-separator">·</span> Statistical analysis in Python</span><span>Your data stays in the current session.</span></footer>
        </section>
      </main>
    </div>
  )
}

function MetricCard({ label, value, note, color, icon }: { label: string; value: string; note: string; color: string; icon: React.ReactNode }) {
  return <article className="metric-card"><div className={`metric-icon ${color}`}>{icon}</div><span className="metric-label">{label}</span><strong className="metric-value">{value}</strong><span className="metric-note">{note}</span></article>
}

function BoxPlot({ stats }: { stats: NonNullable<AnalysisResult['statistics'][string]> }) {
  const range = stats.max - stats.min || 1
  const position = (value: number) => 20 + ((value - stats.min) / range) * 60
  return (
    <div className="boxplot-wrap">
      <div className="boxplot-values"><span>{formatNumber(stats.lower_whisker)}</span><span>{formatNumber(stats.upper_whisker)}</span></div>
      <svg viewBox="0 0 100 56" role="img" aria-label="Box plot with quartiles and outliers">
        <line x1={position(stats.lower_whisker)} y1="27" x2={position(stats.upper_whisker)} y2="27" stroke="#a5a8b6" strokeWidth="1.5"><title>Whisker range</title></line>
        <line x1={position(stats.lower_whisker)} y1="19" x2={position(stats.lower_whisker)} y2="35" stroke="#a5a8b6" strokeWidth="1.5"><title>Lower whisker: {formatNumber(stats.lower_whisker)}</title></line>
        <line x1={position(stats.upper_whisker)} y1="19" x2={position(stats.upper_whisker)} y2="35" stroke="#a5a8b6" strokeWidth="1.5"><title>Upper whisker: {formatNumber(stats.upper_whisker)}</title></line>
        <rect x={position(stats.q1)} y="13" width={Math.max(2, position(stats.q3) - position(stats.q1))} height="28" rx="4" fill="#f0efff" stroke="#7668ee" strokeWidth="1.5"><title>Interquartile range: Q1 {formatNumber(stats.q1)} — Q3 {formatNumber(stats.q3)}</title></rect>
        <line x1={position(stats.median)} y1="13" x2={position(stats.median)} y2="41" stroke="#5145cd" strokeWidth="2.5"><title>Median: {formatNumber(stats.median)}</title></line>
        {stats.outliers.map((value, index) => <circle key={`${value}-${index}`} cx={Math.min(98, Math.max(2, position(value)))} cy={27 + (index % 3 - 1) * 6} r="2" fill="#ed7291"><title>Outlier: {formatNumber(value)}</title></circle>)}
      </svg>
      <div className="boxplot-scale"><span>Lower whisker</span><span>Q1</span><span>Median</span><span>Q3</span><span>Upper whisker</span></div>
    </div>
  )
}

function CorrelationPanel({ columns, correlations, spearman }: { columns: string[]; correlations: AnalysisResult['correlations']; spearman: AnalysisResult['spearman_correlations'] }) {
  const [method, setMethod] = useState<'pearson' | 'spearman'>('pearson')
  if (columns.length < 2) return null
  const matrix = method === 'pearson' ? correlations : spearman
  return (
    <section className="panel details-panel correlation-panel">
      <div className="panel-heading">
        <div><h3>Correlation matrix</h3><p>{method === 'pearson' ? 'Pearson · linear relationship' : 'Spearman · monotonic relationship by rank'}</p></div>
        <div className="correlation-toggle">
          <button className={method === 'pearson' ? 'selected' : ''} onClick={() => setMethod('pearson')}>Pearson</button>
          <button className={method === 'spearman' ? 'selected' : ''} onClick={() => setMethod('spearman')}>Spearman</button>
        </div>
      </div>
      <div className="correlation-table">
        <div className="correlation-row correlation-head" style={{ gridTemplateColumns: `minmax(110px, 1.2fr) repeat(${columns.length}, minmax(74px, 1fr))` }}>
          <span></span>{columns.map((name) => <strong key={name}>{name}</strong>)}
        </div>
        {columns.map((row) => (
          <div className="correlation-row" key={row} style={{ gridTemplateColumns: `minmax(110px, 1.2fr) repeat(${columns.length}, minmax(74px, 1fr))` }}>
            <strong>{row}</strong>
            {columns.map((column) => {
              const value = matrix[row]?.[column]
              return <span key={column} className={value !== null && value !== undefined && Math.abs(value) > 0.7 ? 'correlation-strong' : ''}>{formatNumber(value, 2)}</span>
            })}
          </div>
        ))}
      </div>
      <p className="spearman-note">Values close to +1 or −1 indicate strong associations; null values indicate constant variables.</p>
    </section>
  )
}

function ParameterField({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return <label className="field-label">{label}<input type="number" value={value} onChange={(event) => onChange(event.target.value)} /></label>
}

function ResultValue({ label, value }: { label: string; value: string }) {
  return <div className="test-value"><span>{label}</span><strong>{value}</strong></div>
}

export default App
