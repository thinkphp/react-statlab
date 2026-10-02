export type ColumnStatistics = {
  count: number
  mean: number
  median: number
  mode: number[]
  variance: number
  std_dev: number
  min: number
  max: number
  q1: number
  q3: number
  iqr: number
  lower_fence: number
  upper_fence: number
  lower_whisker: number
  upper_whisker: number
  outliers: number[]
}

export type AnalysisResult = {
  rows: number
  columns: string[]
  statistics: Record<string, ColumnStatistics>
  histograms: Record<string, { bins: { start: number; end: number; count: number; label: string }[] }>
  correlations: Record<string, Record<string, number | null>>
  spearman_correlations: Record<string, Record<string, number | null>>
  regression: {
    target: string
    predictors: string[]
    intercept: number
    coefficients: Record<string, number>
    r_squared: number
    adjusted_r_squared: number
    residual_std_error: number
    observations: number
    residuals: { predicted: number; actual: number; residual: number; predictors: Record<string, number> }[]
  } | null
}

export type DistributionResult = {
  type: 'normal' | 'binomial' | 'poisson'
  label: string
  points: { x: number; density: number; cdf: number }[]
}

export type StatisticalTestResult = {
  type: 'ttest' | 'chi_square'
  label: string
  statistic: number
  p_value: number
  degrees_of_freedom: number
  significant: boolean
  interpretation: string
}

async function post<T>(endpoint: string, body: unknown): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api/${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch {
    throw new Error('Cannot connect to the Python backend. Start the FastAPI server on port 8000.')
  }
  const payload: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const message = payload && typeof payload === 'object' && 'detail' in payload
      ? String(payload.detail)
      : `The backend returned error ${response.status}.`
    throw new Error(message)
  }
  return payload as T
}

export function analyzeData(
  data: Record<string, number[]>,
  target: string,
  predictors: string[],
): Promise<AnalysisResult> {
  return post('analyze', { data, target, predictors })
}

export function calculateDistribution(
  type: 'normal' | 'binomial' | 'poisson',
  values: Record<string, string>,
): Promise<DistributionResult> {
  return post('distribution', { type, ...values })
}

export function calculateTest(
  type: 'ttest' | 'chi_square',
  values_a: number[],
  values_b?: number[],
  expected?: number[],
): Promise<StatisticalTestResult> {
  return post('test', { type, values_a, values_b, expected })
}
