// API Configuration
const API_CONFIG = {
    BASE_URL: 'http://localhost:8000/api',
    ENDPOINTS: {
        // Dashboard endpoints
        MODEL_INFO: '/models/info',
        MODEL_METRICS: '/models/metrics',
        RISK_DISTRIBUTION: '/dashboard/risk-distribution',
        MODEL_EVOLUTION: '/dashboard/model-evolution',
        HIGH_RISK_CUSTOMERS: '/dashboard/high-risk-customers',
        
        // Customer analysis endpoints
        CUSTOMER_DETAILS: '/customers/{id}',
        CUSTOMER_RISK_SCORE: '/customers/{id}/risk-score',
        CUSTOMER_COMPARISON: '/customers/comparison',
        
        // Inference endpoints
        SINGLE_PREDICTION: '/inference/predict',
        BATCH_PREDICTION: '/inference/predict-batch',
        
        // Feature importance
        FEATURE_IMPORTANCE: '/models/feature-importance'
    },
    
    TIMEOUT: 30000,
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000
};

// Chart color configuration
const CHART_COLORS = {
    primary: '#2563eb',
    secondary: '#64748b',
    success: '#16a34a',
    danger: '#dc2626',
    warning: '#ea580c',
    info: '#0891b2',
    
    stable: '#10b981',
    atRisk: '#f59e0b',
    highRisk: '#ef4444',
    
    gradients: {
        blue: ['#dbeafe', '#3b82f6', '#1e40af'],
        green: ['#dcfce7', '#10b981', '#065f46'],
        orange: ['#fed7aa', '#f59e0b', '#92400e'],
        red: ['#fecaca', '#ef4444', '#7f1d1d']
    }
};

// Risk category thresholds - Based on score normalization (0-1 or 0-100)
const RISK_THRESHOLDS = {
    stable: 0.33,
    atRisk: 0.67,
    highRisk: 1.0
};

// Risk categories - Match database values
const RISK_CATEGORIES = {
    STABLE: 'Stable',
    AT_RISK: 'At Risk',
    HIGH_RISK: 'High Risk'
};

// Required CSV headers for batch upload
const REQUIRED_CSV_HEADERS = [
    'AccountWeeks',
    'ContractRenewal',
    'DataPlan',
    'DataUsage',
    'CustServCalls',
    'DayMins',
    'DayCalls',
    'MonthlyCharge',
    'OverageFee',
    'RoamMins'
];

// Feature display names
const FEATURE_NAMES = {
    'AccountWeeks': 'Account Weeks',
    'ContractRenewal': 'Contract Renewal',
    'DataPlan': 'Data Plan',
    'DataUsage': 'Data Usage (GB)',
    'CustServCalls': 'Customer Service Calls',
    'DayMins': 'Day Minutes',
    'DayCalls': 'Day Calls',
    'MonthlyCharge': 'Monthly Charge ($)',
    'OverageFee': 'Overage Fee ($)',
    'RoamMins': 'Roam Minutes'
};

// Utility function to get risk category
function getRiskCategory(score) {
    const s = parseFloat(score) || 0;
    if (s < RISK_THRESHOLDS.stable) return RISK_CATEGORIES.STABLE;
    if (s < RISK_THRESHOLDS.atRisk) return RISK_CATEGORIES.AT_RISK;
    return RISK_CATEGORIES.HIGH_RISK;
}

// Utility function to get risk color
function getRiskColor(category) {
    switch(category) {
        case RISK_CATEGORIES.STABLE:
            return CHART_COLORS.stable;
        case RISK_CATEGORIES.AT_RISK:
            return CHART_COLORS.atRisk;
        case RISK_CATEGORIES.HIGH_RISK:
            return CHART_COLORS.highRisk;
        default:
            return CHART_COLORS.secondary;
    }
}

// Utility function to get CSS class for risk category
function getRiskClass(category) {
    switch(category) {
        case RISK_CATEGORIES.STABLE:
            return 'stable';
        case RISK_CATEGORIES.AT_RISK:
            return 'at-risk';
        case RISK_CATEGORIES.HIGH_RISK:
            return 'high-risk';
        default:
            return '';
    }
}