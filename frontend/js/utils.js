// Utility Functions

// Show loading spinner
function showLoading(message = 'Loading...') {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        const loadingText = spinner.querySelector('p');
        if (loadingText) loadingText.textContent = message;
        spinner.classList.remove('hidden');
    }
}

// Hide loading spinner
function hideLoading() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.classList.add('hidden');
    }
}

// Show error message
function showError(message) {
    const errorElement = document.getElementById('errorMessage');
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.classList.remove('hidden');
        setTimeout(() => {
            errorElement.classList.add('hidden');
        }, 5000);
    }
    console.error('Error:', message);
}

// Show success message
function showSuccess(message) {
    console.log('Success:', message);
}

// Format currency
function formatCurrency(value) {
    if (!value || isNaN(value)) return '$0.00';
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

// Format percentage
function formatPercentage(value) {
    if (!value || isNaN(value)) return '0.00%';
    return (value * 100).toFixed(2) + '%';
}

// Format number with decimal places
function formatNumber(value, decimals = 2) {
    if (!value && value !== 0 || isNaN(value)) return '0.00';
    return parseFloat(value).toFixed(decimals);
}

// Format date
function formatDate(dateString) {
    if (!dateString) return 'N/A';
    try {
        const options = { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
        return new Date(dateString).toLocaleDateString('en-US', options);
    } catch {
        return dateString;
    }
}

// Mock data generators - Match your actual database structure
const MOCK_DATA = {
    modelInfo: {
        model_bundle_id: 1,
        model_name: 'Churn_Autoencoder_KMeans_IsolationForest',
        training_type: 'initial',
        training_date: new Date().toISOString(),
        training_data_size: 2500,
        is_active: true
    },
    
    modelMetrics: {
        reconstruction_error_mean: 0.0245,
        silhouette_score: 0.6234,
        anomaly_contamination_rate: 0.08
    },
    
    riskDistribution: {
        stable_customers: 1200,
        at_risk_customers: 800,
        high_risk_customers: 500
    },
    
    modelEvolution: [
        { date: '2026-03-01', stable: 1100, at_risk: 850, high_risk: 550 },
        { date: '2026-03-02', stable: 1150, at_risk: 820, high_risk: 530 },
        { date: '2026-03-03', stable: 1180, at_risk: 810, high_risk: 510 },
        { date: '2026-03-04', stable: 1190, at_risk: 805, high_risk: 505 },
        { date: '2026-03-05', stable: 1195, at_risk: 802, high_risk: 503 },
        { date: '2026-03-06', stable: 1200, at_risk: 800, high_risk: 500 }
    ],
    
    featureImportance: [
        { feature: 'reconstruction_error', importance: 0.25 },
        { feature: 'cluster_distance', importance: 0.25 },
        { feature: 'anomaly_score', importance: 0.25 },
        { feature: 'MonthlyCharge', importance: 0.10 },
        { feature: 'DataUsage', importance: 0.08 },
        { feature: 'AccountWeeks', importance: 0.07 }
    ],
    
    highRiskCustomers: [
        {
            customer_id: 1,
            final_risk_score: 0.92,
            risk_category: 'High Risk',
            reconstruction_error: 0.85,
            cluster_distance: 0.95,
            anomaly_score: 0.96,
            monthly_charge: 75.5,
            account_weeks: 12,
            prediction_time: new Date().toISOString()
        },
        {
            customer_id: 5,
            final_risk_score: 0.88,
            risk_category: 'High Risk',
            reconstruction_error: 0.82,
            cluster_distance: 0.92,
            anomaly_score: 0.90,
            monthly_charge: 68.0,
            account_weeks: 8,
            prediction_time: new Date().toISOString()
        },
        {
            customer_id: 12,
            final_risk_score: 0.85,
            risk_category: 'High Risk',
            reconstruction_error: 0.78,
            cluster_distance: 0.88,
            anomaly_score: 0.89,
            monthly_charge: 72.5,
            account_weeks: 15,
            prediction_time: new Date().toISOString()
        }
    ],
    
    customerDetails: {
        customer_id: 101,
        AccountWeeks: 45,
        ContractRenewal: 1,
        DataPlan: 1,
        DataUsage: 5.2,
        CustServCalls: 4,
        DayMins: 250.5,
        DayCalls: 85,
        MonthlyCharge: 65.5,
        OverageFee: 12.3,
        RoamMins: 100.2,
        created_at: new Date().toISOString()
    },
    
    riskScore: {
        customer_id: 101,
        final_risk_score: 0.48,
        risk_category: 'At Risk',
        reconstruction_error: 0.35,
        cluster_distance: 0.58,
        anomaly_score: 0.52,
        prediction_time: new Date().toISOString()
    },
    
    batchPredictions: Array.from({ length: 25 }, (_, i) => {
        const scores = [
            { reconstruction_error: 0.35, cluster_distance: 0.58, anomaly_score: 0.52, final_risk_score: 0.48 },
            { reconstruction_error: 0.28, cluster_distance: 0.45, anomaly_score: 0.38, final_risk_score: 0.37 },
            { reconstruction_error: 0.72, cluster_distance: 0.88, anomaly_score: 0.85, final_risk_score: 0.82 },
            { reconstruction_error: 0.42, cluster_distance: 0.52, anomaly_score: 0.48, final_risk_score: 0.47 },
            { reconstruction_error: 0.25, cluster_distance: 0.32, anomaly_score: 0.28, final_risk_score: 0.28 }
        ];
        const score = scores[i % scores.length];
        return {
            ...score,
            risk_category: getRiskCategory(score.final_risk_score)
        };
    })
};

// API Helper - Make API calls with error handling
async function apiCall(endpoint, method = 'GET', data = null, useMockData = true) {
    const url = API_CONFIG.BASE_URL + endpoint;
    
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
            timeout: API_CONFIG.TIMEOUT
        };

        if (data && (method === 'POST' || method === 'PUT')) {
            options.body = JSON.stringify(data);
        }

        const response = await axios(url, options);
        console.log(`API Success [${endpoint}]:`, response.data);
        return { success: true, data: response.data };
    } catch (error) {
        console.error('API Error:', error.message);
        
        // Return mock data as fallback
        if (useMockData) {
            console.warn(`Using mock data for ${endpoint}`);
            
            if (endpoint.includes('/models/info')) {
                return { success: true, data: MOCK_DATA.modelInfo };
            } else if (endpoint.includes('/models/metrics')) {
                return { success: true, data: MOCK_DATA.modelMetrics };
            } else if (endpoint.includes('/models/feature-importance')) {
                return { success: true, data: MOCK_DATA.featureImportance };
            } else if (endpoint.includes('/dashboard/risk-distribution')) {
                return { success: true, data: MOCK_DATA.riskDistribution };
            } else if (endpoint.includes('/dashboard/model-evolution')) {
                return { success: true, data: MOCK_DATA.modelEvolution };
            } else if (endpoint.includes('/dashboard/high-risk-customers')) {
                return { success: true, data: MOCK_DATA.highRiskCustomers };
            } else if (endpoint.includes('/risk-score')) {
                return { success: true, data: MOCK_DATA.riskScore };
            } else if (endpoint.match(/\/customers\/\d+$/)) {
                return { success: true, data: MOCK_DATA.customerDetails };
            } else if (endpoint.includes('/customers/comparison')) {
                return { success: true, data: [MOCK_DATA.riskScore, { ...MOCK_DATA.riskScore, customer_id: 102, final_risk_score: 0.45 }] };
            } else if (endpoint.includes('/inference/predict-batch')) {
                return { success: true, data: MOCK_DATA.batchPredictions };
            } else if (endpoint.includes('/inference/predict')) {
                return { success: true, data: MOCK_DATA.riskScore };
            }
            
            return { success: true, data: {} };
        } else {
            const errorMessage = error.response?.data?.detail || error.message || 'An error occurred';
            return { success: false, error: errorMessage };
        }
    }
}

// Parse CSV file
function parseCSV(csvContent) {
    const lines = csvContent.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];

    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim() === '') continue;
        
        const values = lines[i].split(',').map(v => v.trim());
        const row = {};
        
        headers.forEach((header, index) => {
            const value = values[index];
            row[header] = isNaN(value) ? value : parseFloat(value);
        });
        
        data.push(row);
    }

    return data;
}

// Convert array of objects to CSV
function arrayToCSV(data) {
    if (!data || data.length === 0) return '';

    const headers = Object.keys(data[0]);
    const csvContent = [
        headers.join(','),
        ...data.map(row =>
            headers.map(header => {
                const value = row[header];
                return typeof value === 'string' && value.includes(',')
                    ? `"${value.replace(/"/g, '""')}"` 
                    : value;
            }).join(',')
        )
    ].join('\n');

    return csvContent;
}

// Download CSV file
function downloadCSV(csvContent, filename = 'results.csv') {
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvContent));
    element.setAttribute('download', filename);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}

// Validate form data
function validateFormData(formData, requiredFields) {
    const errors = [];
    requiredFields.forEach(field => {
        if (!formData[field] && formData[field] !== 0) {
            errors.push(`${field} is required`);
        }
    });
    return errors;
}

// Get risk badge HTML
function getRiskBadgeHTML(category) {
    const color = getRiskColor(category);
    const riskClass = getRiskClass(category);
    return `<span class="risk-badge ${riskClass}">${category}</span>`;
}

// Validate CSV headers
function validateCSVHeaders(headers, requiredHeaders) {
    const missingHeaders = requiredHeaders.filter(h => !headers.includes(h));
    return missingHeaders.length === 0 ? null : missingHeaders;
}

// Throttle function
function throttle(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}