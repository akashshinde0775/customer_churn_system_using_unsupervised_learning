// Dashboard Page Logic

document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

async function initializeDashboard() {
    showLoading('Loading dashboard data...');
    
    try {
        // Fetch all dashboard data in parallel
        const [modelInfo, modelMetrics, riskDist, evolution, highRiskCustomers, featureImportance] = await Promise.all([
            apiCall(API_CONFIG.ENDPOINTS.MODEL_INFO),
            apiCall(API_CONFIG.ENDPOINTS.MODEL_METRICS),
            apiCall(API_CONFIG.ENDPOINTS.RISK_DISTRIBUTION),
            apiCall(API_CONFIG.ENDPOINTS.MODEL_EVOLUTION),
            apiCall(API_CONFIG.ENDPOINTS.HIGH_RISK_CUSTOMERS),
            apiCall(API_CONFIG.ENDPOINTS.FEATURE_IMPORTANCE)
        ]);

        if (modelInfo.success) updateModelInfo(modelInfo.data);
        if (modelMetrics.success) updateMetrics(modelMetrics.data);
        if (riskDist.success) createRiskDistributionChart('riskDistributionChart', riskDist.data);
        if (evolution.success) createModelEvolutionChart('modelEvolutionChart', evolution.data);
        if (featureImportance.success) createFeatureImportanceChart('featureImportanceChart', featureImportance.data);
        if (highRiskCustomers.success) updateHighRiskCustomersTable(highRiskCustomers.data);

        hideLoading();
    } catch (error) {
        console.error('Error loading dashboard:', error);
        showError('Failed to load dashboard data');
        hideLoading();
    }
}

function updateModelInfo(modelInfo) {
    document.getElementById('modelName').textContent = modelInfo.model_name || 'Unknown';
    document.getElementById('trainingType').textContent = `Type: ${modelInfo.training_type || '-'}`;
    document.getElementById('trainingDate').textContent = `Date: ${formatDate(modelInfo.training_date) || '-'}`;
    document.getElementById('dataSize').textContent = `Training Size: ${modelInfo.training_data_size || 0}`;
    document.getElementById('trainingDate2').textContent = `Last Training: ${formatDate(modelInfo.training_date) || '-'}`;
}

function updateMetrics(metrics) {
    document.getElementById('reconstructionError').textContent = formatNumber(metrics.reconstruction_error_mean, 6);
    document.getElementById('silhouetteScore').textContent = formatNumber(metrics.silhouette_score, 4);
    document.getElementById('contaminationRate').textContent = formatPercentage(metrics.anomaly_contamination_rate);
}

function updateHighRiskCustomersTable(customers) {
    const tableBody = document.getElementById('highRiskTable');
    
    if (!customers || customers.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="7" class="text-center">No high-risk customers found</td></tr>';
        return;
    }

    tableBody.innerHTML = customers.slice(0, 10).map(customer => `
        <tr>
            <td>${customer.customer_id}</td>
            <td>${formatNumber(customer.final_risk_score, 4)}</td>
            <td>${getRiskBadgeHTML(customer.risk_category)}</td>
            <td>${formatNumber(customer.reconstruction_error, 4)}</td>
            <td>${formatNumber(customer.cluster_distance, 4)}</td>
            <td>${formatNumber(customer.anomaly_score, 4)}</td>
            <td><button class="btn-action" onclick="viewCustomer(${customer.customer_id})">View</button></td>
        </tr>
    `).join('');
}

function viewCustomer(customerId) {
    window.location.href = `customer_analysis.html?id=${customerId}`;
}