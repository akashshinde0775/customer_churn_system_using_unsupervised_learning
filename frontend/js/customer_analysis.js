// Customer Analysis Page Logic

let currentMode = 'search';
let comparisonChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeCustomerAnalysis();
});

function initializeCustomerAnalysis() {
    document.getElementById('searchBtn').addEventListener('click', searchCustomer);
    document.getElementById('customerIdInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') searchCustomer();
    });

    document.getElementById('compareBtn').addEventListener('click', () => {
        currentMode = 'compare';
        document.getElementById('singleCustomerSection').classList.add('hidden');
        document.getElementById('customerComparisonSection').classList.remove('hidden');
    });

    document.getElementById('compareExecuteBtn').addEventListener('click', compareCustomers);
    document.getElementById('closCompareBtn').addEventListener('click', () => {
        currentMode = 'search';
        document.getElementById('customerComparisonSection').classList.add('hidden');
        document.getElementById('singleCustomerSection').classList.add('hidden');
        document.getElementById('comparisonResultsSection').classList.add('hidden');
        document.getElementById('customerIdInput').value = '';
    });

    const urlParams = new URLSearchParams(window.location.search);
    const customerId = urlParams.get('id');
    if (customerId) {
        document.getElementById('customerIdInput').value = customerId;
        searchCustomer();
    }
}

async function searchCustomer() {
    const customerId = document.getElementById('customerIdInput').value.trim();
    
    if (!customerId) {
        showError('Please enter a customer ID');
        return;
    }

    showLoading('Loading customer data...');

    try {
        const [customerDetails, riskScore] = await Promise.all([
            apiCall(API_CONFIG.ENDPOINTS.CUSTOMER_DETAILS.replace('{id}', customerId)),
            apiCall(API_CONFIG.ENDPOINTS.CUSTOMER_RISK_SCORE.replace('{id}', customerId))
        ]);

        if (!customerDetails.success) {
            showError(customerDetails.error || 'Customer not found');
            hideLoading();
            return;
        }

        currentMode = 'search';
        document.getElementById('singleCustomerSection').classList.remove('hidden');
        document.getElementById('customerComparisonSection').classList.add('hidden');

        updateCustomerInfo(customerDetails.data);
        
        if (riskScore.success) {
            updateRiskScores(riskScore.data, customerDetails.data);
        }

        hideLoading();
    } catch (error) {
        console.error('Error searching customer:', error);
        showError('Failed to load customer data');
        hideLoading();
    }
}

function updateCustomerInfo(customer) {
    document.getElementById('custId').textContent = customer.customer_id;
    document.getElementById('custAccountWeeks').textContent = customer.AccountWeeks || '-';
    document.getElementById('custMonthlyCharge').textContent = formatCurrency(customer.MonthlyCharge) || '-';
    document.getElementById('custContractRenewal').textContent = customer.ContractRenewal === 1 ? 'Yes' : 'No';
}

function updateRiskScores(scores, customerData) {
    const riskCategory = getRiskCategory(scores.final_risk_score);
    
    createRiskGaugeChart('riskGaugeChart', scores.final_risk_score, riskCategory);
    
    const categoryBadge = document.getElementById('riskCategory');
    categoryBadge.textContent = riskCategory;
    categoryBadge.className = `category-badge ${getRiskClass(riskCategory)}`;

    document.getElementById('finalRiskScore').textContent = formatNumber(scores.final_risk_score, 4);
    document.getElementById('reconstructionErrorScore').textContent = formatNumber(scores.reconstruction_error, 4);
    document.getElementById('clusterDistanceScore').textContent = formatNumber(scores.cluster_distance, 4);
    document.getElementById('anomalyScoreValue').textContent = formatNumber(scores.anomaly_score, 4);

    createFeatureContributionChart('featureContributionChart', scores);
    createBehaviorVisualizationChart('behaviorVisualizationChart', customerData);
}

async function compareCustomers() {
    const customer1 = document.getElementById('compareCustomer1').value.trim();
    const customer2 = document.getElementById('compareCustomer2').value.trim();

    if (!customer1 || !customer2) {
        showError('Please enter both customer IDs');
        return;
    }

    if (customer1 === customer2) {
        showError('Please select different customers');
        return;
    }

    showLoading('Comparing customers...');

    try {
        const [cust1Details, cust1Scores, cust2Details, cust2Scores] = await Promise.all([
            apiCall(API_CONFIG.ENDPOINTS.CUSTOMER_DETAILS.replace('{id}', customer1)),
            apiCall(API_CONFIG.ENDPOINTS.CUSTOMER_RISK_SCORE.replace('{id}', customer1)),
            apiCall(API_CONFIG.ENDPOINTS.CUSTOMER_DETAILS.replace('{id}', customer2)),
            apiCall(API_CONFIG.ENDPOINTS.CUSTOMER_RISK_SCORE.replace('{id}', customer2))
        ]);

        if (!cust1Details.success || !cust2Details.success) {
            showError('One or both customers not found');
            hideLoading();
            return;
        }

        document.getElementById('comparisonResultsSection').classList.remove('hidden');

        updateComparisonCard(1, cust1Details.data, cust1Scores.data);
        updateComparisonCard(2, cust2Details.data, cust2Scores.data);

        createComparisonRiskChart('comparisonRiskChart', cust1Scores.data, cust2Scores.data);
        createComparisonBehaviorChart('comparisonBehaviorChart', cust1Details.data, cust2Details.data);

        hideLoading();
    } catch (error) {
        console.error('Error comparing customers:', error);
        showError('Failed to compare customers');
        hideLoading();
    }
}

function updateComparisonCard(cardNum, customerData, scoreData) {
    const riskCategory = getRiskCategory(scoreData.final_risk_score);
    
    document.getElementById(`comp${cardNum}Title`).textContent = `Customer ${customerData.customer_id}`;
    document.getElementById(`comp${cardNum}RiskScore`).textContent = formatNumber(scoreData.final_risk_score, 4);
    document.getElementById(`comp${cardNum}Category`).innerHTML = getRiskBadgeHTML(riskCategory);
    document.getElementById(`comp${cardNum}Charge`).textContent = formatCurrency(customerData.MonthlyCharge);
    document.getElementById(`comp${cardNum}Weeks`).textContent = customerData.AccountWeeks;
}