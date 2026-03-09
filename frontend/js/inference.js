// Inference Page Logic

let batchPredictionResults = [];
let selectedFile = null;

document.addEventListener('DOMContentLoaded', function() {
    initializeInference();
});

function initializeInference() {
    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            switchTab(tabName);
        });
    });

    // Single prediction form
    const form = document.getElementById('singlePredictionForm');
    if (form) {
        form.addEventListener('submit', handleSinglePrediction);
    }

    // Batch upload
    const uploadArea = document.getElementById('uploadArea');
    const csvFile = document.getElementById('csvFile');

    if (uploadArea && csvFile) {
        uploadArea.addEventListener('click', () => csvFile.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = 'rgba(37, 99, 235, 0.15)';
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.backgroundColor = 'rgba(37, 99, 235, 0.05)';
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.backgroundColor = 'rgba(37, 99, 235, 0.05)';
            if (e.dataTransfer.files.length > 0) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        csvFile.addEventListener('change', (e) => {
            if (e.target.files && e.target.files[0]) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    const uploadBtn = document.getElementById('uploadBtn');
    const downloadBtn = document.getElementById('downloadResultsBtn');
    
    if (uploadBtn) uploadBtn.addEventListener('click', processBatchUpload);
    if (downloadBtn) downloadBtn.addEventListener('click', downloadBatchResults);
}

function switchTab(tabName) {
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    if (event && event.target) {
        event.target.classList.add('active');
    }

    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    const tabContent = document.getElementById(tabName);
    if (tabContent) {
        tabContent.classList.add('active');
    }
}

async function handleSinglePrediction(e) {
    e.preventDefault();

    const formData = {
        AccountWeeks: parseFloat(document.getElementById('accountWeeks').value) || 0,
        ContractRenewal: parseFloat(document.getElementById('contractRenewal').value) || 0,
        DataPlan: parseFloat(document.getElementById('dataPlan').value) || 0,
        DataUsage: parseFloat(document.getElementById('dataUsage').value) || 0,
        CustServCalls: parseFloat(document.getElementById('custServCalls').value) || 0,
        DayMins: parseFloat(document.getElementById('dayMins').value) || 0,
        DayCalls: parseFloat(document.getElementById('dayCalls').value) || 0,
        MonthlyCharge: parseFloat(document.getElementById('monthlyCharge').value) || 0,
        OverageFee: parseFloat(document.getElementById('overageFee').value) || 0,
        RoamMins: parseFloat(document.getElementById('roamMins').value) || 0
    };

    showLoading('Making prediction...');

    try {
        const response = await apiCall(API_CONFIG.ENDPOINTS.SINGLE_PREDICTION, 'POST', formData, true);

        if (response.success) {
            const prediction = response.data;
            const riskCategory = getRiskCategory(prediction.final_risk_score);

            const resultSection = document.getElementById('singlePredictionResult');
            if (resultSection) resultSection.classList.remove('hidden');

            const scoreElement = document.getElementById('predictionRiskScore');
            if (scoreElement) scoreElement.textContent = formatNumber(prediction.final_risk_score, 4);

            const categoryElement = document.getElementById('predictionRiskCategory');
            if (categoryElement) categoryElement.innerHTML = getRiskBadgeHTML(riskCategory);

            const reconElement = document.getElementById('predictionReconError');
            if (reconElement) reconElement.textContent = formatNumber(prediction.reconstruction_error, 4);

            const clusterElement = document.getElementById('predictionClusterDist');
            if (clusterElement) clusterElement.textContent = formatNumber(prediction.cluster_distance, 4);

            const anomalyElement = document.getElementById('predictionAnomalyScore');
            if (anomalyElement) anomalyElement.textContent = formatNumber(prediction.anomaly_score, 4);

            createFeatureContributionChart('singleComponentChart', prediction);
            showSuccess('Prediction successful!');
        } else {
            showError(response.error || 'Prediction failed');
        }

        hideLoading();
    } catch (error) {
        console.error('Error making prediction:', error);
        showError('Failed to make prediction: ' + error.message);
        hideLoading();
    }
}

function handleFileSelect(file) {
    if (!file.name.endsWith('.csv')) {
        showError('Please select a CSV file');
        return;
    }

    selectedFile = file;
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        uploadArea.innerHTML = `<p style="color: #10b981; font-weight: 600;">✓ ${file.name} selected</p>`;
    }
}

async function processBatchUpload() {
    if (!selectedFile) {
        showError('Please select a CSV file');
        return;
    }

    showLoading('Processing batch predictions...');

    try {
        const text = await selectedFile.text();
        const csvData = parseCSV(text);

        if (!csvData || csvData.length === 0) {
            showError('CSV file is empty');
            hideLoading();
            return;
        }

        // Validate headers
        const csvHeaders = Object.keys(csvData[0]);
        const missingHeaders = validateCSVHeaders(csvHeaders, REQUIRED_CSV_HEADERS);

        if (missingHeaders) {
            showError(`Missing headers: ${missingHeaders.join(', ')}`);
            hideLoading();
            return;
        }

        // Create request payload
        const requestPayload = {
            data: csvData
        };

        const response = await apiCall(API_CONFIG.ENDPOINTS.BATCH_PREDICTION, 'POST', requestPayload, true);

        if (response.success) {
            batchPredictionResults = response.data;
            
            if (!Array.isArray(batchPredictionResults)) {
                batchPredictionResults = [];
            }
            
            if (batchPredictionResults.length > 0) {
                updateBatchResults(batchPredictionResults);
                showSuccess('Batch prediction successful!');
            } else {
                showError('No results returned');
            }
        } else {
            showError(response.error || 'Batch prediction failed');
        }

        hideLoading();
    } catch (error) {
        console.error('Error processing batch:', error);
        showError('Failed to process batch upload: ' + error.message);
        hideLoading();
    }
}

function updateBatchResults(results) {
    if (!Array.isArray(results) || results.length === 0) {
        showError('No results to display');
        return;
    }

    // Count by risk category
    let stableCount = 0;
    let atRiskCount = 0;
    let highRiskCount = 0;

    results.forEach(r => {
        const score = parseFloat(r.final_risk_score) || 0;
        const category = getRiskCategory(score);
        if (category === RISK_CATEGORIES.STABLE) stableCount++;
        else if (category === RISK_CATEGORIES.AT_RISK) atRiskCount++;
        else highRiskCount++;
    });

    // Update statistics
    const statsElement = document.getElementById('resultStats');
    if (statsElement) {
        statsElement.textContent = 
            `Total Predictions: ${results.length} | Stable: ${stableCount} | At Risk: ${atRiskCount} | High Risk: ${highRiskCount}`;
    }

    // Update results table
    const tableBody = document.getElementById('batchResultsTable');
    if (tableBody) {
        tableBody.innerHTML = results.map((result, index) => {
            const riskCategory = getRiskCategory(result.final_risk_score);
            return `
                <tr>
                    <td>${index + 1}</td>
                    <td>${formatNumber(result.final_risk_score, 4)}</td>
                    <td>${getRiskBadgeHTML(riskCategory)}</td>
                    <td>${formatNumber(result.reconstruction_error, 4)}</td>
                    <td>${formatNumber(result.cluster_distance, 4)}</td>
                    <td>${formatNumber(result.anomaly_score, 4)}</td>
                </tr>
            `;
        }).join('');
    }

    // Create distribution chart
    createRiskDistributionChart('batchRiskDistributionChart', {
        stable_customers: stableCount,
        at_risk_customers: atRiskCount,
        high_risk_customers: highRiskCount
    });

    // Show results section
    const resultsSection = document.getElementById('batchResultsSection');
    if (resultsSection) {
        resultsSection.classList.remove('hidden');
    }
}

function downloadBatchResults() {
    if (!Array.isArray(batchPredictionResults) || batchPredictionResults.length === 0) {
        showError('No results to download');
        return;
    }

    const csv = arrayToCSV(batchPredictionResults);
    downloadCSV(csv, 'batch_predictions_' + new Date().getTime() + '.csv');
    showSuccess('Results downloaded successfully!');
}