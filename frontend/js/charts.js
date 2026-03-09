// Chart Utilities and Initialization

let charts = {};

// Initialize Gauge Chart for Risk Score
function createRiskGaugeChart(canvasId, riskScore, riskCategory) {
    // Normalize score to 0-100
    const score = (riskScore * 100).toFixed(1);
    
    // Color based on risk category
    const color = getRiskColor(riskCategory);
    
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    const data = {
        labels: ['Stable', 'At Risk', 'High Risk'],
        datasets: [{
            data: [
                Math.max(0, RISK_THRESHOLDS.stable * 100 - score),
                Math.max(0, (RISK_THRESHOLDS.atRisk - RISK_THRESHOLDS.stable) * 100),
                Math.max(0, (1 - RISK_THRESHOLDS.atRisk) * 100)
            ],
            backgroundColor: [
                CHART_COLORS.stable,
                CHART_COLORS.atRisk,
                CHART_COLORS.highRisk
            ],
            borderColor: ['#fff', '#fff', '#fff'],
            borderWidth: 2
        }]
    };

    charts[canvasId] = new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + formatNumber(context.parsed, 1) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Initialize Risk Distribution Chart (Bar Chart)
function createRiskDistributionChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Stable', 'At Risk', 'High Risk'],
            datasets: [{
                label: 'Number of Customers',
                data: [
                    data.stable_customers || 0,
                    data.at_risk_customers || 0,
                    data.high_risk_customers || 0
                ],
                backgroundColor: [
                    CHART_COLORS.stable,
                    CHART_COLORS.atRisk,
                    CHART_COLORS.highRisk
                ],
                borderRadius: 5,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Initialize Model Evolution Chart (Line Chart)
function createModelEvolutionChart(canvasId, evolutionData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    if (!evolutionData || evolutionData.length === 0) {
        evolutionData = [
            { date: '2026-03-01', stable: 45, at_risk: 30, high_risk: 25 },
            { date: '2026-03-02', stable: 48, at_risk: 28, high_risk: 24 },
            { date: '2026-03-03', stable: 50, at_risk: 26, high_risk: 24 },
            { date: '2026-03-04', stable: 52, at_risk: 25, high_risk: 23 },
            { date: '2026-03-05', stable: 55, at_risk: 23, high_risk: 22 },
            { date: '2026-03-06', stable: 57, at_risk: 22, high_risk: 21 }
        ];
    }

    const dates = evolutionData.map(d => d.date || 'N/A').slice(-12);
    const stableData = evolutionData.map(d => d.stable || 0).slice(-12);
    const atRiskData = evolutionData.map(d => d.at_risk || 0).slice(-12);
    const highRiskData = evolutionData.map(d => d.high_risk || 0).slice(-12);

    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Stable',
                    data: stableData,
                    borderColor: CHART_COLORS.stable,
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: CHART_COLORS.stable
                },
                {
                    label: 'At Risk',
                    data: atRiskData,
                    borderColor: CHART_COLORS.atRisk,
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: CHART_COLORS.atRisk
                },
                {
                    label: 'High Risk',
                    data: highRiskData,
                    borderColor: CHART_COLORS.highRisk,
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: CHART_COLORS.highRisk
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Initialize Feature Importance Chart (Horizontal Bar)
function createFeatureImportanceChart(canvasId, featureData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    // Default data if none provided
    if (!featureData || featureData.length === 0) {
        featureData = [
            { feature: "Monthly Charge", importance: 0.25 },
            { feature: "Data Usage", importance: 0.20 },
            { feature: "Customer Service Calls", importance: 0.18 },
            { feature: "Account Weeks", importance: 0.15 },
            { feature: "Day Minutes", importance: 0.12 },
            { feature: "Overage Fee", importance: 0.10 }
        ];
    }

    const features = featureData.map(f => f.feature);
    const importance = featureData.map(f => f.importance);

    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Importance Score',
                data: importance,
                backgroundColor: [
                    CHART_COLORS.primary,
                    '#3b82f6',
                    '#60a5fa',
                    '#93c5fd',
                    '#bfdbfe',
                    '#dbeafe'
                ],
                borderRadius: 5
            }]
        },
        options: {
            indexAxis: 'y', // This creates horizontal bars
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

// Initialize Feature Contribution Chart (Pie Chart)
function createFeatureContributionChart(canvasId, riskScores) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    const reconstruction = riskScores.reconstruction_error || 0.3;
    const cluster = riskScores.cluster_distance || 0.3;
    const anomaly = riskScores.anomaly_score || 0.4;
    
    const total = reconstruction + cluster + anomaly;
    
    const reconstructionPercent = (reconstruction / total) * 100;
    const clusterPercent = (cluster / total) * 100;
    const anomalyPercent = (anomaly / total) * 100;

    charts[canvasId] = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Reconstruction Error', 'Cluster Distance', 'Anomaly Score'],
            datasets: [{
                data: [reconstructionPercent, clusterPercent, anomalyPercent],
                backgroundColor: [
                    '#3b82f6',
                    '#8b5cf6',
                    '#ec4899'
                ],
                borderColor: '#fff',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return formatNumber(context.parsed, 1) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Initialize Behavior Visualization Chart (Radar Chart)
function createBehaviorVisualizationChart(canvasId, customerData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    // Normalize customer data to 0-100 scale
    const normalize = (value, max) => {
        if (!value || isNaN(value)) return 0;
        return Math.min((value / max) * 100, 100);
    };

    const radarData = {
        labels: ['Account Weeks', 'Data Usage', 'Service Calls', 'Day Minutes', 'Monthly Charge'],
        datasets: [{
            label: 'Customer Metrics',
            data: [
                normalize(customerData.AccountWeeks, 200),
                normalize(customerData.DataUsage, 10),
                normalize(customerData.CustServCalls, 10),
                normalize(customerData.DayMins, 1000),
                normalize(customerData.MonthlyCharge, 100)
            ],
            borderColor: CHART_COLORS.primary,
            backgroundColor: 'rgba(37, 99, 235, 0.2)',
            borderWidth: 2,
            pointRadius: 4,
            pointBackgroundColor: CHART_COLORS.primary
        }]
    };

    charts[canvasId] = new Chart(ctx, {
        type: 'radar',
        data: radarData,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom'
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// Initialize Comparison Charts
function createComparisonRiskChart(canvasId, customer1, customer2) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    charts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Final Risk Score', 'Reconstruction Error', 'Cluster Distance', 'Anomaly Score'],
            datasets: [
                {
                    label: `Customer ${customer1.customer_id}`,
                    data: [
                        customer1.final_risk_score || 0,
                        customer1.reconstruction_error || 0,
                        customer1.cluster_distance || 0,
                        customer1.anomaly_score || 0
                    ],
                    backgroundColor: CHART_COLORS.primary
                },
                {
                    label: `Customer ${customer2.customer_id}`,
                    data: [
                        customer2.final_risk_score || 0,
                        customer2.reconstruction_error || 0,
                        customer2.cluster_distance || 0,
                        customer2.anomaly_score || 0
                    ],
                    backgroundColor: CHART_COLORS.info
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Initialize Comparison Behavior Chart
function createComparisonBehaviorChart(canvasId, customer1, customer2) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    const normalize = (value, max) => {
        if (!value || isNaN(value)) return 0;
        return Math.min((value / max) * 100, 100);
    };

    charts[canvasId] = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Account Weeks', 'Data Usage', 'Service Calls', 'Day Minutes', 'Monthly Charge'],
            datasets: [
                {
                    label: `Customer ${customer1.customer_id}`,
                    data: [
                        normalize(customer1.AccountWeeks, 200),
                        normalize(customer1.DataUsage, 10),
                        normalize(customer1.CustServCalls, 10),
                        normalize(customer1.DayMins, 1000),
                        normalize(customer1.MonthlyCharge, 100)
                    ],
                    borderColor: CHART_COLORS.primary,
                    backgroundColor: 'rgba(37, 99, 235, 0.2)',
                    borderWidth: 2,
                    pointRadius: 4,
                    pointBackgroundColor: CHART_COLORS.primary
                },
                {
                    label: `Customer ${customer2.customer_id}`,
                    data: [
                        normalize(customer2.AccountWeeks, 200),
                        normalize(customer2.DataUsage, 10),
                        normalize(customer2.CustServCalls, 10),
                        normalize(customer2.DayMins, 1000),
                        normalize(customer2.MonthlyCharge, 100)
                    ],
                    borderColor: CHART_COLORS.info,
                    backgroundColor: 'rgba(8, 145, 178, 0.2)',
                    borderWidth: 2,
                    pointRadius: 4,
                    pointBackgroundColor: CHART_COLORS.info
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom'
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// Cleanup all charts
function cleanupCharts() {
    Object.values(charts).forEach(chart => {
        if (chart) chart.destroy();
    });
    charts = {};
}