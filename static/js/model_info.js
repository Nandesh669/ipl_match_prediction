document.addEventListener("DOMContentLoaded", () => {
    renderFeatureImportanceChart();
});

function renderFeatureImportanceChart() {
    const canvas = document.getElementById('features-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');

    // Parse and sort feature importances descending
    const items = Object.keys(featureImportances).map(key => {
        return {
            key: key,
            label: featureLabelsMapping[key] || key,
            value: featureImportances[key]
        };
    });

    items.sort((a, b) => b.value - a.value);

    const labels = items.map(item => item.label);
    const values = items.map(item => item.value);

    // Create a beautiful electric purple gradient
    const gradient = ctx.createLinearGradient(0, 0, 400, 0);
    gradient.addColorStop(0, '#06b6d4'); // Cyan
    gradient.addColorStop(1, '#8b5cf6'); // Purple

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: gradient,
                borderRadius: 6,
                borderWidth: 0,
                barThickness: 12
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(11, 15, 25, 0.95)',
                    titleColor: '#ffffff',
                    bodyColor: '#06b6d4',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: function(context) {
                            return ` Importance: ${(context.raw * 100).toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    max: Math.max(...values) * 1.1, // Leave space for label padding
                    grid: { display: true },
                    ticks: { callback: value => `${(value * 100).toFixed(0)}%` }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });
}
