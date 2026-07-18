document.addEventListener("DOMContentLoaded", () => {
    // Set global Chart.js defaults for dark theme
    Chart.defaults.color = '#9ca3af'; // text-muted
    Chart.defaults.font.family = "'Outfit', sans-serif";
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';

    // Fetch stats data
    fetch("/api/stats")
        .then(response => {
            if (!response.ok) throw new Error("Failed to load statistics");
            return response.json();
        })
        .then(data => {
            renderTeamWinsChart(data.team_stats);
            renderTossTrendsChart(data.toss_decisions);
            renderTossImpactChart(data.toss_win_impact);
            renderPlayerAwardsChart(data.top_players);
            renderVenueComparisonChart(data.venue_stats);
        })
        .catch(err => {
            console.error("Error loading dashboard data:", err);
            alert("Error loading dashboard visualizations. Check server logs.");
        });
});

// 1. Overall Team Win Percentages Chart (Horizontal Bar Chart)
function renderTeamWinsChart(teamStats) {
    const ctx = document.getElementById('team-wins-chart').getContext('2d');
    
    // Sort descending
    teamStats.sort((a, b) => b.win_percentage - a.win_percentage);

    const labels = teamStats.map(item => item.team);
    const percentages = teamStats.map(item => item.win_percentage);

    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, 400, 0);
    gradient.addColorStop(0, '#8b5cf6'); // Purple
    gradient.addColorStop(1, '#06b6d4'); // Cyan

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Win Rate (%)',
                data: percentages,
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
                            return ` ${context.parsed.x}% Win Rate`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    max: 100,
                    grid: { display: true },
                    ticks: { callback: value => `${value}%` }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });
}

// 2. Toss Decision Trends by Season (Stacked Bar Chart)
function renderTossTrendsChart(tossDecisions) {
    const ctx = document.getElementById('toss-trends-chart').getContext('2d');

    const seasons = tossDecisions.map(item => item.season);
    const batCounts = tossDecisions.map(item => item.bat || 0);
    const fieldCounts = tossDecisions.map(item => item.field || 0);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: seasons,
            datasets: [
                {
                    label: 'Chose to Bat',
                    data: batCounts,
                    backgroundColor: '#8b5cf6', // Purple
                    borderRadius: 4
                },
                {
                    label: 'Chose to Field',
                    data: fieldCounts,
                    backgroundColor: '#06b6d4', // Cyan
                    borderRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 12 }
                },
                tooltip: {
                    backgroundColor: 'rgba(11, 15, 25, 0.95)',
                    padding: 10,
                    borderWidth: 1,
                    borderColor: 'rgba(255, 255, 255, 0.1)'
                }
            },
            scales: {
                x: { stacked: true, grid: { display: false } },
                y: { stacked: true, grid: { display: true } }
            }
        }
    });
}

// 3. Toss Winner = Match Winner (Doughnut Chart)
function renderTossImpactChart(tossImpact) {
    const ctx = document.getElementById('toss-impact-chart').getContext('2d');

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Won Match', 'Lost Match'],
            datasets: [{
                data: [tossImpact.toss_winner_won, tossImpact.toss_winner_lost],
                backgroundColor: [
                    '#06b6d4', // Cyan (Won)
                    'rgba(255, 255, 255, 0.05)' // Light Grey (Lost)
                ],
                borderWidth: 1,
                borderColor: 'rgba(255, 255, 255, 0.08)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '75%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { boxWidth: 12, padding: 15 }
                },
                tooltip: {
                    backgroundColor: 'rgba(11, 15, 25, 0.95)',
                    padding: 10,
                    borderWidth: 1,
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    callbacks: {
                        label: function(context) {
                            const val = context.raw;
                            const total = context.dataset.data[0] + context.dataset.data[1];
                            const pct = ((val / total) * 100).toFixed(1);
                            return ` ${val} matches (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
}

// 4. Top 10 Player of the Match Leaders (Vertical Bar Chart)
function renderPlayerAwardsChart(topPlayers) {
    const ctx = document.getElementById('player-awards-chart').getContext('2d');

    const labels = topPlayers.map(item => item.player);
    const awards = topPlayers.map(item => item.awards);

    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, '#f59e0b'); // Amber Gold
    gradient.addColorStop(1, '#8b5cf6'); // Purple

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Awards Count',
                data: awards,
                backgroundColor: gradient,
                borderRadius: 6,
                barThickness: 16
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(11, 15, 25, 0.95)',
                    padding: 10,
                    borderWidth: 1,
                    borderColor: 'rgba(255, 255, 255, 0.1)'
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { maxRotation: 45, minRotation: 45 }
                },
                y: { grid: { display: true } }
            }
        }
    });
}

// 5. Venue Chasing vs Defending Analysis (Grouped Bar Chart)
function renderVenueComparisonChart(venueStats) {
    const ctx = document.getElementById('venue-comparison-chart').getContext('2d');

    // Limit to top 10
    const topVenues = venueStats.slice(0, 10);
    
    // Shorten venue names for labels so they fit nicely
    const labels = topVenues.map(item => {
        let name = item.venue;
        if (name.includes(',')) name = name.split(',')[0];
        if (name.length > 22) name = name.substring(0, 20) + '...';
        return name;
    });
    
    const batFirstWinPct = topVenues.map(item => item.bat_first_win_percentage);
    const fieldFirstWinPct = topVenues.map(item => item.field_first_win_percentage);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Defending wins %',
                    data: batFirstWinPct,
                    backgroundColor: '#8b5cf6', // Purple
                    borderRadius: 4,
                    barPercentage: 0.8,
                    categoryPercentage: 0.7
                },
                {
                    label: 'Chasing wins %',
                    data: fieldFirstWinPct,
                    backgroundColor: '#10b981', // Emerald Green
                    borderRadius: 4,
                    barPercentage: 0.8,
                    categoryPercentage: 0.7
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 12 }
                },
                tooltip: {
                    backgroundColor: 'rgba(11, 15, 25, 0.95)',
                    padding: 10,
                    borderWidth: 1,
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    callbacks: {
                        label: function(context) {
                            return ` ${context.dataset.label}: ${context.raw}%`;
                        }
                    }
                }
            },
            scales: {
                x: { grid: { display: false } },
                y: {
                    max: 100,
                    grid: { display: true },
                    ticks: { callback: value => `${value}%` }
                }
            }
        }
    });
}
