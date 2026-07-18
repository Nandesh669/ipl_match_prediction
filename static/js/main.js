document.addEventListener("DOMContentLoaded", () => {
    const team1Select = document.getElementById("team1");
    const team2Select = document.getElementById("team2");
    const tossWinnerSelect = document.getElementById("toss_winner");
    const venueSelect = document.getElementById("venue");
    const tossDecisionSelect = document.getElementById("toss_decision");
    const form = document.getElementById("predictor-form");
    
    const resultsPanel = document.getElementById("results-panel");
    const placeholder = document.getElementById("results-placeholder");
    const loading = document.getElementById("results-loading");
    const content = document.getElementById("results-content");

    // Dynamic UI Elements for results
    const predictedWinnerName = document.getElementById("predicted-winner-name");
    const predictedConfidence = document.getElementById("predicted-confidence");
    const probT1Name = document.getElementById("prob-t1-name");
    const probT2Name = document.getElementById("prob-t2-name");
    const probT1Bar = document.getElementById("prob-t1-bar");
    const probT2Bar = document.getElementById("prob-t2-bar");
    const probT1Label = document.getElementById("prob-t1-label");
    const probT2Label = document.getElementById("prob-t2-label");

    // Dynamic stats card elements
    const h2hTotalMatches = document.getElementById("h2h-total-matches");
    const h2hT1Name = document.getElementById("h2h-t1-name");
    const h2hT1Wins = document.getElementById("h2h-t1-wins");
    const h2hT2Name = document.getElementById("h2h-t2-name");
    const h2hT2Wins = document.getElementById("h2h-t2-wins");

    const venueSubLabel = document.getElementById("venue-sub-label");
    const venueT1Name = document.getElementById("venue-t1-name");
    const venueT1Record = document.getElementById("venue-t1-record");
    const venueT2Name = document.getElementById("venue-t2-name");
    const venueT2Record = document.getElementById("venue-t2-record");

    const overallT1Header = document.getElementById("overall-t1-header");
    const overallT1Rate = document.getElementById("overall-t1-rate");
    const overallT1Desc = document.getElementById("overall-t1-desc");

    const overallT2Header = document.getElementById("overall-t2-header");
    const overallT2Rate = document.getElementById("overall-t2-rate");
    const overallT2Desc = document.getElementById("overall-t2-desc");

    // Enable/Disable and update Toss Winner selection options based on Teams
    function updateTossWinnerOptions() {
        const t1 = team1Select.value;
        const t2 = team2Select.value;

        if (t1 && t2) {
            tossWinnerSelect.disabled = false;
            // Clear current options
            tossWinnerSelect.innerHTML = '<option value="" disabled selected>Select Toss Winner</option>';
            
            // Add options
            const opt1 = document.createElement("option");
            opt1.value = t1;
            opt1.textContent = t1;
            tossWinnerSelect.appendChild(opt1);

            const opt2 = document.createElement("option");
            opt2.value = t2;
            opt2.textContent = t2;
            tossWinnerSelect.appendChild(opt2);
        } else {
            tossWinnerSelect.disabled = true;
            tossWinnerSelect.innerHTML = '<option value="" disabled selected>Select Teams First</option>';
        }
    }

    team1Select.addEventListener("change", () => {
        // Prevent selecting the same team for Team 2
        const selectedT1 = team1Select.value;
        Array.from(team2Select.options).forEach(opt => {
            if (opt.value === selectedT1 && opt.value !== "") {
                opt.disabled = true;
            } else {
                opt.disabled = false;
            }
        });
        updateTossWinnerOptions();
    });

    team2Select.addEventListener("change", () => {
        // Prevent selecting the same team for Team 1
        const selectedT2 = team2Select.value;
        Array.from(team1Select.options).forEach(opt => {
            if (opt.value === selectedT2 && opt.value !== "") {
                opt.disabled = true;
            } else {
                opt.disabled = false;
            }
        });
        updateTossWinnerOptions();
    });

    // Form submission
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        
        const team1 = team1Select.value;
        const team2 = team2Select.value;
        const venue = venueSelect.value;
        const toss_winner = tossWinnerSelect.value;
        const toss_decision = tossDecisionSelect.value;

        if (!team1 || !team2 || !venue || !toss_winner || !toss_decision) {
            alert("Please select all required parameters.");
            return;
        }

        // Show loading state
        placeholder.classList.add("hidden");
        content.classList.add("hidden");
        loading.classList.remove("hidden");
        
        // Scroll results card into view for mobile users
        if (window.innerWidth <= 900) {
            resultsPanel.scrollIntoView({ behavior: 'smooth' });
        }

        // Fetch prediction
        fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                team1: team1,
                team2: team2,
                venue: venue,
                toss_winner: toss_winner,
                toss_decision: toss_decision
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || "Prediction failed") });
            }
            return response.json();
        })
        .then(data => {
            // Update UI elements
            predictedWinnerName.textContent = data.predicted_winner;
            predictedConfidence.textContent = `${data.confidence}%`;
            
            probT1Name.textContent = data.team1;
            probT2Name.textContent = data.team2;
            
            // Set widths of segments
            probT1Bar.style.width = `${data.team1_prob}%`;
            probT2Bar.style.width = `${data.team2_prob}%`;
            
            probT1Label.textContent = `${data.team1_prob}%`;
            probT2Label.textContent = `${data.team2_prob}%`;

            // Setup head to head stats
            h2hTotalMatches.textContent = data.h2h_stats.total;
            h2hT1Name.textContent = data.team1;
            h2hT1Wins.textContent = `${data.h2h_stats.team1_wins} Wins`;
            h2hT2Name.textContent = data.team2;
            h2hT2Wins.textContent = `${data.h2h_stats.team2_wins} Wins`;

            // Setup venue stats
            venueSubLabel.textContent = data.venue_stats.venue;
            venueT1Name.textContent = data.team1;
            venueT1Record.textContent = `${data.venue_stats.team1_wins} Wins of ${data.venue_stats.team1_matches}`;
            venueT2Name.textContent = data.team2;
            venueT2Record.textContent = `${data.venue_stats.team2_wins} Wins of ${data.venue_stats.team2_matches}`;

            // Setup overall team 1 stats
            overallT1Header.textContent = `${data.team1} Overall`;
            overallT1Rate.textContent = `${data.team1_stats.win_rate}%`;
            overallT1Desc.textContent = `${data.team1_stats.wins} Wins in ${data.team1_stats.matches} Matches`;

            // Setup overall team 2 stats
            overallT2Header.textContent = `${data.team2} Overall`;
            overallT2Rate.textContent = `${data.team2_stats.win_rate}%`;
            overallT2Desc.textContent = `${data.team2_stats.wins} Wins in ${data.team2_stats.matches} Matches`;

            // Reveal content
            loading.classList.add("hidden");
            content.classList.remove("hidden");
        })
        .catch(err => {
            console.error(err);
            alert(`Error: ${err.message}`);
            loading.classList.add("hidden");
            placeholder.classList.remove("hidden");
        });
    });
});
