/**
 * ShadowHawk Dashboard Interaction Logic
 */

let serviceChart;

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    fetchData();
    setInterval(fetchData, 5000); // Polling every 5s
});

function initCharts() {
    const ctx = document.getElementById('serviceChart').getContext('2d');
    serviceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: ['#00F5FF', '#FF3131', '#238636', '#8B949E'],
                borderWidth: 0,
                hoverOffset: 12
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#8B949E', usePointStyle: true, padding: 20 } }
            },
            cutout: '75%'
        }
    });
}

async function fetchData() {
    try {
        // More logs for the attack logs page
        const limit = window.location.pathname === '/logs' ? 500 : 50;
        const response = await fetch(`/api/stats?limit=${limit}`);
        const data = await response.json();

        if (document.getElementById('stat-total')) updateStats(data.stats);
        if (document.getElementById('attack-body')) updateTable(data.recent);
        if (document.getElementById('top-ips-list')) updateTopAdversaries(data.stats.top_ips);
        if (serviceChart && data.stats.services) updateServiceChart(data.stats.services);

    } catch (err) {
        console.error('Error fetching dashboard data:', err);
    }
}


function updateStats(stats) {
    document.getElementById('stat-total').innerText = stats.total;
    document.getElementById('stat-ips').innerText = Object.keys(stats.top_ips).length;
}

function updateTable(attacks) {
    const body = document.getElementById('attack-body');
    // For simplicity, we'll clear and rebuild for now, 
    // but in a production app we'd prepend only new entries.
    body.innerHTML = '';

    attacks.forEach(atk => {
        const row = document.createElement('tr');
        // Structure: (id, ts, ip, svc, payload, user, pw, city, country, lat, lon)
        const ts = new Date(atk[1]).toLocaleString();
        const svc = atk[3];
        const ip = atk[2];
        const location = atk[7] && atk[8] ? `${atk[7]}, ${atk[8]}` : 'Unknown';
        const payload = atk[4] || (atk[5] ? `Login: ${atk[5]}:${atk[6]}` : '-');

        row.innerHTML = `
            <td class="timestamp">${ts}</td>
            <td><span class="badge ${svc.includes('ssh') ? 'ssh' : 'http'}">${svc.toUpperCase()}</span></td>
            <td class="origin-ip">${ip}</td>
            <td>${location}</td>
            <td><code class="payload-snippet" title="${payload}">${payload}</code></td>
        `;
        body.appendChild(row);
    });
}

function updateTopAdversaries(topIps) {
    const list = document.getElementById('top-ips-list');
    list.innerHTML = '';

    Object.entries(topIps).forEach(([ip, count]) => {
        const li = document.createElement('li');
        li.innerHTML = `<span class="ip">${ip}</span> <span class="count">${count}pts</span>`;
        list.appendChild(li);
    });
}

function updateServiceChart(services) {
    serviceChart.data.labels = Object.keys(services);
    serviceChart.data.datasets[0].data = Object.values(services);
    serviceChart.update();
}
