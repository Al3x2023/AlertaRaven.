// app.js - Versión modularizada para componentes
document.addEventListener('DOMContentLoaded', function() {
    // Esperar a que todos los componentes estén cargados
    document.addEventListener('componentsLoaded', function() {
        console.log('🚀 Todos los componentes cargados, inicializando aplicación AlertaRaven...');
        
        // Esperar a que las librerías externas estén disponibles
        if (typeof Chart !== 'undefined' && typeof L !== 'undefined') {
            AlertaRavenApp.initialize();
        } else {
            // Reintentar después de un segundo si las librerías no están cargadas
            setTimeout(() => {
                if (typeof Chart !== 'undefined' && typeof L !== 'undefined') {
                    AlertaRavenApp.initialize();
                } else {
                    console.error('❌ Librerías Chart.js o Leaflet no disponibles');
                    showErrorNotification('Error: Librerías necesarias no cargaron correctamente');
                }
            }, 1000);
        }
    });

    // Fallback: si el evento componentsLoaded no se dispara después de 5 segundos
    setTimeout(() => {
        if (!window.componentsLoaded) {
            console.warn('⚠️ componentsLoaded no se disparó, intentando inicializar de todos modos...');
            if (typeof Chart !== 'undefined' && typeof L !== 'undefined') {
                AlertaRavenApp.initialize();
            }
        }
    }, 5000);
});

// Módulo principal de la aplicación
const AlertaRavenApp = (function() {
    // Estado global de la aplicación
    const state = {
        currentSection: 'dashboard',
        websocket: null,
        reconnectInterval: null,
        currentAlerts: [],
        currentStats: {},
        filters: {
            date: '',
            type: '',
            status: '',
            search: ''
        },
        tableState: {
            allAlerts: [],
            filteredAlerts: [],
            currentPage: 1,
            itemsPerPage: 10,
            currentSort: { field: 'timestamp', direction: 'desc' }
        },
        charts: {
            accidentType: null,
            alertsTrend: null,
            status: null,
            performance: null
        },
        map: null,
        dashboardMap: null,
        heatmapEnabled: false,
        heatmapLayer: null,
        locationFilter: {
            enabled: false,
            userLocation: null,
            radius: 5, // km
            circle: null,
            marker: null
        },
        systemLogs: [],
        componentsReady: false
    };

    // Configuración
    const config = {
        websocket: {
            reconnectDelay: 5000,
            maxReconnectAttempts: 5
        },
        api: {
            baseUrl: '/api',
            endpoints: {
                alerts: '/alerts',
                statistics: '/statistics',
                system: '/system'
            }
        },
        updateIntervals: {
            charts: 30000,
            map: 30000,
            system: 30000,
            logs: 10000
        }
    };

    // Inicialización de la aplicación
    function initialize() {
        console.log('🎯 Inicializando AlertaRaven App...');
        
        // Verificar que los componentes principales estén presentes
        if (!checkRequiredComponents()) {
            console.error('❌ Componentes requeridos no encontrados');
            return;
        }

        setupNavigation();
        setupWebSocket();
        setupEventListeners();
        loadInitialData();
        initializeCharts();
        initializeMap();
        initializeDashboardMap();
        initializeSystemPanel();
        
        startPeriodicUpdates();
        
        state.componentsReady = true;
        console.log('✅ AlertaRaven App inicializada correctamente');
    }

    // Verificar componentes requeridos
    function checkRequiredComponents() {
        const requiredElements = [
            'sidebar-component',
            'header-component',
            'dashboard-component',
            'alerts-component',
            'map-component',
            'statistics-component',
            'system-component'
        ];

        const missingElements = requiredElements.filter(id => !document.getElementById(id));
        
        if (missingElements.length > 0) {
            console.warn('⚠️ Elementos faltantes:', missingElements);
            return false;
        }
        
        return true;
    }

    // Configuración de navegación
    function setupNavigation() {
        // Usar event delegation para manejar la navegación dinámica
        document.addEventListener('click', function(e) {
            const navItem = e.target.closest('.nav-item');
            if (navItem && navItem.dataset.section) {
                e.preventDefault();
                const section = navItem.dataset.section;
                switchSection(section);
            }
        });

        // También configurar listeners directos para elementos existentes
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', function(e) {
                e.preventDefault();
                const section = this.dataset.section;
                switchSection(section);
            });
        });
    }

    function switchSection(section) {
        console.log(`🔄 Cambiando a sección: ${section}`);
        
        // Actualizar navegación activa
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        
        const activeNav = document.querySelector(`[data-section="${section}"]`);
        if (activeNav) {
            activeNav.classList.add('active');
        }

        // Ocultar todas las secciones
        document.querySelectorAll('.content-section').forEach(sec => {
            sec.classList.remove('active');
        });

        // Mostrar sección seleccionada
        const targetSection = document.getElementById(`${section}-section`);
        if (targetSection) {
            targetSection.classList.add('active');
        } else {
            console.warn(`⚠️ Sección ${section}-section no encontrada`);
            return;
        }

        // Actualizar título
        const titles = {
            'dashboard': 'Dashboard',
            'alerts': 'Alertas',
            'map': 'Mapa',
            'statistics': 'Estadísticas',
            'system': 'Sistema'
        };
        
        const pageTitle = document.getElementById('page-title');
        if (pageTitle) {
            pageTitle.textContent = titles[section] || 'AlertaRaven';
        }

        state.currentSection = section;
        
        // Cargar datos específicos de la sección
        loadSectionData(section);
    }

    function loadSectionData(section) {
        console.log(`📊 Cargando datos para sección: ${section}`);
        
        switch(section) {
            case 'dashboard':
                loadDashboardData();
                break;
            case 'alerts':
                loadAlertsTable();
                break;
            case 'map':
                loadMapData();
                break;
            case 'statistics':
                loadStatisticsData();
                break;
            case 'system':
                loadSystemData();
                break;
        }
    }

    // Gestión de WebSocket
    function setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        try {
            state.websocket = new WebSocket(wsUrl);
            
            state.websocket.onopen = function() {
                console.log('🔌 WebSocket conectado');
                updateConnectionStatus(true);
                clearReconnectInterval();
            };
            
            state.websocket.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                } catch (error) {
                    console.error('❌ Error parseando mensaje WebSocket:', error);
                }
            };
            
            state.websocket.onclose = function() {
                console.log('🔌 WebSocket desconectado');
                updateConnectionStatus(false);
                scheduleReconnect();
            };
            
            state.websocket.onerror = function(error) {
                console.error('❌ Error de WebSocket:', error);
                updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('❌ Error al crear WebSocket:', error);
            updateConnectionStatus(false);
            scheduleReconnect();
        }
    }

    function handleWebSocketMessage(data) {
        console.log('📨 Mensaje WebSocket recibido:', data);
        // El backend envía la carga útil en 'data'; mantener compatibilidad con 'alert' si existiera
        const payload = data.data || data.alert || data;
        
        switch (data.type) {
            case 'new_alert':
                // Notificación de nueva alerta enviada desde el backend
                handleNewAlert(payload);
                break;
            case 'alert_status_change':
                // Cambio de estado de alerta (backend usa 'alert_status_change')
                handleAlertUpdated(payload);
                break;
            case 'system_status':
                // Estado del sistema viene en 'data' { status, message }
                handleSystemStatus(payload);
                break;
            default:
                console.log('📨 Tipo de mensaje no manejado:', data.type);
        }
    }

    function handleNewAlert(alert) {
        showNotification('🆕 Nueva alerta recibida', 'info');
        
        // Verificar si es una alerta crítica
        checkCriticalAlerts([alert]);
        
        // Actualizar datos si estamos en las secciones relevantes
        if (['dashboard', 'alerts'].includes(state.currentSection)) {
            loadInitialData();
        }
        
        // Actualizar componentes visuales
        updateVisualComponents();
    }

    function handleAlertUpdated(alert) {
        showNotification('📝 Alerta actualizada', 'info');
        
        // Actualizar datos locales
        const index = state.currentAlerts.findIndex(a => a.alert_id === alert.alert_id);
        if (index !== -1) {
            state.currentAlerts[index] = alert;
        }
        
        // Refrescar estadísticas y lista para reflejar cambios inmediatos
        loadInitialData();

        // Actualizar interfaz
        if (state.currentSection === 'alerts') {
            loadAlertsTable();
        } else {
            updateRecentAlertsDisplay(state.currentAlerts);
        }
    }

    // Manejo básico de estado del sistema recibido por WebSocket
    function handleSystemStatus(statusInfo) {
        try {
            const msg = statusInfo && statusInfo.message ? statusInfo.message : 'Actualización de sistema';
            showNotification(`🔧 ${msg}`, 'info');
            // Podríamos refrescar panel del sistema si está visible
            if (state.currentSection === 'system') {
                updateSystemStatus();
            }
        } catch (e) {
            console.warn('⚠️ Error manejando system_status:', e);
        }
    }

    function updateConnectionStatus(connected) {
        const indicator = document.getElementById('connection-status');
        const text = document.getElementById('connection-text');
        
        if (!indicator || !text) {
            console.warn('⚠️ Elementos de conexión no encontrados');
            return;
        }
        
        if (connected) {
            indicator.classList.add('connected');
            indicator.classList.remove('disconnected');
            text.textContent = 'Conectado';
        } else {
            indicator.classList.remove('connected');
            indicator.classList.add('disconnected');
            text.textContent = 'Desconectado';
        }
    }

    function scheduleReconnect() {
        if (state.reconnectInterval) return;
        
        state.reconnectInterval = setInterval(() => {
            console.log('🔄 Intentando reconectar WebSocket...');
            setupWebSocket();
        }, config.websocket.reconnectDelay);
    }

    function clearReconnectInterval() {
        if (state.reconnectInterval) {
            clearInterval(state.reconnectInterval);
            state.reconnectInterval = null;
        }
    }

    // Gestión de datos
    async function loadInitialData() {
        try {
            await Promise.all([
                loadStatistics(),
                loadRecentAlerts()
            ]);
        } catch (error) {
            console.error('❌ Error cargando datos iniciales:', error);
            showNotification('❌ Error cargando datos', 'error');
        }
    }

    async function loadStatistics() {
        try {
            const response = await fetch(`${config.api.baseUrl}${config.api.endpoints.statistics}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            state.currentStats = await response.json();
            updateStatisticsDisplay();
        } catch (error) {
            console.error('❌ Error cargando estadísticas:', error);
            throw error;
        }
    }

    function updateStatisticsDisplay() {
        const stats = state.currentStats;
        
        updateElementText('total-alerts', stats.total_alerts || 0);
        updateElementText('active-alerts', stats.active_alerts || 0);
        updateElementText('last-24h-alerts', stats.last_24h || 0);
        
        // Calcular alertas resueltas
        const resolved = stats.status_distribution?.COMPLETED || 0;
        updateElementText('resolved-alerts', resolved);
        
        // Actualizar métricas avanzadas si existen
        if (stats.metrics) {
            updateElementText('avg-response-time', `${stats.metrics.avg_response_time || 0} min`);
            updateElementText('alerts-per-hour', stats.metrics.alerts_per_hour || 0);
            updateElementText('critical-alerts', stats.metrics.critical_alerts || 0);
        }
    }

    async function loadRecentAlerts(limit = 10) {
        try {
            const response = await fetch(`${config.api.baseUrl}${config.api.endpoints.alerts}?limit=${limit}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const alerts = await response.json();
            state.currentAlerts = Array.isArray(alerts) ? alerts : [];
            updateRecentAlertsDisplay(state.currentAlerts);
        } catch (error) {
            console.error('❌ Error cargando alertas recientes:', error);
            throw error;
        }
    }

    function updateRecentAlertsDisplay(alerts) {
        const container = document.getElementById('recent-alerts-list');
        if (!container) {
            console.warn('⚠️ Contenedor de alertas recientes no encontrado');
            return;
        }
        
        if (!alerts || alerts.length === 0) {
            container.innerHTML = '<p class="text-center text-secondary">No hay alertas recientes</p>';
            return;
        }
        
        container.innerHTML = alerts.map(alert => `
            <div class="alert-item" onclick="AlertaRavenApp.showAlertDetails('${alert.alert_id}')">
                <div class="alert-header">
                    <span class="alert-type">${formatAccidentType(alert.accident_type)}</span>
                    <span class="alert-status ${(alert.status || '').toLowerCase()}">${formatStatus(alert.status)}</span>
                </div>
                <div class="alert-info">
                    <div>Dispositivo: ${alert.device_id || 'N/A'}</div>
                    <div>Confianza: ${alert.confidence || 0}%</div>
                    <div class="alert-time">${formatDateTime(alert.created_at || alert.timestamp)}</div>
                </div>
            </div>
        `).join('');
    }

    // Funciones del dashboard
    function loadDashboardData() {
        loadStatistics();
        loadRecentAlerts();
        loadModelMetrics();
        updateCharts();
    }

    // Gestión de gráficos
    function initializeCharts() {
        initializeAccidentTypeChart();
        initializeAlertsTrendChart();
        initializeStatusChart();
    }

    function initializeAccidentTypeChart() {
        const ctx = document.getElementById('accident-type-chart');
        if (!ctx) {
            console.warn('⚠️ Canvas para gráfico de tipos no encontrado');
            return;
        }

        state.charts.accidentType = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Colisión', 'Caída', 'Médica', 'Incendio', 'Otros'],
                datasets: [{
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: [
                        '#FF6384',
                        '#36A2EB',
                        '#FFCE56',
                        '#4BC0C0',
                        '#9966FF'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    function initializeAlertsTrendChart() {
        const ctx = document.getElementById('alerts-trend-chart');
        if (!ctx) {
            console.warn('⚠️ Canvas para gráfico de tendencias no encontrado');
            return;
        }

        const hours = Array.from({length: 24}, (_, i) => {
            const hour = new Date();
            hour.setHours(hour.getHours() - (23 - i));
            return hour.getHours() + ':00';
        });

        state.charts.alertsTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: hours,
                datasets: [{
                    label: 'Alertas por Hora',
                    data: new Array(24).fill(0),
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    function initializeStatusChart() {
        const ctx = document.getElementById('status-chart');
        if (!ctx) {
            console.warn('⚠️ Canvas para gráfico de estados no encontrado');
            return;
        }

        state.charts.status = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Pendiente', 'En Progreso', 'Completado', 'Cancelado'],
                datasets: [{
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        '#f39c12',
                        '#3498db',
                        '#27ae60',
                        '#e74c3c'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    // ==============================
    // Métricas de modelo
    // ==============================
    async function loadModelMetrics() {
        try {
            const response = await fetch(`/api/v1/metrics/model`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();
            renderModelMetrics(data);
        } catch (error) {
            console.error('❌ Error cargando métricas de modelo:', error);
            const statusEl = document.getElementById('model-metrics-status');
            if (statusEl) statusEl.textContent = 'Error cargando métricas';
        }
    }

    function renderModelMetrics(data) {
        const statusEl = document.getElementById('model-metrics-status');
        if (!data || data.available === false) {
            updateElementText('model-precision', '--');
            updateElementText('model-recall', '--');
            updateElementText('model-f1', '--');
            if (statusEl) statusEl.textContent = 'Sin datos de predicción';
            const container = document.getElementById('confusion-matrix-container');
            if (container) {
                container.innerHTML = '<p>No hay datos suficientes para construir la matriz de confusión.</p>';
            }
            return;
        }

        if (statusEl) statusEl.textContent = 'Actualizado';

        const overall = data.overall || {};
        updateElementText('model-precision', formatPercentage(overall.precision));
        updateElementText('model-recall', formatPercentage(overall.recall));
        updateElementText('model-f1', formatPercentage(overall.f1));

        renderConfusionMatrix(data.classes, data.confusion_matrix);
    }

    function formatPercentage(value) {
        if (value === undefined || value === null) return '--';
        return `${Math.round(value * 100)}%`;
    }

    function renderConfusionMatrix(classes, matrix) {
        const container = document.getElementById('confusion-matrix-container');
        if (!container) return;

        // Construir tabla HTML
        let html = '<table class="matrix-table">';
        // Header
        html += '<thead><tr><th>Real \\ Pred.</th>' + classes.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr></thead>';
        // Body
        html += '<tbody>';
        classes.forEach(real => {
            html += `<tr><th>${escapeHtml(real)}</th>`;
            classes.forEach(pred => {
                const value = (matrix[real] && matrix[real][pred]) ? matrix[real][pred] : 0;
                const cellClass = real === pred ? 'tp' : (value > 0 ? 'offdiag' : 'zero');
                html += `<td class="${cellClass}">${value}</td>`;
            });
            html += '</tr>';
        });
        html += '</tbody></table>';
        container.innerHTML = html;
    }

    function escapeHtml(text) {
        const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
        return String(text).replace(/[&<>"']/g, function(m) { return map[m]; });
    }

    function updateCharts() {
        if (state.currentAlerts.length === 0) {
            console.log('📊 No hay datos para actualizar gráficos');
            return;
        }
        
        updateAccidentTypeChart(state.currentAlerts);
        updateAlertsTrendChart(state.currentAlerts);
        updateStatusChart(state.currentAlerts);
        updateAdvancedMetrics(state.currentAlerts);
    }

    function updateAccidentTypeChart(alerts) {
        if (!state.charts.accidentType) return;

        const typeCounts = {
            'COLLISION': 0,
            'FALL': 0,
            'MEDICAL': 0,
            'FIRE': 0,
            'OTHER': 0
        };

        alerts.forEach(alert => {
            const type = alert.accident_type || 'OTHER';
            if (typeCounts.hasOwnProperty(type)) {
                typeCounts[type]++;
            } else {
                typeCounts['OTHER']++;
            }
        });

        state.charts.accidentType.data.datasets[0].data = [
            typeCounts.COLLISION,
            typeCounts.FALL,
            typeCounts.MEDICAL,
            typeCounts.FIRE,
            typeCounts.OTHER
        ];
        state.charts.accidentType.update();
    }

    function updateAlertsTrendChart(alerts) {
        if (!state.charts.alertsTrend) return;

        const hourCounts = new Array(24).fill(0);
        const now = new Date();

        alerts.forEach(alert => {
            const alertTime = new Date(alert.timestamp);
            const hoursDiff = Math.floor((now - alertTime) / (1000 * 60 * 60));
            if (hoursDiff >= 0 && hoursDiff < 24) {
                hourCounts[23 - hoursDiff]++;
            }
        });

        state.charts.alertsTrend.data.datasets[0].data = hourCounts;
        state.charts.alertsTrend.update();
    }

    function updateStatusChart(alerts) {
        if (!state.charts.status) return;

        const statusCounts = {
            'PENDING': 0,
            'IN_PROGRESS': 0,
            'COMPLETED': 0,
            'CANCELLED': 0
        };

        alerts.forEach(alert => {
            const status = alert.status || 'PENDING';
            if (statusCounts.hasOwnProperty(status)) {
                statusCounts[status]++;
            }
        });

        state.charts.status.data.datasets[0].data = [
            statusCounts.PENDING,
            statusCounts.IN_PROGRESS,
            statusCounts.COMPLETED,
            statusCounts.CANCELLED
        ];
        state.charts.status.update();
    }

    function updateAdvancedMetrics(alerts) {
        // Tiempo promedio de respuesta
        let totalResponseTime = 0;
        let completedAlerts = 0;

        alerts.forEach(alert => {
            if (alert.status === 'COMPLETED' && alert.response_time) {
                totalResponseTime += alert.response_time;
                completedAlerts++;
            }
        });

        const avgResponseTime = completedAlerts > 0 ? Math.round(totalResponseTime / completedAlerts) : 0;
        updateElementText('avg-response-time', `${avgResponseTime} min`);

        // Alertas por hora (última hora)
        const lastHour = new Date(Date.now() - 60 * 60 * 1000);
        const recentAlerts = alerts.filter(alert => new Date(alert.timestamp) > lastHour);
        updateElementText('alerts-per-hour', recentAlerts.length);

        // Alertas críticas
        const criticalAlerts = alerts.filter(alert => 
            alert.severity === 'CRITICAL' || alert.accident_type === 'MEDICAL'
        );
        updateElementText('critical-alerts', criticalAlerts.length);

        // Conexiones activas (simulado)
        updateElementText('active-connections', Math.floor(Math.random() * 10) + 5);
    }

    // Funciones adicionales del dashboard
    function refreshRecentAlerts() {
        console.log('🔄 Actualizando alertas recientes...');
        loadRecentAlerts()
            .then(() => {
                showNotification('✅ Alertas recientes actualizadas', 'success');
            })
            .catch(error => {
                console.error('❌ Error actualizando alertas recientes:', error);
                showNotification('❌ Error actualizando alertas recientes', 'error');
            });
    }

    function downloadChart(chartId) {
        console.log(`📊 Descargando gráfico: ${chartId}`);
        
        const canvas = document.getElementById(chartId);
        if (!canvas) {
            console.warn(`⚠️ Canvas ${chartId} no encontrado`);
            showNotification('❌ Gráfico no encontrado', 'error');
            return;
        }

        try {
            // Crear enlace de descarga
            const link = document.createElement('a');
            link.download = `${chartId}_${new Date().toISOString().split('T')[0]}.png`;
            link.href = canvas.toDataURL('image/png');
            
            // Simular click para descargar
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            showNotification('📊 Gráfico descargado exitosamente', 'success');
        } catch (error) {
            console.error('❌ Error descargando gráfico:', error);
            showNotification('❌ Error descargando gráfico', 'error');
        }
    }

    // Gestión de alertas
 async function loadAlertsTable() {
    try {
        console.log('🔄 Cargando tabla de alertas...');
        
        const { date, type, status } = state.filters;
        
        let url = `${config.api.baseUrl}${config.api.endpoints.alerts}?limit=100`;
        if (status) url += `&status=${status}`;
        if (type) url += `&accident_type=${type}`;
        
        console.log('📡 URL de API:', url);
        
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        
        const data = await response.json();
        console.log('📨 Datos recibidos:', data);
        
        // Verificar la estructura de los datos
        if (data.alerts && Array.isArray(data.alerts)) {
            state.tableState.allAlerts = data.alerts;
            console.log(`✅ ${data.alerts.length} alertas cargadas`);
        } else if (Array.isArray(data)) {
            state.tableState.allAlerts = data;
            console.log(`✅ ${data.length} alertas cargadas (formato directo)`);
        } else {
            console.warn('⚠️ Formato de datos inesperado:', data);
            state.tableState.allAlerts = [];
        }
        
        applyTableFilters();
        
    } catch (error) {
        console.error('❌ Error cargando tabla de alertas:', error);
        showNotification('❌ Error cargando alertas', 'error');
        
        // Mostrar datos de ejemplo para debugging
        console.log('🔄 Cargando datos de ejemplo...');
        loadSampleData();
    }
}

// Función de ayuda para datos de ejemplo
function loadSampleData() {
    const sampleAlerts = [
        {
            alert_id: 'alert_001',
            device_id: 'device_123',
            accident_type: 'COLLISION',
            status: 'PENDING',
            confidence: 85,
            timestamp: new Date().toISOString(),
            location_data: {
                latitude: 40.4168,
                longitude: -3.7038
            }
        },
        {
            alert_id: 'alert_002', 
            device_id: 'device_456',
            accident_type: 'FALL',
            status: 'IN_PROGRESS',
            confidence: 92,
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            location_data: {
                latitude: 40.4175,
                longitude: -3.7042
            }
        }
    ];
    
    state.tableState.allAlerts = sampleAlerts;
    applyTableFilters();
    showNotification('📋 Mostrando datos de ejemplo', 'info');
}

    function applyTableFilters() {
        let filtered = [...state.tableState.allAlerts];

        // Filtro por fecha
        if (state.filters.date) {
            filtered = filterByDate(filtered, state.filters.date);
        }

        // Filtro por tipo
        if (state.filters.type) {
            filtered = filtered.filter(alert => 
                alert.accident_type === state.filters.type
            );
        }

        // Filtro por estado
        if (state.filters.status) {
            filtered = filtered.filter(alert => 
                alert.status === state.filters.status
            );
        }

        // Filtro por búsqueda
        if (state.filters.search) {
            filtered = filtered.filter(alert => 
                (alert.alert_id && alert.alert_id.toString().includes(state.filters.search)) ||
                (alert.device_id && alert.device_id.toString().includes(state.filters.search)) ||
                (alert.accident_type && alert.accident_type.toLowerCase().includes(state.filters.search))
            );
        }

        state.tableState.filteredAlerts = filtered;
        sortAlertsData();
        displayAlertsTable();
        generatePagination();
    }

    function filterByDate(alerts, dateFilter) {
        const now = new Date();
        let startDate;

        switch (dateFilter) {
            case 'today':
                startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate());
                break;
            case 'yesterday':
                startDate = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 1);
                break;
            case 'week':
                startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
                break;
            case 'month':
                startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
                break;
            default:
                return alerts;
        }

        return alerts.filter(alert => {
            const alertDate = new Date(alert.timestamp || alert.created_at);
            return alertDate >= startDate;
        });
    }

    function searchAlertsTable() {
        const searchInput = document.getElementById('table-search');
        const searchTerm = searchInput?.value.toLowerCase() || '';
        state.filters.search = searchTerm;
        applyTableFilters();
    }

    function sortAlertsTable() {
        const sortSelect = document.getElementById('table-sort');
        const sortValue = sortSelect?.value;
        if (!sortValue) return;
        
        const [field, direction] = sortValue.split('-');
        state.tableState.currentSort = { field, direction };
        sortAlertsData();
        displayAlertsTable();
    }

    function sortTableBy(field) {
        if (state.tableState.currentSort.field === field) {
            state.tableState.currentSort.direction = 
                state.tableState.currentSort.direction === 'asc' ? 'desc' : 'asc';
        } else {
            state.tableState.currentSort.field = field;
            state.tableState.currentSort.direction = 'asc';
        }
        
        sortAlertsData();
        displayAlertsTable();
    }

    function sortAlertsData() {
        const { field, direction } = state.tableState.currentSort;
        
        state.tableState.filteredAlerts.sort((a, b) => {
            let aValue, bValue;
            
            switch (field) {
                case 'id':
                    aValue = a.alert_id || '';
                    bValue = b.alert_id || '';
                    break;
                case 'timestamp':
                    aValue = new Date(a.timestamp || a.created_at);
                    bValue = new Date(b.timestamp || b.created_at);
                    break;
                case 'type':
                    aValue = a.accident_type || '';
                    bValue = b.accident_type || '';
                    break;
                case 'status':
                    aValue = a.status || '';
                    bValue = b.status || '';
                    break;
                case 'location':
                    aValue = a.location_data ? `${a.location_data.latitude},${a.location_data.longitude}` : '';
                    bValue = b.location_data ? `${b.location_data.latitude},${b.location_data.longitude}` : '';
                    break;
                default:
                    return 0;
            }
            
            if (direction === 'asc') {
                return aValue > bValue ? 1 : -1;
            } else {
                return aValue < bValue ? 1 : -1;
            }
        });
    }

    function displayAlertsTable() {
        const tbody = document.getElementById('alerts-table-body');
        if (!tbody) {
            console.warn('⚠️ Tabla de alertas no encontrada');
            return;
        }
        
        const { currentPage, itemsPerPage, filteredAlerts } = state.tableState;
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        const pageAlerts = filteredAlerts.slice(startIndex, endIndex);
        
        if (pageAlerts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center">No se encontraron alertas</td></tr>';
            return;
        }
        
        tbody.innerHTML = pageAlerts.map(alert => `
            <tr>
                <td>${(alert.alert_id || '').substring(0, 8)}...</td>
                <td>${formatDateTime(alert.timestamp || alert.created_at)}</td>
                <td>${formatAccidentType(alert.accident_type)}</td>
                <td>
                    <span class="status-badge status-${(alert.status || 'pending').toLowerCase()}">
                        ${formatStatus(alert.status)}
                    </span>
                </td>
                <td>
                    ${alert.location_data ? 
                        `${(alert.location_data.latitude || 0).toFixed(4)}, ${(alert.location_data.longitude || 0).toFixed(4)}` : 
                        'No disponible'
                    }
                </td>
                <td>
                    <button class="btn-action" onclick="AlertaRavenApp.showAlertDetails('${alert.alert_id}')" title="Ver detalles">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="btn-action" onclick="AlertaRavenApp.updateAlertStatus('${alert.alert_id}', 'IN_PROGRESS')" title="Marcar en progreso">
                        <i class="fas fa-play"></i>
                    </button>
                    <button class="btn-action" onclick="AlertaRavenApp.updateAlertStatus('${alert.alert_id}', 'COMPLETED')" title="Marcar completado">
                        <i class="fas fa-check"></i>
                    </button>
                </td>
            </tr>
        `).join('');
        
        // Actualizar información de la tabla
        updateTableInfo();
    }

    function updateTableInfo() {
        const { filteredAlerts, allAlerts } = state.tableState;
        const countElement = document.getElementById('table-count');
        const totalElement = document.getElementById('table-total');
        
        if (countElement) countElement.textContent = filteredAlerts.length;
        if (totalElement) totalElement.textContent = allAlerts.length;
    }

    function generatePagination() {
        const pagination = document.getElementById('alerts-pagination');
        if (!pagination) return;
        
        const { filteredAlerts, currentPage, itemsPerPage } = state.tableState;
        const totalPages = Math.ceil(filteredAlerts.length / itemsPerPage);
        
        if (totalPages <= 1) {
            pagination.innerHTML = '';
            return;
        }
        
        let paginationHTML = '';
        
        // Botón anterior
        paginationHTML += `
            <button ${currentPage === 1 ? 'disabled' : ''} 
                    onclick="AlertaRavenApp.changePage(${currentPage - 1})">
                <i class="fas fa-chevron-left"></i>
            </button>
        `;
        
        // Números de página
        for (let i = 1; i <= totalPages; i++) {
            if (i === 1 || i === totalPages || (i >= currentPage - 2 && i <= currentPage + 2)) {
                paginationHTML += `
                    <button class="${i === currentPage ? 'active' : ''}" 
                            onclick="AlertaRavenApp.changePage(${i})">
                        ${i}
                    </button>
                `;
            } else if (i === currentPage - 3 || i === currentPage + 3) {
                paginationHTML += '<span>...</span>';
            }
        }
        
        // Botón siguiente
        paginationHTML += `
            <button ${currentPage === totalPages ? 'disabled' : ''} 
                    onclick="AlertaRavenApp.changePage(${currentPage + 1})">
                <i class="fas fa-chevron-right"></i>
            </button>
        `;
        
        pagination.innerHTML = paginationHTML;
    }

    function changePage(page) {
        const totalPages = Math.ceil(state.tableState.filteredAlerts.length / state.tableState.itemsPerPage);
        if (page >= 1 && page <= totalPages) {
            state.tableState.currentPage = page;
            displayAlertsTable();
            generatePagination();
        }
    }

    // Gestión del mapa
    function initializeMap() {
        const mapContainer = document.getElementById('alerts-map');
        if (!mapContainer) {
            console.warn('⚠️ Contenedor del mapa no encontrado');
            return;
        }

        try {
            // Coordenadas por defecto (Madrid, España)
            state.map = L.map('alerts-map').setView([40.4168, -3.7038], 6);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(state.map);

            updateMapMarkers();
            console.log('🗺️ Mapa inicializado correctamente');
        } catch (error) {
            console.error('❌ Error inicializando mapa:', error);
        }
    }

    function updateMapMarkers() {
        if (!state.map) return;

        // Limpiar marcadores existentes
        state.map.eachLayer(layer => {
            if (layer instanceof L.Marker) {
                state.map.removeLayer(layer);
            }
        });

        // Agregar marcadores de alertas actuales
        state.currentAlerts.forEach(alert => {
            if (alert.location_data && alert.location_data.latitude && alert.location_data.longitude) {
                try {
                    const marker = L.marker([alert.location_data.latitude, alert.location_data.longitude])
                        .addTo(state.map);
                    
                    const popupContent = `
                        <div>
                            <strong>${formatAccidentType(alert.accident_type)}</strong><br>
                            <small>${formatDateTime(alert.timestamp)}</small><br>
                            Estado: ${formatStatus(alert.status)}<br>
                            Confianza: ${alert.confidence || 0}%
                        </div>
                    `;
                    marker.bindPopup(popupContent);
                } catch (error) {
                    console.error('❌ Error agregando marcador:', error);
                }
            }
        });
    }

    function loadMapData() {
        updateMapMarkers();
    }

    function loadStatisticsData() {
        console.log('📊 Cargando datos de estadísticas');
        initializeStatistics();
        loadDetailedStatistics();
    }

    function refreshMap() {
        updateMapMarkers();
        showNotification('🗺️ Mapa actualizado', 'success');
    }

    function toggleHeatmap() {
        if (!state.map) {
            console.warn('⚠️ Mapa no inicializado');
            return;
        }

        const button = document.getElementById('heatmap-toggle');
        if (!button) return;

        try {
            if (state.heatmapEnabled) {
                // Desactivar mapa de calor
                if (state.heatmapLayer) {
                    state.map.removeLayer(state.heatmapLayer);
                    state.heatmapLayer = null;
                }
                state.heatmapEnabled = false;
                button.innerHTML = '<i class="fas fa-fire"></i> Mapa de Calor';
                button.classList.remove('btn-danger');
                button.classList.add('btn-primary');
                
                // Mostrar marcadores normales
                updateMapMarkers();
                showNotification('🗺️ Mapa de calor desactivado', 'info');
            } else {
                // Activar mapa de calor
                const heatmapData = state.currentAlerts
                    .filter(alert => alert.location_data && alert.location_data.latitude && alert.location_data.longitude)
                    .map(alert => {
                        const intensity = getAlertIntensity(alert);
                        return [alert.location_data.latitude, alert.location_data.longitude, intensity];
                    });

                if (heatmapData.length > 0) {
                    // Ocultar marcadores normales
                    state.map.eachLayer(layer => {
                        if (layer instanceof L.Marker) {
                            state.map.removeLayer(layer);
                        }
                    });

                    // Crear capa de mapa de calor (simulado con círculos)
                    state.heatmapLayer = L.layerGroup();
                    
                    heatmapData.forEach(([lat, lng, intensity]) => {
                        const circle = L.circle([lat, lng], {
                            color: getHeatmapColor(intensity),
                            fillColor: getHeatmapColor(intensity),
                            fillOpacity: 0.6,
                            radius: intensity * 1000 // Radio basado en intensidad
                        });
                        state.heatmapLayer.addLayer(circle);
                    });

                    state.map.addLayer(state.heatmapLayer);
                    state.heatmapEnabled = true;
                    button.innerHTML = '<i class="fas fa-fire"></i> Vista Normal';
                    button.classList.remove('btn-primary');
                    button.classList.add('btn-danger');
                    showNotification('🔥 Mapa de calor activado', 'success');
                } else {
                    showNotification('❌ No hay datos suficientes para el mapa de calor', 'warning');
                }
            }
        } catch (error) {
            console.error('❌ Error en toggleHeatmap:', error);
            showNotification('❌ Error al cambiar vista del mapa', 'error');
        }
    }

    function getAlertIntensity(alert) {
        // Calcular intensidad basada en tipo de accidente y confianza
        const typeWeights = {
            'TRAFFIC_ACCIDENT': 0.8,
            'MEDICAL_EMERGENCY': 1.0,
            'FIRE': 0.9,
            'CRIME': 0.7,
            'NATURAL_DISASTER': 1.0,
            'OTHER': 0.5
        };
        
        const baseIntensity = typeWeights[alert.accident_type] || 0.5;
        const confidenceMultiplier = (alert.confidence || 50) / 100;
        
        return Math.max(0.3, Math.min(1.0, baseIntensity * confidenceMultiplier));
    }

    function getHeatmapColor(intensity) {
        // Gradiente de colores basado en intensidad
        if (intensity >= 0.8) return '#ff0000'; // Rojo intenso
        if (intensity >= 0.6) return '#ff4500'; // Naranja rojizo
        if (intensity >= 0.4) return '#ffa500'; // Naranja
        if (intensity >= 0.2) return '#ffff00'; // Amarillo
        return '#90ee90'; // Verde claro
    }

    // Funciones de filtrado por zona
    function toggleLocationFilter() {
        const panel = document.getElementById('location-filter-panel');
        if (panel) {
            panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
        }
    }

    function enableLocationFilter() {
        if (!navigator.geolocation) {
            showNotification('❌ Geolocalización no disponible en este navegador', 'error');
            return;
        }

        showNotification('📍 Obteniendo tu ubicación...', 'info');

        navigator.geolocation.getCurrentPosition(
            function(position) {
                const lat = position.coords.latitude;
                const lng = position.coords.longitude;
                
                state.locationFilter.userLocation = { lat, lng };
                state.locationFilter.enabled = true;
                
                // Actualizar UI
                updateFilterStatus();
                
                // Agregar marcador de ubicación del usuario
                addUserLocationMarker(lat, lng);
                
                // Agregar círculo de radio
                addFilterCircle(lat, lng, state.locationFilter.radius);
                
                // Centrar mapa en ubicación del usuario
                state.map.setView([lat, lng], 13);
                
                // Filtrar alertas
                filterAlertsByLocation();
                
                showNotification('✅ Filtro por zona activado', 'success');
            },
            function(error) {
                console.error('Error obteniendo ubicación:', error);
                let message = '❌ Error obteniendo ubicación';
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        message = '❌ Permiso de ubicación denegado';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        message = '❌ Ubicación no disponible';
                        break;
                    case error.TIMEOUT:
                        message = '❌ Tiempo de espera agotado';
                        break;
                }
                showNotification(message, 'error');
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 300000 // 5 minutos
            }
        );
    }

    function disableLocationFilter() {
        state.locationFilter.enabled = false;
        state.locationFilter.userLocation = null;
        
        // Remover marcador del usuario
        if (state.locationFilter.marker) {
            state.map.removeLayer(state.locationFilter.marker);
            state.locationFilter.marker = null;
        }
        
        // Remover círculo de filtrado
        if (state.locationFilter.circle) {
            state.map.removeLayer(state.locationFilter.circle);
            state.locationFilter.circle = null;
        }
        
        // Actualizar UI
        updateFilterStatus();
        
        // Mostrar todas las alertas nuevamente
        updateMapMarkers();
        
        showNotification('🔄 Filtro por zona desactivado', 'info');
    }

    function updateFilterRadius() {
        const select = document.getElementById('radius-select');
        if (select) {
            state.locationFilter.radius = parseInt(select.value);
            
            if (state.locationFilter.enabled && state.locationFilter.userLocation) {
                // Actualizar círculo con nuevo radio
                if (state.locationFilter.circle) {
                    state.map.removeLayer(state.locationFilter.circle);
                }
                
                addFilterCircle(
                    state.locationFilter.userLocation.lat,
                    state.locationFilter.userLocation.lng,
                    state.locationFilter.radius
                );
                
                // Filtrar alertas con nuevo radio
                filterAlertsByLocation();
                
                showNotification(`📏 Radio actualizado a ${state.locationFilter.radius} km`, 'info');
            }
        }
    }

    function addUserLocationMarker(lat, lng) {
        if (state.locationFilter.marker) {
            state.map.removeLayer(state.locationFilter.marker);
        }
        
        state.locationFilter.marker = L.marker([lat, lng], {
            icon: L.divIcon({
                className: 'user-location-marker',
                html: '<i class="fas fa-crosshairs" style="color: #007bff; font-size: 20px;"></i>',
                iconSize: [20, 20],
                iconAnchor: [10, 10]
            })
        }).addTo(state.map);
        
        state.locationFilter.marker.bindPopup('📍 Tu ubicación actual');
    }

    function addFilterCircle(lat, lng, radiusKm) {
        if (state.locationFilter.circle) {
            state.map.removeLayer(state.locationFilter.circle);
        }
        
        state.locationFilter.circle = L.circle([lat, lng], {
            radius: radiusKm * 1000, // Convertir km a metros
            color: '#007bff',
            fillColor: '#007bff',
            fillOpacity: 0.1,
            weight: 2,
            dashArray: '5, 5'
        }).addTo(state.map);
    }

    function filterAlertsByLocation() {
        if (!state.locationFilter.enabled || !state.locationFilter.userLocation) {
            return;
        }
        
        const userLat = state.locationFilter.userLocation.lat;
        const userLng = state.locationFilter.userLocation.lng;
        const radiusKm = state.locationFilter.radius;
        
        // Filtrar alertas por distancia
        const filteredAlerts = state.currentAlerts.filter(alert => {
            if (!alert.latitude || !alert.longitude) return false;
            
            const distance = calculateDistance(
                userLat, userLng,
                alert.latitude, alert.longitude
            );
            
            return distance <= radiusKm;
        });
        
        // Actualizar marcadores solo con alertas filtradas
        updateMapMarkersWithFiltered(filteredAlerts);
        
        // Actualizar contador
        const count = filteredAlerts.length;
        showNotification(`🎯 ${count} alerta${count !== 1 ? 's' : ''} en un radio de ${radiusKm} km`, 'info');
    }

    function calculateDistance(lat1, lng1, lat2, lng2) {
        // Fórmula de Haversine para calcular distancia entre dos puntos
        const R = 6371; // Radio de la Tierra en km
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLng = (lng2 - lng1) * Math.PI / 180;
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                  Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
                  Math.sin(dLng/2) * Math.sin(dLng/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }

    function updateMapMarkersWithFiltered(alerts) {
        if (!state.map) return;
        
        // Limpiar marcadores existentes (excepto ubicación del usuario y círculo)
        state.map.eachLayer(function(layer) {
            if (layer instanceof L.Marker && layer !== state.locationFilter.marker) {
                state.map.removeLayer(layer);
            }
        });
        
        // Agregar marcadores de alertas filtradas
        alerts.forEach(alert => {
            if (alert.latitude && alert.longitude) {
                const marker = L.marker([alert.latitude, alert.longitude], {
                    icon: getAlertIcon(alert.severity)
                }).addTo(state.map);
                
                const popupContent = `
                    <div class="alert-popup">
                        <h4>${alert.accident_type || 'Tipo desconocido'}</h4>
                        <p><strong>Severidad:</strong> ${alert.severity || 'N/A'}</p>
                        <p><strong>Confianza:</strong> ${alert.confidence || 'N/A'}%</p>
                        <p><strong>Hora:</strong> ${new Date(alert.timestamp).toLocaleString()}</p>
                        ${alert.description ? `<p><strong>Descripción:</strong> ${alert.description}</p>` : ''}
                    </div>
                `;
                marker.bindPopup(popupContent);
            }
        });
    }

    function updateFilterStatus() {
        const statusElement = document.getElementById('filter-status');
        const statusText = statusElement?.querySelector('.status-text');
        
        if (statusElement && statusText) {
            if (state.locationFilter.enabled) {
                statusElement.classList.add('active');
                statusText.textContent = `Activo (${state.locationFilter.radius} km)`;
            } else {
                statusElement.classList.remove('active');
                statusText.textContent = 'Desactivado';
            }
        }
    }

    // Mapa del dashboard (versión simplificada)
    function initializeDashboardMap() {
        const dashboardMapContainer = document.getElementById('dashboard-map');
        if (!dashboardMapContainer) {
            console.warn('⚠️ Contenedor del mapa del dashboard no encontrado');
            return;
        }

        try {
            // Coordenadas por defecto (Madrid, España)
            state.dashboardMap = L.map('dashboard-map').setView([40.4168, -3.7038], 6);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(state.dashboardMap);

            updateDashboardMapMarkers();
            console.log('🗺️ Mapa del dashboard inicializado correctamente');
        } catch (error) {
            console.error('❌ Error inicializando mapa del dashboard:', error);
        }
    }

    function updateDashboardMapMarkers() {
        if (!state.dashboardMap) return;

        // Limpiar marcadores existentes
        state.dashboardMap.eachLayer(layer => {
            if (layer instanceof L.Marker) {
                state.dashboardMap.removeLayer(layer);
            }
        });

        // Agregar marcadores de alertas recientes (máximo 10)
        const recentAlerts = state.currentAlerts.slice(0, 10);
        recentAlerts.forEach(alert => {
            if (alert.location_data && alert.location_data.latitude && alert.location_data.longitude) {
                try {
                    const marker = L.marker([alert.location_data.latitude, alert.location_data.longitude])
                        .addTo(state.dashboardMap);
                    
                    const popupContent = `
                        <div>
                            <strong>${formatAccidentType(alert.accident_type)}</strong><br>
                            <small>${formatDateTime(alert.timestamp)}</small><br>
                            Estado: ${formatStatus(alert.status)}
                        </div>
                    `;
                    marker.bindPopup(popupContent);
                } catch (error) {
                    console.error('❌ Error agregando marcador al dashboard:', error);
                }
            }
        });
    }

    function refreshMapData() {
        updateDashboardMapMarkers();
        showNotification('🗺️ Mapa del dashboard actualizado', 'success');
    }

    // Gestión del sistema
    function initializeSystemPanel() {
        initializePerformanceChart();
        updateSystemStatus();
        loadSystemLogs();
    }

    function initializePerformanceChart() {
        const ctx = document.getElementById('performance-chart');
        if (!ctx) {
            console.warn('⚠️ Canvas para gráfico de rendimiento no encontrado');
            return;
        }

        // Destruir gráfico existente si existe
        if (state.charts.performance) {
            state.charts.performance.destroy();
        }

        state.charts.performance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'CPU (%)',
                        data: [],
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Memoria (%)',
                        data: [],
                        borderColor: '#28a745',
                        backgroundColor: 'rgba(40, 167, 69, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Conexiones',
                        data: [],
                        borderColor: '#ffc107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Porcentaje (%)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Conexiones'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }

    function updateSystemStatus() {
        const systemData = generateMockSystemData();
        
        // Actualizar indicadores de estado
        updateStatusIndicator('server-status', systemData.server.status);
        updateStatusIndicator('db-status', systemData.database.status);
        updateStatusIndicator('ws-status', systemData.websocket.status);
        updateStatusIndicator('api-status', systemData.api.status);
        
        // Actualizar métricas
        updateElementText('server-uptime', systemData.server.uptime);
        updateElementText('server-cpu', systemData.server.cpu);
        updateElementText('server-memory', systemData.server.memory);
        
        updateElementText('db-connections', systemData.database.connections);
        updateElementText('db-queries', systemData.database.queries);
        updateElementText('db-latency', systemData.database.latency);
        
        updateElementText('ws-connections', systemData.websocket.connections);
        updateElementText('ws-messages', systemData.websocket.messages);
        updateElementText('ws-last-activity', systemData.websocket.lastActivity);
        
        updateElementText('api-requests', systemData.api.requests);
        updateElementText('api-response-time', systemData.api.responseTime);
        updateElementText('api-errors', systemData.api.errors);
        
        // Actualizar gráfico de rendimiento
        updatePerformanceChart(systemData);
    }

    function updateStatusIndicator(elementId, status) {
        const indicator = document.getElementById(elementId);
        if (!indicator) return;
        
        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('.status-text');
        
        if (!dot || !text) return;
        
        dot.className = 'status-dot';
        
        switch (status) {
            case 'online':
                dot.classList.add('online');
                text.textContent = 'En línea';
                break;
            case 'offline':
                dot.classList.add('offline');
                text.textContent = 'Desconectado';
                break;
            case 'warning':
                dot.classList.add('warning');
                text.textContent = 'Advertencia';
                break;
        }
    }

    function generateMockSystemData() {
        const now = new Date();
        return {
            server: {
                status: 'online',
                uptime: '2d 14h 32m',
                cpu: Math.floor(Math.random() * 30 + 10),
                memory: Math.floor(Math.random() * 40 + 30)
            },
            database: {
                status: 'online',
                connections: Math.floor(Math.random() * 10 + 5),
                queries: Math.floor(Math.random() * 100 + 50),
                latency: Math.floor(Math.random() * 20 + 5)
            },
            websocket: {
                status: 'online',
                connections: Math.floor(Math.random() * 5 + 1),
                messages: Math.floor(Math.random() * 50 + 10),
                lastActivity: 'Hace 2 segundos'
            },
            api: {
                status: 'online',
                requests: Math.floor(Math.random() * 200 + 100),
                responseTime: Math.floor(Math.random() * 100 + 50),
                errors: Math.floor(Math.random() * 3)
            },
            timestamp: now
        };
    }

    function updatePerformanceChart(systemData) {
        if (!state.charts.performance) return;
        
        const timeLabel = new Date().toLocaleTimeString();
        
        // Mantener solo los últimos 20 puntos de datos
        if (state.charts.performance.data.labels.length >= 20) {
            state.charts.performance.data.labels.shift();
            state.charts.performance.data.datasets.forEach(dataset => dataset.data.shift());
        }
        
        state.charts.performance.data.labels.push(timeLabel);
        state.charts.performance.data.datasets[0].data.push(systemData.server.cpu);
        state.charts.performance.data.datasets[1].data.push(systemData.server.memory);
        state.charts.performance.data.datasets[2].data.push(systemData.websocket.connections);
        
        state.charts.performance.update('none');
    }

    function loadSystemLogs() {
        const mockLogs = generateMockLogs();
        state.systemLogs = mockLogs;
        displaySystemLogs();
    }

    function generateMockLogs() {
        const logTypes = ['info', 'warning', 'error'];
        const messages = [
            'Sistema iniciado correctamente',
            'Nueva conexión WebSocket establecida',
            'Alerta procesada exitosamente',
            'Base de datos conectada',
            'Error de conexión temporal',
            'Memoria del sistema al 85%',
            'API endpoint respondió en 45ms',
            'Usuario autenticado correctamente',
            'Backup automático completado',
            'Conexión WebSocket cerrada'
        ];
        
        const logs = [];
        for (let i = 0; i < 15; i++) {
            const type = logTypes[Math.floor(Math.random() * logTypes.length)];
            const message = messages[Math.floor(Math.random() * messages.length)];
            const timestamp = new Date(Date.now() - Math.random() * 3600000).toLocaleString();
            
            logs.push({
                timestamp,
                level: type,
                message
            });
        }
        
        return logs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    }

    function displaySystemLogs() {
        const container = document.getElementById('system-logs-container');
        if (!container) {
            console.warn('⚠️ Contenedor de logs no encontrado');
            return;
        }
        
        const logLevelSelect = document.getElementById('log-level');
        const logLevel = logLevelSelect?.value || 'all';
        
        let filteredLogs = state.systemLogs;
        if (logLevel !== 'all') {
            filteredLogs = state.systemLogs.filter(log => log.level === logLevel);
        }
        
        if (filteredLogs.length === 0) {
            container.innerHTML = '<div class="log-entry">No hay logs disponibles</div>';
            return;
        }
        
        container.innerHTML = filteredLogs.map(log => `
            <div class="log-entry ${log.level}">
                <span class="log-timestamp">${log.timestamp}</span>
                <span class="log-level ${log.level}">[${log.level.toUpperCase()}]</span>
                <span class="log-message">${log.message}</span>
            </div>
        `).join('');
    }

    function clearLogs() {
        state.systemLogs = [];
        displaySystemLogs();
        showNotification('🗑️ Logs limpiados', 'info');
    }

    function refreshLogs() {
        loadSystemLogs();
        showNotification('🔄 Logs actualizados', 'info');
    }

    function loadSystemData() {
        updateSystemStatus();
        loadSystemLogs();
    }

    // Gestión de alertas críticas
    function checkCriticalAlerts(alerts) {
        const criticalAlerts = alerts.filter(alert => 
            alert.severity === 'CRITICAL' || 
            alert.accident_type === 'MEDICAL' ||
            alert.accident_type === 'FIRE'
        );

        criticalAlerts.forEach(alert => {
            if (!alert.notified) {
                showCriticalNotification(alert);
                playAlertSound();
                alert.notified = true;
            }
        });
    }

    function showCriticalNotification(alert) {
        const notification = document.createElement('div');
        notification.className = 'critical-alert-notification';
        notification.innerHTML = `
            <div class="notification-header">
                <i class="fas fa-exclamation-triangle"></i>
                🚨 ALERTA CRÍTICA
            </div>
            <div class="notification-body">
                ${formatAccidentType(alert.accident_type)} - ID: ${alert.alert_id}
                <br>
                ${formatDateTime(alert.timestamp)}
            </div>
        `;

        document.body.appendChild(notification);

        // Remover notificación después de 10 segundos
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 10000);
    }

    function playAlertSound() {
        // Crear un sonido de alerta usando Web Audio API
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
            oscillator.frequency.setValueAtTime(600, audioContext.currentTime + 0.1);
            oscillator.frequency.setValueAtTime(800, audioContext.currentTime + 0.2);

            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);

            oscillator.start(audioContext.currentTime);
            oscillator.stop(audioContext.currentTime + 0.3);
        } catch (error) {
            console.warn('🔇 No se pudo reproducir sonido de alerta:', error);
        }
    }

    // Modal de detalles de alerta
    async function showAlertDetails(alertId) {
        try {
            const response = await fetch(`${config.api.baseUrl}${config.api.endpoints.alerts}/${alertId}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const alert = await response.json();
            displayAlertDetails(alert);
            
            const modal = document.getElementById('alert-modal');
            if (modal) {
                modal.classList.add('active');
            } else {
                console.warn('⚠️ Modal de alerta no encontrado');
            }
            
        } catch (error) {
            console.error('❌ Error cargando detalles de alerta:', error);
            showNotification('❌ Error cargando detalles', 'error');
        }
    }

    function displayAlertDetails(alert) {
        const container = document.getElementById('alert-details');
        if (!container) {
            console.warn('⚠️ Contenedor de detalles de alerta no encontrado');
            return;
        }
        
        container.innerHTML = `
            <div class="alert-detail-grid">
                <div class="detail-item">
                    <div class="detail-label">ID de Alerta</div>
                    <div class="detail-value">${alert.alert_id || 'N/A'}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Dispositivo</div>
                    <div class="detail-value">${alert.device_id || 'N/A'}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Tipo de Accidente</div>
                    <div class="detail-value">${formatAccidentType(alert.accident_type)}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Estado</div>
                    <div class="detail-value">
                        <span class="alert-status ${(alert.status || '').toLowerCase()}">${formatStatus(alert.status)}</span>
                    </div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Confianza</div>
                    <div class="detail-value">${alert.confidence || 0}%</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Fecha/Hora</div>
                    <div class="detail-value">${formatDateTime(alert.timestamp)}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Aceleración</div>
                    <div class="detail-value">${alert.acceleration_magnitude || 0} m/s²</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Giroscopio</div>
                    <div class="detail-value">${alert.gyroscope_magnitude || 0} rad/s</div>
                </div>
                
                ${alert.location_data ? `
                    <div class="detail-item location-info">
                        <div class="detail-label">Ubicación</div>
                        <div class="detail-value">
                            Lat: ${alert.location_data.latitude || 0}<br>
                            Lng: ${alert.location_data.longitude || 0}<br>
                            Precisión: ${alert.location_data.accuracy || 0}m
                        </div>
                    </div>
                ` : ''}
                
                ${alert.medical_info ? `
                    <div class="detail-item medical-info">
                        <div class="detail-label">Información Médica</div>
                        <div class="detail-value">
                            <strong>Condiciones:</strong> ${alert.medical_info.medical_conditions || 'Ninguna'}<br>
                            <strong>Medicamentos:</strong> ${alert.medical_info.medications || 'Ninguno'}<br>
                            <strong>Alergias:</strong> ${alert.medical_info.allergies || 'Ninguna'}<br>
                            <strong>Tipo de sangre:</strong> ${alert.medical_info.blood_type || 'No especificado'}<br>
                            <strong>Contacto de emergencia:</strong> ${alert.medical_info.emergency_contact || 'No especificado'}
                        </div>
                    </div>
                ` : ''}
                
                ${alert.emergency_contacts && alert.emergency_contacts.length > 0 ? `
                    <div class="detail-item emergency-contacts">
                        <div class="detail-label">Contactos de Emergencia</div>
                        <div class="detail-value">
                            ${alert.emergency_contacts.map(contact => `
                                <div class="contact-item">
                                    <strong>${contact.name || 'N/A'}</strong> (${contact.relationship || 'N/A'})<br>
                                    Tel: ${contact.phone_number || 'N/A'}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }

    async function updateAlertStatus(alertId, newStatus) {
        try {
            const response = await fetch(`${config.api.baseUrl}${config.api.endpoints.alerts}/${alertId}/status`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ status: newStatus })
            });
            
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            showNotification('✅ Estado actualizado correctamente', 'success');
            closeModal();
            
            // Recargar datos
            if (state.currentSection === 'alerts') {
                loadAlertsTable();
            } else {
                loadInitialData();
            }
            
        } catch (error) {
            console.error('❌ Error actualizando estado:', error);
            showNotification('❌ Error actualizando estado', 'error');
        }
    }

    function closeModal() {
        const modal = document.getElementById('alert-modal');
        if (modal) {
            modal.classList.remove('active');
        }
    }

    // Funciones de utilidad
    function updateElementText(elementId, text) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = text;
        } else {
            console.warn(`⚠️ Elemento ${elementId} no encontrado para actualizar texto`);
        }
    }

    function formatAccidentType(type) {
        const types = {
            'COLLISION': 'Colisión',
            'SUDDEN_STOP': 'Frenado Brusco',
            'ROLLOVER': 'Volcadura',
            'FALL': 'Caída',
            'MEDICAL': 'Emergencia Médica',
            'FIRE': 'Incendio',
            'UNKNOWN': 'Desconocido'
        };
        return types[type] || type;
    }

    function formatStatus(status) {
        const statuses = {
            'PENDING': 'Pendiente',
            'IN_PROGRESS': 'En Progreso',
            'COMPLETED': 'Completada',
            'CANCELLED': 'Cancelada',
            'FAILED': 'Fallida'
        };
        return statuses[status] || status;
    }

    function formatDateTime(dateString) {
        try {
            const date = new Date(dateString);
            return date.toLocaleString('es-ES', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch (error) {
            return 'Fecha inválida';
        }
    }

    function showNotification(message, type = 'info') {
        const container = document.getElementById('notifications');
        if (!container) {
            console.warn('⚠️ Contenedor de notificaciones no encontrado');
            return;
        }
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" 
                        style="background: none; border: none; font-size: 1.2rem; cursor: pointer;">
                    &times;
                </button>
            </div>
        `;
        
        container.appendChild(notification);
        
        // Auto-remover después de 5 segundos
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }

    function showErrorNotification(message) {
        showNotification(`❌ ${message}`, 'error');
    }

    // Actualizaciones periódicas
    function startPeriodicUpdates() {
        // Actualizar datos cada 30 segundos
        setInterval(() => {
            if (['dashboard', 'alerts'].includes(state.currentSection)) {
                loadInitialData();
            }
            if (state.currentSection === 'map') {
                updateMapMarkers();
            }
            if (state.currentSection === 'system') {
                updateSystemStatus();
            }
        }, config.updateIntervals.charts);
        
        // Actualizar logs cada 10 segundos
        setInterval(() => {
            if (state.currentSection === 'system' && Math.random() > 0.7) {
                loadSystemLogs();
            }
        }, config.updateIntervals.logs);
    }

    function updateVisualComponents() {
        if (typeof Chart !== 'undefined') {
            updateCharts();
        }
        if (typeof L !== 'undefined') {
            if (state.map) {
                updateMapMarkers();
            }
            if (state.dashboardMap) {
                updateDashboardMapMarkers();
            }
        }
    }

    // Configuración de event listeners
    function setupEventListeners() {
        // Cerrar modal al hacer clic fuera
        const modal = document.getElementById('alert-modal');
        if (modal) {
            modal.addEventListener('click', function(e) {
                if (e.target === this) {
                    closeModal();
                }
            });
        }
        
        // Tecla ESC para cerrar modal
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeModal();
            }
        });
        
        // Filtro de logs
        const logLevelSelect = document.getElementById('log-level');
        if (logLevelSelect) {
            logLevelSelect.addEventListener('change', displaySystemLogs);
        }
        
        // Filtros de alertas
        const filterElements = ['date-filter', 'type-filter', 'status-filter', 'search-filter'];
        filterElements.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('change', function() {
                    AlertaRavenApp.applyFilters();
                });
            }
        });

        // Busqueda en tabla
        const tableSearch = document.getElementById('table-search');
        if (tableSearch) {
            tableSearch.addEventListener('input', searchAlertsTable);
        }

        // Ordenamiento de tabla
        const tableSort = document.getElementById('table-sort');
        if (tableSort) {
            tableSort.addEventListener('change', sortAlertsTable);
        }
    }

    // Funciones de estadísticas detalladas
    function initializeStatistics() {
        loadDetailedStatistics();
        initializeMonthlyTrendsChart();
    }

    function loadDetailedStatistics() {
        if (!state.currentAlerts || state.currentAlerts.length === 0) {
            console.log('📊 No hay datos para estadísticas detalladas');
            return;
        }

        updateStatusStats(state.currentAlerts);
        updateTypeStats(state.currentAlerts);
        updateTimeMetrics(state.currentAlerts);
        updateGeographicStats(state.currentAlerts);
    }

    function updateStatusStats(alerts) {
        const statusCounts = {
            'PENDING': 0,
            'IN_PROGRESS': 0,
            'COMPLETED': 0,
            'CANCELLED': 0
        };

        alerts.forEach(alert => {
            const status = alert.status || 'PENDING';
            if (statusCounts.hasOwnProperty(status)) {
                statusCounts[status]++;
            }
        });

        const container = document.getElementById('status-stats');
        if (container) {
            container.innerHTML = Object.entries(statusCounts).map(([status, count]) => `
                <div class="stat-item">
                    <span class="stat-label">${formatStatus(status)}</span>
                    <span class="stat-value">${count}</span>
                    <span class="stat-percentage">${((count / alerts.length) * 100).toFixed(1)}%</span>
                </div>
            `).join('');
        }
    }

    function updateTypeStats(alerts) {
        const typeCounts = {
            'COLLISION': 0,
            'FALL': 0,
            'MEDICAL': 0,
            'FIRE': 0,
            'OTHER': 0
        };

        alerts.forEach(alert => {
            const type = alert.accident_type || 'OTHER';
            if (typeCounts.hasOwnProperty(type)) {
                typeCounts[type]++;
            } else {
                typeCounts['OTHER']++;
            }
        });

        const container = document.getElementById('type-stats');
        if (container) {
            container.innerHTML = Object.entries(typeCounts).map(([type, count]) => `
                <div class="stat-item">
                    <span class="stat-label">${formatAccidentType(type)}</span>
                    <span class="stat-value">${count}</span>
                    <span class="stat-percentage">${((count / alerts.length) * 100).toFixed(1)}%</span>
                </div>
            `).join('');
        }
    }

    function updateTimeMetrics(alerts) {
        // Calcular tiempo promedio de respuesta
        const responseTimes = alerts
            .filter(alert => alert.response_time)
            .map(alert => alert.response_time);
        
        const avgResponseTime = responseTimes.length > 0 
            ? Math.round(responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length)
            : 0;

        // Calcular tiempo promedio de resolución
        const resolutionTimes = alerts
            .filter(alert => alert.status === 'COMPLETED' && alert.resolution_time)
            .map(alert => alert.resolution_time);
        
        const avgResolutionTime = resolutionTimes.length > 0
            ? Math.round(resolutionTimes.reduce((a, b) => a + b, 0) / resolutionTimes.length)
            : 0;

        // Calcular horas pico
        const hourCounts = {};
        alerts.forEach(alert => {
            const hour = new Date(alert.timestamp || alert.created_at).getHours();
            hourCounts[hour] = (hourCounts[hour] || 0) + 1;
        });

        const peakHour = Object.entries(hourCounts)
            .sort(([,a], [,b]) => b - a)[0];
        
        const peakHours = peakHour 
            ? `${peakHour[0]}:00-${parseInt(peakHour[0]) + 1}:00`
            : '14:00-16:00';

        // Actualizar elementos
        updateElementText('avg-response-time-stat', `${avgResponseTime} min`);
        updateElementText('avg-resolution-time', `${avgResolutionTime} min`);
        updateElementText('peak-hours', peakHours);
    }

    function updateGeographicStats(alerts) {
        const locationData = alerts.filter(alert => alert.location_data);
        
        if (locationData.length === 0) {
            const container = document.getElementById('geographic-stats');
            if (container) {
                container.innerHTML = '<p class="text-center text-secondary">No hay datos de ubicación disponibles</p>';
            }
            return;
        }

        // Agrupar por zonas aproximadas (simplificado)
        const zones = {};
        locationData.forEach(alert => {
            const lat = Math.round(alert.location_data.latitude * 100) / 100;
            const lng = Math.round(alert.location_data.longitude * 100) / 100;
            const zone = `${lat}, ${lng}`;
            zones[zone] = (zones[zone] || 0) + 1;
        });

        const sortedZones = Object.entries(zones)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 5);

        const container = document.getElementById('geographic-stats');
        if (container) {
            container.innerHTML = sortedZones.map(([zone, count]) => `
                <div class="geo-stat-item">
                    <span class="geo-zone">${zone}</span>
                    <span class="geo-count">${count} alertas</span>
                </div>
            `).join('');
        }
    }

    function initializeMonthlyTrendsChart() {
        const canvas = document.getElementById('monthly-trends-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        // Datos de ejemplo para los últimos 6 meses
        const monthlyData = generateMonthlyTrendsData();

        state.charts.monthlyTrends = new Chart(ctx, {
            type: 'line',
            data: {
                labels: monthlyData.labels,
                datasets: [{
                    label: 'Alertas por Mes',
                    data: monthlyData.data,
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true
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

    function generateMonthlyTrendsData() {
        const months = [];
        const data = [];
        const now = new Date();

        for (let i = 5; i >= 0; i--) {
            const date = new Date(now.getFullYear(), now.getMonth() - i, 1);
            months.push(date.toLocaleDateString('es-ES', { month: 'short', year: 'numeric' }));
            
            // Filtrar alertas por mes
            const monthAlerts = state.currentAlerts.filter(alert => {
                const alertDate = new Date(alert.timestamp || alert.created_at);
                return alertDate.getMonth() === date.getMonth() && 
                       alertDate.getFullYear() === date.getFullYear();
            });
            
            data.push(monthAlerts.length);
        }

        return { labels: months, data };
    }

    function generateReport() {
        try {
            if (!state.currentAlerts || state.currentAlerts.length === 0) {
                showNotification('⚠️ No hay datos para generar el reporte', 'warning');
                return;
            }

            // Crear contenido del reporte
            const reportData = {
                title: 'Reporte de Estadísticas AlertaRaven',
                date: new Date().toLocaleDateString('es-ES'),
                totalAlerts: state.currentAlerts.length,
                statusStats: calculateStatusStats(state.currentAlerts),
                typeStats: calculateTypeStats(state.currentAlerts),
                timeMetrics: calculateTimeMetrics(state.currentAlerts),
                geographicStats: calculateGeographicStats(state.currentAlerts)
            };

            // Generar PDF (simplificado como texto por ahora)
            const reportContent = generateReportContent(reportData);
            
            // Descargar como archivo de texto
            const blob = new Blob([reportContent], { type: 'text/plain;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            
            link.setAttribute('href', url);
            link.setAttribute('download', `reporte_estadisticas_${new Date().toISOString().split('T')[0]}.txt`);
            link.style.visibility = 'hidden';
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            showNotification('📄 Reporte generado correctamente', 'success');
            
        } catch (error) {
            console.error('❌ Error generando reporte:', error);
            showNotification('❌ Error generando reporte', 'error');
        }
    }

    function calculateStatusStats(alerts) {
        const stats = { 'PENDING': 0, 'IN_PROGRESS': 0, 'COMPLETED': 0, 'CANCELLED': 0 };
        alerts.forEach(alert => {
            const status = alert.status || 'PENDING';
            if (stats.hasOwnProperty(status)) stats[status]++;
        });
        return stats;
    }

    function calculateTypeStats(alerts) {
        const stats = { 'COLLISION': 0, 'FALL': 0, 'MEDICAL': 0, 'FIRE': 0, 'OTHER': 0 };
        alerts.forEach(alert => {
            const type = alert.accident_type || 'OTHER';
            if (stats.hasOwnProperty(type)) stats[type]++;
            else stats['OTHER']++;
        });
        return stats;
    }

    function calculateTimeMetrics(alerts) {
        const responseTimes = alerts.filter(a => a.response_time).map(a => a.response_time);
        const resolutionTimes = alerts.filter(a => a.status === 'COMPLETED' && a.resolution_time).map(a => a.resolution_time);
        
        return {
            avgResponseTime: responseTimes.length > 0 ? Math.round(responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length) : 0,
            avgResolutionTime: resolutionTimes.length > 0 ? Math.round(resolutionTimes.reduce((a, b) => a + b, 0) / resolutionTimes.length) : 0
        };
    }

    function calculateGeographicStats(alerts) {
        const locationData = alerts.filter(alert => alert.location_data);
        return { totalWithLocation: locationData.length, totalAlerts: alerts.length };
    }

    function generateReportContent(data) {
        return `
${data.title}
Fecha: ${data.date}
=====================================

RESUMEN GENERAL
- Total de alertas: ${data.totalAlerts}

ESTADÍSTICAS POR ESTADO
${Object.entries(data.statusStats).map(([status, count]) => 
    `- ${formatStatus(status)}: ${count} (${((count / data.totalAlerts) * 100).toFixed(1)}%)`
).join('\n')}

ESTADÍSTICAS POR TIPO
${Object.entries(data.typeStats).map(([type, count]) => 
    `- ${formatAccidentType(type)}: ${count} (${((count / data.totalAlerts) * 100).toFixed(1)}%)`
).join('\n')}

MÉTRICAS DE TIEMPO
- Tiempo promedio de respuesta: ${data.timeMetrics.avgResponseTime} minutos
- Tiempo promedio de resolución: ${data.timeMetrics.avgResolutionTime} minutos

DATOS GEOGRÁFICOS
- Alertas con ubicación: ${data.geographicStats.totalWithLocation}/${data.geographicStats.totalAlerts}
- Cobertura: ${((data.geographicStats.totalWithLocation / data.geographicStats.totalAlerts) * 100).toFixed(1)}%

=====================================
Reporte generado por AlertaRaven
        `.trim();
    }

    // Funciones públicas
    return {
        // Inicialización
        initialize,
        
        // Navegación
        switchSection,
        
        // Alertas
        showAlertDetails,
        updateAlertStatus,
        closeModal,
        
        // Filtros
        applyFilters: function() {
            state.filters.date = document.getElementById('date-filter')?.value || '';
            state.filters.type = document.getElementById('type-filter')?.value || '';
            state.filters.status = document.getElementById('status-filter')?.value || '';
            state.filters.search = document.getElementById('search-filter')?.value.toLowerCase() || '';
            
            applyTableFilters();
        },
        
        clearFilters: function() {
            state.filters = {
                date: '',
                type: '',
                status: '',
                search: ''
            };
            
            // Resetear elementos del DOM
            const filterElements = ['date-filter', 'type-filter', 'status-filter', 'search-filter'];
            filterElements.forEach(id => {
                const element = document.getElementById(id);
                if (element) element.value = '';
            });
            
            applyTableFilters();
            showNotification('🧹 Filtros limpiados', 'info');
        },
        
        // Tabla
        searchAlertsTable,
        sortAlertsTable,
        sortTableBy,
        changePage,
        
        // Sistema
        refreshData: function() {
            loadInitialData();
            showNotification('🔄 Datos actualizados', 'success');
        },
        exportAlerts: function() {
        try {
            const { filteredAlerts } = state.tableState;
            
            if (filteredAlerts.length === 0) {
                showNotification('❌ No hay alertas para exportar', 'warning');
                return;
            }
            
            // Crear CSV
            const headers = ['ID', 'Fecha/Hora', 'Tipo', 'Estado', 'Dispositivo', 'Confianza', 'Ubicación'];
            const csvData = filteredAlerts.map(alert => [
                alert.alert_id,
                formatDateTime(alert.timestamp),
                formatAccidentType(alert.accident_type),
                formatStatus(alert.status),
                alert.device_id,
                `${alert.confidence}%`,
                alert.location_data ? 
                    `${alert.location_data.latitude}, ${alert.location_data.longitude}` : 
                    'No disponible'
            ]);
            
            const csvContent = [headers, ...csvData]
                .map(row => row.map(field => `"${field}"`).join(','))
                .join('\n');
            
            // Descargar archivo
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            
            link.setAttribute('href', url);
            link.setAttribute('download', `alertas_${new Date().toISOString().split('T')[0]}.csv`);
            link.style.visibility = 'hidden';
            
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            showNotification('📊 Alertas exportadas correctamente', 'success');
            
        } catch (error) {
            console.error('❌ Error exportando alertas:', error);
            showNotification('❌ Error exportando alertas', 'error');
        }
    },
        refreshMap,
        toggleHeatmap,
        refreshMapData,

        // Dashboard
        reloadModelMetrics: function() { loadModelMetrics(); },
        refreshRecentAlerts,
        downloadChart,
        
        // Filtrado por zona
        toggleLocationFilter,
        enableLocationFilter,
        disableLocationFilter,
        updateFilterRadius,
        clearLogs,
        refreshLogs,
        
        // Estadísticas detalladas
        initializeStatistics,
        loadDetailedStatistics,
        generateReport,
        
        // Utilidades
        getState: () => ({ ...state }),
        getConfig: () => ({ ...config }),
        
        // Verificación de estado
        isReady: () => state.componentsReady
    };
})();

// Variable global para verificar carga
window.componentsLoaded = false;
document.addEventListener('componentsLoaded', function() {
    window.componentsLoaded = true;
});

// Función global para recargar componentes (útil para desarrollo)
window.reloadComponent = function(componentId) {
    if (window.componentLoader) {
        window.componentLoader.reloadComponent(componentId);
    }
};

// Función global para forzar reinicio de la app
window.restartApp = function() {
    if (confirm('¿Estás seguro de que quieres reiniciar la aplicación?')) {
        location.reload();
    }
};