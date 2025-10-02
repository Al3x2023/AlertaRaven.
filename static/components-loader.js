// components-loader.js
class ComponentLoader {
    constructor() {
        this.components = {
            'sidebar-component': '/static/components/sidebar.html',
            'header-component': '/static/components/header.html',
            'dashboard-component': '/static/components/dashboard.html',
            'alerts-component': '/static/components/alerts.html',
            'map-component': '/static/components/map.html',
            'statistics-component': '/static/components/statistics.html',
            'system-component': '/static/components/system.html',
            'modals-component': '/static/components/modals.html',
            'notifications-component': '/static/components/notifications.html'
        };
        
        this.loadedComponents = new Set();
    }

    async loadComponent(elementId, filePath) {
        try {
            const element = document.getElementById(elementId);
            if (!element) {
                console.warn(`Elemento ${elementId} no encontrado`);
                return;
            }

            const response = await fetch(filePath);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const html = await response.text();
            element.innerHTML = html;
            
            // Ejecutar scripts dentro del componente
            this.executeScripts(element);
            
            this.loadedComponents.add(elementId);
            console.log(`✅ Componente ${filePath} cargado correctamente`);
            
        } catch (error) {
            console.error(`❌ Error cargando componente ${filePath}:`, error);
            this.showError(elementId, error);
        }
    }

    executeScripts(container) {
        const scripts = container.querySelectorAll('script');
        scripts.forEach(script => {
            const newScript = document.createElement('script');
            
            // Copiar atributos
            Array.from(script.attributes).forEach(attr => {
                newScript.setAttribute(attr.name, attr.value);
            });
            
            // Copiar contenido
            if (script.src) {
                newScript.src = script.src;
            } else {
                newScript.textContent = script.textContent;
            }
            
            // Reemplazar el script original
            script.parentNode.replaceChild(newScript, script);
        });
    }

    showError(elementId, error) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="component-error" style="padding: 2rem; text-align: center; color: #ef4444;">
                    <h3>Error cargando componente</h3>
                    <p>${error.message}</p>
                    <button onclick="location.reload()" class="btn-primary" style="margin-top: 1rem;">
                        Reintentar
                    </button>
                </div>
            `;
        }
    }

    async loadAllComponents() {
        console.log('🔄 Cargando componentes...');
        
        const loadPromises = Object.entries(this.components).map(([elementId, filePath]) => {
            return this.loadComponent(elementId, filePath);
        });

        try {
            await Promise.all(loadPromises);
            console.log('✅ Todos los componentes cargados correctamente');
            
            // Disparar evento personalizado cuando todos los componentes estén cargados
            document.dispatchEvent(new CustomEvent('componentsLoaded'));
            
        } catch (error) {
            console.error('❌ Error cargando componentes:', error);
        }
    }

    isComponentLoaded(elementId) {
        return this.loadedComponents.has(elementId);
    }

    reloadComponent(elementId) {
        const filePath = this.components[elementId];
        if (filePath) {
            return this.loadComponent(elementId, filePath);
        }
    }
}

// Inicializar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    window.componentLoader = new ComponentLoader();
    window.componentLoader.loadAllComponents();
});

// Exportar para uso global
window.ComponentLoader = ComponentLoader;