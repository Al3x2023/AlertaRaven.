const CACHE_NAME = 'alertaraven-v1';
const OFFLINE_URLS = [
  '/',
  '/dashboard',
  '/static/index.html',
  '/static/styles.css',
  '/static/app.js',
  '/static/components-loader.js',
  '/static/components/sidebar.html',
  '/static/components/header.html',
  '/static/components/dashboard.html',
  '/static/components/alerts.html',
  '/static/components/map.html',
  '/static/components/statistics.html',
  '/static/components/system.html',
  '/static/components/modals.html',
  '/static/components/notifications.html',
  '/static/manifest.webmanifest',
  '/static/icons/icon.svg'
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(OFFLINE_URLS);
    })
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME) {
            return caches.delete(key);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Estrategias:
// - Static (HTML/CSS/JS): cache-first, con actualización en segundo plano
// - API (/api/*): network-first, fallback al caché si offline
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Ignorar WebSocket y non-GET
  if (request.method !== 'GET' || url.protocol.startsWith('ws')) {
    return;
  }

  if (url.pathname.startsWith('/api/')) {
    // Network-first para API
    event.respondWith(
      fetch(request)
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          return response;
        })
        .catch(() => caches.match(request))
    );
    return;
  }

  // Static: cache-first
  event.respondWith(
    caches.match(request).then((cached) => {
      const networkFetch = fetch(request)
        .then((response) => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          return response;
        })
        .catch(() => cached);

      return cached || networkFetch;
    })
  );
});

// Mensajes opcionales desde la app para forzar actualización de caché
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'CACHE_FLUSH') {
    caches.delete(CACHE_NAME);
  }
});