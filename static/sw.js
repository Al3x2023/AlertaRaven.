/* AlertaRaven Service Worker */
const CACHE_NAME = 'alertaraven-v1';
const CORE_ASSETS = [
  '/',
  '/dashboard',
  '/static/styles.css',
  '/static/components-loader.js',
  '/static/app.js',
  '/static/index.html',
  '/static/landing.html',
  '/static/login.html',
  '/components/sidebar',
  '/components/header',
  '/components/dashboard',
  '/components/map',
  '/components/alerts',
  '/components/statistics',
  '/components/system',
  '/manifest.webmanifest',
  '/static/icons/icon.svg',
  '/static/icons/maskable_icon.svg'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(CORE_ASSETS);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.map((key) => key !== CACHE_NAME && caches.delete(key))
    ))
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  // No cache for non-GET or WebSocket
  if (request.method !== 'GET' || request.url.includes('/ws')) {
    return;
  }

  const url = new URL(request.url);

  // Navigation requests: network-first, fallback to cached index
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, copy)).catch(() => {});
          return response;
        })
        .catch(() => caches.match('/static/index.html'))
    );
    return;
  }

  // Static assets: cache-first
  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request).then((response) => {
        const copy = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(request, copy)).catch(() => {});
        return response;
      });
    })
  );
});

// Push notifications: display even when app is closed
self.addEventListener('push', (event) => {
  let data = {};
  try {
    data = event.data ? event.data.json() : {};
  } catch (e) {
    // Fallback if payload is not JSON
    data = { title: 'AlertaRaven', body: event.data ? event.data.text() : 'Notificación' };
  }

  const title = data.title || 'AlertaRaven';
  const body = data.body || 'Tienes una nueva alerta';
  const icon = '/static/icons/icon.svg';
  const badge = '/static/icons/maskable_icon.svg';
  const url = data.url || '/dashboard';

  event.waitUntil(
    self.registration.showNotification(title, {
      body,
      icon,
      badge,
      data: { url, alert_id: data.alert_id },
      tag: data.tag || 'alertaraven-alert',
      renotify: true,
      requireInteraction: true
    })
  );
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const targetUrl = event.notification.data?.url || '/dashboard';
  event.waitUntil(
    clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      for (const client of clientList) {
        if ('focus' in client) {
          const url = new URL(client.url);
          if (url.pathname === targetUrl) return client.focus();
        }
      }
      if (clients.openWindow) return clients.openWindow(targetUrl);
    })
  );
});