import catalog from './catalog.json';
import icons from './icons.json';
import charts from './charts.json';

interface Env {}

const CORS_HEADERS = {
  'Content-Type': 'application/json',
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), { status, headers: CORS_HEADERS });
}

// GET /api/v1/appstore/hash?version=X
function handleHash(url: URL): Response {
  const version = url.searchParams.get('version') || '1.12.3';
  return json({
    hash: catalog.hash,
    last_updated: new Date().toISOString(),
    version,
  });
}

// GET /api/v1/appstore/info?version=X
function handleInfo(url: URL): Response {
  const version = url.searchParams.get('version') || '1.12.3';
  const now = new Date().toISOString();

  // Build app ID list for topic_lists (comma-separated)
  const appIds = Object.keys(catalog.summaries).join(',');

  return json({
    version,
    hash: catalog.hash,
    last_updated: now,
    data: {
      apps: catalog.summaries,
      recommends: {},
      pages: {
        AI: {
          category: 'AI',
          content: JSON.stringify([
            { type: 'Topic', id: 'Featured apps in AI' },
            { type: 'Default Topic', id: 'Newest' },
          ]),
        },
      },
      topics: {},
      topic_lists: {
        'Featured apps in AI': {
          name: 'Featured apps in AI',
          type: 'Category',
          content: appIds,
          title: { 'en-US': 'Featured apps in AI', 'zh-CN': 'AI精选应用' },
        },
      },
      tops: [],
      latest: catalog.latest,
      tags: {
        AI: {
          _id: 'ai_custom_source',
          name: 'AI',
          title: { 'en-US': 'AI', 'zh-CN': 'AI' },
          icon: 'https://app.cdn.olares.com/icons/market/sidebar/neurology.svg',
          sort: 7,
          source: 0,
          updated_at: now,
          createdAt: '2025-11-07T05:14:01.765Z',
        },
      },
    },
    stats: {
      appstore_data: {
        apps: Object.keys(catalog.summaries).length,
        pages: 1,
        recommends: 0,
        tags: 1,
        topic_lists: 1,
        topics: 0,
      },
      last_updated: now,
    },
  });
}

// POST /api/v1/applications/info
async function handleDetail(request: Request): Promise<Response> {
  const body = (await request.json()) as { app_ids: string[]; version: string };
  const version = body.version || '1.12.3';
  const apps: Record<string, unknown> = {};
  const notFound: string[] = [];

  const details = catalog.details as Record<string, unknown>;

  for (const id of body.app_ids || []) {
    if (details[id]) {
      apps[id] = details[id];
    } else {
      notFound.push(id);
    }
  }

  return json({
    apps,
    version,
    ...(notFound.length > 0 ? { not_found: notFound } : {}),
  });
}

export default {
  async fetch(request: Request, _env: Env): Promise<Response> {
    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    const path = url.pathname;

    if (path === '/api/v1/appstore/hash' && request.method === 'GET') {
      return handleHash(url);
    }

    if (path === '/api/v1/appstore/info' && request.method === 'GET') {
      return handleInfo(url);
    }

    if (path === '/api/v1/applications/info' && request.method === 'POST') {
      return handleDetail(request);
    }

    // Serve charts: /api/v1/applications/{app_name}/chart?fileName=xxx.tgz
    const chartMatch = path.match(/^\/api\/v1\/applications\/(.+)\/chart$/);
    if (chartMatch && request.method === 'GET') {
      const fileName = url.searchParams.get('fileName') || chartMatch[1];
      const data = (charts as Record<string, string>)[fileName];
      if (data) {
        const binary = Uint8Array.from(atob(data), c => c.charCodeAt(0));
        return new Response(binary, {
          headers: {
            'Content-Type': 'application/gzip',
            'Content-Disposition': `attachment; filename="${fileName}"`,
            'Cache-Control': 'public, max-age=86400',
            'Access-Control-Allow-Origin': '*',
          },
        });
      }
      return json({ error: 'Chart not found' }, 404);
    }

    // Serve icons
    if (path.startsWith('/icons/') && request.method === 'GET') {
      const name = path.slice('/icons/'.length).replace(/\.png$/, '');
      const data = (icons as Record<string, string>)[name];
      if (data) {
        const binary = Uint8Array.from(atob(data), c => c.charCodeAt(0));
        return new Response(binary, {
          headers: {
            'Content-Type': 'image/png',
            'Cache-Control': 'public, max-age=86400',
            'Access-Control-Allow-Origin': '*',
          },
        });
      }
      return json({ error: 'Icon not found' }, 404);
    }

    // Health check
    if (path === '/' || path === '/health') {
      return json({
        name: 'orales-one-market',
        status: 'ok',
        apps: Object.keys(catalog.summaries).length,
      });
    }

    return json({ error: 'Not Found' }, 404);
  },
} satisfies ExportedHandler<Env>;
