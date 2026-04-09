/**
 * Credential proxy for container isolation.
 * Containers connect here instead of directly to the Anthropic API.
 * The proxy injects real credentials so containers never see them.
 *
 * Two operating modes:
 *
 *   Claude mode (default):
 *     API key:  Proxy injects x-api-key on every request.
 *     OAuth:    Container CLI exchanges its placeholder token for a temp
 *               API key via /api/oauth/claude_cli/create_api_key.
 *               Proxy injects real OAuth token on that exchange request;
 *               subsequent requests carry the temp key which is valid as-is.
 *
 *   Ollama mode (OLLAMA_MODEL is set):
 *     Requests to /v1/messages are translated from Anthropic format to
 *     Ollama's native /api/chat format and forwarded to the Ollama endpoint.
 *     Responses are translated back to Anthropic format. No Anthropic
 *     API key is needed. Uses native API (not OpenAI-compat) because it
 *     supports think:false and returns pre-parsed tool arguments.
 *     All other endpoints return canned responses.
 */
import { createServer, IncomingMessage, Server, ServerResponse } from 'http';
import { request as httpsRequest } from 'https';
import { request as httpRequest, RequestOptions } from 'http';

import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { readEnvFile } from './env.js';
import { logger } from './logger.js';
import {
  OllamaTranslatorConfig,
  StreamTranslator,
  parseNDJSONBuffer,
  translateRequest,
  translateResponse,
} from './ollama-translator.js';

export type AuthMode = 'api-key' | 'oauth';

export interface ProxyConfig {
  authMode: AuthMode;
}

// ---------------------------------------------------------------------------
// Ollama mode: translate Anthropic API ↔ Ollama native API
// ---------------------------------------------------------------------------

function handleOllamaRequest(
  req: IncomingMessage,
  res: ServerResponse,
  body: Buffer,
  config: OllamaTranslatorConfig,
): void {
  const url = req.url || '';

  logger.info(
    { url, method: req.method, bodySize: body.length },
    'Ollama proxy request',
  );

  // Only /v1/messages carries LLM traffic. Everything else gets canned responses.
  if (url === '/v1/messages' || url.startsWith('/v1/messages?')) {
    handleOllamaMessages(req, res, body, config);
    return;
  }

  // OAuth token exchange — return a dummy success so the SDK doesn't error
  if (url.includes('/oauth/') || url.includes('/create_api_key')) {
    logger.info({ url }, 'Ollama proxy: returning canned OAuth response');
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ api_key: 'ollama-placeholder', expires_at: '2099-01-01T00:00:00Z' }));
    return;
  }

  // Any other endpoint — 200 with empty response
  logger.info({ url }, 'Ollama proxy: returning empty response for unknown endpoint');
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end('{}');
}

function handleOllamaMessages(
  _req: IncomingMessage,
  res: ServerResponse,
  body: Buffer,
  config: OllamaTranslatorConfig,
): void {
  let anthropicReq: Parameters<typeof translateRequest>[0];
  try {
    anthropicReq = JSON.parse(body.toString());
  } catch (err) {
    res.writeHead(400, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: 'Invalid JSON in request body' }));
    return;
  }

  const isStreaming = anthropicReq.stream === true;
  const ollamaReq = translateRequest(anthropicReq, config);

  const ollamaUrl = new URL(config.ollamaHost);
  const isHttps = ollamaUrl.protocol === 'https:';
  const makeReq = isHttps ? httpsRequest : httpRequest;

  const requestBody = JSON.stringify(ollamaReq);

  logger.info(
    {
      streaming: isStreaming,
      model: ollamaReq.model,
      messageCount: ollamaReq.messages.length,
      toolCount: ollamaReq.tools?.length || 0,
      bodySize: Buffer.byteLength(requestBody),
      numCtx: ollamaReq.options?.num_ctx,
    },
    'Forwarding to Ollama /api/chat',
  );

  const upstreamOpts: RequestOptions = {
    hostname: ollamaUrl.hostname,
    port: ollamaUrl.port || (isHttps ? 443 : 80),
    path: '/api/chat',
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'content-length': Buffer.byteLength(requestBody),
    },
  };

  // For streaming: send SSE headers + keep-alive comments immediately so the
  // SDK doesn't time out during Ollama's slow CPU-based prompt eval.
  let keepAliveTimer: ReturnType<typeof setInterval> | undefined;
  if (isStreaming) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    // SSE comment lines (starting with ':') serve as keep-alive pings
    keepAliveTimer = setInterval(() => {
      if (!res.writableEnded) {
        res.write(': keep-alive\n\n');
      }
    }, 15000);
  }

  const upstream = makeReq(upstreamOpts, (upRes) => {
    if (keepAliveTimer) clearInterval(keepAliveTimer);
    if (isStreaming) {
      handleStreamingResponse(upRes, res);
    } else {
      handleNonStreamingResponse(upRes, res);
    }
  });

  upstream.on('error', (err) => {
    if (keepAliveTimer) clearInterval(keepAliveTimer);
    logger.error({ err, host: config.ollamaHost }, 'Ollama upstream error');
    if (!res.headersSent) {
      res.writeHead(502, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          type: 'error',
          error: {
            type: 'api_error',
            message: `Failed to connect to Ollama at ${config.ollamaHost}: ${err.message}`,
          },
        }),
      );
    } else if (!res.writableEnded) {
      res.end();
    }
  });

  upstream.write(requestBody);
  upstream.end();
}

function handleNonStreamingResponse(
  upRes: IncomingMessage,
  res: ServerResponse,
): void {
  const chunks: Buffer[] = [];
  upRes.on('data', (c) => chunks.push(c));
  upRes.on('end', () => {
    const raw = Buffer.concat(chunks).toString();

    if (upRes.statusCode && upRes.statusCode >= 400) {
      logger.warn(
        { status: upRes.statusCode, body: raw.slice(0, 500) },
        'Ollama returned error',
      );
      res.writeHead(upRes.statusCode, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          type: 'error',
          error: {
            type: 'api_error',
            message: `Ollama error (${upRes.statusCode}): ${raw.slice(0, 200)}`,
          },
        }),
      );
      return;
    }

    try {
      const ollamaRes = JSON.parse(raw);
      const anthropicRes = translateResponse(ollamaRes);
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(anthropicRes);
    } catch (err) {
      logger.error(
        { err, raw: raw.slice(0, 500) },
        'Failed to translate Ollama response',
      );
      res.writeHead(502, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          type: 'error',
          error: {
            type: 'api_error',
            message: 'Failed to parse Ollama response',
          },
        }),
      );
    }
  });
}

function handleStreamingResponse(
  upRes: IncomingMessage,
  res: ServerResponse,
): void {
  if (upRes.statusCode && upRes.statusCode >= 400) {
    // Collect error body and return as Anthropic error
    const chunks: Buffer[] = [];
    upRes.on('data', (c) => chunks.push(c));
    upRes.on('end', () => {
      const raw = Buffer.concat(chunks).toString();
      logger.warn(
        { status: upRes.statusCode, body: raw.slice(0, 500) },
        'Ollama streaming error',
      );
      // Headers may already be sent (keep-alive mode); send error as SSE event
      if (!res.headersSent) {
        res.writeHead(upRes.statusCode!, { 'content-type': 'application/json' });
      }
      res.end(
        JSON.stringify({
          type: 'error',
          error: {
            type: 'api_error',
            message: `Ollama error (${upRes.statusCode}): ${raw.slice(0, 200)}`,
          },
        }),
      );
    });
    return;
  }

  // Headers already sent by keep-alive logic in handleOllamaMessages
  if (!res.headersSent) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
  }

  const translator = new StreamTranslator();
  let ndjsonBuffer = '';

  upRes.on('data', (chunk) => {
    ndjsonBuffer += chunk.toString();
    const { lines, remainder } = parseNDJSONBuffer(ndjsonBuffer);
    ndjsonBuffer = remainder;

    for (const line of lines) {
      const events = translator.processChunk(line);
      for (const event of events) {
        res.write(event);
      }
    }
  });

  upRes.on('end', () => {
    // Process any remaining buffer
    if (ndjsonBuffer.trim()) {
      const events = translator.processChunk(ndjsonBuffer.trim());
      for (const event of events) {
        res.write(event);
      }
    }
    if (!res.writableEnded) {
      res.end();
    }
  });

  upRes.on('error', (err) => {
    logger.error({ err }, 'Ollama streaming upstream error');
    if (!res.writableEnded) {
      res.end();
    }
  });
}

// ---------------------------------------------------------------------------
// Claude mode: credential injection passthrough (original behavior)
// ---------------------------------------------------------------------------

function handleClaudeRequest(
  req: IncomingMessage,
  res: ServerResponse,
  body: Buffer,
  upstreamUrl: URL,
  secrets: Record<string, string>,
  authMode: AuthMode,
): void {
  const isHttps = upstreamUrl.protocol === 'https:';
  const makeRequest = isHttps ? httpsRequest : httpRequest;
  const oauthToken =
    secrets.CLAUDE_CODE_OAUTH_TOKEN || secrets.ANTHROPIC_AUTH_TOKEN;

  const headers: Record<string, string | number | string[] | undefined> = {
    ...(req.headers as Record<string, string>),
    host: upstreamUrl.host,
    'content-length': body.length,
  };

  // Strip hop-by-hop headers that must not be forwarded by proxies
  delete headers['connection'];
  delete headers['keep-alive'];
  delete headers['transfer-encoding'];

  if (authMode === 'api-key') {
    // API key mode: inject x-api-key on every request
    delete headers['x-api-key'];
    headers['x-api-key'] = secrets.ANTHROPIC_API_KEY;
  } else {
    // OAuth mode: replace placeholder Bearer token with the real one
    // only when the container actually sends an Authorization header
    // (exchange request + auth probes). Post-exchange requests use
    // x-api-key only, so they pass through without token injection.
    if (headers['authorization']) {
      delete headers['authorization'];
      if (oauthToken) {
        headers['authorization'] = `Bearer ${oauthToken}`;
      }
    }
  }

  const upstream = makeRequest(
    {
      hostname: upstreamUrl.hostname,
      port: upstreamUrl.port || (isHttps ? 443 : 80),
      path: req.url,
      method: req.method,
      headers,
    } as RequestOptions,
    (upRes) => {
      res.writeHead(upRes.statusCode!, upRes.headers);
      upRes.pipe(res);
    },
  );

  upstream.on('error', (err) => {
    logger.error({ err, url: req.url }, 'Credential proxy upstream error');
    if (!res.headersSent) {
      res.writeHead(502);
      res.end('Bad Gateway');
    }
  });

  upstream.write(body);
  upstream.end();
}

// ---------------------------------------------------------------------------
// Dashboard: runtime Ollama config switching
// ---------------------------------------------------------------------------

function fetchJson(url: string): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const makeReq = parsed.protocol === 'https:' ? httpsRequest : httpRequest;
    const req = makeReq(url, (res) => {
      const chunks: Buffer[] = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        try {
          resolve(JSON.parse(Buffer.concat(chunks).toString()));
        } catch {
          resolve(null);
        }
      });
    });
    req.on('error', reject);
    req.setTimeout(5000, () => { req.destroy(); reject(new Error('timeout')); });
    req.end();
  });
}

function handleDashboardStatus(
  res: ServerResponse,
  config: OllamaTranslatorConfig | undefined,
): void {
  if (!config) {
    res.writeHead(404, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not in Ollama mode' }));
    return;
  }

  const host = config.ollamaHost;
  Promise.all([
    fetchJson(`${host}/api/tags`).catch(() => ({ models: [] })),
    fetchJson(`${host}/api/ps`).catch(() => ({ models: [] })),
  ]).then(([tags, ps]) => {
    const available = ((tags as { models?: { name: string }[] }).models || []).map(
      (m) => m.name,
    );
    const running = ((ps as { models?: { name: string }[] }).models || []).map(
      (m) => m.name,
    );
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(
      JSON.stringify({
        currentModel: config.ollamaModel,
        think: config.think ?? false,
        contextLength: config.ollamaContextLength ?? 8192,
        availableModels: available,
        runningModels: running,
      }),
    );
  });
}

function handleDashboardConfigUpdate(
  body: Buffer,
  res: ServerResponse,
  config: OllamaTranslatorConfig | undefined,
): void {
  if (!config) {
    res.writeHead(404, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not in Ollama mode' }));
    return;
  }

  let update: { model?: string; think?: boolean; contextLength?: number };
  try {
    update = JSON.parse(body.toString());
  } catch {
    res.writeHead(400, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: 'Invalid JSON' }));
    return;
  }

  if (update.model !== undefined) {
    if (typeof update.model !== 'string' || !update.model) {
      res.writeHead(400, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'model must be a non-empty string' }));
      return;
    }
    config.ollamaModel = update.model;
  }
  if (update.think !== undefined) {
    if (typeof update.think !== 'boolean') {
      res.writeHead(400, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'think must be a boolean' }));
      return;
    }
    config.think = update.think;
  }
  if (update.contextLength !== undefined) {
    const n = Number(update.contextLength);
    if (!Number.isInteger(n) || n <= 0) {
      res.writeHead(400, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ error: 'contextLength must be a positive integer' }));
      return;
    }
    config.ollamaContextLength = n;
  }

  logger.info(
    { model: config.ollamaModel, think: config.think, contextLength: config.ollamaContextLength },
    'Ollama config updated via dashboard',
  );

  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(
    JSON.stringify({
      ok: true,
      currentModel: config.ollamaModel,
      think: config.think ?? false,
      contextLength: config.ollamaContextLength ?? 8192,
    }),
  );
}

function serveDashboard(res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  res.end(DASHBOARD_HTML);
}

const DASHBOARD_HTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NanoClaw — Ollama Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 2rem; }
  h1 { color: #7fdbca; margin-bottom: 1.5rem; font-size: 1.4rem; }
  .card { background: #16213e; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
  .card h2 { color: #82aaff; font-size: 1rem; margin-bottom: 1rem; }
  label { display: block; margin-bottom: 0.75rem; font-size: 0.9rem; }
  label span { display: inline-block; width: 120px; color: #888; }
  select, input[type=number] { background: #0f3460; color: #e0e0e0; border: 1px solid #444; border-radius: 4px; padding: 0.4rem 0.6rem; font-size: 0.9rem; }
  select { min-width: 220px; }
  input[type=number] { width: 100px; }
  .toggle { position: relative; display: inline-block; width: 44px; height: 24px; vertical-align: middle; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .toggle .slider { position: absolute; inset: 0; background: #333; border-radius: 24px; cursor: pointer; transition: .2s; }
  .toggle .slider:before { content: ""; position: absolute; height: 18px; width: 18px; left: 3px; bottom: 3px; background: #e0e0e0; border-radius: 50%; transition: .2s; }
  .toggle input:checked + .slider { background: #7fdbca; }
  .toggle input:checked + .slider:before { transform: translateX(20px); }
  button { background: #7fdbca; color: #1a1a2e; border: none; border-radius: 4px; padding: 0.5rem 1.5rem; font-size: 0.9rem; font-weight: 600; cursor: pointer; margin-top: 0.5rem; }
  button:hover { background: #64c4aa; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .status { font-size: 0.85rem; color: #888; margin-top: 0.5rem; }
  .status.ok { color: #7fdbca; }
  .status.err { color: #ff6b6b; }
  .running { display: inline-block; background: #0f3460; border-radius: 4px; padding: 0.2rem 0.5rem; font-size: 0.8rem; margin-top: 0.25rem; }
  .running.active { color: #7fdbca; border: 1px solid #7fdbca; }
  .running.idle { color: #888; border: 1px solid #444; }
</style>
</head>
<body>
<h1>NanoClaw — Ollama Dashboard</h1>

<div class="card">
  <h2>Model Configuration</h2>
  <label><span>Model</span> <select id="model"></select></label>
  <label><span>Think mode</span>
    <span class="toggle"><input type="checkbox" id="think"><span class="slider"></span></span>
  </label>
  <label><span>Context length</span> <input type="number" id="ctx" min="1024" step="1024"></label>
  <button id="save">Apply</button> <button id="refresh" style="background:#3a506b">Refresh Models</button>
  <div class="status" id="status"></div>
</div>

<div class="card">
  <h2>Running Models</h2>
  <div id="running">Loading...</div>
</div>

<script>
const $model = document.getElementById('model');
const $think = document.getElementById('think');
const $ctx   = document.getElementById('ctx');
const $save  = document.getElementById('save');
const $status = document.getElementById('status');
const $running = document.getElementById('running');

async function load() {
  try {
    const r = await fetch('/api/ollama/status');
    const d = await r.json();
    if (d.error) { $status.textContent = d.error; $status.className = 'status err'; return; }

    $model.innerHTML = d.availableModels.map(m =>
      '<option' + (m === d.currentModel ? ' selected' : '') + '>' + m + '</option>'
    ).join('');
    $think.checked = d.think;
    $ctx.value = d.contextLength;

    $running.innerHTML = d.availableModels.map(m => {
      const isRunning = d.runningModels.includes(m);
      return '<span class="running ' + (isRunning ? 'active' : 'idle') + '">' + m + (isRunning ? ' (loaded)' : '') + '</span> ';
    }).join('');
  } catch(e) { $status.textContent = 'Failed to load: ' + e.message; $status.className = 'status err'; }
}

$save.onclick = async () => {
  $save.disabled = true;
  $status.textContent = 'Saving...';
  $status.className = 'status';
  try {
    const r = await fetch('/api/ollama/config', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: $model.value,
        think: $think.checked,
        contextLength: parseInt($ctx.value, 10),
      }),
    });
    const d = await r.json();
    if (d.ok) {
      $status.textContent = 'Applied: ' + d.currentModel + ' | think=' + d.think + ' | ctx=' + d.contextLength;
      $status.className = 'status ok';
      setTimeout(load, 1000);
    } else {
      $status.textContent = d.error || 'Unknown error';
      $status.className = 'status err';
    }
  } catch(e) { $status.textContent = 'Error: ' + e.message; $status.className = 'status err'; }
  $save.disabled = false;
};

document.getElementById('refresh').onclick = load;
load();
</script>
</body>
</html>`;

// ---------------------------------------------------------------------------
// Server startup
// ---------------------------------------------------------------------------

export function startCredentialProxy(
  port: number,
  host = '127.0.0.1',
): Promise<Server> {
  const secrets = readEnvFile([
    'ANTHROPIC_API_KEY',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'ANTHROPIC_AUTH_TOKEN',
    'ANTHROPIC_BASE_URL',
    'OLLAMA_MODEL',
    'OLLAMA_HOST',
    'OLLAMA_CONTEXT_LENGTH',
    'OLLAMA_SYSTEM_PROMPT_FILE',
  ]);

  const ollamaModel = secrets.OLLAMA_MODEL;
  const isOllamaMode = !!ollamaModel;

  let ollamaConfig: OllamaTranslatorConfig | undefined;
  if (isOllamaMode) {
    const contextLength = parseInt(secrets.OLLAMA_CONTEXT_LENGTH || '8192', 10);

    // Load custom system prompt to replace the CLI's massive 70KB one.
    // Falls back to groups/global/CLAUDE.md if no file specified.
    let systemPromptOverride: string | undefined;
    const promptFile = secrets.OLLAMA_SYSTEM_PROMPT_FILE
      || join(process.cwd(), 'groups', 'global', 'CLAUDE.md');
    if (existsSync(promptFile)) {
      systemPromptOverride = readFileSync(promptFile, 'utf-8');
      logger.info({ file: promptFile, size: systemPromptOverride.length },
        'Loaded custom system prompt for Ollama');
    }

    ollamaConfig = {
      ollamaModel,
      ollamaHost: secrets.OLLAMA_HOST || 'http://localhost:11434',
      ollamaContextLength: contextLength,
      systemPromptOverride,
      // Only forward essential tools — the CLI sends 30+ tools totaling ~75KB.
      allowedToolNames: [
        'Bash', 'Read', 'Write', 'Edit', 'Glob', 'Grep',
        'mcp__nanoclaw__*',
      ],
      think: false,
    };
    logger.info(
      { model: ollamaModel, host: ollamaConfig.ollamaHost },
      'Ollama mode enabled — all LLM traffic will be routed to Ollama',
    );
  }

  // Claude mode config (only used when not in Ollama mode)
  const authMode: AuthMode = secrets.ANTHROPIC_API_KEY ? 'api-key' : 'oauth';
  const upstreamUrl = new URL(
    secrets.ANTHROPIC_BASE_URL || 'https://api.anthropic.com',
  );

  return new Promise((resolve, reject) => {
    const server = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', (c) => chunks.push(c));
      req.on('end', () => {
        const body = Buffer.concat(chunks);
        const url = req.url || '';

        // Dashboard routes
        if (req.method === 'GET' && url === '/dashboard') {
          serveDashboard(res);
          return;
        }
        if (req.method === 'GET' && url === '/api/ollama/status') {
          handleDashboardStatus(res, ollamaConfig);
          return;
        }
        if (req.method === 'POST' && url === '/api/ollama/config') {
          handleDashboardConfigUpdate(body, res, ollamaConfig);
          return;
        }

        if (isOllamaMode) {
          handleOllamaRequest(req, res, body, ollamaConfig!);
        } else {
          handleClaudeRequest(req, res, body, upstreamUrl, secrets, authMode);
        }
      });
    });

    server.listen(port, host, () => {
      logger.info(
        { port, host, mode: isOllamaMode ? 'ollama' : authMode },
        'Credential proxy started',
      );
      resolve(server);
    });

    server.on('error', reject);
  });
}

/** Detect which auth mode the host is configured for. */
export function detectAuthMode(): AuthMode {
  const secrets = readEnvFile(['ANTHROPIC_API_KEY', 'OLLAMA_MODEL']);
  // In Ollama mode, use api-key mode with placeholder (simplest path)
  if (secrets.OLLAMA_MODEL) return 'api-key';
  return secrets.ANTHROPIC_API_KEY ? 'api-key' : 'oauth';
}
