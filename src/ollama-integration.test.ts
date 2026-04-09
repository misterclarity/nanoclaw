/**
 * Integration tests for the Ollama translation proxy.
 * Requires Ollama running locally with gemma4:e2b installed.
 *
 * Run: npx vitest run src/ollama-integration.test.ts
 */
import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import http from 'http';
import type { AddressInfo } from 'net';

// Check if Ollama is actually running before tests
async function isOllamaRunning(): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get('http://localhost:11434/api/tags', (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(2000, () => {
      req.destroy();
      resolve(false);
    });
  });
}

const mockEnv: Record<string, string> = {};
vi.mock('./env.js', () => ({
  readEnvFile: vi.fn(() => ({ ...mockEnv })),
}));

vi.mock('./logger.js', () => ({
  logger: {
    info: vi.fn(),
    error: vi.fn(),
    debug: vi.fn(),
    warn: vi.fn(),
    fatal: vi.fn(),
  },
}));

import { startCredentialProxy } from './credential-proxy.js';

function makeRequest(
  port: number,
  body: unknown,
  streaming = false,
): Promise<{
  statusCode: number;
  body: string;
  headers: http.IncomingHttpHeaders;
}> {
  const payload = JSON.stringify(body);
  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: '127.0.0.1',
        port,
        method: 'POST',
        path: '/v1/messages',
        headers: {
          'content-type': 'application/json',
          'content-length': Buffer.byteLength(payload),
          'x-api-key': 'placeholder',
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          resolve({
            statusCode: res.statusCode!,
            body: Buffer.concat(chunks).toString(),
            headers: res.headers,
          });
        });
      },
    );
    req.on('error', reject);
    req.write(payload);
    req.end();
  });
}

describe('Ollama integration', () => {
  let proxyServer: http.Server;
  let proxyPort: number;
  let ollamaAvailable = false;

  beforeAll(async () => {
    ollamaAvailable = await isOllamaRunning();
    if (!ollamaAvailable) return;

    Object.assign(mockEnv, {
      OLLAMA_MODEL: 'gemma4:e2b',
      OLLAMA_HOST: 'http://localhost:11434',
    });
    proxyServer = await startCredentialProxy(0);
    proxyPort = (proxyServer.address() as AddressInfo).port;
  });

  afterAll(async () => {
    if (proxyServer) {
      await new Promise<void>((r) => proxyServer.close(() => r()));
    }
  });

  it('translates a simple text request and gets valid Anthropic response', async () => {
    if (!ollamaAvailable) {
      console.log('  ⏭ Skipped: Ollama not running');
      return;
    }

    const res = await makeRequest(proxyPort, {
      model: 'claude-sonnet-4-20250514',
      max_tokens: 50,
      messages: [{ role: 'user', content: 'Say hello in one word.' }],
      stream: false,
    });

    expect(res.statusCode).toBe(200);

    const parsed = JSON.parse(res.body);
    expect(parsed.type).toBe('message');
    expect(parsed.role).toBe('assistant');
    expect(parsed.stop_reason).toBe('end_turn');
    expect(parsed.content).toBeInstanceOf(Array);
    expect(parsed.content.length).toBeGreaterThanOrEqual(1);
    expect(parsed.content[0].type).toBe('text');
    expect(parsed.content[0].text.length).toBeGreaterThan(0);
    expect(parsed.usage).toBeDefined();
    expect(parsed.usage.input_tokens).toBeGreaterThan(0);
    expect(parsed.usage.output_tokens).toBeGreaterThan(0);
    expect(parsed.id).toMatch(/^msg_/);
  }, 30000);

  it('handles tool use round-trip', async () => {
    if (!ollamaAvailable) {
      console.log('  ⏭ Skipped: Ollama not running');
      return;
    }

    const res = await makeRequest(proxyPort, {
      model: 'claude-sonnet-4-20250514',
      max_tokens: 100,
      messages: [
        { role: 'user', content: 'List the files in /tmp using the Bash tool.' },
      ],
      tools: [
        {
          name: 'Bash',
          description: 'Run a bash command and return the output',
          input_schema: {
            type: 'object',
            properties: {
              command: {
                type: 'string',
                description: 'The bash command to execute',
              },
            },
            required: ['command'],
          },
        },
      ],
      stream: false,
    });

    expect(res.statusCode).toBe(200);

    const parsed = JSON.parse(res.body);
    expect(parsed.type).toBe('message');
    expect(parsed.role).toBe('assistant');

    // Should have a tool_use content block
    const toolUse = parsed.content.find(
      (b: { type: string }) => b.type === 'tool_use',
    );
    expect(toolUse).toBeDefined();
    expect(toolUse.name).toBe('Bash');
    expect(toolUse.input).toBeDefined();
    expect(typeof toolUse.input.command).toBe('string');
    expect(toolUse.id).toBeDefined();
    expect(parsed.stop_reason).toBe('tool_use');
  }, 30000);

  it('handles streaming text response', async () => {
    if (!ollamaAvailable) {
      console.log('  ⏭ Skipped: Ollama not running');
      return;
    }

    const res = await makeRequest(proxyPort, {
      model: 'claude-sonnet-4-20250514',
      max_tokens: 50,
      messages: [{ role: 'user', content: 'Say hi.' }],
      stream: true,
    });

    expect(res.statusCode).toBe(200);
    expect(res.headers['content-type']).toBe('text/event-stream');

    const events = res.body
      .split('\n')
      .filter((l) => l.startsWith('event: '))
      .map((l) => l.slice(7));

    // Must have the full Anthropic SSE event sequence
    expect(events).toContain('message_start');
    expect(events).toContain('content_block_start');
    expect(events).toContain('content_block_delta');
    expect(events).toContain('content_block_stop');
    expect(events).toContain('message_delta');
    expect(events).toContain('message_stop');

    // Parse the message_start to verify structure
    const messageStartLine = res.body
      .split('\n')
      .find((l) => l.startsWith('data: ') && l.includes('message_start'));
    expect(messageStartLine).toBeDefined();
    const messageStart = JSON.parse(messageStartLine!.slice(6));
    expect(messageStart.message.role).toBe('assistant');
    expect(messageStart.message.id).toMatch(/^msg_/);

    // Parse message_delta to verify stop reason
    const messageDeltaLine = res.body
      .split('\n')
      .find((l) => l.startsWith('data: ') && l.includes('message_delta'));
    expect(messageDeltaLine).toBeDefined();
    const messageDelta = JSON.parse(messageDeltaLine!.slice(6));
    expect(messageDelta.delta.stop_reason).toBe('end_turn');
  }, 30000);

  it('handles streaming tool use response', async () => {
    if (!ollamaAvailable) {
      console.log('  ⏭ Skipped: Ollama not running');
      return;
    }

    const res = await makeRequest(proxyPort, {
      model: 'claude-sonnet-4-20250514',
      max_tokens: 100,
      messages: [
        { role: 'user', content: 'Run the command: echo test' },
      ],
      tools: [
        {
          name: 'Bash',
          description: 'Run a bash command',
          input_schema: {
            type: 'object',
            properties: {
              command: { type: 'string', description: 'The command to run' },
            },
            required: ['command'],
          },
        },
      ],
      stream: true,
    });

    expect(res.statusCode).toBe(200);

    const events = res.body
      .split('\n')
      .filter((l) => l.startsWith('event: '))
      .map((l) => l.slice(7));

    expect(events).toContain('message_start');
    expect(events).toContain('message_stop');

    // Should have tool_use content block
    const blockStartLines = res.body
      .split('\n')
      .filter(
        (l) =>
          l.startsWith('data: ') && l.includes('content_block_start'),
      );

    const hasToolBlock = blockStartLines.some((l) => {
      const data = JSON.parse(l.slice(6));
      return data.content_block?.type === 'tool_use';
    });
    expect(hasToolBlock).toBe(true);

    // message_delta should have tool_use stop reason
    const messageDeltaLine = res.body
      .split('\n')
      .find((l) => l.startsWith('data: ') && l.includes('message_delta'));
    const messageDelta = JSON.parse(messageDeltaLine!.slice(6));
    expect(messageDelta.delta.stop_reason).toBe('tool_use');
  }, 30000);

  it('returns canned response for OAuth endpoints', async () => {
    if (!ollamaAvailable) {
      console.log('  ⏭ Skipped: Ollama not running');
      return;
    }

    const payload = '{}';
    const res = await new Promise<{ statusCode: number; body: string }>(
      (resolve, reject) => {
        const req = http.request(
          {
            hostname: '127.0.0.1',
            port: proxyPort,
            method: 'POST',
            path: '/api/oauth/claude_cli/create_api_key',
            headers: {
              'content-type': 'application/json',
              'content-length': Buffer.byteLength(payload),
            },
          },
          (res) => {
            const chunks: Buffer[] = [];
            res.on('data', (c) => chunks.push(c));
            res.on('end', () => {
              resolve({
                statusCode: res.statusCode!,
                body: Buffer.concat(chunks).toString(),
              });
            });
          },
        );
        req.on('error', reject);
        req.write(payload);
        req.end();
      },
    );

    expect(res.statusCode).toBe(200);
    const parsed = JSON.parse(res.body);
    expect(parsed.api_key).toBeDefined();
  });
});
