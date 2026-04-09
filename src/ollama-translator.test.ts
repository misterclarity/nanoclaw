import { describe, it, expect } from 'vitest';
import {
  translateRequest,
  translateResponse,
  StreamTranslator,
  parseNDJSONBuffer,
  OllamaTranslatorConfig,
} from './ollama-translator.js';

const config: OllamaTranslatorConfig = {
  ollamaModel: 'gemma4:e2b',
  ollamaHost: 'http://localhost:11434',
  systemPromptTransforms: false,
};

describe('translateRequest', () => {
  it('translates a basic text message', () => {
    const result = translateRequest(
      {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        system: 'You are helpful.',
        messages: [{ role: 'user', content: 'Hello' }],
      },
      config,
    );

    expect(result.model).toBe('gemma4:e2b');
    expect(result.stream).toBe(false);
    expect(result.think).toBe(false);
    expect(result.options?.num_predict).toBe(1024);
    expect(result.messages).toEqual([
      { role: 'system', content: 'You are helpful.' },
      { role: 'user', content: 'Hello' },
    ]);
  });

  it('translates system prompt from array format', () => {
    const result = translateRequest(
      {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        system: [
          { type: 'text', text: 'Part one.' },
          { type: 'text', text: 'Part two.' },
        ],
        messages: [{ role: 'user', content: 'Hi' }],
      },
      config,
    );

    expect(result.messages[0]).toEqual({
      role: 'system',
      content: 'Part one.\nPart two.',
    });
  });

  it('translates tool definitions', () => {
    const result = translateRequest(
      {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [
          {
            name: 'get_weather',
            description: 'Get weather',
            input_schema: {
              type: 'object',
              properties: { city: { type: 'string' } },
              required: ['city'],
            },
          },
        ],
      },
      config,
    );

    expect(result.tools).toEqual([
      {
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get weather',
          parameters: {
            type: 'object',
            properties: { city: { type: 'string' } },
            required: ['city'],
          },
        },
      },
    ]);
  });

  it('translates assistant messages with tool_use blocks', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 1024,
        messages: [
          { role: 'user', content: 'Run ls' },
          {
            role: 'assistant',
            content: [
              { type: 'text', text: 'Running ls.' },
              {
                type: 'tool_use',
                id: 'toolu_123',
                name: 'Bash',
                input: { command: 'ls' },
              },
            ],
          },
        ],
      },
      config,
    );

    const assistantMsg = result.messages.find((m) => m.role === 'assistant')!;
    expect(assistantMsg.content).toBe('Running ls.');
    expect(assistantMsg.tool_calls).toEqual([
      {
        id: 'toolu_123',
        function: {
          name: 'Bash',
          arguments: { command: 'ls' },
        },
      },
    ]);
  });

  it('translates tool_result blocks to role:tool messages', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'toolu_123',
                content: 'file1.txt\nfile2.txt',
              },
            ],
          },
        ],
      },
      config,
    );

    expect(result.messages).toEqual([
      {
        role: 'tool',
        content: 'file1.txt\nfile2.txt',
      },
    ]);
  });

  it('translates tool_result with is_error flag', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'toolu_456',
                content: 'command not found',
                is_error: true,
              },
            ],
          },
        ],
      },
      config,
    );

    expect(result.messages[0].content).toBe('[ERROR] command not found');
  });

  it('splits mixed user message: tool_results + text', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 1024,
        messages: [
          {
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'toolu_789',
                content: 'done',
              },
              { type: 'text', text: 'Now do the next thing' },
            ],
          },
        ],
      },
      config,
    );

    expect(result.messages).toEqual([
      { role: 'tool', content: 'done' },
      { role: 'user', content: 'Now do the next thing' },
    ]);
  });

  it('maps parameters to Ollama options', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 2048,
        messages: [],
        temperature: 0.5,
        top_p: 0.9,
        top_k: 40,
        stop_sequences: ['\n\nHuman:'],
      },
      config,
    );
    expect(result.options).toEqual({
      temperature: 0.5,
      top_p: 0.9,
      top_k: 40,
      stop: ['\n\nHuman:'],
      num_predict: 2048,
    });
  });

  it('defaults think to false', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 1,
        messages: [],
      },
      config,
    );
    expect(result.think).toBe(false);
  });

  it('uses think from config when set to true', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 1,
        messages: [],
      },
      { ...config, think: true },
    );
    expect(result.think).toBe(true);
  });

  it('preserves streaming flag', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 1,
        messages: [],
        stream: true,
      },
      config,
    );
    expect(result.stream).toBe(true);
  });

  it('uses systemPromptOverride instead of incoming system prompt', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 1024,
        system: 'This is a very long system prompt that should be replaced.',
        messages: [{ role: 'user', content: 'Hi' }],
      },
      { ...config, systemPromptOverride: 'Short prompt.' },
    );

    expect(result.messages[0]).toEqual({
      role: 'system',
      content: 'Short prompt.',
    });
  });

  it('omits system message when systemPromptOverride is empty string', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 1024,
        system: 'Should be dropped.',
        messages: [{ role: 'user', content: 'Hi' }],
      },
      { ...config, systemPromptOverride: '' },
    );

    expect(result.messages).toEqual([{ role: 'user', content: 'Hi' }]);
  });

  it('filters tools by allowedToolNames with exact match', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [
          { name: 'Bash', description: 'Run bash', input_schema: { type: 'object' } },
          { name: 'Read', description: 'Read file', input_schema: { type: 'object' } },
          { name: 'Agent', description: 'Launch agent', input_schema: { type: 'object' } },
          { name: 'WebSearch', description: 'Search web', input_schema: { type: 'object' } },
        ],
      },
      { ...config, allowedToolNames: ['Bash', 'Read'] },
    );

    expect(result.tools).toHaveLength(2);
    expect(result.tools!.map((t) => t.function.name)).toEqual(['Bash', 'Read']);
  });

  it('filters tools by allowedToolNames with wildcard prefix', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [
          { name: 'Bash', description: 'Run bash', input_schema: { type: 'object' } },
          { name: 'mcp__nanoclaw__send_message', description: 'Send', input_schema: { type: 'object' } },
          { name: 'mcp__nanoclaw__list_tasks', description: 'List', input_schema: { type: 'object' } },
          { name: 'mcp__other__foo', description: 'Other', input_schema: { type: 'object' } },
        ],
      },
      { ...config, allowedToolNames: ['Bash', 'mcp__nanoclaw__*'] },
    );

    expect(result.tools).toHaveLength(3);
    expect(result.tools!.map((t) => t.function.name)).toEqual([
      'Bash',
      'mcp__nanoclaw__send_message',
      'mcp__nanoclaw__list_tasks',
    ]);
  });

  it('passes all tools when allowedToolNames is not set', () => {
    const result = translateRequest(
      {
        model: 'x',
        max_tokens: 1024,
        messages: [{ role: 'user', content: 'Hi' }],
        tools: [
          { name: 'Bash', description: 'a', input_schema: { type: 'object' } },
          { name: 'Agent', description: 'b', input_schema: { type: 'object' } },
        ],
      },
      config,
    );

    expect(result.tools).toHaveLength(2);
  });
});

describe('translateResponse', () => {
  it('translates a basic text response', () => {
    const result = JSON.parse(
      translateResponse({
        model: 'gemma4:e2b',
        created_at: '2026-04-07T07:15:49Z',
        message: { role: 'assistant', content: 'Hello!' },
        done: true,
        done_reason: 'stop',
        prompt_eval_count: 10,
        eval_count: 5,
      }),
    );

    expect(result.type).toBe('message');
    expect(result.role).toBe('assistant');
    expect(result.content).toEqual([{ type: 'text', text: 'Hello!' }]);
    expect(result.stop_reason).toBe('end_turn');
    expect(result.usage).toEqual({ input_tokens: 10, output_tokens: 5 });
  });

  it('translates tool_calls response', () => {
    const result = JSON.parse(
      translateResponse({
        model: 'gemma4:e2b',
        created_at: '2026-04-07T07:16:22Z',
        message: {
          role: 'assistant',
          content: '',
          tool_calls: [
            {
              id: 'call_abc',
              function: {
                index: 0,
                name: 'Bash',
                arguments: { command: 'ls -la' },
              },
            },
          ],
        },
        done: true,
        done_reason: 'stop',
        prompt_eval_count: 70,
        eval_count: 13,
      }),
    );

    expect(result.stop_reason).toBe('tool_use');
    // Tool call with no text content — should only have tool_use block
    expect(result.content).toHaveLength(1);
    expect(result.content[0].type).toBe('tool_use');
    expect(result.content[0].name).toBe('Bash');
    expect(result.content[0].input).toEqual({ command: 'ls -la' });
  });

  it('translates tool_calls with text content', () => {
    const result = JSON.parse(
      translateResponse({
        model: 'gemma4:e2b',
        created_at: '2026-04-07T07:16:22Z',
        message: {
          role: 'assistant',
          content: 'Let me check.',
          tool_calls: [
            {
              id: 'call_xyz',
              function: {
                index: 0,
                name: 'Bash',
                arguments: { command: 'ls' },
              },
            },
          ],
        },
        done: true,
        done_reason: 'stop',
      }),
    );

    expect(result.stop_reason).toBe('tool_use');
    expect(result.content).toHaveLength(2);
    expect(result.content[0]).toEqual({ type: 'text', text: 'Let me check.' });
    expect(result.content[1].type).toBe('tool_use');
    expect(result.content[1].input).toEqual({ command: 'ls' });
  });

  it('maps done_reason correctly', () => {
    const test = (reason: string, expected: string) => {
      const r = JSON.parse(
        translateResponse({
          model: 'test',
          created_at: '',
          message: { role: 'assistant', content: 'x' },
          done: true,
          done_reason: reason,
        }),
      );
      expect(r.stop_reason).toBe(expected);
    };

    test('stop', 'end_turn');
    test('length', 'max_tokens');
  });

  it('handles empty response', () => {
    const result = JSON.parse(
      translateResponse({
        model: 'test',
        created_at: '',
        message: { role: 'assistant', content: '' },
        done: true,
        done_reason: 'stop',
      }),
    );
    expect(result.content).toEqual([{ type: 'text', text: '' }]);
  });
});

describe('StreamTranslator', () => {
  it('translates a simple text stream', () => {
    const translator = new StreamTranslator();
    const allEvents: string[] = [];

    // Token chunks
    allEvents.push(
      ...translator.processChunk(
        '{"model":"gemma4:e2b","created_at":"2026-04-07T07:16:32Z","message":{"role":"assistant","content":"Hello"},"done":false}',
      ),
    );
    allEvents.push(
      ...translator.processChunk(
        '{"model":"gemma4:e2b","created_at":"2026-04-07T07:16:32Z","message":{"role":"assistant","content":" world!"},"done":false}',
      ),
    );

    // Final chunk with done:true
    allEvents.push(
      ...translator.processChunk(
        '{"model":"gemma4:e2b","created_at":"2026-04-07T07:16:33Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":11,"eval_count":3}',
      ),
    );

    const raw = allEvents.join('');

    expect(raw).toContain('event: message_start');
    expect(raw).toContain('event: content_block_start');
    expect(raw).toContain('"text_delta"');
    expect(raw).toContain('"Hello"');
    expect(raw).toContain('" world!"');
    expect(raw).toContain('event: content_block_stop');
    expect(raw).toContain('"stop_reason":"end_turn"');
    expect(raw).toContain('event: message_stop');
  });

  it('translates a tool call stream', () => {
    const translator = new StreamTranslator();
    const allEvents: string[] = [];

    // Tool call comes as a complete object in one chunk
    allEvents.push(
      ...translator.processChunk(
        '{"model":"gemma4:e2b","created_at":"2026-04-07T07:16:57Z","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_abc","function":{"index":0,"name":"Bash","arguments":{"command":"ls /tmp"}}}]},"done":false}',
      ),
    );

    // Final chunk
    allEvents.push(
      ...translator.processChunk(
        '{"model":"gemma4:e2b","created_at":"2026-04-07T07:16:57Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":67,"eval_count":15}',
      ),
    );

    const raw = allEvents.join('');

    expect(raw).toContain('event: message_start');
    expect(raw).toContain('"tool_use"');
    expect(raw).toContain('"Bash"');
    expect(raw).toContain('"input_json_delta"');
    expect(raw).toContain('"stop_reason":"tool_use"');
    expect(raw).toContain('event: message_stop');
  });

  it('handles text followed by tool call', () => {
    const translator = new StreamTranslator();
    const allEvents: string[] = [];

    // Text first
    allEvents.push(
      ...translator.processChunk(
        '{"model":"test","created_at":"","message":{"role":"assistant","content":"Let me check."},"done":false}',
      ),
    );

    // Then tool call
    allEvents.push(
      ...translator.processChunk(
        '{"model":"test","created_at":"","message":{"role":"assistant","content":"","tool_calls":[{"id":"call_1","function":{"name":"Bash","arguments":{"command":"ls"}}}]},"done":false}',
      ),
    );

    // Done
    allEvents.push(
      ...translator.processChunk(
        '{"model":"test","created_at":"","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}',
      ),
    );

    const raw = allEvents.join('');

    // Should have text block, then tool block
    expect(raw).toContain('"text_delta"');
    expect(raw).toContain('"Let me check."');
    expect(raw).toContain('"tool_use"');
    expect(raw).toContain('"stop_reason":"tool_use"');
  });
});

describe('parseNDJSONBuffer', () => {
  it('extracts complete lines from buffer', () => {
    const { lines, remainder } = parseNDJSONBuffer(
      '{"test":1}\n{"test":2}\n',
    );
    expect(lines).toEqual(['{"test":1}', '{"test":2}']);
    expect(remainder).toBe('');
  });

  it('handles incomplete lines', () => {
    const { lines, remainder } = parseNDJSONBuffer(
      '{"complete":true}\n{"incom',
    );
    expect(lines).toEqual(['{"complete":true}']);
    expect(remainder).toBe('{"incom');
  });

  it('skips empty lines', () => {
    const { lines } = parseNDJSONBuffer('{"a":1}\n\n{"b":2}\n');
    expect(lines).toEqual(['{"a":1}', '{"b":2}']);
  });
});
