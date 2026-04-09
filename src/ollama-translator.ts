/**
 * Anthropic Messages API ↔ Ollama native API translator.
 *
 * Sits inside the credential proxy. When OLLAMA_MODEL is set, requests
 * to /v1/messages are translated to Ollama's /api/chat endpoint, and
 * responses are translated back to Anthropic format.
 *
 * Uses the native Ollama API (not the OpenAI-compatible endpoint) because:
 * - The native API supports `think: false` to disable thinking mode
 * - Tool call arguments come pre-parsed (no JSON string wrangling)
 * - Streaming is simpler (newline-delimited JSON, not SSE)
 *
 * The Claude Agent SDK and CLI are treated as black boxes — they speak
 * Anthropic API, and this module makes Ollama look like Anthropic to them.
 */

// ---------------------------------------------------------------------------
// Types: Anthropic API
// ---------------------------------------------------------------------------

interface AnthropicRequest {
  model: string;
  max_tokens: number;
  system?: string | Array<{ type: string; text: string }>;
  messages: AnthropicMessage[];
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop_sequences?: string[];
  stream?: boolean;
  tools?: AnthropicTool[];
  tool_choice?: { type: string; name?: string };
  metadata?: { user_id?: string };
}

interface AnthropicMessage {
  role: string;
  content: string | AnthropicContentBlock[];
}

interface AnthropicContentBlock {
  type: string;
  text?: string;
  id?: string;
  name?: string;
  input?: unknown;
  tool_use_id?: string;
  content?: string | Array<{ type: string; text: string }>;
  is_error?: boolean;
  source?: { type: string; media_type: string; data: string };
}

interface AnthropicTool {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Types: Ollama native API
// ---------------------------------------------------------------------------

interface OllamaRequest {
  model: string;
  messages: OllamaMessage[];
  stream: boolean;
  think: boolean;
  tools?: OllamaTool[];
  options?: {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    stop?: string[];
    num_predict?: number;
    num_ctx?: number;
  };
}

interface OllamaMessage {
  role: string;
  content: string;
  tool_calls?: OllamaToolCall[];
  images?: string[];
}

interface OllamaToolCall {
  id?: string;
  function: {
    index?: number;
    name: string;
    arguments: Record<string, unknown>;
  };
}

interface OllamaTool {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

interface OllamaResponse {
  model: string;
  created_at: string;
  message: OllamaMessage;
  done: boolean;
  done_reason?: string;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

export interface OllamaTranslatorConfig {
  ollamaModel: string;
  ollamaHost: string;
  ollamaContextLength?: number;
  systemPromptTransforms?: boolean;
  /** When set, replaces the incoming system prompt entirely.
   *  The Claude Code CLI always sends its full ~70KB system prompt,
   *  which is far too large for CPU-only local models. This allows
   *  the proxy to substitute a minimal system prompt instead. */
  systemPromptOverride?: string;
  /** When set, only tools whose names match this list are forwarded.
   *  The CLI sends 30+ tool definitions (~75KB) regardless of
   *  allowedTools. This filters them at the proxy level. */
  allowedToolNames?: string[];
  /** Enable thinking/reasoning mode. Default false. */
  think?: boolean;
}

// ---------------------------------------------------------------------------
// Request translation: Anthropic → Ollama
// ---------------------------------------------------------------------------

function translateSystemPrompt(
  system: string | Array<{ type: string; text: string }> | undefined,
): string | undefined {
  if (!system) return undefined;
  if (typeof system === 'string') return system;
  return system
    .filter((b) => b.type === 'text')
    .map((b) => b.text)
    .join('\n');
}

function patchSystemPromptForOllama(text: string): string {
  // Strip Claude-specific extended thinking instructions
  let patched = text.replace(
    /You should think step-by-step.*?before responding\./gs,
    '',
  );
  // Add tool-calling discipline reminder at the end
  patched +=
    '\n\nIMPORTANT: When you need to use a tool, you MUST use the function calling format. Never describe tool calls in plain text.';
  return patched;
}

function translateTools(
  tools: AnthropicTool[] | undefined,
): OllamaTool[] | undefined {
  if (!tools || tools.length === 0) return undefined;
  return tools.map((t) => ({
    type: 'function' as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: t.input_schema,
    },
  }));
}

/**
 * Flatten Anthropic content blocks to a plain string.
 */
function flattenTextContent(
  content: string | AnthropicContentBlock[],
): string {
  if (typeof content === 'string') return content;
  return content
    .filter((b) => b.type === 'text')
    .map((b) => b.text || '')
    .join('');
}

/**
 * Extract base64 image data from content blocks.
 */
function extractImages(content: AnthropicContentBlock[]): string[] {
  return content
    .filter((b) => b.type === 'image' && b.source?.data)
    .map((b) => b.source!.data);
}

/**
 * Translate Anthropic messages to Ollama native format.
 *
 * Key differences:
 * - Anthropic tool_use blocks → Ollama tool_calls on assistant messages
 * - Anthropic tool_result blocks (in user messages) → Ollama role:tool messages
 * - Anthropic image blocks → Ollama images array (base64)
 * - Ollama tool call arguments are objects (not JSON strings)
 */
function translateMessages(messages: AnthropicMessage[]): OllamaMessage[] {
  const result: OllamaMessage[] = [];

  for (const msg of messages) {
    if (msg.role === 'assistant') {
      const blocks =
        typeof msg.content === 'string'
          ? [{ type: 'text', text: msg.content }]
          : msg.content;

      const textParts = blocks.filter((b) => b.type === 'text');
      const toolUseParts = blocks.filter((b) => b.type === 'tool_use');

      const ollamaMsg: OllamaMessage = {
        role: 'assistant',
        content: textParts.map((b) => b.text || '').join(''),
      };

      if (toolUseParts.length > 0) {
        ollamaMsg.tool_calls = toolUseParts.map((b) => ({
          id: b.id || `toolu_${Math.random().toString(36).slice(2)}`,
          function: {
            name: b.name!,
            arguments:
              typeof b.input === 'object' && b.input !== null
                ? (b.input as Record<string, unknown>)
                : {},
          },
        }));
      }

      result.push(ollamaMsg);
    } else if (msg.role === 'user') {
      const blocks =
        typeof msg.content === 'string'
          ? [{ type: 'text', text: msg.content }]
          : msg.content;

      const toolResults = blocks.filter((b) => b.type === 'tool_result');
      const otherBlocks = blocks.filter((b) => b.type !== 'tool_result');

      // Tool results → role:tool messages
      for (const tr of toolResults) {
        let content: string;
        if (typeof tr.content === 'string') {
          content = tr.content;
        } else if (Array.isArray(tr.content)) {
          content = tr.content
            .filter((c) => c.type === 'text')
            .map((c) => c.text)
            .join('');
        } else {
          content = '';
        }
        if (tr.is_error) {
          content = `[ERROR] ${content}`;
        }
        result.push({ role: 'tool', content });
      }

      // Remaining user content
      if (otherBlocks.length > 0) {
        const userMsg: OllamaMessage = {
          role: 'user',
          content: flattenTextContent(otherBlocks as AnthropicContentBlock[]),
        };
        const images = extractImages(otherBlocks as AnthropicContentBlock[]);
        if (images.length > 0) {
          userMsg.images = images;
        }
        result.push(userMsg);
      }
    } else {
      result.push({
        role: msg.role,
        content: flattenTextContent(msg.content),
      });
    }
  }

  return result;
}

export function translateRequest(
  anthropicReq: AnthropicRequest,
  config: OllamaTranslatorConfig,
): OllamaRequest {
  const messages: OllamaMessage[] = [];

  // System prompt → first system message.
  // When systemPromptOverride is set, use it instead of the (potentially huge)
  // system prompt from the SDK/CLI. This is critical for CPU-only inference
  // where the 70KB Claude Code system prompt takes 40+ min for prompt eval.
  let systemText: string | undefined;
  if (config.systemPromptOverride !== undefined) {
    systemText = config.systemPromptOverride || undefined;
  } else {
    systemText = translateSystemPrompt(anthropicReq.system);
  }
  if (systemText) {
    const finalSystem =
      config.systemPromptTransforms !== false
        ? patchSystemPromptForOllama(systemText)
        : systemText;
    messages.push({ role: 'system', content: finalSystem });
  }

  messages.push(...translateMessages(anthropicReq.messages));

  const ollamaReq: OllamaRequest = {
    model: config.ollamaModel,
    messages,
    stream: anthropicReq.stream ?? false,
    think: config.think ?? false,
  };

  // Map parameters to Ollama options
  const options: OllamaRequest['options'] = {};
  let hasOptions = false;

  if (anthropicReq.temperature != null) {
    options.temperature = anthropicReq.temperature;
    hasOptions = true;
  }
  if (anthropicReq.top_p != null) {
    options.top_p = anthropicReq.top_p;
    hasOptions = true;
  }
  if (anthropicReq.top_k != null) {
    options.top_k = anthropicReq.top_k;
    hasOptions = true;
  }
  if (anthropicReq.stop_sequences) {
    options.stop = anthropicReq.stop_sequences;
    hasOptions = true;
  }
  if (anthropicReq.max_tokens) {
    options.num_predict = anthropicReq.max_tokens;
    hasOptions = true;
  }
  if (config.ollamaContextLength) {
    options.num_ctx = config.ollamaContextLength;
    hasOptions = true;
  }
  if (hasOptions) {
    ollamaReq.options = options;
  }

  if (anthropicReq.tools) {
    let tools = anthropicReq.tools;
    if (config.allowedToolNames && config.allowedToolNames.length > 0) {
      const exact = new Set<string>();
      const prefixes: string[] = [];
      for (const name of config.allowedToolNames) {
        if (name.endsWith('*')) prefixes.push(name.slice(0, -1));
        else exact.add(name);
      }
      tools = tools.filter((t) =>
        exact.has(t.name) || prefixes.some((p) => t.name.startsWith(p)),
      );
    }
    ollamaReq.tools = translateTools(tools);
  }

  return ollamaReq;
}

// ---------------------------------------------------------------------------
// Response translation: Ollama → Anthropic (non-streaming)
// ---------------------------------------------------------------------------

const DONE_REASON_MAP: Record<string, string> = {
  stop: 'end_turn',
  length: 'max_tokens',
  load: 'end_turn',
};

function generateMsgId(): string {
  return `msg_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

export function translateResponse(ollamaRes: OllamaResponse): string {
  const content: AnthropicContentBlock[] = [];
  const hasToolCalls =
    ollamaRes.message.tool_calls && ollamaRes.message.tool_calls.length > 0;

  // Text content
  if (ollamaRes.message.content) {
    content.push({
      type: 'text',
      text: ollamaRes.message.content,
    });
  }

  // Tool calls — arguments are already parsed objects from Ollama
  if (hasToolCalls) {
    for (const tc of ollamaRes.message.tool_calls!) {
      content.push({
        type: 'tool_use',
        id:
          tc.id ||
          `toolu_${Math.random().toString(36).slice(2)}${Math.random().toString(36).slice(2)}`,
        name: tc.function.name,
        input: tc.function.arguments,
      });
    }
  }

  // Ensure at least one content block
  if (content.length === 0) {
    content.push({ type: 'text', text: '' });
  }

  // Map stop reason
  let stopReason: string;
  if (hasToolCalls) {
    stopReason = 'tool_use';
  } else {
    stopReason =
      DONE_REASON_MAP[ollamaRes.done_reason || 'stop'] || 'end_turn';
  }

  // Estimate token counts from Ollama's duration-based stats
  const inputTokens = ollamaRes.prompt_eval_count || 0;
  const outputTokens = ollamaRes.eval_count || 0;

  const anthropicRes = {
    id: generateMsgId(),
    type: 'message',
    role: 'assistant',
    model: ollamaRes.model,
    content,
    stop_reason: stopReason,
    usage: {
      input_tokens: inputTokens,
      output_tokens: outputTokens,
    },
  };

  return JSON.stringify(anthropicRes);
}

// ---------------------------------------------------------------------------
// Streaming response translation: Ollama NDJSON → Anthropic SSE
// ---------------------------------------------------------------------------

/**
 * Translates Ollama's streaming response (newline-delimited JSON) into
 * Anthropic SSE format.
 *
 * Ollama streaming is simpler than OpenAI:
 * - Each line is a complete JSON object
 * - Text comes as `message.content` tokens
 * - Tool calls come as complete objects (not fragmented)
 * - Final chunk has `done: true` with usage stats
 */
export class StreamTranslator {
  private msgId = generateMsgId();
  private model = '';
  private contentIndex = 0;
  private textBlockOpen = false;
  private inputTokens = 0;
  private outputTokens = 0;
  private firstChunk = true;
  private output: string[] = [];
  private hasToolCalls = false;

  /**
   * Process a single line from the Ollama NDJSON stream.
   * Returns Anthropic SSE events to emit.
   */
  processChunk(line: string): string[] {
    let chunk: OllamaResponse;
    try {
      chunk = JSON.parse(line);
    } catch {
      return [];
    }

    this.output = [];

    if (this.firstChunk) {
      this.firstChunk = false;
      this.model = chunk.model || '';
      this.emitMessageStart();
    }

    // Text content token
    if (chunk.message?.content) {
      if (!this.textBlockOpen) {
        this.emitContentBlockStart('text');
        this.textBlockOpen = true;
      }
      this.emitTextDelta(chunk.message.content);
    }

    // Tool calls — come as complete objects in a single chunk
    if (chunk.message?.tool_calls && chunk.message.tool_calls.length > 0) {
      this.hasToolCalls = true;

      // Close text block if open
      if (this.textBlockOpen) {
        this.emitContentBlockStop();
        this.textBlockOpen = false;
      }

      for (const tc of chunk.message.tool_calls) {
        const toolId =
          tc.id ||
          `toolu_${Math.random().toString(36).slice(2)}${Math.random().toString(36).slice(2)}`;
        this.emitContentBlockStart('tool_use', toolId, tc.function.name);
        this.emitInputJsonDelta(JSON.stringify(tc.function.arguments));
        this.emitContentBlockStop();
      }
    }

    // Final chunk
    if (chunk.done) {
      this.inputTokens = chunk.prompt_eval_count || 0;
      this.outputTokens = chunk.eval_count || 0;
      return this.finalize();
    }

    return this.output;
  }

  private finalize(): string[] {
    // Close text block if still open
    if (this.textBlockOpen) {
      this.emitContentBlockStop();
      this.textBlockOpen = false;
    }

    const stopReason = this.hasToolCalls ? 'tool_use' : 'end_turn';

    this.emit('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReason },
      usage: { output_tokens: this.outputTokens },
    });

    this.emit('message_stop', { type: 'message_stop' });

    return this.output;
  }

  private emitMessageStart(): void {
    this.emit('message_start', {
      type: 'message_start',
      message: {
        id: this.msgId,
        type: 'message',
        role: 'assistant',
        content: [],
        model: this.model,
        stop_reason: null,
        usage: { input_tokens: this.inputTokens, output_tokens: 0 },
      },
    });
  }

  private emitContentBlockStart(
    type: string,
    toolId?: string,
    toolName?: string,
  ): void {
    const block: Record<string, unknown> = { type };
    if (type === 'tool_use') {
      block.id = toolId;
      block.name = toolName;
      block.input = {};
    } else {
      block.text = '';
    }
    this.emit('content_block_start', {
      type: 'content_block_start',
      index: this.contentIndex,
      content_block: block,
    });
  }

  private emitContentBlockStop(): void {
    this.emit('content_block_stop', {
      type: 'content_block_stop',
      index: this.contentIndex,
    });
    this.contentIndex++;
  }

  private emitTextDelta(text: string): void {
    this.emit('content_block_delta', {
      type: 'content_block_delta',
      index: this.contentIndex,
      delta: { type: 'text_delta', text },
    });
  }

  private emitInputJsonDelta(json: string): void {
    this.emit('content_block_delta', {
      type: 'content_block_delta',
      index: this.contentIndex,
      delta: { type: 'input_json_delta', partial_json: json },
    });
  }

  private emit(event: string, data: unknown): void {
    this.output.push(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  }
}

// ---------------------------------------------------------------------------
// NDJSON line parser for streaming
// ---------------------------------------------------------------------------

/**
 * Parse a raw NDJSON stream buffer into individual lines.
 * Returns { lines, remainder } where remainder is an incomplete
 * line that needs more data.
 */
export function parseNDJSONBuffer(buffer: string): {
  lines: string[];
  remainder: string;
} {
  const parts = buffer.split('\n');
  const remainder = parts.pop() || '';
  const lines = parts.filter((l) => l.trim().length > 0);
  return { lines, remainder };
}
