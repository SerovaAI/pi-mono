import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type AssistantMessage,
	type Context,
	createAssistantMessageEventStream,
	type Model,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { convertToLlm } from "../src/core/messages.ts";
import { type CreateAgentSessionOptions, createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createInMemoryModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";

describe("SDK agent customization", () => {
	let cwd: string;
	const cleanups: Array<() => void> = [];
	beforeEach(() => {
		cwd = mkdtempSync(join(tmpdir(), "pi-sdk-customization-"));
	});
	afterEach(() => {
		for (const cleanup of cleanups.splice(0).reverse()) cleanup();
		rmSync(cwd, { recursive: true, force: true });
	});

	async function setup(options: Partial<CreateAgentSessionOptions> = {}, toolCall = false) {
		const registry = await createInMemoryModelRegistry(AuthStorage.inMemory());
		const model: Model<"openai-completions"> = {
			id: "customization",
			name: "Customization",
			api: "openai-completions",
			provider: "sdk-customization",
			baseUrl: "https://unused.invalid",
			reasoning: false,
			input: ["text", "image"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 128000,
			maxTokens: 4096,
		};
		const requests: Context[] = [];
		registry.registerProvider(model.provider, {
			api: model.api,
			apiKey: "test-key",
			streamSimple: (_model, context) => {
				requests.push(context);
				const callTool = toolCall && requests.length === 1;
				const message: AssistantMessage = {
					role: "assistant",
					api: model.api,
					provider: model.provider,
					model: model.id,
					content: callTool
						? [{ type: "toolCall", id: "handoff-1", name: "handoff", arguments: {} }]
						: [{ type: "text", text: "done" }],
					stopReason: callTool ? "toolUse" : "stop",
					timestamp: Date.now(),
					usage: {
						input: 1,
						output: 1,
						cacheRead: 0,
						cacheWrite: 0,
						totalTokens: 2,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
				};
				const stream = createAssistantMessageEventStream();
				stream.push({ type: "done", reason: callTool ? "toolUse" : "stop", message });
				return stream;
			},
		});
		cleanups.push(() => registry.unregisterProvider(model.provider));
		const { session } = await createAgentSession({
			cwd,
			agentDir: cwd,
			model,
			modelRuntime: getModelRuntime(registry),
			sessionManager: SessionManager.inMemory(cwd),
			settingsManager: SettingsManager.inMemory({ compaction: { enabled: false }, retry: { enabled: false } }),
			tools: [],
			...options,
		});
		cleanups.push(() => session.dispose());
		return { session, requests };
	}

	it("runs extension context hooks before host conversion", async () => {
		mkdirSync(join(cwd, "extensions"));
		writeFileSync(
			join(cwd, "extensions", "context.ts"),
			`export default function (pi) {
			pi.on("context", (event) => ({ messages: [...event.messages, {
				role: "user", content: "extension context", timestamp: 1,
			}] }));
		}`,
		);
		const { session, requests } = await setup({
			convertToLlm: (messages) =>
				convertToLlm(messages).map((message) =>
					message.role === "user" && message.content === "extension context"
						? { ...message, content: "converted extension context" }
						: message,
				),
		});
		await session.prompt("hello");
		expect(requests[0].messages.at(-1)?.content).toBe("converted extension context");
	});

	it("applies image blocking after asynchronous host conversion without changing history", async () => {
		const settingsManager = SettingsManager.inMemory({
			images: { blockImages: true },
			compaction: { enabled: false },
		});
		const { session, requests } = await setup({
			settingsManager,
			convertToLlm: async (messages) => [
				...convertToLlm(messages),
				{
					role: "user",
					timestamp: 1,
					content: [{ type: "image", data: "aGVsbG8=", mimeType: "image/png" }],
				},
			],
		});
		await session.prompt("hello");
		expect(requests[0].messages.at(-1)?.content).toEqual([{ type: "text", text: "Image reading is disabled." }]);
		expect(session.messages.filter((message) => message.role === "user")).toHaveLength(1);
		settingsManager.setBlockImages(false);
		await session.prompt("again");
		expect(requests[1].messages.at(-1)?.content).toEqual([
			{ type: "image", data: "aGVsbG8=", mimeType: "image/png" },
		]);
	});

	it.each([false, true])("stops after a completed handoff only when requested (%s)", async (stop) => {
		let handedOff = false;
		const { session, requests } = await setup(
			{
				tools: ["handoff"],
				customTools: [
					{
						name: "handoff",
						label: "Handoff",
						description: "Hand work back to host",
						parameters: Type.Object({}),
						execute: async () => {
							handedOff = true;
							return { content: [{ type: "text", text: "handed off" }], details: {} };
						},
					},
				],
				shouldStopAfterTurn: stop ? async () => handedOff : undefined,
			},
			true,
		);
		await session.prompt("hand off");
		expect(handedOff).toBe(true);
		expect(requests).toHaveLength(stop ? 1 : 2);
		expect(session.messages.some((message) => message.role === "toolResult" && message.toolName === "handoff")).toBe(
			true,
		);
	});
});
