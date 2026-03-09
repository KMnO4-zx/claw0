"""
第05节: 网关与路由 -- "每条消息都能找到归宿" (OpenRouter版本)

Gateway 是消息枢纽: 每条入站消息解析为 (agent_id, session_key)。
路由系统是一个五层绑定表, 从最具体到最通用进行匹配。

    入站消息 (channel, account_id, peer_id, text)
           |
    +------v------+     +----------+
    |   Gateway    | <-- | WS/REPL  |  JSON-RPC 2.0
    +------+------+     +----------+
           |
    +------v------+
    |   Routing    |  5层: peer > guild > account > channel > default
    +------+------+
           |
     (agent_id, session_key)
           |
    +------v------+
    | AgentManager |  每个 agent 的配置 / 工作区 / 会话
    +------+------+
           |
        LLM API

运行方法:  cd claw0 && python sessions/zh/s05_gateway_routing_openrouter.py

需要在 .env 中配置:
    OPEN_ROUTER_API_KEY=sk-or-xxxxx
    OPEN_ROUTER_BASE_URL=https://openrouter.ai/api/v1
    OPEN_ROUTER_MODEL_ID=anthropic/claude-opus-4-20250514
"""

# ---------------------------------------------------------------------------
# 导入 & 配置
# ---------------------------------------------------------------------------
import os, re, sys, json, time, asyncio, threading
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

MODEL_ID = os.getenv("OPEN_ROUTER_MODEL_ID", "anthropic/claude-opus-4-20250514")
client = OpenAI(
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    base_url=os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
)
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent / "workspace"
AGENTS_DIR = WORKSPACE_DIR / ".agents"

# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------
CYAN, GREEN, YELLOW, DIM, RESET = "\033[36m", "\033[32m", "\033[33m", "\033[2m", "\033[0m"
BOLD, MAGENTA, RED, BLUE = "\033[1m", "\033[35m", "\033[31m", "\033[34m"
MAX_TOOL_OUTPUT = 30000

# ---------------------------------------------------------------------------
# Agent ID 标准化
# ---------------------------------------------------------------------------

VALID_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
INVALID_CHARS_RE = re.compile(r"[^a-z0-9_-]+")
DEFAULT_AGENT_ID = "main"

def normalize_agent_id(value: str) -> str:
    """标准化 agent ID: 转小写, 移除非法字符"""
    trimmed = value.strip()
    if not trimmed:
        return DEFAULT_AGENT_ID
    if VALID_ID_RE.match(trimmed):
        return trimmed.lower()
    cleaned = INVALID_CHARS_RE.sub("-", trimmed.lower()).strip("-")[:64]
    return cleaned or DEFAULT_AGENT_ID

# ---------------------------------------------------------------------------
# 绑定: 五层路由解析
# ---------------------------------------------------------------------------
# 第1层: peer_id    -- 将特定用户路由到某个 agent
# 第2层: guild_id   -- guild/服务器级别
# 第3层: account_id -- bot 账号级别
# 第4层: channel    -- 整个通道 (如所有 Telegram)
# 第5层: default    -- 兜底

@dataclass
class Binding:
    """路由绑定规则: 定义什么情况下使用哪个 agent"""
    agent_id: str       # 目标 agent ID
    tier: int           # 优先级层级 (1-5, 越小越具体)
    match_key: str      # 匹配类型: peer_id/guild_id/account_id/channel/default
    match_value: str    # 匹配值: 例如 "telegram:12345"
    priority: int = 0   # 同层级内的优先级 (越大越优先)

    def display(self) -> str:
        """显示绑定规则的可读格式"""
        names = {1: "peer", 2: "guild", 3: "account", 4: "channel", 5: "default"}
        label = names.get(self.tier, f"tier-{self.tier}")
        return f"[{label}] {self.match_key}={self.match_value} -> agent:{self.agent_id} (pri={self.priority})"

class BindingTable:
    """路由表: 管理所有绑定规则, 提供 5 层匹配解析"""
    def __init__(self) -> None:
        self._bindings: list[Binding] = []

    def add(self, binding: Binding) -> None:
        """添加绑定规则并按层级和优先级排序"""
        self._bindings.append(binding)
        self._bindings.sort(key=lambda b: (b.tier, -b.priority))

    def remove(self, agent_id: str, match_key: str, match_value: str) -> bool:
        """移除匹配的绑定规则"""
        before = len(self._bindings)
        self._bindings = [
            b for b in self._bindings
            if not (b.agent_id == agent_id and b.match_key == match_key
                    and b.match_value == match_value)
        ]
        return len(self._bindings) < before

    def list_all(self) -> list[Binding]:
        """列出所有绑定规则"""
        return list(self._bindings)

    def resolve(self, channel: str = "", account_id: str = "",
                guild_id: str = "", peer_id: str = "") -> tuple[str | None, Binding | None]:
        """解析路由: 从第 1 层到第 5 层遍历, 第一个匹配的获胜

        返回: (agent_id, matched_binding)
        """
        for b in self._bindings:
            if b.tier == 1 and b.match_key == "peer_id":
                if ":" in b.match_value:
                    if b.match_value == f"{channel}:{peer_id}":
                        return b.agent_id, b
                elif b.match_value == peer_id:
                    return b.agent_id, b
            elif b.tier == 2 and b.match_key == "guild_id" and b.match_value == guild_id:
                return b.agent_id, b
            elif b.tier == 3 and b.match_key == "account_id" and b.match_value == account_id:
                return b.agent_id, b
            elif b.tier == 4 and b.match_key == "channel" and b.match_value == channel:
                return b.agent_id, b
            elif b.tier == 5 and b.match_key == "default":
                return b.agent_id, b
        return None, None

# ---------------------------------------------------------------------------
# 会话键构建
# ---------------------------------------------------------------------------
# dm_scope 控制私聊隔离粒度:
#   main                      -> agent:{id}:main
#   per-peer                  -> agent:{id}:direct:{peer}
#   per-channel-peer          -> agent:{id}:{ch}:direct:{peer}
#   per-account-channel-peer  -> agent:{id}:{ch}:{acc}:direct:{peer}

def build_session_key(agent_id: str, channel: str = "", account_id: str = "",
                      peer_id: str = "", dm_scope: str = "per-peer") -> str:
    """构建会话键: 根据 dm_scope 控制会话隔离粒度

    dm_scope 选项:
    - main: 所有人共享一个会话
    - per-peer: 每个用户独立会话
    - per-channel-peer: 每个平台的用户独立
    - per-account-channel-peer: 最大隔离度
    """
    aid = normalize_agent_id(agent_id)
    ch = (channel or "unknown").strip().lower()
    acc = (account_id or "default").strip().lower()
    pid = (peer_id or "").strip().lower()
    if dm_scope == "per-account-channel-peer" and pid:
        return f"agent:{aid}:{ch}:{acc}:direct:{pid}"
    if dm_scope == "per-channel-peer" and pid:
        return f"agent:{aid}:{ch}:direct:{pid}"
    if dm_scope == "per-peer" and pid:
        return f"agent:{aid}:direct:{pid}"
    return f"agent:{aid}:main"

# ---------------------------------------------------------------------------
# Agent 配置 & 管理器
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Agent 配置: 定义 agent 的性格、模型、会话隔离策略"""
    id: str              # Agent ID
    name: str            # 显示名称
    personality: str = ""  # 性格描述
    model: str = ""      # 模型 ID (空则使用全局 MODEL_ID)
    dm_scope: str = "per-peer"  # 会话隔离粒度

    @property
    def effective_model(self) -> str:
        """获取实际使用的模型 ID"""
        return self.model or MODEL_ID

    def system_prompt(self) -> str:
        """生成系统提示词"""
        parts = [f"You are {self.name}."]
        if self.personality:
            parts.append(f"Your personality: {self.personality}")
        parts.append("Answer questions helpfully and stay in character.")
        return " ".join(parts)

class AgentManager:
    """Agent 管理器: 注册 agent, 管理会话历史和工作区"""
    def __init__(self, agents_base: Path | None = None) -> None:
        self._agents: dict[str, AgentConfig] = {}
        self._agents_base = agents_base or AGENTS_DIR
        self._sessions: dict[str, list[dict]] = {}

    def register(self, config: AgentConfig) -> None:
        """注册一个新的 agent"""
        aid = normalize_agent_id(config.id)
        config.id = aid
        self._agents[aid] = config
        agent_dir = self._agents_base / aid
        (agent_dir / "sessions").mkdir(parents=True, exist_ok=True)
        (WORKSPACE_DIR / f"workspace-{aid}").mkdir(parents=True, exist_ok=True)

    def get_agent(self, agent_id: str) -> AgentConfig | None:
        """获取指定的 agent 配置"""
        return self._agents.get(normalize_agent_id(agent_id))

    def list_agents(self) -> list[AgentConfig]:
        """列出所有已注册的 agent"""
        return list(self._agents.values())

    def get_session(self, session_key: str) -> list[dict]:
        """获取或创建会话历史"""
        if session_key not in self._sessions:
            self._sessions[session_key] = []
        return self._sessions[session_key]

    def list_sessions(self, agent_id: str = "") -> dict[str, int]:
        """列出所有会话及其消息数量"""
        aid = normalize_agent_id(agent_id) if agent_id else ""
        return {k: len(v) for k, v in self._sessions.items()
                if not aid or k.startswith(f"agent:{aid}:")}

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "required": ["file_path"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time in UTC.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
]

def _tool_read(file_path: str) -> str:
    """读取文件内容的工具实现"""
    try:
        p = Path(file_path).resolve()
        if not p.exists():
            return f"Error: File not found: {file_path}"
        content = p.read_text(encoding="utf-8")
        if len(content) > MAX_TOOL_OUTPUT:
            return content[:MAX_TOOL_OUTPUT] + f"\n... [truncated, {len(content)} total chars]"
        return content
    except Exception as exc:
        return f"Error: {exc}"

TOOL_HANDLERS: dict[str, Any] = {
    "read_file": lambda file_path: _tool_read(file_path),
    "get_current_time": lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
}

def process_tool_call(name: str, inp: dict) -> str:
    """处理工具调用并返回结果"""
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return f"Error: Unknown tool '{name}'"
    try:
        return handler(**inp)
    except Exception as exc:
        return f"Error: {name} failed: {exc}"

# ---------------------------------------------------------------------------
# 共享事件循环 (持久化后台线程)
# ---------------------------------------------------------------------------

_event_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None

def get_event_loop() -> asyncio.AbstractEventLoop:
    """获取或创建持久化的事件循环"""
    global _event_loop, _loop_thread
    if _event_loop is not None and _event_loop.is_running():
        return _event_loop
    _event_loop = asyncio.new_event_loop()
    def _run():
        asyncio.set_event_loop(_event_loop)
        _event_loop.run_forever()
    _loop_thread = threading.Thread(target=_run, daemon=True)
    _loop_thread.start()
    return _event_loop

def run_async(coro):
    """在持久化事件循环中运行协程"""
    loop = get_event_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()

# ---------------------------------------------------------------------------
# 路由解析
# ---------------------------------------------------------------------------

def resolve_route(bindings: BindingTable, mgr: AgentManager,
                  channel: str, peer_id: str,
                  account_id: str = "", guild_id: str = "") -> tuple[str, str]:
    """路由解析: 根据消息来源决定使用哪个 agent 和会话

    返回: (agent_id, session_key)
    """
    agent_id, matched = bindings.resolve(
        channel=channel, account_id=account_id,
        guild_id=guild_id, peer_id=peer_id,
    )
    if not agent_id:
        agent_id = DEFAULT_AGENT_ID
        print(f"  {DIM}[route] No binding matched, default: {agent_id}{RESET}")
    elif matched:
        print(f"  {DIM}[route] Matched: {matched.display()}{RESET}")
    agent = mgr.get_agent(agent_id)
    dm_scope = agent.dm_scope if agent else "per-peer"
    sk = build_session_key(agent_id, channel=channel, account_id=account_id,
                           peer_id=peer_id, dm_scope=dm_scope)
    return agent_id, sk

# ---------------------------------------------------------------------------
# Agent 运行器
# ---------------------------------------------------------------------------

_agent_semaphore: asyncio.Semaphore | None = None

async def run_agent(mgr: AgentManager, agent_id: str, session_key: str,
                    user_text: str, on_typing: Any = None) -> str:
    """运行 agent: 使用信号量控制并发 (最多 4 个), 调用 LLM API

    参数:
    - on_typing: 可选的回调函数, 用于通知"正在输入"状态
    """
    global _agent_semaphore
    if _agent_semaphore is None:
        _agent_semaphore = asyncio.Semaphore(4)
    agent = mgr.get_agent(agent_id)
    if not agent:
        return f"Error: agent '{agent_id}' not found"
    messages = mgr.get_session(session_key)
    messages.append({"role": "user", "content": user_text})
    async with _agent_semaphore:
        if on_typing:
            on_typing(agent_id, True)
        try:
            return await _agent_loop(agent.effective_model, agent.system_prompt(), messages)
        finally:
            if on_typing:
                on_typing(agent_id, False)

async def _agent_loop(model: str, system: str, messages: list[dict]) -> str:
    """Agent 循环: 调用 LLM API, 处理工具调用, 最多 15 轮"""
    for _ in range(15):
        try:
            # 调用 OpenAI API (通过 OpenRouter)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                max_tokens=4096,
                messages=[{"role": "system", "content": system}] + messages,
                tools=TOOLS,
            )
        except Exception as exc:
            # API 错误: 回滚消息历史
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages:
                messages.pop()
            return f"API Error: {exc}"

        # 提取响应
        finish_reason = response.choices[0].finish_reason
        assistant_message = response.choices[0].message
        assistant_content = assistant_message.content or ""

        # 添加助手消息到历史
        messages.append({"role": "assistant", "content": assistant_content})

        if finish_reason == "stop":
            # 正常结束
            return assistant_content or "[no text]"

        elif finish_reason == "tool_calls":
            # 处理工具调用
            tool_calls = assistant_message.tool_calls

            # 更新最后一条消息，包含 tool_calls
            messages[-1]["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in tool_calls
            ]

            # 执行每个工具并添加结果
            for tc in tool_calls:
                print(f"  {DIM}[tool: {tc.function.name}]{RESET}")
                tool_input = json.loads(tc.function.arguments)
                result = process_tool_call(tc.function.name, tool_input)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            continue

        else:
            # 其他结束原因
            return assistant_content or f"[stop={finish_reason}]"

    return "[max iterations reached]"

# ---------------------------------------------------------------------------
# Gateway 服务器 (WebSocket, JSON-RPC 2.0)
# ---------------------------------------------------------------------------

class GatewayServer:
    """WebSocket 网关服务器: 使用 JSON-RPC 2.0 协议处理远程消息"""
    def __init__(self, mgr: AgentManager, bindings: BindingTable,
                 host: str = "localhost", port: int = 8765) -> None:
        self._mgr = mgr
        self._bindings = bindings
        self._host, self._port = host, port
        self._clients: set[Any] = set()
        self._start_time = time.monotonic()
        self._server: Any = None
        self._running = False

    async def start(self) -> None:
        try:
            import websockets
        except ImportError:
            print(f"{RED}websockets not installed. pip install websockets{RESET}"); return
        self._start_time = time.monotonic()
        self._running = True
        self._server = await websockets.serve(self._handle, self._host, self._port)
        print(f"{GREEN}Gateway started ws://{self._host}:{self._port}{RESET}")

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._running = False

    async def _handle(self, ws: Any, path: str = "") -> None:
        self._clients.add(ws)
        try:
            async for raw in ws:
                resp = await self._dispatch(raw)
                if resp:
                    await ws.send(json.dumps(resp))
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    def _typing_cb(self, agent_id: str, typing: bool) -> None:
        msg = json.dumps({"jsonrpc": "2.0", "method": "typing",
                          "params": {"agent_id": agent_id, "typing": typing}})
        for ws in list(self._clients):
            try:
                asyncio.ensure_future(ws.send(msg))
            except Exception:
                self._clients.discard(ws)

    async def _dispatch(self, raw: str) -> dict | None:
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            return {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None}
        rid, method, params = req.get("id"), req.get("method", ""), req.get("params", {})
        methods = {
            "send": self._m_send, "bindings.set": self._m_bind_set,
            "bindings.list": self._m_bind_list, "sessions.list": self._m_sessions,
            "agents.list": self._m_agents, "status": self._m_status,
        }
        handler = methods.get(method)
        if not handler:
            return {"jsonrpc": "2.0", "error": {"code": -32601, "message": f"Unknown: {method}"}, "id": rid}
        try:
            return {"jsonrpc": "2.0", "result": await handler(params), "id": rid}
        except Exception as exc:
            return {"jsonrpc": "2.0", "error": {"code": -32000, "message": str(exc)}, "id": rid}

    async def _m_send(self, p: dict) -> dict:
        text = p.get("text", "")
        if not text:
            raise ValueError("text is required")
        ch, pid = p.get("channel", "websocket"), p.get("peer_id", "ws-client")
        if p.get("agent_id"):
            aid = normalize_agent_id(p["agent_id"])
            a = self._mgr.get_agent(aid)
            sk = build_session_key(aid, channel=ch, peer_id=pid,
                                   dm_scope=a.dm_scope if a else "per-peer")
        else:
            aid, sk = resolve_route(self._bindings, self._mgr, ch, pid)
        reply = await run_agent(self._mgr, aid, sk, text, on_typing=self._typing_cb)
        return {"agent_id": aid, "session_key": sk, "reply": reply}

    async def _m_bind_set(self, p: dict) -> dict:
        b = Binding(agent_id=normalize_agent_id(p.get("agent_id", "")),
                    tier=int(p.get("tier", 5)), match_key=p.get("match_key", "default"),
                    match_value=p.get("match_value", "*"), priority=int(p.get("priority", 0)))
        self._bindings.add(b)
        return {"ok": True, "binding": b.display()}

    async def _m_bind_list(self, p: dict) -> list[dict]:
        return [{"agent_id": b.agent_id, "tier": b.tier, "match_key": b.match_key,
                 "match_value": b.match_value, "priority": b.priority}
                for b in self._bindings.list_all()]

    async def _m_sessions(self, p: dict) -> dict:
        return self._mgr.list_sessions(p.get("agent_id", ""))

    async def _m_agents(self, p: dict) -> list[dict]:
        return [{"id": a.id, "name": a.name, "model": a.effective_model,
                 "dm_scope": a.dm_scope, "personality": a.personality}
                for a in self._mgr.list_agents()]

    async def _m_status(self, p: dict) -> dict:
        return {"running": self._running,
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
                "connected_clients": len(self._clients),
                "agent_count": len(self._mgr.list_agents()),
                "binding_count": len(self._bindings.list_all())}

# ---------------------------------------------------------------------------
# 演示: 双 agent (luna + sage) + 路由绑定
# ---------------------------------------------------------------------------

def setup_demo() -> tuple[AgentManager, BindingTable]:
    mgr = AgentManager()
    mgr.register(AgentConfig(
        id="luna", name="Luna",
        personality="warm, curious, and encouraging. You love asking follow-up questions.",
    ))
    mgr.register(AgentConfig(
        id="sage", name="Sage",
        personality="direct, analytical, and concise. You prefer facts over opinions.",
    ))
    bt = BindingTable()
    bt.add(Binding(agent_id="luna", tier=5, match_key="default", match_value="*"))
    bt.add(Binding(agent_id="sage", tier=4, match_key="channel", match_value="telegram"))
    bt.add(Binding(agent_id="sage", tier=1, match_key="peer_id",
                   match_value="discord:admin-001", priority=10))
    return mgr, bt

# ---------------------------------------------------------------------------
# REPL + 命令
# ---------------------------------------------------------------------------

def cmd_bindings(bt: BindingTable) -> None:
    all_b = bt.list_all()
    if not all_b:
        print(f"  {DIM}(no bindings){RESET}"); return
    print(f"\n{BOLD}Route Bindings ({len(all_b)}):{RESET}")
    for b in all_b:
        c = [MAGENTA, BLUE, CYAN, GREEN, DIM][min(b.tier - 1, 4)]
        print(f"  {c}{b.display()}{RESET}")
    print()

def cmd_route(bt: BindingTable, mgr: AgentManager, args: str) -> None:
    parts = args.strip().split()
    if len(parts) < 2:
        print(f"  {YELLOW}Usage: /route <channel> <peer_id> [account_id] [guild_id]{RESET}"); return
    ch, pid = parts[0], parts[1]
    acc = parts[2] if len(parts) > 2 else ""
    gid = parts[3] if len(parts) > 3 else ""
    aid, sk = resolve_route(bt, mgr, channel=ch, peer_id=pid, account_id=acc, guild_id=gid)
    a = mgr.get_agent(aid)
    print(f"\n{BOLD}Route Resolution:{RESET}")
    print(f"  {DIM}Input:   ch={ch} peer={pid} acc={acc or '-'} guild={gid or '-'}{RESET}")
    print(f"  {CYAN}Agent:   {aid} ({a.name if a else '?'}){RESET}")
    print(f"  {GREEN}Session: {sk}{RESET}\n")

def cmd_agents(mgr: AgentManager) -> None:
    agents = mgr.list_agents()
    if not agents:
        print(f"  {DIM}(no agents){RESET}"); return
    print(f"\n{BOLD}Agents ({len(agents)}):{RESET}")
    for a in agents:
        print(f"  {CYAN}{a.id}{RESET} ({a.name})  model={a.effective_model}  dm_scope={a.dm_scope}")
        if a.personality:
            print(f"    {DIM}{a.personality[:70]}{'...' if len(a.personality) > 70 else ''}{RESET}")
    print()

def cmd_sessions(mgr: AgentManager) -> None:
    s = mgr.list_sessions()
    if not s:
        print(f"  {DIM}(no sessions){RESET}"); return
    print(f"\n{BOLD}Sessions ({len(s)}):{RESET}")
    for k, n in sorted(s.items()):
        print(f"  {GREEN}{k}{RESET} ({n} msgs)")
    print()

def repl() -> None:
    mgr, bindings = setup_demo()
    print(f"{DIM}{'=' * 64}{RESET}")
    print(f"{DIM}  claw0  |  Section 05: Gateway & Routing{RESET}")
    print(f"{DIM}  Model: {MODEL_ID}{RESET}")
    print(f"{DIM}{'=' * 64}{RESET}")
    print(f"{DIM}  /bindings  /route <ch> <peer>  /agents  /sessions  /switch <id>  /gateway{RESET}")
    print()

    ch, pid = "cli", "repl-user"
    force_agent = ""
    gw_started = False

    while True:
        try:
            user_input = input(f"{CYAN}{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye.{RESET}"); break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}Goodbye.{RESET}"); break

        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            args = user_input[len(cmd):].strip()
            if cmd == "/bindings":
                cmd_bindings(bindings)
            elif cmd == "/route":
                cmd_route(bindings, mgr, args)
            elif cmd == "/agents":
                cmd_agents(mgr)
            elif cmd == "/sessions":
                cmd_sessions(mgr)
            elif cmd == "/switch":
                if not args:
                    print(f"  {DIM}force={force_agent or '(off)'}{RESET}")
                elif args.lower() == "off":
                    force_agent = ""
                    print(f"  {DIM}Routing mode restored.{RESET}")
                else:
                    aid = normalize_agent_id(args)
                    if mgr.get_agent(aid):
                        force_agent = aid
                        print(f"  {GREEN}Forcing: {aid}{RESET}")
                    else:
                        print(f"  {YELLOW}Not found: {aid}{RESET}")
            elif cmd == "/gateway":
                if gw_started:
                    print(f"  {DIM}Already running.{RESET}")
                else:
                    gw = GatewayServer(mgr, bindings)
                    asyncio.run_coroutine_threadsafe(gw.start(), get_event_loop())
                    print(f"{GREEN}Gateway running in background on ws://localhost:8765{RESET}\n")
                    gw_started = True
            else:
                print(f"  {YELLOW}Unknown: {cmd}{RESET}")
            continue

        if force_agent:
            agent_id = force_agent
            a = mgr.get_agent(agent_id)
            session_key = build_session_key(agent_id, channel=ch, peer_id=pid,
                                            dm_scope=a.dm_scope if a else "per-peer")
        else:
            agent_id, session_key = resolve_route(bindings, mgr, channel=ch, peer_id=pid)

        agent = mgr.get_agent(agent_id)
        name = agent.name if agent else agent_id
        print(f"  {DIM}-> {name} ({agent_id}) | {session_key}{RESET}")

        try:
            reply = run_async(run_agent(mgr, agent_id, session_key, user_input))
        except Exception as exc:
            print(f"\n{RED}Error: {exc}{RESET}\n"); continue
        print(f"\n{GREEN}{BOLD}{name}:{RESET} {reply}\n")

# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("OPEN_ROUTER_API_KEY"):
        print(f"{YELLOW}Error: OPEN_ROUTER_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)
    repl()

if __name__ == "__main__":
    main()
