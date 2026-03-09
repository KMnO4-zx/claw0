"""
Section 01: Agent 循环
"Agent 就是 while True + stop_reason"

    用户输入 --> [messages[]] --> LLM API --> stop_reason?
                                               /        \
                                         "end_turn"  "tool_use"
                                             |           |
                                          打印回复    (下一节)

用法:
    cd claw0
    python zh/s01_agent_loop_openrouter.py

需要在 .env 中配置:
    OPEN_ROUTER_API_KEY=sk-or-xxxxx
    OPEN_ROUTER_BASE_URL=https://openrouter.ai/api/v1
    OPEN_ROUTER_MODEL_ID=stepfun/step-3.5-flash:free
"""

# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

MODEL_ID = os.getenv("OPEN_ROUTER_MODEL_ID", "stepfun/step-3.5-flash:free")
client = OpenAI(
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    base_url=os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
)

SYSTEM_PROMPT = "You are a helpful AI assistant. Answer questions directly."

# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


# ---------------------------------------------------------------------------
# 核心: Agent 循环
# ---------------------------------------------------------------------------
#   1. 收集用户输入, 追加到 messages
#   2. 调用 API
#   3. 检查 stop_reason 决定下一步
#
#   本节 stop_reason 永远是 "end_turn" (没有工具).
#   下一节加入 "tool_use" -- 循环结构保持不变.
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """主 agent 循环 -- 对话式 REPL."""

    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 01: Agent 循环")
    print_info(f"  Model: {MODEL_ID}")
    print_info("  输入 'quit' 或 'exit' 退出. Ctrl+C 同样有效.")
    print_info("=" * 60)
    print()

    while True:
        # --- 获取用户输入 ---
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}再见.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}再见.{RESET}")
            break

        # --- 追加到历史 ---
        messages.append({
            "role": "user",
            "content": user_input,
        })

        # --- 调用 LLM ---
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                max_tokens=8096,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            )
        except Exception as exc:
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            messages.pop()
            continue

        # --- 检查 stop_reason ---
        finish_reason = response.choices[0].finish_reason
        assistant_content = response.choices[0].message.content or ""

        if finish_reason == "stop":
            print_assistant(assistant_content)
            messages.append({
                "role": "assistant",
                "content": assistant_content,
            })

        elif finish_reason == "tool_calls":
            print_info("[finish_reason=tool_calls] 本节没有可用工具.")
            print_info("参见 s02_tool_use.py 了解工具支持.")
            messages.append({
                "role": "assistant",
                "content": assistant_content,
            })

        else:
            print_info(f"[finish_reason={finish_reason}]")
            if assistant_content:
                print_assistant(assistant_content)
            messages.append({
                "role": "assistant",
                "content": assistant_content,
            })


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("OPEN_ROUTER_API_KEY"):
        print(f"{YELLOW}Error: OPEN_ROUTER_API_KEY 未设置.{RESET}")
        print(f"{DIM}将 .env.example 复制为 .env 并填入你的 key.{RESET}")
        sys.exit(1)

    agent_loop()


if __name__ == "__main__":
    main()
