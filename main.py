"""适合在 PyCharm、VS Code 等 Python 工作台中直接运行的入口。"""

import sys

from gaokao_agent.cli import main


if __name__ == "__main__":
    # 直接点击“运行”时默认进入连续对话；仍可在命令行传入 ingest/ask/search。
    if len(sys.argv) == 1:
        sys.argv.append("chat")
    main()
