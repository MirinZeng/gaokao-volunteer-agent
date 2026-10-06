from pathlib import Path

from gaokao_agent.memory import ConversationMemory


def test_four_layer_memory_is_isolated_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    memory = ConversationMemory(path, window_messages=4)
    first = memory.create_session("第一位考生")
    second = memory.create_session("第二位考生")

    memory.add_message(first["id"], "user", "江西物理类，位次30000")
    memory.update_profile(first["id"], {
        "subject": "物理", "rank": 30000, "target_major": "计算机",
        "preferences": ["就业"], "profile": {"province": "江西"},
    })
    for index in range(6):
        memory.add_message(first["id"], "assistant" if index % 2 else "user", f"消息{index}")
    memory.add_message(second["id"], "user", "历史类想学法学")

    # 重新实例化，验证刷新/进程重启后仍能恢复，而不是只存在内存里。
    restored = ConversationMemory(path, window_messages=4)
    context = restored.context(first["id"])
    other = restored.context(second["id"])

    assert context["metadata"]["title"] == "第一位考生"
    assert context["profile"]["province"] == "江西"
    assert context["profile"]["rank"] == 30000
    assert len(context["window"]) == 4
    assert "江西物理类" in context["summary"]
    assert other["profile"] == {}
    assert len(other["window"]) == 1


def test_delete_session_cascades_messages(tmp_path: Path) -> None:
    memory = ConversationMemory(tmp_path / "memory.sqlite3")
    session = memory.create_session()
    memory.add_message(session["id"], "user", "测试")
    memory.delete_session(session["id"])

    assert memory.list_sessions() == []
