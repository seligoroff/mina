from datetime import datetime, timezone
import json

from app.adapters.output.storage import FileMessageStore
from app.domain.models import ChatMessage


def test_jsonl_serializes_datetime_iso(tmp_path):
    store = FileMessageStore()
    dest = tmp_path / "out.jsonl"
    msg = ChatMessage(
        id=1,
        date=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        from_id=None,
        from_name=None,
        text="hi",
    )
    store.write([msg], str(dest), format="jsonl")

    content = dest.read_text(encoding="utf-8").strip()
    obj = json.loads(content)
    assert obj["date"].endswith("+00:00")
    assert obj["text"] == "hi"

