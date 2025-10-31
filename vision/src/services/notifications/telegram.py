from __future__ import annotations

import json
from typing import Optional, List

import requests

from src.services.notifications.base import NotificationEvent, NotificationSender
from src.schemas.detection import Detection


def _format_detections(dets: List[Detection], max_items: int = 10) -> str:
    if not dets:
        return "(no detections)"
    lines = []
    for i, d in enumerate(dets[:max_items]):
        x1, y1, x2, y2 = d.bbox.to_xyxy()
        lines.append(f"{d.label} {d.score:.2f} [{x1},{y1},{x2},{y2}]")
    if len(dets) > max_items:
        lines.append(f"… and {len(dets) - max_items} more")
    return "\n".join(lines)


class TelegramNotifier(NotificationSender):
    """Sends notifications to a Telegram chat via Bot API.

    If `image_bytes` is provided, uses sendPhoto with caption; otherwise sendMessage.
    """

    def __init__(self, bot_token: str, chat_id: str, timeout: float = 10.0) -> None:
        self._token = bot_token
        self._chat = chat_id
        self._timeout = timeout

    def _base(self) -> str:
        return f"https://api.telegram.org/bot{self._token}"

    def _compose_text(self, event: NotificationEvent) -> str:
        parts: List[str] = []
        if event.title:
            parts.append(f"*{event.title}*")
        if event.text:
            parts.append(event.text)
        if event.detections:
            parts.append("Detections:\n" + _format_detections(event.detections))
        return "\n\n".join(parts) or "Notification"

    def send(self, event: NotificationEvent) -> None:
        text = self._compose_text(event)
        if event.image_bytes:
            url = f"{self._base()}/sendPhoto"
            files = {"photo": ("frame.jpg", event.image_bytes, "image/jpeg")}
            data = {"chat_id": self._chat, "caption": text, "parse_mode": "Markdown"}
            r = requests.post(url, data=data, files=files, timeout=self._timeout)
        else:
            url = f"{self._base()}/sendMessage"
            data = {"chat_id": self._chat, "text": text, "parse_mode": "Markdown"}
            r = requests.post(url, json=data, timeout=self._timeout)
        r.raise_for_status()

