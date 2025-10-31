from __future__ import annotations

import json
from typing import List

from src.config import Config
from src.services.notifications import NotificationSender, TelegramNotifier, HttpPostNotifier


def create_notifiers_from_config(cfg: Config) -> List[NotificationSender]:
    notifiers: List[NotificationSender] = []
    if cfg.telegram_bot_token and cfg.telegram_chat_id:
        notifiers.append(TelegramNotifier(bot_token=cfg.telegram_bot_token, chat_id=str(cfg.telegram_chat_id)))
    if cfg.http_post_url:
        headers = None
        if cfg.http_post_headers_json:
            try:
                headers = json.loads(cfg.http_post_headers_json)
            except Exception:
                headers = None
        notifiers.append(HttpPostNotifier(url=cfg.http_post_url, headers=headers))
    return notifiers

