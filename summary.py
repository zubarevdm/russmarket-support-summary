"""Ежедневная сводка для support-отдела: CRM + рабочие чаты -> список дел.

Как устроено:
  1. Правила по полям CRM решают, нужно ли вмешаться и насколько срочно.
     Это детерминированная часть, ей LLM не управляет.
  2. Сообщения из чатов разбираются на «заявления» (кто что сказал, новое время,
     нарушение процесса). По умолчанию это делают простые правила (offline).
     С флагом --llm разбор делает модель: local (Qwen в Ollama), anthropic
     (Claude) или openai-compat (любая облачная модель с OpenAI-совместимым
     API). Промпт один, prompt.md. Перед отправкой в модель телефоны,
     паспорта, СНИЛС и почта маскируются; для полной обезличенности
     (ФИО, адреса) запросы можно пустить через pii-gateway.
  3. Заявления сверяются с CRM. Переписка может ПОВЫСИТЬ срочность или добавить
     проверку, но никогда не понижает её и не превращает что-то в «подтверждено».
  4. На выходе Markdown для людей и JSON для интеграций.

Запуск:
  python summary.py data/cases_2026-11-02.json
  python summary.py data/cases_2026-11-02.json --llm local --json out/summary.json
  (для local: ollama pull qwen2.5:14b; адрес и модель в LLM_BASE_URL и LLM_MODEL)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

# Уровни срочности: чем меньше число, тем выше в сводке.
CRITICAL, TODAY, WATCH, MANUAL, OK = 0, 1, 2, 3, 4
LEVEL_LABEL = {
    CRITICAL: "Срочно",
    TODAY: "Сегодня",
    WATCH: "Наблюдать",
    MANUAL: "Проверить вручную, не хватает данных",
    OK: "Вмешательство не нужно",
}

ROLE_LABEL = {
    "employee": "сотрудник",
    "manager": "руководитель",
    "courier": "курьер",
    "recipient": "получатель на точке",
    "support": "support",
}

# Сколько часов CRM может не обновляться, пока по кейсу есть активность.
STALE_CRM_HOURS = 24


@dataclass
class Claim:
    """Одно заявление из переписки. Это всегда «со слов», не факт."""
    source: str
    at: str
    text: str
    kind: str                     # done | in_progress | delay | unknown
    about: str = ""               # кратко, о чём заявление
    eta: datetime | None = None   # новое время, если названо
    third_party: bool = False     # говорит за другого человека
    process_flags: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    id: str
    title: str
    level: int
    crm_facts: list[str] = field(default_factory=list)
    chat_claims: list[str] = field(default_factory=list)
    discrepancies: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    drafts: list[str] = field(default_factory=list)
    sort_key: datetime | None = None

    def needs_action(self) -> bool:
        return self.level != OK


# ---------- даты ----------

def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _d(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _human_day(d: date, today: date) -> str:
    delta = (d - today).days
    names = {-1: "вчера", 0: "сегодня", 1: "завтра"}
    return f"{names.get(delta, d.strftime('%d.%m'))} ({d.strftime('%d.%m')})"


def _hm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def _duration(td: timedelta) -> str:
    minutes = int(td.total_seconds() // 60)
    h, m = divmod(abs(minutes), 60)
    return f"{h} ч {m:02d} мин" if h else f"{m} мин"


# ---------- разбор переписки: offline-правила ----------

DONE_WORDS = r"(отправил|отправила|передал|передала|получили|получил|приняли|принял|подписал|согласовал|обсудили|в курсе|сдал|привез|привёз|доставил)"
DELAY_WORDS = r"(задерж|опазд|не успе|раньше \d|позже|перенос|не приеду|застрял|пробк)"
THIRD_PARTY = r"(руководител|начальник|менеджер|он в курсе|она в курсе|согласовал[аи]? с)"
PRIVATE_CHANNEL = r"(в личк|в лс\b|лично|в whatsapp|в ватсап|в телеграм|на почту|фото|фотографи|скрин)"


def parse_eta(text: str, now: datetime) -> datetime | None:
    m = re.search(r"(\d{1,2})[:.](\d{2})", text)
    if not m:
        m2 = re.search(r"\b(?:к|в|до|после|раньше)\s+(\d{1,2})\b", text)
        if not m2:
            return None
        h, mi = int(m2.group(1)), 0
    else:
        h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h < 24 and 0 <= mi < 60):
        return None
    eta = now.replace(hour=h, minute=mi, second=0, microsecond=0)
    if "завтра" in text.lower():
        eta += timedelta(days=1)
    return eta


def analyze_message_offline(msg: dict, now: datetime) -> Claim:
    text = msg.get("text", "")
    low = text.lower()
    claim = Claim(source=msg.get("from_role", "unknown"), at=msg.get("at", ""),
                  text=text, kind="unknown")
    if re.search(DELAY_WORDS, low):
        claim.kind = "delay"
        claim.eta = parse_eta(text, now)
    elif re.search(DONE_WORDS, low):
        claim.kind = "done"
    if claim.source == "employee" and re.search(THIRD_PARTY, low):
        claim.third_party = True
    if re.search(PRIVATE_CHANNEL, low):
        if re.search(r"(в личк|в лс\b|лично|whatsapp|ватсап|телеграм)", low):
            claim.process_flags.append("отправлены в личный чат")
        if re.search(r"(фото|фотографи|скрин)", low):
            claim.process_flags.append("это фото, а не документы в системе")
    return claim


# ---------- разбор переписки: LLM ----------

CLAIMS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims"],
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["message_index", "kind", "about", "eta", "third_party", "process_flags"],
        "properties": {
            "message_index": {"type": "integer"},
            "kind": {"type": "string", "enum": ["done", "in_progress", "delay", "unknown"]},
            "about": {"type": "string"},
            "eta": {"type": ["string", "null"]},
            "third_party": {"type": "boolean"},
            "process_flags": {"type": "array", "items": {"type": "string"}},
        },
    }}},
}

# Маскировка персональных данных перед отправкой в модель. Для разбора
# «кто что заявил» модели не нужны телефоны, номера паспортов и почта.
PD_PATTERNS = [
    (re.compile(r"\b\d{4}\s?\d{6}\b"), "[паспорт]"),
    (re.compile(r"(?:\+7|\b8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}\b"), "[телефон]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[email]"),
    (re.compile(r"\b\d{3}-\d{3}-\d{3}\s?\d{2}\b"), "[СНИЛС]"),
]


def mask_pd(text: str) -> str:
    for rx, repl in PD_PATTERNS:
        text = rx.sub(repl, text)
    return text


def _llm_payload(case: dict, now: datetime) -> str:
    # В модель уходит минимум: тип операции, поля CRM и тексты без ПДн.
    msgs = [{"from_role": m.get("from_role"), "at": m.get("at"), "text": mask_pd(m.get("text", ""))}
            for m in case.get("messages", [])]
    return json.dumps({"now": now.isoformat(), "case": {
        "type": case.get("type"), "crm": case.get("crm"), "messages": msgs}}, ensure_ascii=False)


def _call_openai_compat(system: str, user: str) -> str:
    """Любая модель с OpenAI-совместимым API.

    По умолчанию локальная Qwen в Ollama: данные не покидают контур. Тот же код
    работает с LM Studio, vLLM, GigaChat, YandexGPT, DeepSeek, OpenAI, OpenRouter
    и с pii-gateway, который обезличивает запрос перед облачной моделью.
    Настройка: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY.
    """
    import os
    import urllib.request

    base = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("LLM_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["LLM_API_KEY"]
    body = {
        "model": os.environ.get("LLM_MODEL", "qwen2.5:14b"),
        "temperature": 0,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "claims", "schema": CLAIMS_SCHEMA}},
    }
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode("utf-8"),
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def _call_anthropic(system: str, user: str) -> str:
    """Claude через API Anthropic. Только если передача данных за рубеж согласована."""
    import anthropic  # импорт здесь, чтобы остальные режимы работали без пакета

    resp = anthropic.Anthropic().beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=4000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": CLAIMS_SCHEMA}},
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("модель отказалась отвечать")
    return "".join(b.text for b in resp.content if b.type == "text")


# local и openai-compat: один код, разница в настройках. local смотрит в Ollama
# на этой машине, openai-compat требует LLM_BASE_URL (облако или pii-gateway).
PROVIDERS = {"local": _call_openai_compat, "openai-compat": _call_openai_compat,
             "anthropic": _call_anthropic}
KINDS = {"done", "in_progress", "delay", "unknown"}


def analyze_messages_llm(case: dict, now: datetime, provider: str = "local") -> list[Claim]:
    """Разбор сообщений моделью. Возвращает те же Claim, что и offline-режим."""
    prompt = (Path(__file__).parent / "prompt.md").read_text(encoding="utf-8")
    raw = PROVIDERS[provider](prompt, _llm_payload(case, now))
    # Локальные модели иногда оборачивают JSON в текст: берём от первой { до последней }.
    data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])

    msgs = case.get("messages", [])
    claims = []
    for c in data.get("claims", []):
        i = c.get("message_index")
        if not isinstance(i, int) or not 0 <= i < len(msgs):
            continue  # модель сослалась на несуществующее сообщение: игнорируем
        m = msgs[i]
        try:
            eta = _dt(c.get("eta"))
        except ValueError:
            eta = None
        claims.append(Claim(
            source=m.get("from_role", "unknown"), at=m.get("at", ""), text=m.get("text", ""),
            kind=c.get("kind") if c.get("kind") in KINDS else "unknown",
            about=str(c.get("about", "")), eta=eta,
            third_party=bool(c.get("third_party")),
            process_flags=[str(f) for f in c.get("process_flags", [])],
        ))
    return claims


# ---------- правила по CRM и сверка ----------

def _claim_line(c: Claim) -> str:
    who = ROLE_LABEL.get(c.source, c.source)
    when = f", {_hm(_dt(c.at))}" if c.at else ""
    return f"{who}{when}: «{c.text}»"


def rule_onboarding(case, crm, claims, now, r: CaseResult):
    first = _d(crm.get("first_shift_date"))
    docs = crm.get("docs_status")
    if first is None or docs is None:
        r.level = MANUAL
        r.checks.append("В CRM нет даты первого выхода или статуса документов: заполнить карточку.")
        return
    today = now.date()
    r.crm_facts.append(f"первый выход {_human_day(first, today)}")
    r.crm_facts.append("документы получены" if docs == "received" else "документы не получены (статус «ожидаются»)")
    r.sort_key = datetime.combine(first, datetime.min.time())
    if docs == "received":
        r.level = OK
        return
    days_left = (first - today).days
    if days_left <= 0:
        r.level = CRITICAL
        r.actions.append("Решить с руководителем точки, допускать ли сотрудника к смене без оформленных документов (по правилам, скорее всего, нет).")
    elif days_left == 1:
        r.level = TODAY
    elif days_left <= 3:
        r.level = WATCH
    else:
        r.level = OK
        return

    said_sent = [c for c in claims if c.kind == "done"]
    if said_sent:
        r.discrepancies.append("В чате говорят, что документы отправлены, а в CRM их нет.")
        for c in said_sent:
            if c.process_flags:
                r.discrepancies.append("Документы ушли не по процессу: " + "; ".join(c.process_flags) + ".")
        r.checks.append("Получил ли руководитель документы на самом деле и все ли они читаются.")
        r.actions.append("Попросить руководителя переслать документы в support по рабочему каналу (или загрузить в CRM) и удалить копии из личной переписки: это персональные данные.")
        r.actions.append("После загрузки сменить статус в CRM на «получены».")
        r.drafts.append(f"Руководителю: «Добрый день! Сотрудник {r.id} выходит {_human_day(first, today)}, в CRM его документов пока нет. Он пишет, что отправил их вам. Перешлите, пожалуйста, сегодня до 15:00 в support (или загрузите в CRM), а из личной переписки потом удалите: там персональные данные.»")
    else:
        r.actions.append("Напомнить сотруднику, какие документы нужны и куда их отправить, срок сегодня.")
    r.actions.append("Если к концу дня документов в CRM нет: предупредить руководителя точки, что выход завтра под вопросом.")


def rule_offboarding(case, crm, claims, now, r: CaseResult):
    last = _d(crm.get("last_day"))
    confirmed = crm.get("manager_confirmation")
    if last is None or confirmed is None:
        r.level = MANUAL
        r.checks.append("В CRM нет даты последнего дня или поля подтверждения руководителя.")
        return
    today = now.date()
    r.crm_facts.append(f"последний рабочий день {_human_day(last, today)}")
    r.crm_facts.append("подтверждение руководителя есть" if confirmed else "подтверждения ответственного руководителя нет")
    r.sort_key = datetime.combine(last, datetime.min.time())
    if confirmed:
        r.level = OK
        return
    days_left = (last - today).days
    if days_left <= 0:
        r.level = CRITICAL
    elif days_left == 1:
        r.level = TODAY
    elif days_left <= 3:
        r.level = WATCH
    else:
        r.level = OK
        return

    hearsay = [c for c in claims if c.third_party or c.kind == "done"]
    if hearsay:
        r.discrepancies.append("Сотрудник говорит за руководителя («он в курсе»). Это слова третьего лица, подтверждением не считается.")
    if days_left < 0:
        r.discrepancies.append("Последний день уже прошёл, а увольнение не подтверждено.")
    r.checks.append("Знает ли руководитель именно о сегодняшней дате, нет ли претензий по передаче дел, имущества, карт.")
    r.actions.append("Написать или позвонить руководителю напрямую и получить подтверждение в CRM (или письменно в рабочем чате, приложить к карточке) до конца дня.")
    r.actions.append("Если до 16:00 ответа нет: эскалировать старшему по проекту, чтобы увольнение не зависло после последнего дня.")
    r.drafts.append(f"Руководителю: «Добрый день! У сотрудника {r.id} последний рабочий день {_human_day(last, today)}, а в CRM нет вашего подтверждения увольнения. Подтвердите, пожалуйста, в CRM до 16:00 или ответьте здесь, я приложу. Если что-то не закрыто (дела, имущество), напишите.»")


def rule_card_delivery(case, crm, claims, now, r: CaseResult):
    status = crm.get("status")
    deadline = _dt(crm.get("deadline"))
    if status is None:
        r.level = MANUAL
        r.checks.append("В CRM нет статуса доставки.")
        return
    if status == "accepted":
        accepted = _dt(crm.get("accepted_at"))
        r.crm_facts.append("статус «Принято»" + (f", приёмка подтверждена в {_hm(accepted)}" if accepted else ""))
        r.level = OK
        if accepted and deadline and accepted > deadline:
            r.crm_facts.append(f"принято с опозданием на {_duration(accepted - deadline)}")
            r.level = WATCH
            r.actions.append("Зафиксировать опоздание у службы доставки (для статистики и претензий).")
        if any(c.kind == "delay" for c in claims):
            r.discrepancies.append("В чате пишут о задержке, хотя в CRM «Принято». Уточнить, нет ли второй партии или ошибки в статусе.")
            r.level = min(r.level, WATCH)
        return

    if deadline is None:
        r.level = MANUAL
        r.crm_facts.append(f"статус «{STATUS_LABEL.get(status, status)}»")
        r.checks.append("В CRM нет срока доставки, срочность определить нельзя.")
        return

    r.crm_facts.append(f"доставка запланирована до {_hm(deadline)}" + ("" if deadline.date() == now.date() else f" {deadline:%d.%m}"))
    r.crm_facts.append(f"статус «{STATUS_LABEL.get(status, status)}»")
    r.sort_key = deadline
    if now >= deadline:
        r.level = CRITICAL
        r.crm_facts.append("срок наступил, приёмки нет" if now - deadline < timedelta(minutes=1)
                           else f"срок прошёл {_duration(now - deadline)} назад, приёмки нет")
    elif deadline - now <= timedelta(hours=2):
        r.level = TODAY
    elif deadline.date() == now.date():
        r.level = WATCH
    else:
        r.level = OK

    for c in claims:
        if c.kind == "delay" and c.eta:
            late = c.eta - deadline
            r.discrepancies.append(
                f"Со слов курьера, приедет не раньше {_hm(c.eta)}, это на {_duration(late)} позже срока."
                if late > timedelta(0) else f"Курьер называет время {_hm(c.eta)}, в срок укладывается.")
            if late > timedelta(0):
                r.level = min(r.level, TODAY) if r.level != CRITICAL else CRITICAL
        elif c.kind == "delay":
            r.discrepancies.append("Курьер пишет о задержке, но время не назвал.")
            r.level = min(r.level, TODAY)
        elif c.kind == "done":
            r.discrepancies.append("В чате пишут, что карты доставлены, а в CRM приёмки нет.")
            r.checks.append("Попросить точку подтвердить приёмку в CRM.")

    if r.level in (CRITICAL, TODAY):
        eta = max((c.eta for c in claims if c.eta), default=None)
        when = f"не раньше {_hm(eta)}" if eta else "позже срока, время уточняем"
        r.checks.append(f"Не нужны ли карты на точке до {_hm(eta)} (выдача клиентам, встречи)." if eta else "Где сейчас курьер и когда реально приедет.")
        r.actions.append("Предупредить получателя на точке о новом времени, чтобы кто-то был на месте для приёмки.")
        r.actions.append("Уточнить у службы доставки, официальная ли это задержка, и внести новое время в CRM. Статус «В пути» не трогать, пока нет приёмки.")
        r.actions.append("Если карты нужны раньше: обсудить с логистикой ускорение.")
        r.drafts.append(f"Точке: «Добрый день! Доставка карт {r.id} задерживается: курьер будет {when}. Кто-то сможет принять? Если карты нужны раньше, напишите, попробуем ускорить.»")


STATUS_LABEL = {"in_transit": "В пути", "accepted": "Принято", "created": "Создано", "cancelled": "Отменено"}

RULES = {
    "onboarding": rule_onboarding,
    "offboarding": rule_offboarding,
    "card_delivery": rule_card_delivery,
}


def evaluate(case: dict, now: datetime, llm: str | None = None) -> CaseResult:
    r = CaseResult(id=case.get("id", "?"), title=case.get("title", case.get("type", "")), level=MANUAL)
    crm = case.get("crm") or {}
    msgs = case.get("messages") or []

    if llm and msgs:
        try:
            claims = analyze_messages_llm(case, now, llm)
        except Exception as e:  # LLM недоступна: не теряем кейс, падаем на правила
            print(f"[warn] {r.id}: LLM недоступна ({e}), разбираю правилами", file=sys.stderr)
            claims = [analyze_message_offline(m, now) for m in msgs]
    else:
        claims = [analyze_message_offline(m, now) for m in msgs]

    r.chat_claims = [_claim_line(c) for c in claims]

    rule = RULES.get(case.get("type"))
    if rule is None:
        r.level = MANUAL
        r.checks.append(f"Неизвестный тип операции «{case.get('type')}»: правил нет, разобрать вручную.")
    else:
        rule(case, crm, claims, now, r)

    # Общие проверки качества данных: применяются к любому типу.
    upd = _dt(crm.get("updated_at"))
    if upd and r.level != OK and now - upd > timedelta(hours=STALE_CRM_HOURS):
        r.checks.append(f"Карточка в CRM не обновлялась с {upd:%d.%m %H:%M}: данные могли устареть.")
    if r.level != OK and claims and all(not c.text.strip() for c in claims):
        r.checks.append("Сообщения пустые: проверить выгрузку чатов.")
    return r


# ---------- вывод ----------

def render_markdown(results: list[CaseResult], now: datetime) -> str:
    order = sorted(results, key=lambda x: (x.level, x.sort_key or datetime.max))
    act = [r for r in order if r.needs_action()]
    lines = [f"# Сводка support-отдела на {now:%d.%m.%Y %H:%M}", ""]
    lines.append(f"Кейсов: {len(results)}. Нужно вмешаться: {len(act)}. В порядке: {len(results) - len(act)}.")
    lines.append("")
    if act:
        lines.append("С чего начать: " + " → ".join(f"{r.id}" for r in act))
        lines.append("")
    lines.append("Строка «CRM» означает факт из карточки. Строка «Переписка» означает слова участников: пока этого нет в CRM, считаем неподтверждённым.")
    lines.append("")
    for r in order:
        lines.append(f"## {r.id}, {r.title.lower()}: {LEVEL_LABEL[r.level].lower()}")
        if r.crm_facts:
            lines.append("CRM (подтверждено): " + "; ".join(r.crm_facts) + ".")
        if r.chat_claims:
            label = "Переписка (совпадает с CRM)" if r.level == OK else "Переписка (не подтверждено)"
            lines.append(f"{label}: " + " | ".join(r.chat_claims))
        for d in r.discrepancies:
            lines.append(f"- Расхождение. {d}")
        if r.checks:
            lines.append("Проверить:")
            lines += [f"- {c}" for c in r.checks]
        if r.actions:
            lines.append("Сделать:")
            lines += [f"{i}. {a}" for i, a in enumerate(r.actions, 1)]
        if r.drafts:
            lines.append("Черновик сообщения:")
            lines += [f"> {d}" for d in r.drafts]
        if not r.needs_action():
            lines.append("Ничего делать не нужно.")
        lines.append("")
    return "\n".join(lines)


def to_json(results: list[CaseResult]) -> list[dict]:
    order = sorted(results, key=lambda x: (x.level, x.sort_key or datetime.max))
    return [{
        "id": r.id, "title": r.title, "priority": i + 1,
        "level": LEVEL_LABEL[r.level], "needs_action": r.needs_action(),
        "crm_facts": r.crm_facts, "chat_claims_unconfirmed": r.chat_claims,
        "discrepancies": r.discrepancies, "checks": r.checks,
        "actions": r.actions, "drafts": r.drafts,
    } for i, r in enumerate(order)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="JSON с полями now и cases")
    p.add_argument("--llm", choices=sorted(PROVIDERS), help="разбирать переписку моделью: local (Qwen в Ollama, "
                   "данные не уходят из контура), openai-compat (любой OpenAI-совместимый API, адрес в LLM_BASE_URL) "
                   "или anthropic (Claude, нужен ANTHROPIC_API_KEY)")
    p.add_argument("--now", help="переопределить текущее время, ISO 8601")
    p.add_argument("--md", help="куда сохранить Markdown (по умолчанию stdout)")
    p.add_argument("--json", help="куда сохранить JSON")
    a = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # консоль Windows по умолчанию не в UTF-8
        stream.reconfigure(encoding="utf-8")

    data = json.loads(Path(a.input).read_text(encoding="utf-8"))
    now = _dt(a.now or data.get("now")) or datetime.now()
    results = [evaluate(c, now, a.llm) for c in data["cases"]]

    md = render_markdown(results, now)
    if a.md:
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text(md, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(md)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(to_json(results), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
