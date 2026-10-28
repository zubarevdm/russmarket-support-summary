"""Проверки правил: четыре кейса из задания и граничные случаи.

python -m unittest test_summary -v
"""

import contextlib
import copy
import io
import json
import os
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import summary
from summary import CRITICAL, MANUAL, OK, TODAY, WATCH, evaluate, mask_pd, render_markdown

DATA = json.loads((Path(__file__).parent / "data" / "cases_2026-11-02.json").read_text(encoding="utf-8"))
NOW = datetime.fromisoformat(DATA["now"])
CASES = {c["id"]: c for c in DATA["cases"]}


def run(case, now=NOW):
    return evaluate(case, now)


class TaskCases(unittest.TestCase):
    def test_levels(self):
        self.assertEqual(run(CASES["С-24"]).level, CRITICAL)
        self.assertEqual(run(CASES["П-08"]).level, CRITICAL)
        self.assertEqual(run(CASES["С-17"]).level, TODAY)
        self.assertEqual(run(CASES["П-09"]).level, OK)

    def test_order(self):
        md = render_markdown([run(c) for c in DATA["cases"]], NOW)
        self.assertIn("С-24 → П-08 → С-17", md)
        self.assertNotIn("П-09 →", md)

    def test_hearsay_is_flagged(self):
        r = run(CASES["С-24"])
        self.assertTrue(any("третьего лица" in d for d in r.discrepancies))

    def test_private_channel_flagged(self):
        r = run(CASES["С-17"])
        self.assertTrue(any("личный чат" in d for d in r.discrepancies))

    def test_courier_eta(self):
        r = run(CASES["П-08"])
        self.assertTrue(any("13:00" in d and "3 ч" in d for d in r.discrepancies))


class EdgeCases(unittest.TestCase):
    def test_chat_cannot_lower_priority(self):
        # «Всё внесено, всё ок» в чате не делает кейс зелёным.
        c = copy.deepcopy(CASES["С-24"])
        c["messages"] = [{"from_role": "employee", "at": "", "text": "Руководитель всё подтвердил, в CRM уже внесено, статус ок"}]
        self.assertEqual(run(c).level, CRITICAL)

    def test_delivery_before_deadline_no_chat(self):
        c = copy.deepcopy(CASES["П-08"])
        c["messages"] = []
        self.assertEqual(run(c, NOW.replace(hour=7)).level, WATCH)

    def test_courier_delay_before_deadline_raises(self):
        c = copy.deepcopy(CASES["П-08"])
        self.assertEqual(run(c, NOW.replace(hour=7)).level, TODAY)

    def test_accepted_late(self):
        c = copy.deepcopy(CASES["П-09"])
        c["crm"]["deadline"] = "2026-11-02T09:00:00"
        r = run(c)
        self.assertEqual(r.level, WATCH)
        self.assertTrue(any("опозданием" in f for f in r.crm_facts))

    def test_accepted_but_chat_says_delay(self):
        c = copy.deepcopy(CASES["П-09"])
        c["messages"] = [{"from_role": "courier", "at": "", "text": "Задерживаюсь, буду к 12:00"}]
        self.assertEqual(run(c).level, WATCH)

    def test_offboarding_overdue(self):
        c = copy.deepcopy(CASES["С-24"])
        c["crm"]["last_day"] = "2026-10-30"
        r = run(c)
        self.assertEqual(r.level, CRITICAL)
        self.assertTrue(any("уже прошёл" in d for d in r.discrepancies))

    def test_offboarding_confirmed(self):
        c = copy.deepcopy(CASES["С-24"])
        c["crm"]["manager_confirmation"] = True
        self.assertEqual(run(c).level, OK)

    def test_onboarding_today_without_docs(self):
        c = copy.deepcopy(CASES["С-17"])
        c["crm"]["first_shift_date"] = "2026-11-02"
        self.assertEqual(run(c).level, CRITICAL)

    def test_onboarding_far_away(self):
        c = copy.deepcopy(CASES["С-17"])
        c["crm"]["first_shift_date"] = "2026-11-20"
        self.assertEqual(run(c).level, OK)

    def test_missing_crm_fields_go_to_manual(self):
        for cid in ("С-17", "С-24"):
            c = copy.deepcopy(CASES[cid])
            c["crm"] = {}
            self.assertEqual(run(c).level, MANUAL, cid)

    def test_unknown_type(self):
        self.assertEqual(run({"id": "X-1", "type": "equipment", "crm": {}, "messages": []}).level, MANUAL)

    def test_stale_crm(self):
        c = copy.deepcopy(CASES["П-08"])
        c["crm"]["updated_at"] = "2026-10-30T08:00:00"
        self.assertTrue(any("не обновлялась" in x for x in run(c).checks))


class LlmLayer(unittest.TestCase):
    def test_mask_pd(self):
        out = mask_pd("паспорт 4510 123456, тел +7 (912) 345-67-89, почта a.b@mail.ru")
        self.assertNotIn("123456", out)
        self.assertNotIn("345-67-89", out)
        self.assertNotIn("a.b@mail.ru", out)
        self.assertIn("[паспорт]", out)

    def test_payload_has_no_pd_and_no_id(self):
        c = copy.deepcopy(CASES["С-17"])
        c["messages"][0]["text"] = "Отправил паспорт 4510 123456 в личку"
        payload = summary._llm_payload(c, NOW)
        self.assertNotIn("123456", payload)

    def test_model_output_is_sanitized(self):
        # Модель вернула текст вокруг JSON, чужой индекс и неизвестный kind.
        fake = ('Вот ответ: {"claims": [{"message_index": 0, "kind": "maybe", "about": "x", '
                '"eta": "не знаю", "third_party": true, "process_flags": []}, '
                '{"message_index": 5, "kind": "done", "about": "", "eta": null, '
                '"third_party": false, "process_flags": []}]}')
        with mock.patch.dict(summary.PROVIDERS, {"local": lambda s, u: fake}):
            claims = summary.analyze_messages_llm(CASES["С-24"], NOW, "local")
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].kind, "unknown")
        self.assertIsNone(claims[0].eta)

    def test_model_cannot_lower_priority(self):
        fake = json.dumps({"claims": [{"message_index": 0, "kind": "done", "about": "подтверждено",
                                       "eta": None, "third_party": False, "process_flags": []}]})
        with mock.patch.dict(summary.PROVIDERS, {"local": lambda s, u: fake}):
            self.assertEqual(evaluate(CASES["С-24"], NOW, "local").level, CRITICAL)

    def test_local_server_down_falls_back_to_rules(self):
        with mock.patch.dict(os.environ, {"LLM_BASE_URL": "http://127.0.0.1:9"}), \
             contextlib.redirect_stderr(io.StringIO()):
            r = evaluate(CASES["П-08"], NOW, "local")
        self.assertEqual(r.level, CRITICAL)
        self.assertTrue(any("13:00" in d for d in r.discrepancies))


if __name__ == "__main__":
    unittest.main()
