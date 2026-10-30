"""
Tests for app/agent.py. Run from the project root with:

    python -m unittest tests.test_agent -v

No network and no API key: the Gemini client and the SDK's data classes are
replaced by small fakes, so these tests check OUR code: that it runs the tool
the model asks for, hands the result back, and protects itself from bad input.
"""

import pathlib
import sys
import tempfile
import types as pytypes
import unittest
from types import SimpleNamespace
from unittest import mock

from app import agent, seed_db, tools


# ---- fake google-genai SDK -------------------------------------------------
class FakePart:
    def __init__(self, text=None, function_response=None):
        self.text = text
        self.function_response = function_response

    @classmethod
    def from_function_response(cls, name, response, id=None):
        return cls(function_response={"name": name, "response": response, "id": id})


class FakeContent:
    def __init__(self, role=None, parts=None):
        self.role = role
        self.parts = parts or []


class FakeTool:
    def __init__(self, function_declarations=None):
        self.function_declarations = function_declarations


class FakeConfig:
    def __init__(self, system_instruction=None, tools=None):
        self.system_instruction = system_instruction
        self.tools = tools


def fake_sdk_modules():
    fake_types = SimpleNamespace(
        Part=FakePart, Content=FakeContent, Tool=FakeTool, GenerateContentConfig=FakeConfig
    )
    genai = pytypes.ModuleType("google.genai")
    genai.types = fake_types
    google = pytypes.ModuleType("google")
    google.genai = genai
    return {"google": google, "google.genai": genai}


def model_reply(text=None, calls=None):
    """What the fake model returns."""
    return SimpleNamespace(
        text=text,
        function_calls=calls or [],
        candidates=[SimpleNamespace(content=FakeContent(role="model"))],
    )


def tool_request(name, args, call_id=None):
    return SimpleNamespace(name=name, args=args, id=call_id)


class FakeClient:
    """Plays back scripted replies and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.sent.append({"model": model, "contents": list(contents), "config": config})
        return self.replies.pop(0)


# ---- run_tool: the safety layer between the model and the real functions ---
class RunToolTests(unittest.TestCase):
    def test_unknown_tool_returns_an_error_not_an_exception(self):
        payload = agent.run_tool("delete_everything", {})
        self.assertIn("Unknown tool", payload["error"])

    def test_runs_the_real_function_and_wraps_the_result(self):
        spec = {"fn": lambda keyword="": [keyword.upper()], "args": {"keyword": str}}
        with mock.patch.dict(agent.TOOL_FUNCTIONS, {"find_projects": spec}):
            self.assertEqual(agent.run_tool("find_projects", {"keyword": "go"}),
                             {"result": ["GO"]})

    def test_arguments_the_card_does_not_declare_are_dropped(self):
        spec = {"fn": lambda keyword="": [keyword], "args": {"keyword": str}}
        with mock.patch.dict(agent.TOOL_FUNCTIONS, {"find_projects": spec}):
            payload = agent.run_tool(
                "find_projects", {"keyword": "x", "db_path": "/etc/passwd", "limit": 9999}
            )
        self.assertEqual(payload, {"result": ["x"]})

    def test_argument_values_are_converted_to_the_declared_type(self):
        spec = {"fn": lambda keyword="": [keyword], "args": {"keyword": str}}
        with mock.patch.dict(agent.TOOL_FUNCTIONS, {"find_projects": spec}):
            self.assertEqual(agent.run_tool("find_projects", {"keyword": 5})["result"], ["5"])

    def test_a_failing_tool_becomes_an_error_payload(self):
        def broken():
            raise RuntimeError("GitHub rate limit reached")

        with mock.patch.dict(agent.TOOL_FUNCTIONS, {"github_repos": {"fn": broken, "args": {}}}):
            payload = agent.run_tool("github_repos", {})
        self.assertEqual(payload, {"error": "RuntimeError: GitHub rate limit reached"})

    def test_missing_arguments_are_treated_as_none_given(self):
        spec = {"fn": lambda: ["ok"], "args": {}}
        with mock.patch.dict(agent.TOOL_FUNCTIONS, {"github_repos": spec}):
            self.assertEqual(agent.run_tool("github_repos", None), {"result": ["ok"]})


class CardTests(unittest.TestCase):
    def test_every_card_has_a_matching_function(self):
        self.assertEqual({c["name"] for c in agent.TOOL_CARDS}, set(agent.TOOL_FUNCTIONS))

    def test_the_model_can_only_set_declared_parameters(self):
        for card in agent.TOOL_CARDS:
            declared = set(card.get("parameters", {}).get("properties", {}))
            allowed = set(agent.TOOL_FUNCTIONS[card["name"]]["args"])
            self.assertEqual(declared, allowed, card["name"])

    def test_cards_have_descriptions(self):
        for card in agent.TOOL_CARDS:
            self.assertGreater(len(card["description"]), 30, card["name"])


# ---- ask(): the full round trip, against a scripted fake model -------------
class AskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = pathlib.Path(cls.tmp.name) / "portfolio.db"
        seed_db.seed(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def real_find_projects(self):
        return {"fn": lambda keyword="": tools.find_projects(keyword, db_path=self.db),
                "args": {"keyword": str}}

    def ask(self, replies, question="q", **kwargs):
        client = FakeClient(replies)
        log = []
        with mock.patch.dict(sys.modules, fake_sdk_modules()), \
             mock.patch.dict(agent.TOOL_FUNCTIONS, {"find_projects": self.real_find_projects()}):
            answer = agent.ask(question, client=client, model="m", say=log.append, **kwargs)
        return answer, client, "\n".join(log)

    def test_model_asks_for_a_tool_we_run_it_and_the_model_answers(self):
        answer, client, log = self.ask([
            model_reply(calls=[tool_request("find_projects", {"keyword": "ultrasonic"}, "c1")]),
            model_reply(text="Three projects used ultrasonic sensors."),
        ])
        self.assertEqual(answer, "Three projects used ultrasonic sensors.")
        self.assertEqual(len(client.sent), 2)  # one call to ask, one to answer
        self.assertIn("find_projects", log)
        self.assertIn("3 result(s)", log)

    def test_the_tool_result_is_what_goes_back_to_the_model(self):
        _, client, _ = self.ask([
            model_reply(calls=[tool_request("find_projects", {"keyword": "ultrasonic"}, "c1")]),
            model_reply(text="done"),
        ])
        second_request = client.sent[1]["contents"]
        returned = second_request[-1].parts[0].function_response
        self.assertEqual(returned["name"], "find_projects")
        self.assertEqual(returned["id"], "c1")  # id is passed back to the model
        self.assertEqual(len(returned["response"]["result"]), 3)
        # the model's own request is kept in the conversation before our result
        self.assertEqual(second_request[-2].role, "model")

    def test_every_request_carries_the_tool_cards_and_instructions(self):
        _, client, _ = self.ask([model_reply(text="hi")])
        config = client.sent[0]["config"]
        self.assertEqual(config.system_instruction, agent.SYSTEM_PROMPT)
        self.assertEqual(config.tools[0].function_declarations, agent.TOOL_CARDS)

    def test_question_without_a_tool_is_answered_directly(self):
        answer, client, log = self.ask([model_reply(text="Canberra.")])
        self.assertEqual(answer, "Canberra.")
        self.assertEqual(len(client.sent), 1)
        self.assertIn("without using a tool", log)

    def test_a_failing_tool_is_reported_to_the_model_not_raised(self):
        def broken():
            raise RuntimeError("no network")

        client = FakeClient([
            model_reply(calls=[tool_request("github_repos", {})]),
            model_reply(text="I could not reach GitHub."),
        ])
        log = []
        with mock.patch.dict(sys.modules, fake_sdk_modules()), \
             mock.patch.dict(agent.TOOL_FUNCTIONS, {"github_repos": {"fn": broken, "args": {}}}):
            answer = agent.ask("q", client=client, model="m", say=log.append)
        self.assertEqual(answer, "I could not reach GitHub.")
        sent_back = client.sent[1]["contents"][-1].parts[0].function_response["response"]
        self.assertIn("no network", sent_back["error"])
        self.assertIn("the tool failed", "\n".join(log))

    def test_several_requests_in_one_reply_are_all_run(self):
        _, client, _ = self.ask([
            model_reply(calls=[
                tool_request("find_projects", {"keyword": "docker"}, "a"),
                tool_request("find_projects", {"keyword": "go"}, "b"),
            ]),
            model_reply(text="ok"),
        ])
        parts = client.sent[1]["contents"][-1].parts
        self.assertEqual([p.function_response["id"] for p in parts], ["a", "b"])

    def test_the_loop_runs_a_second_round_when_the_model_asks_again(self):
        answer, client, log = self.ask([
            model_reply(calls=[tool_request("find_projects", {"keyword": "python"}, "a")]),
            model_reply(calls=[tool_request("find_projects", {"keyword": "go"}, "b")]),
            model_reply(text="Both lookups done."),
        ])
        self.assertEqual(answer, "Both lookups done.")
        self.assertEqual(len(client.sent), 3)
        self.assertIn("STEP 1", log)
        self.assertIn("STEP 2", log)
        self.assertIn("STEP 3: the model has what it needs", log)

    def test_the_conversation_grows_with_every_round(self):
        _, client, _ = self.ask([
            model_reply(calls=[tool_request("find_projects", {"keyword": "python"}, "a")]),
            model_reply(calls=[tool_request("find_projects", {"keyword": "go"}, "b")]),
            model_reply(text="done"),
        ])
        lengths = [len(request["contents"]) for request in client.sent]
        # question | + model request + our result | + model request + our result
        self.assertEqual(lengths, [1, 3, 5])

    def test_step_limit_takes_the_tools_away_and_forces_an_answer(self):
        answer, client, log = self.ask(
            [
                model_reply(calls=[tool_request("find_projects", {"keyword": "a"}, "1")]),
                model_reply(calls=[tool_request("find_projects", {"keyword": "b"}, "2")]),
                model_reply(text="Best answer from what I have."),  # the forced final call
            ],
            max_rounds=2,
        )
        self.assertEqual(answer, "Best answer from what I have.")
        self.assertEqual(len(client.sent), 3)  # 2 rounds of tools + 1 forced answer
        forced_config = client.sent[2]["config"]
        self.assertIsNone(forced_config.tools)
        self.assertIn("no tools left", forced_config.system_instruction)
        self.assertIn("Step limit reached", log)

    def test_a_model_that_never_stops_asking_cannot_run_forever(self):
        endless = [model_reply(calls=[tool_request("find_projects", {"keyword": "x"}, str(i))])
                   for i in range(10)]
        endless.append(model_reply(text="finally"))
        _, client, _ = self.ask(endless[:3] + [model_reply(text="finally")], max_rounds=3)
        self.assertEqual(len(client.sent), 4)  # 3 rounds + 1 forced answer, then it stops

    def test_a_forced_answer_that_still_has_tool_requests_returns_empty_text(self):
        answer, _, _ = self.ask(
            [
                model_reply(calls=[tool_request("find_projects", {"keyword": "a"}, "1")]),
                model_reply(text="ignored", calls=[tool_request("find_projects", {"keyword": "b"}, "2")]),
            ],
            max_rounds=1,
        )
        self.assertEqual(answer, "")


class MainTests(unittest.TestCase):
    def test_a_failing_model_call_is_reported_not_raised(self):
        with mock.patch.object(agent, "ask", side_effect=RuntimeError("429 rate limit")), \
             mock.patch.object(sys, "argv", ["agent", "some question"]), \
             mock.patch("builtins.print") as fake_print:
            agent.main()  # must not raise
        self.assertIn("429 rate limit", str(fake_print.call_args_list))


if __name__ == "__main__":
    unittest.main()
