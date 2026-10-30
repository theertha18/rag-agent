"""
The agent: a loop where the model asks for tools until it can answer.

Run:
    python -m app.agent "Which of my projects used ultrasonic sensors?"
    python -m app.agent                      (runs three example questions)

What happens (the printout shows each step):
  1. We send the model your question plus a "card" describing each tool.
  2. The model replies with a request ("run find_projects with keyword
     ultrasonic") instead of an answer. It cannot run anything itself.
  3. OUR code runs the real function and sends the result back.
  4. The model either asks for another tool (back to step 3) or writes the
     final answer.

Safety limits: the model can only use the tools listed here, can only pass the
arguments their cards declare, and gets at most MAX_ROUNDS rounds of tool use.
If it is still asking after that, it is told to answer with what it has.

Needs GEMINI_API_KEY (and optionally GEMINI_MODEL) set, like /ask.
"""

import os
import sys

from app import tools
from app.llm import DEFAULT_GEMINI_MODEL

GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "theertha18")
MAX_ROUNDS = 5  # how many times the model may ask for tools before it must answer

SYSTEM_PROMPT = (
    "You answer questions about one person's background, projects and code, "
    "using the tools provided. Choose the tool that fits the question: "
    "search_documents to explain what they did and how, find_projects for lists "
    "and counts of projects, github_repos for questions about their GitHub "
    "repositories. Use only what the tools return. If the tools return nothing "
    "relevant, say you could not find it. Do not guess."
)

# ---------------------------------------------------------------------------
# The "cards": the only thing the model knows about each tool. Its choice of
# tool is based entirely on these descriptions, so they are worth reading.
# ---------------------------------------------------------------------------
TOOL_CARDS = [
    {
        "name": "search_documents",
        "description": (
            "Searches the person's CV, thesis, project write-ups and work notes by "
            "meaning and returns the most relevant passages. Use it to explain what "
            "they did, how something worked, or why. Not suited to complete lists or counts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, as a short natural-language phrase.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_projects",
        "description": (
            "Looks up projects in a database by a keyword in the name, description or "
            "technologies, and returns every match. Use it for lists and counts, for "
            "example all projects that used a given technology."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "One technology or term, such as 'Docker' or 'ultrasonic'. "
                                   "Leave empty to list every project.",
                }
            },
        },
    },
    {
        "name": "github_repos",
        "description": (
            "Lists the person's public GitHub repositories, most recently updated first. "
            "Use it for questions about their repositories or recent code."
        ),
    },
]


def _github_repos():
    return tools.github_repos(GITHUB_USERNAME)


# The real functions. "args" lists the only arguments the model may pass (and
# their types): anything else it sends is dropped, so it cannot, for example,
# point find_projects at a different database file.
TOOL_FUNCTIONS = {
    "search_documents": {"fn": tools.search_documents, "args": {"query": str}},
    "find_projects": {"fn": tools.find_projects, "args": {"keyword": str}},
    "github_repos": {"fn": _github_repos, "args": {}},
}


def run_tool(name: str, args: dict | None) -> dict:
    """Run one tool request from the model. Never raises: failures come back as
    {"error": ...} so the model can tell the user instead of crashing the program."""
    spec = TOOL_FUNCTIONS.get(name)
    if spec is None:
        return {"error": f"Unknown tool '{name}'."}

    allowed = spec["args"]
    clean = {key: allowed[key](value) for key, value in (args or {}).items() if key in allowed}
    try:
        return {"result": spec["fn"](**clean)}
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def _response_part(types, call, payload):
    """Wrap a tool result for the model. Newer models number each call (call.id)
    and expect the same id back."""
    call_id = getattr(call, "id", None)
    if call_id:
        try:
            return types.Part.from_function_response(
                name=call.name, response=payload, id=call_id
            )
        except TypeError:
            pass  # older SDK without the id argument
    return types.Part.from_function_response(name=call.name, response=payload)


def _describe(payload: dict) -> str:
    if "error" in payload:
        return f"the tool failed: {payload['error']}"
    result = payload["result"]
    return f"{len(result)} result(s)" if isinstance(result, list) else "done"


def _final_text(response) -> str:
    # Reading .text on a reply that still contains tool requests makes the SDK
    # print a warning, so only read it when there are none.
    return "" if response.function_calls else (response.text or "")


def ask(question: str, client=None, model: str | None = None, say=print,
        max_rounds: int = MAX_ROUNDS) -> str:
    """Ask one question. The model may use tools for up to max_rounds rounds."""
    from google.genai import types

    if client is None:
        from google import genai

        client = genai.Client()  # reads GEMINI_API_KEY
    model = model or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=TOOL_CARDS)],
    )
    contents = [types.Content(role="user", parts=[types.Part(text=question)])]

    say("=" * 78)
    say(f"QUESTION: {question}\n")

    for round_number in range(1, max_rounds + 1):
        response = client.models.generate_content(model=model, contents=contents, config=config)
        calls = response.function_calls or []

        if not calls:
            if round_number == 1:
                say("The model answered without using a tool:")
            else:
                say(f"STEP {round_number}: the model has what it needs and ANSWERS:")
            say(f"    {response.text}\n")
            return response.text or ""

        # Keep the model's own request in the conversation, then add our results.
        contents.append(response.candidates[0].content)
        result_parts = []
        for call in calls:
            args = dict(call.args or {})
            say(f"STEP {round_number}: the model ASKS for a tool:  {call.name}({args})")
            payload = run_tool(call.name, args)
            say(f"        your code RAN it: {_describe(payload)}")
            say("        the result goes back to the model\n")
            result_parts.append(_response_part(types, call, payload))
        contents.append(types.Content(role="user", parts=result_parts))

    # Step limit reached: take the tools away and ask for an answer from what we have.
    say(f"Step limit reached ({max_rounds} rounds). Asking the model to answer with what it has.")
    final_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT
        + " You have no tools left. Answer now using only the information already returned above."
    )
    response = client.models.generate_content(model=model, contents=contents, config=final_config)
    say("The model ANSWERS:")
    say(f"    {_final_text(response)}\n")
    return _final_text(response)


EXAMPLES = [
    "Which of my projects used ultrasonic sensors?",
    "Which of my projects use Python, and what is my newest GitHub repo?",
    "What did I do with MQTT at Bosch Rexroth?",
]


def main():
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else EXAMPLES
    for question in questions:
        try:
            ask(question)
        except Exception as error:  # e.g. rate limit, bad key, no network
            print(f"Stopped: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
