"""
A tiny, runnable picture of how an agent works. No AI model is involved.

Run it from the project root:   python demo_agent_loop.py

Read the printout top to bottom. In this demo a PUPPET plays the part of the
AI model: `pretend_model` is just a few if-statements. Everything else (the
loop, running the tools, sending results back) is exactly what a real agent
does. In the next step, `pretend_model` is replaced by a call to Gemini.

Needs: python -m app.seed_db  (once, to create the project database)
       internet access (for the GitHub question)
"""

import re

from app import tools

# The tools your code is able to run. The "model" can only ASK for these.
TOOLS = {
    "find_projects": tools.find_projects,
    "github_repos": tools.github_repos,
}

KNOWN_WORDS = ["ultrasonic", "python", "docker", "fastapi", "mqtt", "go", "flutter"]
MAX_STEPS = 5  # safety limit so the loop can never run forever


# ---------------------------------------------------------------------------
# 1. THE PUPPET: decides what to do next.
#    A real model reads the question and decides. This one uses simple rules.
#    It returns either ("ask", tool_name, arguments) or ("answer", text).
# ---------------------------------------------------------------------------
def pretend_model(question, tool_results):
    q = question.lower()

    needed = []
    if "project" in q:
        needed.append("find_projects")
    if "github" in q or "repo" in q:
        needed.append("github_repos")

    if not needed:
        return ("answer", "I have no tool that fits this question.")

    already_used = [name for name, _ in tool_results]
    for tool_name in needed:
        if tool_name not in already_used:
            return ("ask", tool_name, arguments_for(tool_name, q))

    return ("answer", write_answer(tool_results))


def arguments_for(tool_name, q):
    if tool_name == "github_repos":
        return {"username": "theertha18"}
    for word in KNOWN_WORDS:
        if re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", q):
            return {"keyword": word}
    return {"keyword": ""}


def write_answer(tool_results):
    parts = []
    for name, result in tool_results:
        if name == "find_projects":
            if result:
                names = "; ".join(p["name"] for p in result)
                parts.append(f"{len(result)} matching project(s): {names}")
            else:
                parts.append("no matching projects")
        elif name == "github_repos":
            if result:
                parts.append(f"your most recently updated public repo is '{result[0]['name']}'")
            else:
                parts.append("no public repos found")
    return "; and ".join(parts) + "."


# ---------------------------------------------------------------------------
# 2. THE LOOP: this is the agent. Ask the "model", run what it asks for,
#    hand the result back, repeat until it answers.
# ---------------------------------------------------------------------------
def run(question):
    print("=" * 78)
    print(f"QUESTION: {question}\n")

    tool_results = []  # everything looked up so far; the model sees all of it

    for step in range(1, MAX_STEPS + 1):
        decision = pretend_model(question, tool_results)

        if decision[0] == "answer":
            print(f"STEP {step}: the model has what it needs and ANSWERS:")
            print(f"        {decision[1]}\n")
            return

        _, tool_name, arguments = decision
        print(f"STEP {step}: the model ASKS for a tool:  {tool_name}({arguments})")

        try:
            result = TOOLS[tool_name](**arguments)  # YOUR code does the real work
        # Deliberate catch-all: a failing tool is reported, not fatal.
        except Exception as error:  # noqa: BLE001
            result = []
            print(f"        the tool failed: {error}")
        else:
            print(f"        your code RAN it and got {len(result)} result(s)")

        tool_results.append((tool_name, result))
        print("        the result goes back to the model\n")

    print(f"Stopped after {MAX_STEPS} steps (safety limit).\n")


if __name__ == "__main__":
    run("Which of my projects used ultrasonic sensors?")
    run("What is my newest GitHub repo?")
    run("Which of my projects use Python, and what is my newest GitHub repo?")
    run("What did I do at Bosch?")
