# HireFlow Learnings

## Day 1
**What is the difference between an LlmAgent and a SequentialAgent? When would you use each?**

An `LlmAgent` is essentially a single "brain" powered by a large language model. It takes an instruction, looks at the tools available to it, and figures out how to answer the user's prompt by calling those tools autonomously. You use an `LlmAgent` for a specific, narrow job—like checking for bias or parsing a document.

A `SequentialAgent` is like a manager orchestrating an assembly line. It doesn't do the deep thinking itself; instead, it passes data from one agent to the next in a strict, predefined order. You use a `SequentialAgent` when you have a multi-step process (like our HireFlow pipeline) where the output of one specialized `LlmAgent` needs to become the input for the next one, without them stepping on each other's toes.

## Day 2
**Why does the tool return dict instead of a numpy array? What would break if you returned the numpy array directly?**

ADK tools are executed by an LLM agent which requires text-based, structured context to 'understand' the tool's output. An LLM cannot process a raw numerical 
umpy array object natively—it just sees unstructured numbers. By returning a well-structured dict with named fields (like 
ame, match_score, skills), the LLM can easily interpret who the best candidate is and why, allowing it to reason about the next steps. If we returned a numpy array directly, the LLM would likely fail to parse the meaning of the data, break the orchestration flow, or hallucinate an interpretation.

## Day 3
**Why does the docstring matter so much in ADK tools? What happens if the docstring is vague?**

In Google ADK, the docstring acts as the tool's "API documentation" for the LLM. Unlike traditional code where a developer reads the docstring, the LLM reads it at runtime to understand what the function does, what arguments it requires, and when it is appropriate to use it. If the docstring is vague, the LLM might hallucinate arguments, fail to call the tool when it should, or call it at the wrong time (e.g., trying to draft outreach before candidates are even scored). A precise, descriptive docstring is essentially the prompt that controls the tool's usage.

## Day 4
**You have two tools in the JD Analyst agent. How does the agent decide which to call first? What would happen if it called them in the wrong order?**

The LLM agent relies heavily on the instructions provided in its prompt and the semantic meaning of the tool descriptions. In the `instruction` field of the `jd_analyst_agent`, we explicitly listed steps: "1. Call parse_job_description... 2. Call detect_bias_in_jd...". The LLM understands these textual step-by-step guidelines and plans its tool calls accordingly. If it were to call them in the wrong order, it might try to summarize the requirements before it actually parses them, resulting in a hallucinated or empty summary.

## Day 5
**What is the difference between `SequentialAgent` and `LlmAgent`? Why is the orchestrator a `SequentialAgent` and not an `LlmAgent`? What would break if you made it an `LlmAgent`?**

An `LlmAgent` is an autonomous decision-maker that uses an LLM to decide which tools to call and what to output based on its instructions. A `SequentialAgent` does not use an LLM directly to make decisions; instead, it enforces a strict, deterministic execution order across multiple sub-agents. The orchestrator is a `SequentialAgent` because we need a guaranteed pipeline: JD Analysis -> Bias Check -> Candidate Matching -> Outreach Drafting. If we made the orchestrator an `LlmAgent`, it might try to get creative and skip the bias check, run matching before parsing the JD, or draft outreach for candidates it hasn't scored yet. The `SequentialAgent` provides safety and predictability at the top level.

## Day 6
**Why do we test skills separately from agents? What makes agent tests harder to write and less reliable than unit tests?**

Skills are standard, deterministic Python functions. We test them separately to ensure the underlying logic (like the TF-IDF math or JSON extraction structure) works perfectly. Agent tests, on the other hand, require a live LLM to interpret a prompt and generate a dynamic response. Because LLMs are non-deterministic, agent tests are inherently flaky—the model might phrase things differently each time, or occasionally skip a minor detail. This makes them much harder to write (you have to test for general concepts using `in` or regex rather than exact string matches) and less reliable. If a skill is broken, the agent will fail, so testing skills in isolation narrows down where the bug actually is.

## Day 7
**What is the biggest limitation of the current system? What would you add if you had 2 more weeks?**

The biggest limitation of this system is that it currently operates entirely in-memory on static CSV data, without any persistent state tracking or real-world integrations. If I had two more weeks, I would integrate the MCP (Model Context Protocol) sourcing agent we skipped so it could actively search LinkedIn and GitHub for candidates, rather than relying on a static CSV. I would also hook it up to a proper Applicant Tracking System (ATS) API (like Greenhouse or Lever) to automatically move candidates between stages based on the LLM’s scores, and perhaps add a human-in-the-loop Slack integration for final approval before outreach emails are dispatched.
