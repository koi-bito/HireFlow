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
