# HireFlow Learnings

## Day 1
**What is the difference between an LlmAgent and a SequentialAgent? When would you use each?**

An `LlmAgent` is essentially a single "brain" powered by a large language model. It takes an instruction, looks at the tools available to it, and figures out how to answer the user's prompt by calling those tools autonomously. You use an `LlmAgent` for a specific, narrow job—like checking for bias or parsing a document.

A `SequentialAgent` is like a manager orchestrating an assembly line. It doesn't do the deep thinking itself; instead, it passes data from one agent to the next in a strict, predefined order. You use a `SequentialAgent` when you have a multi-step process (like our HireFlow pipeline) where the output of one specialized `LlmAgent` needs to become the input for the next one, without them stepping on each other's toes.
