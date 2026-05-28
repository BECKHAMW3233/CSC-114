# Projects vs. Platform — A Builder's Comparison

**Same bot, two platforms. Here's what I noticed.**

## What was easier on claude.ai (Session 0)?

I have done chatbots using claude.ai and now using it for an agent I find
the agent is better, but I have also done a form of an agent using my own
local setup using Ollama with AnythingLLM as an interface and system prompts
that create data that is always referenced before any output is created.
Claude.ai is simpler to get started with for that kind of setup.

## What was easier on platform.claude.com (Session 1)?

The setup is tricky at first but I am new to this. Once you have it running
and properly set up the possibilities seem endless on what you could do with
it. The fact that it can have files you can set up for it to reference and
use for the data means it will either confirm or deny based on facts and
logic, unlike a normal claude.ai session that can fabricate and also just
assume based on what it seems to be feeling like at the time, or if its
internal training data says something that is proven false it will argue with
you over it and assume its data is correct even when proven false.

## If I had to deploy CSC-114-InfoBot to 500 students next semester, which platform would I use, and why?

I would use the agent on platform.claude.com as it is solid and you can give
students access with an API key so they can only use it to do certain tasks.
It will not fall for prompt injection attacks that would allow it to give
answers it is not programmed for.
