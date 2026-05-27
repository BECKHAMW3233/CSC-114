# Tool Rationale for web-secplus-bot

| Tool | Enabled? | Why / Why Not |
|------|----------|---------------|
| bash | No | An exam prep bot has no need to execute shell commands; enabling bash creates unnecessary attack surface with no functional benefit. |
| read | Yes | The agent needs to read the mounted SY0-701 objectives file at /workspace/sy0-701-objectives.md to answer domain and objective-specific questions accurately. |
| write | No | The bot answers questions and generates practice content in chat; it has no legitimate reason to write files to the environment. |
| edit | No | No files need to be modified by the agent; disabling reduces risk of unintended file system changes. |
| glob | No | File pattern matching is not needed for an exam prep bot that reads a single known knowledge file. |
| grep | No | The agent can search its mounted knowledge file through read and its own reasoning; grep adds no value and increases tool surface. |
| web_search | Yes | Allows the agent to verify current exam information, check for objective updates, and pull supplementary explanations when the knowledge file is insufficient. |
| web_fetch | Yes | Enables the agent to retrieve specific pages from CompTIA or trusted study resources when web_search returns a relevant URL. |

## Rationale Summary

The tool selection follows the principle of minimum necessary access. The core function of this agent is reading a knowledge file and answering exam questions — that requires `read`, `web_search`, and `web_fetch` only. All file modification tools (`write`, `edit`) and filesystem traversal tools (`bash`, `glob`, `grep`) are disabled because an exam prep bot operating in a student environment has no legitimate use for them and their presence increases the blast radius of any prompt injection attempt.
