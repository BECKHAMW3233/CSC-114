# SecPlus-Bot — System Prompt Version History

## Version 1 (initial)

```yaml
name: web-secplus-bot
model:
  id: claude-sonnet-4-6
  speed: standard
description: CompTIA Security+ SY0-701 exam prep agent for IT students
system: |
  ## Section 1: Identity and Role
  You are SecPlus-Bot, an exam preparation tool for the CompTIA Security+
  SY0-701 certification exam.
  ...
  ## Section 3: Domain Knowledge
  You have access to the official SY0-701 exam objectives at
  /workspace/sy0-701-objectives.md. Read it when answering questions
  about specific objective numbers or domain breakdowns.
tools:
  - configs: []
    default_config:
      enabled: true
      permission_policy:
        type: always_allow
    type: agent_toolset_20260401
```

**Problems with Version 1:**
- `speed: standard` is not a valid field in the Claude API spec — causes errors
- `/workspace/sy0-701-objectives.md` references a file that does not exist;
  the agent would hallucinate trying to read it
- `agent_toolset_20260401` tool structure was present but files were not
  actually uploaded yet — no knowledge base existed

---

## Version 2 (working deployment)

```yaml
name: web-secplus-bot
model:
  id: claude-sonnet-4-6
description: CompTIA Security+ SY0-701 exam prep agent for IT students
system: |
  ## Section 3: Domain Knowledge
  You have access to uploaded study notes mounted at /uploads/:
  - /uploads/Section 1_ Introduction.md
  - /uploads/Section 2_ Fundamentals of Security.md
  - /uploads/Section 3_ Threat Actors.md
  - /uploads/Section 4_ Physical Security.md
  - /uploads/Section 5_ Social Engineering.md
  - /uploads/Section 6_ Malware.md
  - /uploads/Section 7_ Data Protection.md
  - /uploads/Section 8_ Cryptographic Solutions.md

  Read the relevant file when answering questions about specific topics,
  objective numbers, or domain breakdowns.
tools:
  - type: agent_toolset_20260401
    configs: []
    default_config:
      enabled: true
      permission_policy:
        type: always_allow
```

**Changes from Version 1:**
- Removed `speed: standard` — invalid field
- Removed `tools: []` placeholder — replaced with properly structured
  `agent_toolset_20260401` required for file read access
- Replaced `/workspace/sy0-701-objectives.md` with actual `/uploads/` paths
  matching the 8 section files uploaded via the Files API
- Updated date to 2026-06-01

**File upload process:**
Files cannot be uploaded through the platform.claude.com console UI.
They must be uploaded programmatically via the API using the Files API
endpoint (`/v1/files`) with the `files-api-2025-04-14` beta header.
Each file returns a File ID (e.g., `file_011CbcgJq39DBHeU7fP73H9t`) which
is then mounted into the session under Resources with a `/uploads/` path.
The `agent_toolset_20260401` tool must be enabled on the agent for file
resources to be readable — without it the session returns:
"Missing required tool: file resources require the read tool to be usable."
