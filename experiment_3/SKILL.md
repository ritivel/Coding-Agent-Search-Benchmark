---
name: web-research-before-building
description: >
  Research the web before writing code from scratch. Use this skill whenever the user asks you to
  build, implement, create, or code anything non-trivial — a feature, a service, an integration,
  a pipeline, an app, a component, a tool, or an automation. Before writing a single line of code,
  this skill makes you research GitHub for high-quality existing implementations, find startups or
  open-source projects that have already solved the problem well, and locate the best official
  documentation. Triggers include: "build me a...", "implement...", "create a...", "code a...",
  "set up a...", "write a service that...", "I need a tool for...", "make an app that...",
  "integrate X with Y", "automate...", or any request where the user wants working software built.
  Also triggers when the user says "find the best way to build...", "what's the best library for...",
  "how do others implement...", or "don't reinvent the wheel". Even if the user doesn't explicitly
  ask for research, use this skill for any implementation task that would take more than ~50 lines
  of code — researching first almost always saves time and produces better results. Do NOT use for
  simple one-liners, quick scripts under 50 lines, pure debugging of existing code, or conceptual
  questions that don't involve building something.
---

# Web Research Before Building

## Purpose

When an AI coding agent (Claude, Cursor, Copilot, etc.) is asked to build something, the default
behavior is to start writing code from scratch based on training data. This is wasteful. In most
cases, someone has already built what you need — or something very close to it — and published it
on GitHub, written about it in a blog post, or documented it in official docs.

This skill forces a **research-first workflow**: before writing any code, you go find the best
existing implementations, libraries, and documentation. Then you build on top of what already
exists rather than reinventing the wheel.

## The Research-First Workflow

When you receive a build/implementation request, follow these phases in order.

### Phase 1: Understand the Request (1-2 minutes)

Before searching anything, clarify what you're building:

1. **What is the core problem?** State it in one sentence.
2. **What are the key technical requirements?** (language, framework, scale, constraints)
3. **What are the integration points?** (APIs, databases, services it must connect to)
4. **What is the quality bar?** (prototype vs production, internal tool vs public-facing)

If anything is ambiguous, ask the user before proceeding. A wrong assumption here wastes all
the research time.

### Phase 2: Search GitHub for Existing Implementations (primary research)

This is the most important phase. You are looking for repositories that have already solved
the problem or a close variant of it.

**How to search effectively:**

Run multiple targeted searches. Don't rely on a single query — you need to triangulate.

Search pattern 1 — Direct implementation search:
```
Search: "{thing you're building} {language/framework}"
Example: "webhook handler typescript express"
Example: "PDF invoice generator python"
Example: "real-time chat websocket react"
```

Search pattern 2 — "Awesome list" search:
```
Search: "awesome {domain}" github
Example: "awesome authentication" github
Example: "awesome pdf" github
Example: "awesome websocket" github
```
Awesome lists are community-curated collections of the best libraries and tools in a domain.
They are gold mines.

Search pattern 3 — Specific library/framework search:
```
Search: "{framework} {feature} example"
Example: "nextjs stripe integration example"
Example: "fastapi background tasks example"
Example: "django celery boilerplate"
```

Search pattern 4 — Problem-pattern search:
```
Search: "{problem pattern} implementation"
Example: "rate limiter implementation"
Example: "event sourcing example"
Example: "job queue with retries"
```

**How to evaluate repositories:**

When you find repos, evaluate them on these signals (in order of importance):

1. **Stars + recent activity** — A repo with 5k+ stars and commits in the last 3 months is
   usually a safe bet. But don't ignore gems with 200-500 stars that are laser-focused on
   your exact problem.
2. **README quality** — A clear, well-structured README with examples, installation steps,
   and architecture explanation signals a mature project.
3. **Code quality** — Skim the source. Is it well-organized? Are there tests? Is it typed?
4. **Issue tracker** — Open issues reveal known limitations. Closed issues show active
   maintenance.
5. **License** — Make sure the license is compatible with your use case (MIT, Apache 2.0,
   BSD are usually safe; GPL has restrictions; check with user if unclear).
6. **Dependencies** — Fewer is better. Heavy dependency trees introduce risk.

**What to extract from good repos:**

Don't just note the repo URL. Extract actionable intelligence:

- **Architecture patterns** — How did they structure the code? What design patterns?
- **Key libraries used** — What dependencies did they choose and why?
- **Configuration approach** — How do they handle config, env vars, secrets?
- **Error handling patterns** — How do they handle edge cases and failures?
- **Testing approach** — What testing strategy did they use?

### Phase 3: Find Startups and Production Examples

Search for companies and products that solve the same or adjacent problem. The goal is to learn
from production-grade implementations.

**How to search:**

```
Search: "{problem domain} startup"
Search: "{problem domain} open source alternative"
Search: "{product type} SaaS"
Search: "{problem domain} YC company"   (Y Combinator companies often open-source tools)
Search: "{specific tool} vs alternatives"
```

**What to look for:**

- **Open-source companies** — Many startups are open-source (Supabase, Cal.com, Documenso,
  Plausible, etc.). Their repos are production-grade reference implementations.
- **Technical blog posts** — Startups often write about how they solved hard problems.
  Search: "{company} engineering blog {topic}"
- **Architecture decisions** — Look for ADRs (Architecture Decision Records), technical docs,
  or blog posts explaining why they chose their tech stack.

**What to extract:**

- The tech stack they chose and why
- Architecture patterns that worked at scale
- Pitfalls they encountered and how they solved them
- Libraries or services they recommend

### Phase 4: Gather Official Documentation

Find the best documentation for the specific libraries, APIs, and frameworks you'll use.

**Priority order for documentation:**

1. **Official docs** — Always the primary source. Search: "{library} documentation"
2. **Official examples/recipes** — Many frameworks have example directories.
   Search: "{framework} examples github" or check the repo's `/examples` folder.
3. **Migration guides** — If using a newer version, find the migration guide to avoid
   deprecated patterns. Search: "{library} v{X} migration guide"
4. **API references** — For external services, find the API reference.
   Search: "{service} API reference" or "{service} developer docs"

**What to extract:**

- The correct, current way to do things (not the v2 pattern when v4 is out)
- Required configuration and setup steps
- Known gotchas or common mistakes (often in a "Troubleshooting" section)
- Rate limits, quotas, or constraints

### Phase 5: Synthesize and Present Findings

Before writing any code, present your research to the user. Structure it like this:

```
## Research Summary

### Recommended Approach
[1-2 sentence recommendation with rationale]

### Best Existing Implementations Found
1. [Repo name](url) — [stars] stars, [last updated]
   - Why it's relevant: [one sentence]
   - Key takeaway: [what to learn/borrow from it]
   
2. [Repo/Product name](url)
   - Why it's relevant: [one sentence]
   - Key takeaway: [what to learn/borrow from it]

### Key Libraries to Use
- [library]: [why this one over alternatives]

### Documentation References
- [doc name](url): [what it covers]

### Proposed Architecture
[Brief description of how you plan to build it, informed by research]

### Risks / Open Questions
- [anything that needs user input before proceeding]
```

Only after the user confirms the approach should you start writing code.

## How Many Searches to Run

Scale your research effort to the complexity of the task:

- **Simple integration** (e.g., "add Stripe checkout") — 3-5 searches. Focus on official docs
  and one good example repo.
- **Medium feature** (e.g., "build a notification system") — 5-8 searches. Cover GitHub repos,
  one startup example, and official docs.
- **Complex system** (e.g., "build a real-time collaboration engine") — 8-15 searches. Full
  research across all four phases. Consider suggesting Claude's Deep Research feature if the
  scope is very large.

## Common Mistakes to Avoid

1. **Don't just find repos — read them.** Use web_fetch to actually look at the README,
   the source code structure, and key files. A repo URL alone is not research.

2. **Don't default to the most popular solution.** The repo with 50k stars might be overkill
   for your use case. A focused 500-star repo might be a better fit.

3. **Don't ignore the date.** A tutorial from 2019 may use deprecated patterns. Always check
   when content was published and cross-reference with current docs.

4. **Don't copy blindly.** Research informs your implementation — it doesn't replace your
   judgment. Adapt patterns to the user's specific context.

5. **Don't skip the license check.** Using GPL code in a commercial project can create legal
   issues. Always note the license.

## Adapting This Skill for Cursor / Other Agents

If you're using this skill in Cursor, Windsurf, or another coding agent:

- The agent should read this SKILL.md at the start of any implementation task.
- The agent should use its web browsing / search capabilities to execute the searches
  described above.
- If the agent doesn't have web access, it should tell the user what to search for and ask
  them to paste the relevant findings back.
- The research summary should be added as a comment block at the top of the main file or
  in a `RESEARCH.md` file in the project root, so future agents/developers have context.

## Example: Full Research Flow

**User request:** "Build me a webhook handler service in Node.js that receives events from
Stripe, validates signatures, processes them idempotently, and stores them in PostgreSQL."

**Phase 1 — Understand:**
- Core problem: Reliable Stripe webhook processing
- Tech: Node.js, PostgreSQL
- Key requirements: signature validation, idempotency, persistence
- Quality bar: Production-grade (Stripe webhooks = money)

**Phase 2 — GitHub searches:**
1. "stripe webhook handler node" → Find repos, check stars/activity
2. "stripe webhook idempotent" → Find idempotency patterns
3. "awesome stripe" github → Find curated Stripe tooling lists
4. "webhook processing queue node" → Find queuing patterns for reliable processing

**Phase 3 — Startup/production examples:**
5. "stripe webhook best practices engineering blog" → Find battle-tested advice
6. "stripe webhook processing production" → Find production patterns

**Phase 4 — Documentation:**
7. "stripe webhook documentation" → Official Stripe webhook docs
8. "stripe node SDK documentation" → stripe-node library docs
9. "node postgres documentation" → pg or drizzle/prisma docs

**Phase 5 — Synthesize and present findings before writing code.**
