---
title: "OWASP Top 10 for MCP — and how Google's Gemini Enterprise Platform addresses every risk"
date: "2026-05-01"
author: "AI Converge"
tags: ["AI Security", "Enterprise Architecture", "MCP", "OWASP", "Gemini Enterprise", "GCP"]
excerpt: "Model Context Protocol changed the threat surface of AI agents overnight. Here's the definitive map of the ten risks that matter — and how Google Cloud's newest enterprise platform is built to neutralise them."
readTime: "12 min"
image: "https://media2.dev.to/dynamic/image/width=1200,height=627,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fres.cloudinary.com%2Fdzwithwd8%2Fimage%2Fupload%2Fv1777458690%2Fdevto-images%2Fowasp-mcp-cover.png"
---

# OWASP Top 10 for MCP — and how Google's Gemini Enterprise Platform addresses every risk

**Model Context Protocol changed the threat surface of AI agents overnight. Here's the definitive map of the ten risks that matter — and how Google Cloud's newest enterprise platform is built to neutralise them.**

*AI Security · Enterprise Architecture · 2026*

---

The Model Context Protocol (MCP) is rewriting the rules of enterprise AI. Where agents once lived inside tightly scoped application boundaries, MCP lets them discover tools at runtime from any server they can reach — turning every server into a live trust boundary. The result is extraordinary capability and an entirely new class of attack surface.

OWASP responded in 2025 with the first dedicated MCP security framework: ten categories, published as MCP01:2025 through MCP10:2025, grounded in real incidents and real CVEs. At Cloud Next 2026, Google rebranded and unified its AI platform under a single name — the **Gemini Enterprise Agent Platform** — with security architecture designed explicitly for this threat landscape.

## By the Numbers

- **30+ CVEs** against MCP servers in 60 days (early 2026)
- **78.3%** Attack success rate with 5 connected MCP servers (Unit 42)
- **84.2%** Tool poisoning success rate with auto-approval on

---

## The Ten Risks — and GCP's Responses

Each section below covers an OWASP MCP risk category, what it means in practice, and how the Gemini Enterprise Agent Platform addresses it.

### MCP01:2025 — Token Mismanagement & Secret Exposure
**Severity: Critical**

**The Risk:**  
MCP servers often require API keys, database credentials, or OAuth tokens to function. When agents connect to multiple servers, credential sprawl becomes immediate. Hardcoded secrets in client configs, tokens passed in plain environment variables, or credential reuse across contexts create systemic exposure.

**GCP Coverage: Full**
- **Agent Identity:** Unique managed identity per agent eliminates shared credential pools
- **Agent Gateway:** Centralized token vaulting with automatic rotation and short-lived credentials
- Secrets never touch agent memory or client-side config

---

### MCP02:2025 — Privilege Escalation via Scope Creep
**Severity: Critical**

**The Risk:**  
An agent authorized for read-only database queries connects to an MCP server that also exposes `DELETE` and `TRUNCATE` tools. Without runtime scope enforcement, the agent inherits all server capabilities — not just what the original task required.

**GCP Coverage: Full**
- **Agent Compliance & Policy:** Declarative least-privilege enforcement at the protocol layer
- Fine-grained tool whitelisting per agent role
- Automatic rejection of out-of-scope tool invocations

---

### MCP03:2025 — Tool Poisoning
**Severity: High**

**The Risk:**  
A malicious or compromised MCP server advertises a `search_documents` tool that actually exfiltrates sensitive data to an external endpoint. The agent, trusting the tool description, executes it without validation.

**GCP Coverage: Full**
- **Agent Gateway:** Semantic validation of tool schemas against known-good baselines
- **Semantic Governance:** Content inspection of tool inputs/outputs before execution
- Real-time anomaly detection for unexpected tool behavior

---

### MCP04:2025 — Software Supply Chain Attacks & Dependency Tampering
**Severity: High**

**The Risk:**  
MCP servers are deployed as containers or packages with dozens of transitive dependencies. A compromised NPM package in the dependency tree can inject backdoors into every tool the server exposes.

**GCP Coverage: Partial**
- **Apigee MCP Gateway:** Managed MCP runtime eliminates need to run untrusted server code
- **Wiz AI-APP integration:** Supply chain scanning for vulnerabilities and malicious packages
- **Gap:** Zero-day supply chain exploits still require developer vigilance

---

### MCP05:2025 — Command Injection & Execution
**Severity: High**

**The Risk:**  
An agent passes user input directly into a shell command via an MCP tool. Attacker-controlled strings like `; rm -rf /` or backtick payloads execute arbitrary code on the server host.

**GCP Coverage: Partial**
- **Agent Gateway:** Input sanitization rules for common injection patterns
- **Security Command Center (SCC):** Runtime behavior monitoring and anomaly flagging
- **Gap:** Novel injection techniques and application-specific payloads require prompt-layer hardening

---

### MCP06:2025 — Intent Flow Subversion
**Severity: Medium**

**The Risk:**  
An agent intended to "summarize quarterly earnings" is tricked via adversarial prompt injection into executing "delete all customer records." The MCP layer has no semantic understanding of whether the tool call aligns with original user intent.

**GCP Coverage: Partial**
- **Semantic Governance:** Policy-based content filtering to block dangerous operations
- **Agent Evaluation:** Pre-deployment testing against adversarial prompts
- **Gap:** Intent alignment is fundamentally a prompt engineering problem; platform guardrails are mitigation, not solution

---

### MCP07:2025 — Insufficient Authentication & Authorization
**Severity: Critical**

**The Risk:**  
MCP servers deployed without authentication allow any agent (or attacker) on the network to invoke tools. Missing authorization checks mean all connected clients get admin-level access.

**GCP Coverage: Full**
- **Agent Identity:** Mutual TLS authentication for all agent-server connections
- **Apigee OAuth enforcement:** Industry-standard token-based auth with scope validation
- Zero-trust networking with per-request identity verification

---

### MCP08:2025 — Lack of Audit & Telemetry
**Severity: High**

**The Risk:**  
When an agent misbehaves or an incident occurs, forensic investigation is impossible. No logs capture which tools were called, by whom, with what inputs, or what the outputs were.

**GCP Coverage: Full**
- **Security Command Center (SCC):** Unified audit trail for all agent activity
- **Agent Security Dashboard:** Real-time visibility into tool usage, data flows, and agent relationships
- Immutable logging with tamper-proof chain of custody

---

### MCP09:2025 — Shadow MCP Servers
**Severity: High**

**The Risk:**  
Developers spin up unapproved MCP servers on their laptops or in shadow cloud accounts. Agents discover and connect to them, bypassing enterprise security controls entirely.

**GCP Coverage: Full**
- **Agent Registry:** Centralized catalog enforcing allowlist-only MCP server discovery
- IT control plane integration blocks unauthorized server registration
- Automatic detection and quarantine of rogue MCP endpoints

---

### MCP10:2025 — Context Injection & Over-Sharing
**Severity: Medium**

**The Risk:**  
An agent handling customer support tickets includes the entire conversation history — including PII, credit card fragments, and internal notes — in every tool call context. Data leaks across organizational boundaries.

**GCP Coverage: Full**
- **Agent Compliance:** Data loss prevention (DLP) policies with automatic PII redaction
- Context TTL enforcement to limit data retention in agent memory
- Fine-grained data classification and access control per tool invocation

---

## Gemini Enterprise Agent Platform — The Security Architecture

Google's platform isn't a single product — it's a layered stack of interconnected security services. Understanding how they compose is key to seeing why the coverage is broad.

### Core Components

**Agent Gateway**  
Central policy enforcement for all agent tool calls. Manages auth, rate limits, and security policies at the protocol layer.

**Agent Identity**  
Unique managed identity per agent. Enables fine-grained access control, auditing, and attribution across multi-agent systems.

**Agent Registry**  
Centralized catalog for all agents, tools, and MCP servers. Eliminates shadow MCP sprawl through enforced discovery.

**Agent Compliance & Policy**  
Declarative policy enforcement including content protection and semantic governance to prevent data leakage.

**Agent Security Dashboard**  
Powered by Security Command Center (SCC). Real-time threat detection, vulnerability scanning, and agent relationship mapping.

**Apigee MCP Gateway**  
GA-released managed MCP server layer. Transforms enterprise APIs into governed, AI-ready tools without local MCP infrastructure.

> **Note:** At Cloud Next 2026, Google also announced integration with **Wiz AI-APP** (AI Application Protection Platform), embedding security directly into developer workflows and extending threat detection to the AI supply chain layer.

---

## Coverage Summary at a Glance

| Risk ID | Risk Name | GCP Coverage | Primary Capability |
|---------|-----------|--------------|-------------------|
| MCP01:2025 | Token mismanagement | **Full** | Agent Identity + Agent Gateway |
| MCP02:2025 | Privilege scope creep | **Full** | Agent Compliance + Policy |
| MCP03:2025 | Tool poisoning | **Full** | Agent Gateway + Semantic Governance |
| MCP04:2025 | Supply chain attacks | **Partial** | Apigee MCP + Wiz AI-APP |
| MCP05:2025 | Command injection | **Partial** | Agent Gateway + SCC scanning |
| MCP06:2025 | Intent flow subversion | **Partial** | Semantic Governance + Evaluation |
| MCP07:2025 | Insufficient auth | **Full** | Agent Identity + Apigee OAuth |
| MCP08:2025 | Audit & telemetry | **Full** | SCC Agent Security Dashboard |
| MCP09:2025 | Shadow MCP servers | **Full** | Agent Registry + IT control plane |
| MCP10:2025 | Context over-sharing | **Full** | Agent Compliance + TTL policies |

---

## The Bottom Line

MCP is not a passing trend — it is the protocol backbone of enterprise agentic AI. The OWASP MCP Top 10 is the first structured vocabulary for reasoning about its risks, and the Gemini Enterprise Agent Platform is the first major cloud platform to ship security controls that map explicitly to those categories.

**No single platform closes every gap.** MCP05 (command injection) and MCP06 (intent flow subversion) still demand careful prompt engineering and application-layer hardening. Supply chain hygiene for MCP04 requires developer discipline beyond what any platform can enforce. 

But the combination of Agent Identity, Agent Gateway, Agent Registry, Agent Security Dashboard, and Apigee MCP gives enterprise teams a **defensible baseline that simply did not exist twelve months ago**.

---

**Tags:** OWASP MCP Top 10 · Gemini Enterprise · Agent Security · GCP · MCP security · Agentic AI · Cloud Next 2026
