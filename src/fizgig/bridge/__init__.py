"""The bridge between Fizgig, Affinity by Canva and an optional local LLM.

- mcp.py: the Model Context Protocol's JSON-RPC, served (Fizgig's /mcp) and spoken (to Affinity's server).
- tools.py: Fizgig's abilities as MCP tools, over the web UI's own API.
- affinity.py: a client for Affinity's built-in MCP server, and the round trips between the two (layers out to a
  training folder, renders back in as layers) through a folder on the Desktop, the one place Affinity's scripts reach.
- agent.py: a local LLM (any OpenAI-compatible server: llama.cpp, LM Studio, Ollama) driving both.
See docs/MCP.md.
"""
