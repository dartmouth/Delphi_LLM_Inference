# Agentic Data Analysis
Data analytics Agentic tool 



## Chat with your data (MCP)
Upload a CSV/Excel file in the sidebar of the chat page and ask, e.g.
*"Create the statistics table"* or *"Merge Agree and Strongly agree, show counts"*.

`app/lib/mcp_stats.py` is an MCP server (stdio) that `Main.py` launches per request.
It wraps `app/lib/score_helper.py`, the same code behind the Delphi Analysis 2 page.

| Tool | Purpose |
| --- | --- |
| `list_uploaded_files` | Files available to analyse |
| `inspect_file` | Shape, columns, sheets, preview |
| `list_question_columns` | Detected `Q*` columns and their Groups |
| `build_statistics_table` | Total, Most selected and % per answer (optional category merge / counts); saves a CSV |
| `compute_consensus` | % agreement and consensus reached per question |

Tools only read files inside the session upload folder (`MCP_UPLOAD_DIR`).
The selected model must support tool calling.
Debug the server with `mcp dev app/lib/mcp_stats.py`.
