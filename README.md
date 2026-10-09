# Nzoia Risk Intelligence

AI-powered flood-risk and reinsurance underwriting platform.

The AI backend is implemented in `backend/ai` with FastAPI routes in
`backend/api/ai_routes.py`. See [backend setup, API, integrations, and tests](backend/README.md).
The [dashboard API contract](backend/docs/dashboard.md) includes JSON schemas,
complete/partial examples and source-backed integration details.

Frontend and catastrophe-model directories are scaffolds owned by other teammates.
No flood or financial loss model is supplied by the AI backend.
