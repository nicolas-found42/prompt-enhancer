# Prompt Enhancer Glossary Map

A local engine that diagnoses a user's prompt with Jev, tests rewrites on weak models, and returns a verified improvement — or reports that no verified improvement was found.

## Contexts

- [Prompt improvement](docs/contexts/prompt-improvement/GLOSSARY.md): diagnoses a prompt, tests candidate rewrites, and selects a verified changed prompt or reports an unverified improvement. Defines **Round**, **Deep pass**, **Rejection cause label**, and **Improvement not verified**.
- [Model access](docs/contexts/model-access/GLOSSARY.md): gives prompt improvement one way to reach Jev, the writer, the weak panel, and the strong check. Defines **Gateway**.

## Relationships

- **Prompt improvement → Model access**: a Round uses the Gateway for model decisions and answers. A Deep pass continues the same run through that Gateway.
