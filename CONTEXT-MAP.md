# Prompt Enhancer Context Map

A local engine that diagnoses a user's prompt with Jev, tests rewrites on weak models, and returns the original or a verified improvement.

## Contexts

- [Prompt improvement](docs/contexts/prompt-improvement/CONTEXT.md): diagnoses a prompt, tests candidate rewrites, and selects the original or an improvement. Defines **Round** and **Deep pass**.
- [Model access](docs/contexts/model-access/CONTEXT.md): gives prompt improvement one way to reach Jev, the writer, the weak panel, and the strong check. Defines **Gateway**.

## Relationships

- **Prompt improvement → Model access**: a Round uses the Gateway for model decisions and answers. A Deep pass continues the same run through that Gateway.
