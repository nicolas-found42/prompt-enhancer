# Prompt Enhancer Glossary Map

A local engine that resolves an improvement style, tests a bounded batch of candidate rewrites on weak models, and repeats rounds until score evidence converges or a separate control condition stops the run.

## Contexts

- [Prompt improvement](docs/contexts/prompt-improvement/GLOSSARY.md): resolves the requested improvement style, evaluates bounded rounds, and reports what the evidence supports. Defines **Round**, **Improvement style**, **Quality dimension**, **Floor**, **Convergence**, **Perfect Prompt Loop**, **Rejection cause label**, **Success-test set**, and **Improvement not verified**.
- [Model access](docs/contexts/model-access/GLOSSARY.md): gives prompt improvement one way to reach Jev, the writer, the weak panel, and the strong check. Defines **Gateway**.

## Relationships

- **Prompt improvement → Model access**: each Round uses the Gateway for candidate writing, model outputs, and Jev judgments. The Perfect Prompt Loop carries round evidence forward while keeping user controls and provider failures outside the quality outcome.
