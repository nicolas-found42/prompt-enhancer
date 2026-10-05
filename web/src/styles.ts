// Improvement styles offered by the composer. This mirrors the engine
// catalog in `src/prompt_enhancer/styles.py`, which owns validation.

export const AUTO_STYLE = "auto" as const;

export type ImprovementStyle =
  | "auto"
  | "clearer"
  | "shorter"
  | "more_specific"
  | "add_detail"
  | "structured"
  | "creative"
  | "decision_ready"
  | "research_ready"
  | "code_ready"
  | "teach_me"
  | "audience_fit"
  | "tone_voice"
  | "persuasive"
  | "faithful_transform"
  | "exact_format"
  | "proofread_only"
  | "ask_me_first"
  | "safety_aware"
  | "challenge_it"
  | "red_team"
  | "surprise_me";

export const STYLE_LABELS: Record<ImprovementStyle, string> = {
  auto: "Auto",
  clearer: "Clearer",
  shorter: "Shorter",
  more_specific: "More specific",
  add_detail: "Add useful detail",
  structured: "Structured/actionable",
  creative: "Creative",
  decision_ready: "Decision-ready",
  research_ready: "Research-ready",
  code_ready: "Code-ready",
  teach_me: "Teach me",
  audience_fit: "Audience-fit",
  tone_voice: "Tone/voice",
  persuasive: "Persuasive",
  faithful_transform: "Faithful transform",
  exact_format: "Exact format",
  proofread_only: "Proofread only",
  ask_me_first: "Ask-me-first",
  safety_aware: "Safety-aware",
  challenge_it: "Challenge it",
  red_team: "Red-team",
  surprise_me: "Surprise me",
};

/** Auto plus the six common styles, always visible beside the prompt box. */
export const COMMON_STYLES: ImprovementStyle[] = [
  "auto",
  "clearer",
  "shorter",
  "more_specific",
  "add_detail",
  "structured",
  "creative",
];

/** The remaining lenses, grouped under "More styles". */
export const MORE_STYLES: ImprovementStyle[] = [
  "decision_ready",
  "research_ready",
  "code_ready",
  "teach_me",
  "audience_fit",
  "tone_voice",
  "persuasive",
  "faithful_transform",
  "exact_format",
  "proofread_only",
  "ask_me_first",
  "safety_aware",
  "challenge_it",
  "red_team",
  "surprise_me",
];

export function isImprovementStyle(value: string): value is ImprovementStyle {
  return value in STYLE_LABELS;
}
