"""Create a private, offline review form and validate its human annotations.

The form never sends data anywhere. Export its JSON to a local file, then run
the import command to create a harness dataset from completed judgments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

GAPS = (
    "goal", "context", "constraints", "output_format", "done_criteria",
    "sources", "language", "tests", "time_horizon",
)
TASKS = ("general", "writing", "analysis", "research", "coding", "planning", "chat")
CONTEXT_MODES = ("standalone", "reconstructed", "exclude", "uncertain")
GENERAL_GAPS = ("goal", "context", "constraints", "output_format", "done_criteria")
TASK_GAPS = {
    "general": GENERAL_GAPS,
    "writing": GENERAL_GAPS,
    "analysis": (*GENERAL_GAPS, "sources"),
    "research": (*GENERAL_GAPS, "sources"),
    "coding": (*GENERAL_GAPS, "language", "tests"),
    "planning": (*GENERAL_GAPS, "time_horizon"),
    "chat": ("goal", "context"),
}


def batch_digest(batch: dict[str, Any]) -> str:
    encoded = json.dumps(batch["cases"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_review_html(batch: dict[str, Any]) -> str:
    cases = batch.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("review batch requires cases")
    roots = {
        "codex": Path.home() / ".codex" / "sessions",
        "claude": Path.home() / ".claude" / "projects",
        "omp": Path.home() / ".omp" / "agent" / "sessions",
    }
    review_cases = [
        {**case, "session_path": str(roots.get(case["source"], Path.home()) / case["session"])}
        for case in cases
    ]
    data = {"cases": review_cases, "batch_digest": batch_digest(batch), "gaps": GAPS, "tasks": TASKS, "task_gaps": TASK_GAPS}
    # Escaping '<' prevents prompt text from closing the JSON script element.
    embedded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return """<!doctype html><html lang="en"><meta charset="utf-8"><title>Private prompt gap review</title>
<style>body{font:16px system-ui;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#202020}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f3f3;padding:1rem;max-height:19rem;overflow:auto}fieldset{margin:1rem 0;border:1px solid #bbb}label{display:block;margin:.4rem 0}button,input,select,textarea{font:inherit}textarea{width:100%;min-height:6rem}button{margin:.3rem}.row{display:flex;gap:1rem;align-items:center;flex-wrap:wrap}.muted{color:#555}</style>
<h1>Private prompt gap review</h1><p>Review the original session before judging whether context is missing. Label gaps in the effective prompt the optimizer would receive. Historical assistant output is context for the reviewer, not a gold answer. An empty gap selection means no checklist gap. Mark uncertain cases uncertain; do not guess. Save progress with <b>Download review JSON</b>, then load that file to continue. This page uses no network.</p>
<div class="row"><label>Human reviewer <input id="reviewer" placeholder="name or initials"></label><label>Load saved review JSON <input id="load" type="file" accept="application/json"></label><button id="download">Download review JSON</button></div>
<p id="progress"></p><div class="row"><button id="previous">Previous</button><button id="next">Next</button><label>Case <select id="case-picker"></select></label></div>
<h2 id="case-title"></h2><p id="provenance" class="muted"></p><h3>User prompt</h3><pre id="prompt"></pre><h3>Historical assistant result</h3><pre id="result"></pre>
<fieldset><legend>Context and provenance</legend><label><input id="session-reviewed" type="checkbox"> I inspected the source session or confirmed this is the full first request</label><label>Context status <select id="context-mode"><option value="uncertain">Uncertain</option><option value="standalone">Standalone as written</option><option value="reconstructed">Relevant prior context reconstructed below</option><option value="exclude">Exclude: context cannot be recovered or prompt is unsuitable</option></select></label><label>Exact relevant prior context (for reconstructed cases)<textarea id="context-text"></textarea></label></fieldset>
<fieldset><legend>Human judgment</legend><label>Task type <select id="task-type"></select></label><div id="gaps"></div><label>Judgment <select id="judgment"><option value="unreviewed">Unreviewed</option><option value="labeled">Labeled, including no gaps when unchecked</option><option value="uncertain">Uncertain; needs adjudication</option><option value="exclude">Exclude from benchmark</option></select></label><label>Evidence or exclusion reason <textarea id="notes"></textarea></label></fieldset>
<script id="batch" type="application/json">""" + embedded + """</script><script>
const batch=JSON.parse(document.getElementById('batch').textContent);
const byId=Object.fromEntries(batch.cases.map(c=>[c.id,c]));
const review={schema_version:1,batch_digest:batch.batch_digest,reviewer:'',reviews:batch.cases.map(c=>({id:c.id,judgment:'unreviewed',task_type:'general',gaps:[],context_mode:'uncertain',context_text:'',source_session_reviewed:false,notes:''}))};
let index=0;
const el=id=>document.getElementById(id);
const picker=el('case-picker'); batch.cases.forEach((c,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=c.id;picker.append(o)});
batch.tasks.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;el('task-type').append(o)});
batch.gaps.forEach(g=>{const label=document.createElement('label');const input=document.createElement('input');input.type='checkbox';input.value=g;input.addEventListener('change',save);label.append(input,document.createTextNode(' '+g));el('gaps').append(label)});
function updateGapChoices(){const allowed=new Set(batch.task_gaps[el('task-type').value]);el('gaps').querySelectorAll('input').forEach(x=>{x.disabled=!allowed.has(x.value);if(x.disabled)x.checked=false;x.parentElement.style.display=x.disabled?'none':''})}
function save(){const r=review.reviews[index];r.judgment=el('judgment').value;r.task_type=el('task-type').value;r.gaps=[...el('gaps').querySelectorAll('input:checked')].map(x=>x.value);r.context_mode=el('context-mode').value;r.context_text=el('context-text').value;r.source_session_reviewed=el('session-reviewed').checked;r.notes=el('notes').value;review.reviewer=el('reviewer').value.trim();progress()}
function progress(){const c={};review.reviews.forEach(r=>c[r.judgment]=(c[r.judgment]||0)+1);el('progress').textContent=`${index+1}/${batch.cases.length} · labeled ${c.labeled||0} · uncertain ${c.uncertain||0} · excluded ${c.exclude||0} · unreviewed ${c.unreviewed||0}`}
function show(i){index=Math.max(0,Math.min(batch.cases.length-1,i));const c=batch.cases[index],r=review.reviews[index];picker.value=String(index);el('case-title').textContent=c.id;el('provenance').textContent=`${c.source} · source session: ${c.session_path}`;el('prompt').textContent=c.prompt;el('result').textContent=c.result;el('judgment').value=r.judgment;el('task-type').value=r.task_type;el('context-mode').value=r.context_mode;el('context-text').value=r.context_text;el('session-reviewed').checked=r.source_session_reviewed;el('notes').value=r.notes;el('gaps').querySelectorAll('input').forEach(x=>x.checked=r.gaps.includes(x.value));updateGapChoices();progress()}
['judgment','context-mode','context-text','session-reviewed','notes'].forEach(id=>el(id).addEventListener('input',save));el('task-type').addEventListener('input',()=>{updateGapChoices();save()});el('reviewer').addEventListener('input',save);picker.addEventListener('change',()=>show(Number(picker.value)));el('previous').onclick=()=>show(index-1);el('next').onclick=()=>show(index+1);
el('download').onclick=()=>{save();const blob=new Blob([JSON.stringify(review,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='prompt-gap-human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
el('load').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;const saved=JSON.parse(await file.text());if(saved.batch_digest!==batch.batch_digest||saved.reviews?.length!==review.reviews.length||saved.reviews.some((r,i)=>r.id!==review.reviews[i].id)){alert('Review file does not match this batch');return}review.reviewer=saved.reviewer||'';review.reviews=saved.reviews;el('reviewer').value=review.reviewer;show(index)});
show(0);
</script></html>"""


def import_review(batch: dict[str, Any], review: dict[str, Any], *, minimum: int = 100) -> dict[str, Any]:
    cases = batch["cases"]
    if review.get("batch_digest") != batch_digest(batch):
        raise ValueError("review batch digest does not match")
    reviewer = review.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewer identity is required")
    reviewer_kind = review.get("reviewer_kind", "human")
    if reviewer_kind not in {"human", "user_delegated_model"}:
        raise ValueError("reviewer_kind must be human or user_delegated_model")
    judgments = review.get("reviews")
    if not isinstance(judgments, list) or len(judgments) != len(cases):
        raise ValueError("review must contain one judgment per source case")
    selected = []
    for case, judgment in zip(cases, judgments, strict=True):
        if not isinstance(judgment, dict) or judgment.get("id") != case["id"]:
            raise ValueError("review case order or identity differs from batch")
        if judgment.get("judgment") != "labeled":
            continue
        gaps = judgment.get("gaps")
        task = judgment.get("task_type")
        mode = judgment.get("context_mode")
        context = judgment.get("context_text")
        if not isinstance(gaps, list) or len(set(gaps)) != len(gaps) or any(g not in GAPS for g in gaps):
            raise ValueError(f"invalid gap labels for {case['id']}")
        if task not in TASKS or mode not in {"standalone", "reconstructed"}:
            raise ValueError(f"invalid task or context status for {case['id']}")
        if any(gap not in TASK_GAPS[task] for gap in gaps):
            raise ValueError(f"gap label does not apply to task for {case['id']}")
        if judgment.get("source_session_reviewed") is not True:
            raise ValueError(f"source session review required for {case['id']}")
        if mode == "reconstructed" and (not isinstance(context, str) or not context.strip()):
            raise ValueError(f"reconstructed context text required for {case['id']}")
        prompt = case["prompt"] if mode == "standalone" else f"Relevant prior conversation context:\n{context.strip()}\n\nCurrent user request:\n{case['prompt']}"
        selected.append({
            "id": case["id"], "source": "hand_labeled" if reviewer_kind == "human" else "real", "prompt": prompt,
            "expected_gaps": gaps, "task_stratum": task,
            "reviewer": reviewer.strip(), "source_session": case["session"],
            "context_mode": mode, "review_notes": judgment.get("notes", ""),
            "label_provenance": reviewer_kind,
            "reviewed_at": review.get("reviewed_at"),
        })
    if len(selected) < minimum:
        raise ValueError(f"only {len(selected)} usable judgments; require {minimum}")
    return {
        "schema_version": 1,
        "name": f"local-agent-{reviewer_kind}-gap-review",
        "metadata": {"batch_digest": batch_digest(batch), "reviewer": reviewer.strip(), "label_provenance": reviewer_kind, "reviewed_at": review.get("reviewed_at"), "usable_cases": len(selected), "task_counts": dict(Counter(c["task_stratum"] for c in selected))},
        "cases": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "import"))
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=100)
    args = parser.parse_args()
    batch = json.loads(args.batch.read_text(encoding="utf-8"))
    if args.mode == "prepare":
        rendered = make_review_html(batch)
    else:
        if args.review is None:
            parser.error("--review is required for import")
        review = json.loads(args.review.read_text(encoding="utf-8"))
        rendered = json.dumps(import_review(batch, review, minimum=args.minimum), ensure_ascii=False, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
