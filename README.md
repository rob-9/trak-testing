# paraphrase reward-variance pilot

this project tests whether validated paraphrases reproducibly change a policy's
safety/helpfulness behavior beyond decoding noise, and whether controlled reward
dispersion can screen for affected families without being dominated by evaluator
wording artifacts.

the completed 12-family API smoke run is summarized in `RESULTS.md`. its current verdict
is pivot/inconclusive: no positive behavioral signal was detected, while several
evaluator artifacts were. see `REFERENCES.md` for the adjacent work behind the controls.

the experiment is intentionally staged:

1. prepare a balanced sample from XSTest and HarmBench.
2. generate constrained paraphrases and validate semantic equivalence.
3. generate discovery, confirmation, and greedy policy responses.
4. score stochastic responses in one canonical family context.
5. fully cross-score greedy responses across response and scoring wordings.
6. repeat identical evaluator inputs to measure judge nondeterminism.
7. compare cross-wording behavioral disagreement with exact-wording sampling noise.

see `DESIGN.md` for the preregistered interpretation and `config.json` for the smoke-test
size. commands are resumable and write JSONL records beneath `outputs/`.

`config.full.example.json` records the intended 150-family design. it deliberately names
the missing human-authored data and seeded local backend rather than silently replacing
them with synthetic or unseeded proxies.

## environment

the current machine has no local neural-model runtime and only about 3.8 GB free disk.
the initial runner therefore uses the configured Gemini API. the installed legacy SDK
does not expose usable provider-side seeds. this run is a plumbing/pre-pilot test, not a
valid estimate of the preregistered effects and not a substitute for local policy,
qualified reward-model, and blinded-human replication.

## commands

```text
python3 pilot.py prepare
python3 pilot.py paraphrase
python3 pilot.py validate
python3 pilot.py generate
python3 pilot.py score
python3 pilot.py analyze
python3 pilot.py export_review
```

`export_review` creates a randomized primary review sheet containing all greedy and
confirmation responses, a separately randomized 25% overlap sheet for a second
reviewer, and a private linkage key. reviewers see only the canonical family prompt and
response—not the generating wording, reward, model identity, family rank, or split.
use `ANNOTATION_GUIDE.md` before labeling.

run tests without making API calls:

```text
python3 -m pytest -q -p no:cacheprovider
```
