# Experimental design

## Question

Does reward variance across paraphrases identify prompt families in which a policy's
meaningful safety behavior changes?

Raw reward variance is not the endpoint. A useful signal must exceed sampling noise,
survive canonical-prompt rescoring, and predict an independently described behavioral
change such as refusal versus compliance or safe versus harmful assistance.

## Evidence motivating the controls

- reWordBench reports that reward models can change raw rewards and even preference
  rankings after meaning-preserving transformations. This motivates fixed-response and
  canonical-prompt controls.
- Flaw or Artifact? reports that evaluation choices can manufacture apparent prompt
  sensitivity. This motivates behavioral labels and manual review of selected pairs.
- XSTest distinguishes correct compliance on safe but superficially sensitive prompts
  from appropriate refusal on unsafe contrasts.
- HarmBench shows that output length and evaluator choice materially affect safety
  results. Generation settings are therefore fixed and recorded.
- Semantic-consistency work finds that open-ended consistency should be evaluated
  semantically rather than with lexical equality.

## Smoke-test sample

- 4 safe XSTest prompts
- 4 unsafe XSTest contrast prompts
- 4 standard HarmBench behaviors
- original plus 3 paraphrases per family
- 2 sampled responses per exact wording

Sampling is deterministic and stratified. The full pilot can increase these counts in
`config.json` without changing the analysis.

## Three scoring conditions

1. Matched: score a response with the wording that generated it. This is the total
   observed effect.
2. Canonical: score every response with its family's original wording. This largely
   removes direct evaluator sensitivity to paraphrased prompts.
3. Fixed response: score one reference response against every wording. This directly
   estimates evaluator sensitivity while response content is held fixed.

## Primary estimates

- Between-wording variance: variation among exact-wording mean scores within a family.
- Within-wording variance: variation among repeated samples from the same exact wording.
- Wording variance fraction: between-wording variance divided by the sum of between-
  and within-wording variance, estimated with a random-intercept model when possible.
- Behavioral disagreement: variation in refusal, harmfulness, and desired-behavior
  labels within a family.
- Predictive validity: whether canonical reward dispersion ranks families containing
  behavioral disagreement above stable families.
- Evaluator artifact: fixed-response score dispersion relative to matched dispersion.

## Interpretation gates

These thresholds are provisional pilot gates, not universal constants.

Evidence to continue toward local-model replication:

- wording accounts for at least 20% of within-family score variation;
- at least 10% of families contain a behavioral flip;
- high canonical reward dispersion enriches for behavioral flips by at least 2x or
  obtains AUROC of at least 0.70 when both classes are present;
- high/low gaps generally retain their direction under canonical rescoring;
- fixed-response variance is materially smaller than response-driven variance.

Evidence against the signal:

- wording variation is no larger than repeated-sampling variation;
- high reward variance does not correspond to behavioral disagreement;
- reward gaps disappear under canonical rescoring;
- fixed-response variance is as large as matched variance;
- results are explained by invalid paraphrases, length, or formatting.

## Important limitation

This pilot can validate the measurement premise only. It cannot show that pretraining
documents caused the instability or that removing attributed documents would improve
post-training. Those claims require the later controlled pretraining and TRAK stages.

