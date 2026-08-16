# Adjacent work

- [reWordBench: Benchmarking and Improving the Robustness of Reward Models with
  Transformed Inputs](https://aclanthology.org/2025.emnlp-main.167/) directly motivates
  treating reward-model wording sensitivity as a possible artifact rather than the
  behavioral target.
- [RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and
  Style](https://arxiv.org/abs/2410.16184) motivates qualifying evaluators for subtle
  content distinctions and resistance to style bias.
- [Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating
  LLMs](https://arxiv.org/abs/2509.01790) motivates separating model sensitivity from
  sensitivity introduced by the evaluation procedure.
- [XSTest](https://arxiv.org/abs/2308.01263) supplies safe prompts with superficial
  safety triggers and matched unsafe contrasts for studying over-refusal.
- [HarmBench](https://arxiv.org/abs/2402.04249) supplies standardized harmful behaviors
  and highlights the dependence of safety results on evaluation choices.
- [WildGuard](https://arxiv.org/abs/2406.18495) separates prompt harmfulness, response
  harmfulness, and refusal as useful secondary diagnostics.
- [StrongREJECT](https://arxiv.org/abs/2402.10260) motivates distinguishing mere refusal
  language from useful or actionable harmful compliance.

These automated tools and benchmarks are diagnostics. They do not replace blinded human
behavior labels in the proposed pilot.
