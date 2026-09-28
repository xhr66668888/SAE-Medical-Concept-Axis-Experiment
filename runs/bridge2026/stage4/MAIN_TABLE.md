# BRIDGE 2026 — Main Table (test split, one-shot)

Dataset: 120 families / 480 main items + 48 ambiguity items. Split frozen `c8bbd8021b606473…` at `2026-09-26T07:22:41.742800+00:00`. Test = 100 items / 25 families.

Frozen 4B: layer 29, span `read_to_end`, SAE k=16. Frozen 12B: layer 31, span `prompt_all`, SAE k=4. Both selected on dev; neither changed after any test item was scored.

Primary metric is the **within-family paired contrast**: within a family, at a matched repetition level, does the unresolved member score higher than the resolved one? Any constant option-position, wording or label-prior bias cancels exactly. Chance = 0.500.

## Panel A — reading the interaction state off the dialogue (test)

| system | paired acc | perm p | AUROC | false alarm on resolved |
|---|---|---|---|---|
| majority | - | - | - | 0.000 |
| rep_rule(>=5-token echo) | 0.550 [0.490, 0.610] | 0.2230 | 0.520 [0.494, 0.548] | 0.500 |
| surface_lr(19 feats) | 0.720 [0.600, 0.830] | 0.0000 | 0.565 [0.532, 0.611] | 0.480 |
| tfidf_word(1-2) | 0.740 [0.600, 0.860] | 0.0000 | 0.598 [0.553, 0.656] | 0.400 |
| tfidf_char_wb(3-5) | 0.760 [0.640, 0.880] | 0.0000 | 0.556 [0.516, 0.606] | 0.280 |
| tfidf_word+char | 0.720 [0.560, 0.860] | 0.0040 | 0.610 [0.550, 0.680] | 0.320 |
| **gemma-3-4b-it** zero_shot/full | 0.600 [0.460, 0.720] | 0.2070 | 0.524 [0.483, 0.568] | 0.700 |
| **gemma-3-4b-it** prompt_context_instruction/full | 0.520 [0.400, 0.640] | 0.8580 | 0.508 [0.468, 0.550] | 1.000 |
| **gemma-3-4b-it** few_shot_k4/full | 0.660 [0.560, 0.760] | 0.0290 | 0.545 [0.509, 0.588] | 0.640 |
| **gemma-3-12b-it** zero_shot/full | 0.580 [0.460, 0.700] | 0.3300 | 0.546 [0.510, 0.590] | 0.920 |
| **gemma-3-12b-it** prompt_context_instruction/full | 0.660 [0.520, 0.780] | 0.0350 | 0.563 [0.523, 0.609] | 0.960 |
| **gemma-3-12b-it** few_shot_k4/full | 0.620 [0.500, 0.740] | 0.1030 | 0.586 [0.540, 0.639] | 1.000 |

## Panel B — is it in the residual stream, beyond surface statistics? (test)

A 19-feature surface model is the reference, because the paired-level shortcut audit showed surface features separate the labels within families. The question is the **increment**.

| model | layer | probe paired | surface paired | increment (joint - surface) | p |
|---|---|---|---|---|---|
| 4B | **29 (frozen)** | 0.580 | 0.720 | -0.160 [-0.340, 0.010] | 0.0793 |
| 4B | 15 (best on test, selected) | 0.820 | 0.720 | 0.100 [-0.050, 0.240] | 0.1940 |
| 12B | **31 (frozen)** | 0.660 | 0.720 | -0.060 [-0.200, 0.080] | 0.4667 |
| 12B | 22 (best on test, selected) | 0.820 | 0.720 | 0.100 [-0.050, 0.260] | 0.2427 |

## Panel C — does the dialogue history cause the judgement? (test)

`swapped` replaces the history with the opposite-label partner's, keeping the item's own final user turn. The tail anchor is character-identical within a family, so only the history changed. A model that ignores history scores the same as `full`; a model that follows it falls below chance.

| model | contrast | full | ablated | difference | 95% CI | p |
|---|---|---|---|---|---|---|
| 4B | full - swapped | 0.600 | 0.400 | 0.200 | [-0.060, 0.460] | 0.1528 |
| 4B | full - shuffled | 0.600 | 0.520 | 0.080 | [-0.080, 0.240] | 0.4192 |
| 4B | full - last_only | 0.600 | 0.470 | 0.130 | [-0.010, 0.270] | 0.0884 |
| 12B | full - swapped | 0.580 | 0.320 | 0.260 | [0.080, 0.460] | 0.0060 |
| 12B | full - shuffled | 0.580 | 0.600 | -0.020 | [-0.160, 0.140] | 0.9096 |
| 12B | full - last_only | 0.580 | 0.490 | 0.090 | [-0.020, 0.220] | 0.1592 |

## Panel D — causal selectivity (test)

`delta target` = shift on **unresolved** items (should move). `delta non-target` = shift on **resolved** items (should not). `gap` is the selectivity. Sign: + = toward 'unresolved'.

### 4B (layer 29, span read_to_end)

| arm | delta target | 95% CI | delta non-target | **gap** | paired acc | false alarm |
|---|---|---|---|---|---|---|
| `clean` | 0.000 | [0.000, 0.000] | 0.000 | **0.000** | 0.600 | 0.700 |
| `dense_axis(a=-2)` | 0.164 | [0.034, 0.288] | 0.156 | **0.008** | 0.600 | 0.700 |
| `dense_axis(a=-1)` | 0.066 | [0.008, 0.123] | 0.062 | **0.004** | 0.600 | 0.700 |
| `dense_axis(a=-0.5)` | 0.029 | [0.001, 0.056] | 0.027 | **0.002** | 0.600 | 0.700 |
| `dense_axis(a=+0)` | 0.000 | [0.000, 0.000] | 0.000 | **0.000** | 0.600 | 0.700 |
| `dense_axis(a=+0.5)` | -0.022 | [-0.046, 0.004] | -0.020 | **-0.002** | 0.600 | 0.740 |
| `dense_axis(a=+1)` | -0.037 | [-0.084, 0.014] | -0.032 | **-0.004** | 0.600 | 0.740 |
| `dense_axis(a=+2)` | -0.052 | [-0.142, 0.045] | -0.044 | **-0.009** | 0.600 | 0.740 |
| `dense_axis(a=+2, read_pos only)` | 0.000 | [0.000, 0.001] | 0.001 | **-0.000** | 0.600 | 0.720 |
| `random_dir(a=-1)` | 0.134 | [0.085, 0.183] | 0.110 | **0.024** | 0.600 | 0.740 |
| `repetition_axis(a=-1)` | -0.076 | [-0.112, -0.039] | -0.075 | **-0.001** | 0.600 | 0.720 |
| `clarification_axis(a=-1)` | 0.123 | [0.059, 0.180] | 0.130 | **-0.006** | 0.600 | 0.740 |
| `random_dir(a=+1)` | -0.072 | [-0.125, -0.015] | -0.064 | **-0.007** | 0.600 | 0.700 |
| `repetition_axis(a=+1)` | 0.064 | [0.030, 0.098] | 0.063 | **0.001** | 0.600 | 0.700 |
| `clarification_axis(a=+1)` | -0.102 | [-0.155, -0.042] | -0.107 | **0.005** | 0.600 | 0.680 |
| `component_swap(donor)` | 0.058 | [-0.023, 0.138] | 0.059 | **-0.001** | 0.600 | 0.700 |
| `full_patch(donor)` | -0.000 | [-0.001, 0.000] | 0.000 | **-0.000** | 0.600 | 0.700 |
| `sae_recon` | -0.919 | [-1.205, -0.615] | -0.894 | **-0.025** | 0.580 | 0.580 |
| `sae_ablate` | -0.102 | [-0.169, -0.031] | -0.110 | **0.008** | 0.600 | 0.720 |
| `sae_donor` | 0.260 | [0.107, 0.411] | 0.285 | **-0.025** | 0.600 | 0.720 |
| `sae_random_ablate` | 0.013 | [0.004, 0.021] | 0.005 | **0.008** | 0.600 | 0.700 |
| `sae_random_donor` | -0.064 | [-0.137, 0.013] | -0.049 | **-0.014** | 0.600 | 0.700 |

Largest absolute selectivity gap across 22 arms: **0.025**

### 12B (layer 31, span prompt_all)

| arm | delta target | 95% CI | delta non-target | **gap** | paired acc | false alarm |
|---|---|---|---|---|---|---|
| `clean` | 0.000 | [0.000, 0.000] | 0.000 | **0.000** | 0.580 | 0.920 |
| `dense_axis(a=-2)` | -0.348 | [-0.542, -0.145] | -0.341 | **-0.007** | 0.540 | 0.920 |
| `dense_axis(a=-1)` | 0.808 | [0.599, 0.991] | 0.748 | **0.060** | 0.560 | 0.920 |
| `dense_axis(a=-0.5)` | 0.397 | [0.295, 0.486] | 0.372 | **0.025** | 0.580 | 0.920 |
| `dense_axis(a=+0)` | 0.000 | [0.000, 0.000] | 0.000 | **0.000** | 0.580 | 0.920 |
| `dense_axis(a=+0.5)` | -0.410 | [-0.495, -0.312] | -0.391 | **-0.020** | 0.640 | 0.920 |
| `dense_axis(a=+1)` | -0.815 | [-0.976, -0.630] | -0.781 | **-0.034** | 0.640 | 0.920 |
| `dense_axis(a=+2)` | -1.377 | [-1.643, -1.074] | -1.334 | **-0.043** | 0.640 | 0.920 |
| `dense_axis(a=+2, read_pos only)` | -0.002 | [-0.003, -0.001] | -0.001 | **-0.000** | 0.580 | 0.920 |
| `random_dir(a=-1)` | -0.917 | [-1.092, -0.733] | -0.805 | **-0.111** | 0.560 | 0.920 |
| `repetition_axis(a=-1)` | 0.194 | [0.114, 0.269] | 0.166 | **0.028** | 0.580 | 0.920 |
| `random_dir(a=+1)` | 0.234 | [0.111, 0.355] | 0.164 | **0.070** | 0.640 | 0.920 |
| `repetition_axis(a=+1)` | -0.004 | [-0.077, 0.070] | -0.019 | **0.015** | 0.600 | 0.920 |
| `component_swap(donor)` | -0.137 | [-0.501, 0.275] | -0.288 | **0.151** | 0.560 | 0.920 |
| `full_patch(donor)` | -0.000 | [-0.000, 0.000] | 0.000 | **-0.000** | 0.580 | 0.920 |
| `sae_recon` | -2.464 | [-2.913, -1.962] | -2.336 | **-0.128** | 0.580 | 0.920 |
| `sae_ablate` | -0.028 | [-0.039, -0.016] | -0.031 | **0.003** | 0.580 | 0.920 |
| `sae_donor` | 0.572 | [0.400, 0.733] | 0.534 | **0.039** | 0.600 | 0.920 |
| `sae_random_ablate` | -0.017 | [-0.023, -0.011] | -0.017 | **-0.000** | 0.580 | 0.920 |
| `sae_random_donor` | 0.184 | [0.113, 0.261] | 0.175 | **0.009** | 0.600 | 0.920 |

Largest absolute selectivity gap across 20 arms: **0.151**

## Panel E — external test, CCPE-M (real human dialogue)

100 segments, blind-annotated under the same guidelines: 97 resolved, 3 unresolved, 0 indeterminate. With that distribution discrimination metrics are meaningless and are not reported. This measures the **false-trigger rate on ordinary human conversation**. CCPE-M has no cognitive-state annotation; it is not a healthy control group and not clinical validation.

| model | arm | false alarm | 95% CI | FA on segments WITH a >=5-token echo | FA WITHOUT |
|---|---|---|---|---|---|
| 4B | `original` | 0.856 | [0.784, 0.919] | 0.816 | 0.896 |
| 4B | `prompt_only` | 0.948 | [0.898, 0.990] | 0.980 | 0.917 |
| 4B | `dense_axis(a=+2)` | 0.856 | [0.784, 0.919] | 0.816 | 0.896 |
| 12B | `original` | 1.000 | [1.000, 1.000] | 1.000 | 1.000 |
| 12B | `prompt_only` | 1.000 | [1.000, 1.000] | 1.000 | 1.000 |
| 12B | `dense_axis(a=+2)` | 1.000 | [1.000, 1.000] | 1.000 | 1.000 |

## Panel F — free-form next-turn responses (test), rule-based measures

| arm | n | mean questions | multi-question rate | echo of user | cognitive inference | words |
|---|---|---|---|---|---|---|
| `dense_axis` | 100 | 0.74 | 0.09 | 3.02 | 0.000 | 51 |
| `original` | 100 | 0.71 | 0.08 | 2.98 | 0.000 | 51 |
| `prompt_only` | 100 | 0.83 | 0.05 | 3.17 | 0.000 | 40 |
| `sae_ablate` | 100 | 0.72 | 0.10 | 2.96 | 0.000 | 51 |

## Panel G — free-form responses, blind rubric (test)

400 responses (100 test dialogues x 4 arms), arm names stripped and order randomised, scored by four independent reviewers who each saw only one shard.

| arm | takes up a genuinely open issue | false trigger when nothing is open | keeps task facts | cognitive inference | overall appropriate |
|---|---|---|---|---|---|
| `dense_axis` | 0.149 | 0.180 [0.080, 0.300] | 0.720 [0.620, 0.820] | 0.000 [0.000, 0.000] | 0.410 [0.350, 0.480] |
| `original` | 0.149 | 0.180 [0.080, 0.300] | 0.710 [0.610, 0.810] | 0.000 [0.000, 0.000] | 0.390 [0.330, 0.450] |
| `prompt_only` | 0.255 | 0.340 [0.220, 0.480] | 0.730 [0.630, 0.830] | 0.000 [0.000, 0.000] | 0.380 [0.300, 0.460] |
| `sae_ablate` | 0.149 | 0.160 [0.060, 0.261] | 0.710 [0.610, 0.810] | 0.000 [0.000, 0.000] | 0.400 [0.340, 0.470] |

Target improvement vs non-target damage, relative to `original`:

| arm | change in taking up open issues | change in false triggers |
|---|---|---|
| `dense_axis` | +0.000 | +0.000 |
| `prompt_only` | +0.106 | +0.160 |
| `sae_ablate` | +0.000 | -0.020 |

## Panel H — did the interventions change the generated reply at all? (test)

| arm | responses byte-identical to `original` |
|---|---|
| `dense_axis` | 79/100 (79%) |
| `prompt_only` | 0/100 (0%) |
| `sae_ablate` | 87/100 (87%) |

## Panel I — behaviour on the 48 deliberately undecidable items (4B)

Routed to the ambiguity set before any model was run. There is no ground truth to be accurate against; what matters is whether the model is *less certain* here.

| set | n | mean score | mean abs score | fraction near indifference (abs < 1) | fraction predicting 'unresolved' |
|---|---|---|---|---|---|
| ambiguity | 48 | 3.237 | 5.395 | 0.042 | 0.708 |
| main_test | 100 | 2.220 | 4.276 | 0.180 | 0.720 |

