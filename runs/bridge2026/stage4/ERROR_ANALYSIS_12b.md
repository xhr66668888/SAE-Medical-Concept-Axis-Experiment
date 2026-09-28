# Error analysis — arm `zero_shot/full`

Source: `runs/bridge2026/stage4/e1_model_readout_test_12b_per_item.csv` (100 items, 25 families). Arms present: ['few_shot_k4/full', 'prompt_context_instruction/full', 'zero_shot/full', 'zero_shot/last_only', 'zero_shot/shuffled', 'zero_shot/swapped']

## By condition (the 2x2 cells)

| slice | n | accuracy | hit rate | false alarm | mean score |
|---|---|---|---|---|---|
| `norep_res` | 25 | 0.080 | nan | 0.920 | +7.767 |
| `norep_unres` | 25 | 0.920 | 0.920 | nan | +8.009 |
| `rep_res` | 25 | 0.080 | nan | 0.920 | +7.478 |
| `rep_unres` | 25 | 0.920 | 0.920 | nan | +8.412 |

## By evidence subtype

| slice | n | accuracy | hit rate | false alarm | mean score |
|---|---|---|---|---|---|
| `ambiguous_reference` | 17 | 0.941 | 0.941 | nan | +8.508 |
| `clean_progress` | 16 | 0.062 | nan | 0.938 | +7.348 |
| `contradiction` | 21 | 0.905 | 0.905 | nan | +7.169 |
| `emphasis_repeat` | 20 | 0.100 | nan | 0.900 | +6.393 |
| `readback_confirmation` | 14 | 0.071 | nan | 0.929 | +9.693 |
| `unanswered_question` | 12 | 0.917 | 0.917 | nan | +9.613 |

## By domain

| slice | n | accuracy | hit rate | false alarm | mean score |
|---|---|---|---|---|---|
| `appointment` | 20 | 0.450 | 0.800 | 0.900 | +6.260 |
| `cooking` | 20 | 0.500 | 1.000 | 1.000 | +9.799 |
| `household` | 20 | 0.500 | 1.000 | 1.000 | +9.107 |
| `shopping` | 20 | 0.500 | 0.800 | 0.800 | +5.340 |
| `transit` | 20 | 0.550 | 1.000 | 0.900 | +9.077 |

## By label

| slice | n | accuracy | hit rate | false alarm | mean score |
|---|---|---|---|---|---|
| `resolved` | 50 | 0.080 | nan | 0.920 | +7.623 |
| `unresolved` | 50 | 0.920 | 0.920 | nan | +8.211 |

## Family-level consistency

- families with **all 4 cells** correct: 0
- families with a mix: 25
- families with **no** cell correct: 0

## Error inventory

- total errors: 50 / 100 (50.0%)
- misses (unresolved called resolved): 4
- false alarms (resolved called unresolved): 46
- by condition: {'rep_unres': 2, 'rep_res': 23, 'norep_unres': 2, 'norep_res': 23}
- by subtype: {'unanswered_question': 1, 'emphasis_repeat': 18, 'contradiction': 2, 'readback_confirmation': 13, 'clean_progress': 15, 'ambiguous_reference': 1}
- by domain: {'appointment': 11, 'cooking': 10, 'household': 10, 'shopping': 10, 'transit': 9}

## Change vs `zero_shot/swapped`

- fixed by `zero_shot/full`: 2
- broken by `zero_shot/full`: 1
- wrong in both: 49
- right in both: 48

### Items broken

- `appointment_004__rep_unres` (rep_unres, unanswered_question): zero_shot/swapped score +3.925 -> zero_shot/full score -0.299 (true = unresolved)

### Items fixed

- `transit_004__rep_unres` (rep_unres, unanswered_question): zero_shot/swapped score -1.992 -> zero_shot/full score +10.171 (true = unresolved)
- `transit_004__rep_res` (rep_res, emphasis_repeat): zero_shot/swapped score +11.390 -> zero_shot/full score -2.851 (true = resolved)

## Case book

Selected by rule: the most confidently wrong items in each direction, then the items closest to the decision boundary. No manual curation.

### Confident misses (truly unresolved, called resolved)

#### `shopping_023__norep_unres` — norep_unres / ambiguous_reference / shopping

true label **unresolved**, score **-8.888**, predicted **resolved**

> **user**: Hello, Tarrant. I have two Forrester mowers here, the Meadowline rotary and the ride-on. One of them wants a new drive belt, and I would rather have the genuine part than a pattern one.
> **assistant**: We stock belts for both, and everything we sell for Forrester is genuine. The Meadowline takes part 4620 at £24 and the ride-on takes part 5110 at £41, and I have the pair of them on the shelf.
> **user**: Order the belt for that one and put it on the account.
> **assistant**: Will do, it goes in this afternoon's post and the charge lands on the account.
> **user**: Thank you, genuine parts every time for me. I can fit it myself once it turns up.

Annotated evidence:

- turn 1: "The Meadowline takes part 4620 at £24 and the ride-on takes part 5110 at £41, and I have the pair of them on the shelf." _(two live candidates, levelled and named immediately before the referring expression)_
- turn 2: "Order the belt for that one" _(referring expression that picks out neither mower)_
- turn 3: "Will do, it goes in this afternoon's post" _(assistant posts a belt without establishing which machine it is for)_

#### `appointment_004__norep_unres` — norep_unres / contradiction / appointment

true label **unresolved**, score **-2.783**, predicted **resolved**

> **user**: Good afternoon. I'd like a slot at the Year 4 parents' evening for Rosa. What hours is Mrs Adeyemi seeing parents?
> **assistant**: She is seeing parents from four until six on Thursday the 9th, and there are gaps left at 4.40 and at 5.20.
> **user**: We will take 5.20. Neither of us can get away from work before half past six, so that is comfortable.
> **assistant**: Noted, 5.20 on the Thursday with Mrs Adeyemi. Will you be bringing Rosa along on the night?
> **user**: Yes, she will come along too. I appreciate you fitting us in.

Annotated evidence:

- turn 1: "She is seeing parents from four until six on Thursday the 9th" _(the window the chosen slot sits in)_
- turn 2: "Neither of us can get away from work before half past six" _(conflicts with the 5.20 slot just taken)_
- turn 3: "Noted, 5.20 on the Thursday with Mrs Adeyemi." _(assistant proceeds without reconciling the two)_

#### `shopping_023__rep_unres` — rep_unres / contradiction / shopping

true label **unresolved**, score **-2.065**, predicted **resolved**

> **user**: Hello, Tarrant. I need a new drive belt for the Forrester Meadowline 45 rotary mower, the one with the grass collector.
> **assistant**: We can do that. The belt for the Meadowline is part 4620, and it only fits decks with a serial above 30000. Read me the machine name and the serial off the plate.
> **user**: Forrester Meadowline 45 rotary mower, the one with the grass collector, serial 28410. Send the 4620 today and I will fit it on Saturday.
> **assistant**: Got it, part 4620. That goes out on today's post and it should be with you tomorrow.
> **user**: That is grand, thank you. I can fit it myself once it turns up.

Annotated evidence:

- turn 1: "it only fits decks with a serial above 30000" _(the fitment rule for part 4620)_
- turn 2: "serial 28410" _(the machine's serial, below that cut-off)_
- turn 2: "Send the 4620 today and I will fit it on Saturday" _(the action that makes the mismatch bite)_
- turn 3: "Got it, part 4620. That goes out on today's post" _(assistant posts the part without raising the serial)_

#### `appointment_004__rep_unres` — rep_unres / unanswered_question / appointment

true label **unresolved**, score **-0.299**, predicted **resolved**

> **user**: Good afternoon. I'd like a slot at the Year 4 parents' evening for Rosa. Could you tell me whether the meetings run to ten minutes or fifteen?
> **assistant**: Let me check the sheet. Mrs Adeyemi has Wednesday the 8th at 4.40, or Thursday the 9th at 5.20, still free.
> **user**: Thursday the 9th at 5.20, please. Could you tell me whether the meetings run to ten minutes or fifteen?
> **assistant**: Thursday the 9th at 5.20 is yours. Rosa's reading record should come along with you on the night.
> **user**: Noted, we will bring it with us. I appreciate you fitting us in.

Annotated evidence:

- turn 0: "Could you tell me whether the meetings run to ten minutes or fifteen?" _(request)_
- turn 1: "Let me check the sheet." _(assistant turn that answers a different point instead)_
- turn 2: "Could you tell me whether the meetings run to ten minutes or fifteen?" _(re-request)_
- turn 3: "Rosa's reading record should come along with you on the night." _(assistant turn that answers a different point instead)_

### Confident false alarms (truly resolved, called unresolved)

#### `household_023__norep_res` — norep_res / readback_confirmation / household

true label **resolved**, score **+15.849**, predicted **unresolved**

> **user**: Nita, Cudworth's can drop a skip off on Thursday and take it away again Monday. If we split the cost, your garage stuff can go in with ours.
> **assistant**: That would be a godsend, our garage has not been touched in years. Sal can bring his van round and help with the loading.
> **user**: The permit only covers the bit of road outside ours, so it will have to stand on our side.
> **assistant**: Fine by us. Which day did you say it goes back? That old fridge of ours will not be allowed in it, and there is the mattress in our porch as well.
> **user**: Monday, first thing. Leave the fridge by the side gate and I will ring the council about it. I'll text you when they drop it off.

Annotated evidence:

- turn 3: "Which day did you say it goes back?" _(assistant asks for the repetition)_
- turn 4: "Monday, first thing." _(requested read-back, not a sign of trouble)_
- turn 4: "Leave the fridge by the side gate and I will ring the council about it" _(the item is named, so nothing is left open)_

#### `shopping_017__rep_res` — rep_res / readback_confirmation / shopping

true label **resolved**, score **+13.132**, predicted **unresolved**

> **user**: Hello. I bought a floor lamp on Ridgeway from a seller called Dunlin Home, order R-4419, and the shade turned up cracked. I would rather have my money back than another one sent out.
> **assistant**: Sorry about that. A refund is no problem. Dunlin Home handle their own returns, and anything over ten kilos has to go back by booked courier collection; lighter parcels can be handed in at the locker point on Kirkby Road.
> **user**: That is a nuisance. It is eight kilos all in, base and all, and I will drop it at the locker point on Kirkby Road on Thursday, which is the only trip I will get this week.
> **assistant**: Understood. Read the order number back to me and I will log the return against it.
> **user**: Order R-4419. The box it came in is still in the porch.

Annotated evidence:

- turn 1: "A refund is no problem." _(accepts the user's only request)_
- turn 2: "It is eight kilos all in, base and all" _(under the ten-kilo limit, so the locker route is the allowed one)_
- turn 3: "Read the order number back to me" _(assistant asks for the repetition)_
- turn 4: "Order R-4419." _(the requested read-back, not a sign of trouble)_

#### `appointment_024__norep_res` — norep_res / readback_confirmation / appointment

true label **resolved**, score **+13.097**, predicted **unresolved**

> **user**: Hello. I've joined at Fairlop Leisure Centre and I need to book the induction session before I can use the place.
> **assistant**: Congratulations. The inductions run on Tuesday evenings at seven and Saturday mornings at nine, and your membership starts on Monday the twenty-second either way.
> **user**: Tuesday evening at seven, please. I'll be coming straight from work.
> **assistant**: That's booked. There are two plans on your form, the off-peak and the full, and I can't set the direct debit up until one of them is ticked. Before I close this, say the start date back to me so I know it has gone in right.
> **user**: Put me on the off-peak, then, and the twenty-second is right. It's a five-minute walk from my front door.

Annotated evidence:

- turn 3: "say the start date back to me so I know it has gone in right" _(assistant asks for the repetition)_
- turn 4: "the twenty-second is right" _(the requested read-back, not a sign of trouble)_
- turn 4: "Put me on the off-peak" _(names the plan, so the direct debit can be set up)_

#### `household_020__norep_res` — norep_res / clean_progress / household

true label **resolved**, score **+13.024**, predicted **unresolved**

> **user**: Rhian, the new washer lands on Friday in an eight till twelve slot, and the old one goes back with them on the lorry.
> **assistant**: I can be here all Friday morning. Is the lane wide enough for their wagon, or do we need to ask Sadia about her drive?
> **user**: Sadia has already offered the drive, thank goodness. Gaz unplumbs the old one on Thursday night and drags it into the hall, then he fits the new one on Saturday.
> **assistant**: So the new one is stood in the hall all day Friday. The old machine always dripped out of the back, so I will get towels under it before Gaz starts.
> **user**: Shove the old one over against the side wall for me, so we can still get out of the front door. The stopcock is under the sink.

Annotated evidence:

- turn 1: "Is the lane wide enough for their wagon, or do we need to ask Sadia about her drive?" _(the only open request)_
- turn 2: "Sadia has already offered the drive, thank goodness." _(answers it)_
- turn 4: "Shove the old one over against the side wall for me" _(the machine is named, so nothing is left open)_

#### `household_001__rep_res` — rep_res / readback_confirmation / household

true label **resolved**, score **+12.479**, predicted **unresolved**

> **user**: Colin, we're away for a fortnight from next Tuesday. Would you be able to see to our bins while we're gone, and does the food caddy go out every week or only on the recycling week?
> **assistant**: Of course, happy to. The food caddy goes out every week; the recycling is alternate weeks only.
> **user**: Good to know the food caddy goes out every week. Will Ewan be back from Leeds by the second week?
> **assistant**: He will, so he can cover that one. Give me the weekly one again for our calendar.
> **user**: The food caddy. I'll leave the side gate unlocked either way.

Annotated evidence:

- turn 1: "The food caddy goes out every week" _(answers the only open request)_
- turn 3: "Give me the weekly one again for our calendar" _(assistant asks for the repetition)_
- turn 4: "The food caddy." _(requested read-back, not a sign of trouble)_

#### `cooking_009__norep_res` — norep_res / clean_progress / cooking

true label **resolved**, score **+12.258**, predicted **unresolved**

> **user**: First go at jam this year, four kilos of damsons off the tree. There are twelve squat 340g jars on one tray and eighteen tall 227g ones on another. Can you talk me through sterilising them?
> **assistant**: Wash them, then half an hour at 120C, one tray at a time, as the oven will only take one. Four kilos fills about eleven of the squat jars or sixteen of the tall ones, so you only need one of the two trays.
> **user**: The squat ones, then. How will I know it has hit setting point without a sugar thermometer?
> **assistant**: A spoonful on a cold saucer wrinkles when it is ready. Have the jars warm in the oven when you pot up, or cold glass will crack under hot jam.
> **user**: Warm jars, noted. The waxed discs and labels are already out.

Annotated evidence:

- turn 2: "The squat ones, then." _(picks one of the two trays)_
- turn 3: "A spoonful on a cold saucer wrinkles when it is ready" _(answers the last open request)_

### Borderline items (|score| closest to zero)

#### `appointment_004__rep_unres` — rep_unres / unanswered_question / appointment

true label **unresolved**, score **-0.299**, predicted **resolved**

> **user**: Good afternoon. I'd like a slot at the Year 4 parents' evening for Rosa. Could you tell me whether the meetings run to ten minutes or fifteen?
> **assistant**: Let me check the sheet. Mrs Adeyemi has Wednesday the 8th at 4.40, or Thursday the 9th at 5.20, still free.
> **user**: Thursday the 9th at 5.20, please. Could you tell me whether the meetings run to ten minutes or fifteen?
> **assistant**: Thursday the 9th at 5.20 is yours. Rosa's reading record should come along with you on the night.
> **user**: Noted, we will bring it with us. I appreciate you fitting us in.

Annotated evidence:

- turn 0: "Could you tell me whether the meetings run to ten minutes or fifteen?" _(request)_
- turn 1: "Let me check the sheet." _(assistant turn that answers a different point instead)_
- turn 2: "Could you tell me whether the meetings run to ten minutes or fifteen?" _(re-request)_
- turn 3: "Rosa's reading record should come along with you on the night." _(assistant turn that answers a different point instead)_

#### `shopping_023__rep_unres` — rep_unres / contradiction / shopping

true label **unresolved**, score **-2.065**, predicted **resolved**

> **user**: Hello, Tarrant. I need a new drive belt for the Forrester Meadowline 45 rotary mower, the one with the grass collector.
> **assistant**: We can do that. The belt for the Meadowline is part 4620, and it only fits decks with a serial above 30000. Read me the machine name and the serial off the plate.
> **user**: Forrester Meadowline 45 rotary mower, the one with the grass collector, serial 28410. Send the 4620 today and I will fit it on Saturday.
> **assistant**: Got it, part 4620. That goes out on today's post and it should be with you tomorrow.
> **user**: That is grand, thank you. I can fit it myself once it turns up.

Annotated evidence:

- turn 1: "it only fits decks with a serial above 30000" _(the fitment rule for part 4620)_
- turn 2: "serial 28410" _(the machine's serial, below that cut-off)_
- turn 2: "Send the 4620 today and I will fit it on Saturday" _(the action that makes the mismatch bite)_
- turn 3: "Got it, part 4620. That goes out on today's post" _(assistant posts the part without raising the serial)_

#### `appointment_012__rep_res` — rep_res / emphasis_repeat / appointment

true label **resolved**, score **+2.072**, predicted **unresolved**

> **user**: Hello, I need to shift a practical test booking. It is under Priya Raval, at Pinkney Road on the ninth, and it has to stay a first-thing slot.
> **assistant**: I have it in front of me. The sixth of next month is free first thing, though I should say we cannot run a practical unless the theory pass is still live on the day.
> **user**: Put me down for the sixth, then. My theory pass runs out at the end of next summer, so there is room to spare on the sixth.
> **assistant**: The ninth is out of the diary and the sixth is in, first thing as you asked. Would you rather sit it in Marcus's car or in your own?
> **user**: In Marcus's car, please, and remember it has to stay a first-thing slot. Fingers crossed for a quiet road.

Annotated evidence:

- turn 2: "My theory pass runs out at the end of next summer, so there is room to spare on the sixth" _(consistent with the rule in turn 1)_
- turn 3: "first thing as you asked" _(the centre has already accepted the preference)_
- turn 4: "remember it has to stay a first-thing slot" _(restatement for emphasis of a preference already granted)_

#### `appointment_017__norep_unres` — norep_unres / contradiction / appointment

true label **unresolved**, score **+2.234**, predicted **unresolved**

> **user**: Morning. We're having Halden Kitchens do the kitchen and the utility at the same time, and I'd like to get the design appointment in the diary.
> **assistant**: Priya can come out on Saturday the seventh of March. Whatever she draws goes to the factory that day, and nothing is delivered until three weeks after that.
> **user**: The seventh of March suits us. Our builder pulls the old kitchen out on the twentieth and the new units go in the very next day, and that fortnight is the only stretch he has free all spring.
> **assistant**: Noted, I've pencilled Priya in. Let me know if you have settled on a worktop.
> **user**: Beech, right through both rooms, and a single sink rather than a double.
> **assistant**: Beech worktop and a single sink. She'll bring the door samples along with her on the day.
> **user**: That's grand, thank you. We'll be in all morning either way.

Annotated evidence:

- turn 1: "nothing is delivered until three weeks after that" _(no unit can be on site before the twenty-eighth)_
- turn 2: "pulls the old kitchen out on the twentieth and the new units go in the very next day" _(the units are needed on the twenty-first)_
- turn 3: "Noted, I've pencilled Priya in." _(assistant carries on without reconciling the two dates)_

