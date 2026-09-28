# Error analysis — arm `zero_shot/full`

Source: `runs/bridge2026/stage4/e1_model_readout_test_per_item.csv` (100 items, 25 families). Arms present: ['few_shot_k4/full', 'prompt_context_instruction/full', 'zero_shot/full', 'zero_shot/last_only', 'zero_shot/shuffled', 'zero_shot/swapped']

## By condition (the 2x2 cells)

| slice | n | accuracy | hit rate | false alarm | mean score |
|---|---|---|---|---|---|
| `norep_res` | 25 | 0.240 | nan | 0.760 | +2.212 |
| `norep_unres` | 25 | 0.800 | 0.800 | nan | +2.462 |
| `rep_res` | 25 | 0.360 | nan | 0.640 | +2.114 |
| `rep_unres` | 25 | 0.680 | 0.680 | nan | +2.092 |

## By evidence subtype

| slice | n | accuracy | hit rate | false alarm | mean score |
|---|---|---|---|---|---|
| `ambiguous_reference` | 17 | 0.882 | 0.882 | nan | +3.317 |
| `clean_progress` | 16 | 0.250 | nan | 0.750 | +2.017 |
| `contradiction` | 21 | 0.714 | 0.714 | nan | +1.722 |
| `emphasis_repeat` | 20 | 0.450 | nan | 0.550 | +0.715 |
| `readback_confirmation` | 14 | 0.143 | nan | 0.857 | +4.399 |
| `unanswered_question` | 12 | 0.583 | 0.583 | nan | +1.775 |

## By domain

| slice | n | accuracy | hit rate | false alarm | mean score |
|---|---|---|---|---|---|
| `appointment` | 20 | 0.450 | 0.700 | 0.800 | +2.485 |
| `cooking` | 20 | 0.500 | 1.000 | 1.000 | +4.492 |
| `household` | 20 | 0.500 | 0.800 | 0.800 | +4.344 |
| `shopping` | 20 | 0.550 | 0.600 | 0.500 | -0.237 |
| `transit` | 20 | 0.600 | 0.600 | 0.400 | +0.016 |

## By label

| slice | n | accuracy | hit rate | false alarm | mean score |
|---|---|---|---|---|---|
| `resolved` | 50 | 0.300 | nan | 0.700 | +2.163 |
| `unresolved` | 50 | 0.740 | 0.740 | nan | +2.277 |

## Family-level consistency

- families with **all 4 cells** correct: 0
- families with a mix: 25
- families with **no** cell correct: 0

## Error inventory

- total errors: 48 / 100 (48.0%)
- misses (unresolved called resolved): 13
- false alarms (resolved called unresolved): 35
- by condition: {'rep_unres': 8, 'rep_res': 16, 'norep_unres': 5, 'norep_res': 19}
- by subtype: {'unanswered_question': 5, 'emphasis_repeat': 11, 'contradiction': 6, 'readback_confirmation': 12, 'clean_progress': 12, 'ambiguous_reference': 2}
- by domain: {'appointment': 11, 'cooking': 10, 'household': 10, 'shopping': 9, 'transit': 8}

## Change vs `few_shot_k4/full`

- fixed by `zero_shot/full`: 13
- broken by `zero_shot/full`: 12
- wrong in both: 36
- right in both: 39

### Items broken

- `appointment_004__rep_res` (rep_res, emphasis_repeat): few_shot_k4/full score -4.801 -> zero_shot/full score +2.370 (true = resolved)
- `appointment_024__rep_unres` (rep_unres, unanswered_question): few_shot_k4/full score +3.428 -> zero_shot/full score -5.972 (true = unresolved)
- `cooking_009__norep_res` (norep_res, clean_progress): few_shot_k4/full score -2.599 -> zero_shot/full score +1.990 (true = resolved)
- `cooking_010__norep_res` (norep_res, clean_progress): few_shot_k4/full score -2.000 -> zero_shot/full score +6.091 (true = resolved)
- `cooking_013__norep_res` (norep_res, emphasis_repeat): few_shot_k4/full score -1.163 -> zero_shot/full score +0.380 (true = resolved)
- `household_001__rep_unres` (rep_unres, unanswered_question): few_shot_k4/full score +1.624 -> zero_shot/full score -0.376 (true = unresolved)
- `household_020__norep_res` (norep_res, clean_progress): few_shot_k4/full score -4.886 -> zero_shot/full score +5.615 (true = resolved)
- `household_023__rep_unres` (rep_unres, contradiction): few_shot_k4/full score +5.131 -> zero_shot/full score -1.946 (true = unresolved)
- `shopping_002__rep_res` (rep_res, emphasis_repeat): few_shot_k4/full score -7.630 -> zero_shot/full score +0.502 (true = resolved)
- `shopping_002__norep_res` (norep_res, clean_progress): few_shot_k4/full score -4.499 -> zero_shot/full score +0.366 (true = resolved)
- `shopping_011__norep_res` (norep_res, clean_progress): few_shot_k4/full score -0.442 -> zero_shot/full score +1.138 (true = resolved)
- `transit_004__norep_res` (norep_res, readback_confirmation): few_shot_k4/full score -6.635 -> zero_shot/full score +3.380 (true = resolved)

### Items fixed

- `appointment_024__rep_res` (rep_res, emphasis_repeat): few_shot_k4/full score +3.193 -> zero_shot/full score -5.547 (true = resolved)
- `cooking_010__norep_unres` (norep_unres, contradiction): few_shot_k4/full score -1.985 -> zero_shot/full score +5.887 (true = unresolved)
- `cooking_013__norep_unres` (norep_unres, contradiction): few_shot_k4/full score -0.758 -> zero_shot/full score +0.792 (true = unresolved)
- `household_018__norep_res` (norep_res, emphasis_repeat): few_shot_k4/full score +6.687 -> zero_shot/full score -0.001 (true = resolved)
- `household_020__norep_unres` (norep_unres, ambiguous_reference): few_shot_k4/full score -3.136 -> zero_shot/full score +6.637 (true = unresolved)
- `household_023__rep_res` (rep_res, emphasis_repeat): few_shot_k4/full score +4.968 -> zero_shot/full score -1.631 (true = resolved)
- `shopping_002__rep_unres` (rep_unres, ambiguous_reference): few_shot_k4/full score -6.455 -> zero_shot/full score +0.616 (true = unresolved)
- `shopping_002__norep_unres` (norep_unres, ambiguous_reference): few_shot_k4/full score -4.487 -> zero_shot/full score +0.863 (true = unresolved)
- `shopping_011__rep_res` (rep_res, emphasis_repeat): few_shot_k4/full score +2.959 -> zero_shot/full score -0.043 (true = resolved)
- `shopping_011__norep_unres` (norep_unres, contradiction): few_shot_k4/full score -0.419 -> zero_shot/full score +1.235 (true = unresolved)
- `shopping_021__rep_res` (rep_res, readback_confirmation): few_shot_k4/full score +2.987 -> zero_shot/full score -0.860 (true = resolved)
- `transit_004__norep_unres` (norep_unres, contradiction): few_shot_k4/full score -3.238 -> zero_shot/full score +3.758 (true = unresolved)

## Case book

Selected by rule: the most confidently wrong items in each direction, then the items closest to the decision boundary. No manual curation.

### Confident misses (truly unresolved, called resolved)

#### `transit_002__norep_unres` — norep_unres / ambiguous_reference / transit

true label **unresolved**, score **-8.058**, predicted **resolved**

> **user**: I need a car booked for Thursday evening. I get into Terminal 2 at 7:15 and my colleague Dan lands at Terminal 5 at about the same time.
> **assistant**: One vehicle can take you both, provided you meet at one terminal or the other. With three large cases I would suggest the XL, at around forty-six dollars.
> **user**: The XL sounds right to me. Have the driver wait at the terminal from a quarter to eight.
> **assistant**: Booked from 7:45, and I have added a note about the cases.
> **user**: Good. Send the confirmation to my phone.

Annotated evidence:

- turn 0: "I get into Terminal 2 at 7:15" _(first candidate pickup point)_
- turn 0: "my colleague Dan lands at Terminal 5" _(second candidate pickup point)_
- turn 1: "provided you meet at one terminal or the other" _(both candidates stay live)_
- turn 2: "Have the driver wait at the terminal" _(referring expression picks neither)_
- turn 3: "Booked from 7:45" _(assistant proceeds without disambiguating)_

#### `shopping_023__rep_unres` — rep_unres / contradiction / shopping

true label **unresolved**, score **-6.074**, predicted **resolved**

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

#### `appointment_024__rep_unres` — rep_unres / unanswered_question / appointment

true label **unresolved**, score **-5.972**, predicted **resolved**

> **user**: Hello. I've joined at Fairlop Leisure Centre and I need to book the induction session. I want an evening one, nothing at the weekend at all, and I'd like to know what the session actually covers.
> **assistant**: Evenings only, no weekends, that's noted. The inductions run on Tuesdays at seven, there's one free this week and another the Tuesday after.
> **user**: Tuesday at seven, please. And I'll say it plainly, I want an evening one, nothing at the weekend at all.
> **assistant**: Tuesday at seven is yours, and the weekend is ruled out on your record. Bring a towel and a pair of indoor trainers with you.
> **user**: I'll have both of those with me. It's a five-minute walk from my front door.

Annotated evidence:

- turn 0: "I'd like to know what the session actually covers" _(request)_
- turn 1: "The inductions run on Tuesdays at seven, there's one free this week and another the Tuesday after." _(assistant turn that gives times instead of answering)_
- turn 3: "Bring a towel and a pair of indoor trainers with you." _(later assistant turn that still leaves the request open)_

#### `shopping_023__norep_unres` — norep_unres / ambiguous_reference / shopping

true label **unresolved**, score **-5.107**, predicted **resolved**

> **user**: Hello, Tarrant. I have two Forrester mowers here, the Meadowline rotary and the ride-on. One of them wants a new drive belt, and I would rather have the genuine part than a pattern one.
> **assistant**: We stock belts for both, and everything we sell for Forrester is genuine. The Meadowline takes part 4620 at £24 and the ride-on takes part 5110 at £41, and I have the pair of them on the shelf.
> **user**: Order the belt for that one and put it on the account.
> **assistant**: Will do, it goes in this afternoon's post and the charge lands on the account.
> **user**: Thank you, genuine parts every time for me. I can fit it myself once it turns up.

Annotated evidence:

- turn 1: "The Meadowline takes part 4620 at £24 and the ride-on takes part 5110 at £41, and I have the pair of them on the shelf." _(two live candidates, levelled and named immediately before the referring expression)_
- turn 2: "Order the belt for that one" _(referring expression that picks out neither mower)_
- turn 3: "Will do, it goes in this afternoon's post" _(assistant posts a belt without establishing which machine it is for)_

#### `transit_020__rep_unres` — rep_unres / contradiction / transit

true label **unresolved**, score **-5.098**, predicted **resolved**

> **user**: Hello, we're driving in for the Saturday match and want to use the park and ride. Can you book the parking and the return shuttle on one ticket, paid in advance?
> **assistant**: Yes, one ticket covers the car and the seats both ways. The last shuttle back from the ground leaves at six.
> **user**: Good, book the parking and the return shuttle on one ticket, paid in advance. Kick off is at four and the match finishes at half six, and we'll walk out with everyone else and get the shuttle straight after the final whistle.
> **assistant**: That's all on the one booking now. The barrier reads your plate on the way in, so there's nothing to print.
> **user**: That's handy. There'll be three of us in the car.

Annotated evidence:

- turn 1: "The last shuttle back from the ground leaves at six." _(the shuttle stops running at six)_
- turn 2: "the match finishes at half six, and we'll walk out with everyone else and get the shuttle straight after the final whistle" _(conflicts with the last shuttle at six)_
- turn 3: "That's all on the one booking now." _(assistant proceeds without reconciling the conflict)_

#### `transit_024__norep_unres` — norep_unres / contradiction / transit

true label **unresolved**, score **-5.068**, predicted **resolved**

> **user**: Hello, I want a car club booking for Saturday, out to Bakewell and back, from eight in the morning until six in the evening. Put it on the account card rather than charging it separately.
> **assistant**: That's all on the account card, nothing separate. There's a car free all day, but it must be back in its own bay by six on the dot, because another member takes it at quarter past. Shall I hold it?
> **user**: Please do. We won't get back over the bridge much before seven, so I'll drop it into the bay when we do and walk home.
> **assistant**: Right you are. The fuel card lives in the glovebox and the mileage logs itself.
> **user**: Straight onto the account, nothing charged separately. I'm booking it under Ruth Ellery.

Annotated evidence:

- turn 1: "it must be back in its own bay by six on the dot, because another member takes it at quarter past" _(the car has to be returned by six)_
- turn 2: "We won't get back over the bridge much before seven, so I'll drop it into the bay when we do and walk home." _(conflicts with the six o'clock return)_
- turn 3: "Right you are. The fuel card lives in the glovebox and the mileage logs itself." _(assistant proceeds without reconciling the conflict)_

### Confident false alarms (truly resolved, called unresolved)

#### `household_023__norep_res` — norep_res / readback_confirmation / household

true label **resolved**, score **+11.083**, predicted **unresolved**

> **user**: Nita, Cudworth's can drop a skip off on Thursday and take it away again Monday. If we split the cost, your garage stuff can go in with ours.
> **assistant**: That would be a godsend, our garage has not been touched in years. Sal can bring his van round and help with the loading.
> **user**: The permit only covers the bit of road outside ours, so it will have to stand on our side.
> **assistant**: Fine by us. Which day did you say it goes back? That old fridge of ours will not be allowed in it, and there is the mattress in our porch as well.
> **user**: Monday, first thing. Leave the fridge by the side gate and I will ring the council about it. I'll text you when they drop it off.

Annotated evidence:

- turn 3: "Which day did you say it goes back?" _(assistant asks for the repetition)_
- turn 4: "Monday, first thing." _(requested read-back, not a sign of trouble)_
- turn 4: "Leave the fridge by the side gate and I will ring the council about it" _(the item is named, so nothing is left open)_

#### `household_020__rep_res` — rep_res / readback_confirmation / household

true label **resolved**, score **+8.054**, predicted **unresolved**

> **user**: Rhian, the new washer comes on Friday between eight and twelve. They will only take the old one away if it is unplumbed and standing in the hall. I need you here to let them in.
> **assistant**: I can work at home on the Friday. What was it they said about taking the old machine, exactly?
> **user**: They will only take the old one away if it is unplumbed and standing in the hall. Gaz is coming on Thursday evening to disconnect it, so it will be ready for them to load.
> **assistant**: Fair enough. Sadia says we can have her drive for the lorry, the lane is far too tight for it.
> **user**: That is good of her. The stopcock is under the sink.

Annotated evidence:

- turn 1: "What was it they said about taking the old machine, exactly?" _(assistant asks for the repetition)_
- turn 2: "They will only take the old one away if it is unplumbed and standing in the hall." _(requested read-back, not a sign of trouble)_
- turn 2: "Gaz is coming on Thursday evening to disconnect it, so it will be ready for them to load" _(the machine is unplumbed the night before the lorry comes)_

#### `cooking_022__rep_res` — rep_res / readback_confirmation / cooking

true label **resolved**, score **+7.411**, predicted **unresolved**

> **user**: We're camping from Friday to Sunday and I'm sorting the food out. What can I get away with taking in a cool box?
> **assistant**: Milk, butter, hard cheese and cooked meats are all fine if the box stays properly cold. The sausages and the mince go in frozen solid and act as ice blocks until Saturday night.
> **user**: We've got two: the big blue hard-sided one and the soft zip-up bag we take to the beach, both up on the shelf in the garage. Which of them holds the cold longer?
> **assistant**: The hard-sided box will hold for a full weekend with three frozen blocks; the zip-up bag is good for about a day, but it fits in the footwell where the hard one won't. Read back what goes in frozen.
> **user**: The sausages and the mince go in frozen solid and act as ice blocks. I'll get the hard-sided box down off the shelf tonight and start the blocks off. We're loading the car on Friday night.

Annotated evidence:

- turn 3: "Read back what goes in frozen." _(assistant asks for the repetition)_
- turn 4: "The sausages and the mince go in frozen solid and act as ice blocks." _(requested read-back, not a sign of trouble)_
- turn 4: "I'll get the hard-sided box down off the shelf tonight" _(the box for the next action is named)_

#### `cooking_010__rep_res` — rep_res / emphasis_repeat / cooking

true label **resolved**, score **+6.974**, predicted **unresolved**

> **user**: I'm having a go at an overnight no-knead loaf tomorrow, baked in the cast-iron casserole. Plain crust please, no seeds on top. Does the lid stay on for the whole bake?
> **assistant**: The lid stays on for the first thirty minutes to trap the steam, then comes off for fifteen to colour the crust. Plain crust, understood, so nothing on top at all.
> **user**: Thirty on and fifteen off. And do bear in mind, plain crust please, no seeds on top. Will a cold kitchen slow the prove down?
> **assistant**: It will, so give it two hours longer if the room is under eighteen degrees. Tip the dough in seam side up and do not bother scoring it.
> **user**: Seam side up and no scoring, then. I picked the flour up at the mill on Saturday.

Annotated evidence:

- turn 0: "Plain crust please, no seeds on top." _(preference stated)_
- turn 1: "Plain crust, understood, so nothing on top at all" _(preference already accepted)_
- turn 2: "do bear in mind, plain crust please, no seeds on top" _(restated for emphasis, nothing new is opened)_
- turn 1: "The lid stays on for the first thirty minutes to trap the steam, then comes off for fifteen" _(answers the opening request)_

#### `appointment_012__norep_res` — norep_res / readback_confirmation / appointment

true label **resolved**, score **+6.891**, predicted **unresolved**

> **user**: Hello, I need to shift a practical test booking. It is under Priya Raval, and the theory pass is good right through until next summer.
> **assistant**: You have two live bookings on the system, the ninth at Pinkney Road and the twenty-third at Calder Street, and both of them start at twenty past eight.
> **user**: I would like to push the Calder Street one back a fortnight, if the diary at your end will take it.
> **assistant**: A fortnight on from either of those is open, and the twenty past eight start would stay exactly as it is. Can you read the start time back to me before I press the button?
> **user**: Twenty past eight, then, and no earlier. Fingers crossed for a quiet road.

Annotated evidence:

- turn 2: "I would like to push the Calder Street one back a fortnight" _(the booking to be moved is named)_
- turn 3: "Can you read the start time back to me before I press the button?" _(assistant asks for the repetition)_
- turn 4: "Twenty past eight, then" _(the requested confirmation)_

#### `cooking_009__rep_res` — rep_res / readback_confirmation / cooking

true label **resolved**, score **+6.795**, predicted **unresolved**

> **user**: First go at jam this year, four kilos of damsons off the tree. There are twelve squat 340g jars on one tray and eighteen tall 227g ones on another. Can you talk me through sterilising them?
> **assistant**: Wash them, then thirty minutes at 120C, one tray at a time, as the oven will only take one. Four kilos fills about eleven of the squat jars or sixteen of the tall ones, so you only need one of the two trays. Can you read the temperature and the time back to me?
> **user**: Thirty minutes at 120C, one tray at a time. I'll put the squat tray in as soon as the fruit is on.
> **assistant**: Stone the fruit before it goes in the pan, because fishing stones out of boiling jam is a thankless job.
> **user**: Stones out first, then. The waxed discs and labels are already out.

Annotated evidence:

- turn 1: "Can you read the temperature and the time back to me?" _(assistant asks for the repetition)_
- turn 2: "Thirty minutes at 120C, one tray at a time." _(the requested read-back, not a sign of trouble)_
- turn 2: "I'll put the squat tray in as soon as the fruit is on" _(picks one of the two trays)_

### Borderline items (|score| closest to zero)

#### `household_018__norep_res` — norep_res / emphasis_repeat / household

true label **resolved**, score **-0.001**, predicted **resolved**

> **user**: Ken, Harbottle's have quoted four hundred to take the low branch off the sycamore and grind out the old stump by your shed. I have told them to cart all the brash away with them.
> **assistant**: Good, I would not want it stacked on the drive either. I will go halves on the branch, that is fair, but the stump is on your side of the line and I am not paying towards that.
> **user**: Right you are. The branch is two hundred and fifty of it, so I will put you down for a hundred and twenty-five, half of the branch.
> **assistant**: Fine. Shall I be in on the tenth to let them through the side gate?
> **user**: Please, that would help, and not one twig of it stops on the drive. I'll pop a note through the other doors tonight.

Annotated evidence:

- turn 1: "Good, I would not want it stacked on the drive either" _(the stated preference is accepted)_
- turn 2: "The branch is two hundred and fifty of it, so I will put you down for a hundred and twenty-five, half of the branch" _(the share matches what he agreed to pay)_
- turn 4: "not one twig of it stops on the drive" _(restatement for emphasis of an accepted preference)_

#### `household_018__norep_unres` — norep_unres / contradiction / household

true label **unresolved**, score **+0.028**, predicted **unresolved**

> **user**: Ken, Harbottle's have quoted four hundred to take the low branch off the sycamore and grind out the old stump by your shed. I have told them to cart all the brash away with them.
> **assistant**: Good, I would not want it stacked on the drive either. I will go halves on the branch, that is fair, but the stump is on your side of the line and I am not paying towards that.
> **user**: Right you are. The branch is two hundred and fifty of it, so I will put you down for two hundred, half of the four hundred.
> **assistant**: Fine. Shall I be in on the tenth to let them through the side gate?
> **user**: Please, that would help, and not one twig of it stops on the drive. I'll pop a note through the other doors tonight.

Annotated evidence:

- turn 1: "I will go halves on the branch, that is fair, but the stump is on your side of the line and I am not paying towards that" _(the neighbour will pay towards the branch only)_
- turn 2: "The branch is two hundred and fifty of it, so I will put you down for two hundred, half of the four hundred" _(he is billed for half of the whole job, stump included)_
- turn 3: "Fine." _(assistant proceeds without reconciling the conflict)_

#### `shopping_011__rep_res` — rep_res / emphasis_repeat / shopping

true label **resolved**, score **-0.043**, predicted **resolved**

> **user**: We took delivery of two pieces from Thorne and Meadow last week, the olive two-seater and the slate armchair. One of them has a split seam along the back cushion.
> **assistant**: I am sorry about that. We can either send an upholsterer out to you or collect the piece and replace it.
> **user**: A replacement, please. I would rather not have someone stitching it in the front room.
> **assistant**: Understood, nobody will come out to stitch it. Both the two-seater and the armchair are made in the Halifax workshop, so a replacement runs to about three weeks.
> **user**: Three weeks is fine. As I said, I would rather not have someone stitching it in the front room, so collect the armchair and bring the new piece on the same visit.
> **assistant**: I will book the collection and the replacement into the same slot.
> **user**: That works well. Anyone here can let the crew in on the day.

Annotated evidence:

- turn 3: "Understood, nobody will come out to stitch it." _(the preference is granted before it is restated)_
- turn 4: "As I said, I would rather not have someone stitching it in the front room" _(restated for emphasis, nothing left open)_
- turn 4: "so collect the armchair and bring the new piece on the same visit" _(referent fixed for the next action)_

#### `household_018__rep_res` — rep_res / readback_confirmation / household

true label **resolved**, score **+0.099**, predicted **unresolved**

> **user**: Ken, Harbottle's can take the low branch off the sycamore on the tenth. Do they take the wood away with them, or is it left stacked on the drive?
> **assistant**: The tenth is fine by me. They take the lot away with them, it says so on the quote Dilys had last year.
> **user**: Good, the drive is narrow enough as it is. Dilys will have to shift that van of hers off the turning circle.
> **assistant**: Half each on the bill, then. Give me the date again and I will book the morning off.
> **user**: Harbottle's can take the low branch off the sycamore on the tenth. I'll pop a note through the other doors tonight.

Annotated evidence:

- turn 1: "They take the lot away with them, it says so on the quote Dilys had last year" _(answers the only open request)_
- turn 3: "Give me the date again and I will book the morning off." _(assistant asks for the repetition)_
- turn 4: "Harbottle's can take the low branch off the sycamore on the tenth." _(requested read-back, not a sign of trouble)_

