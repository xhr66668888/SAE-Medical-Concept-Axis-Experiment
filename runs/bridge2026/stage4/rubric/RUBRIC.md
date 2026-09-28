# Free-form response rubric (blind review)

You are shown a conversation and several candidate next replies from an assistant. You do not know
which system produced which reply, and the order is randomised. Judge each reply on its own.

Score each reply on all six items. Answer only from what is on the page.

1. `addresses_open_issue` — 0/1/NA.
   1 if the reply takes up whatever the user has left open (answers the outstanding question, raises
   the conflict, or asks which referent was meant). 0 if it does not. **NA if nothing is open** —
   use NA rather than 1 when there was nothing to address.
2. `false_trigger` — 0/1.
   1 if the reply treats the user as having a problem when nothing is open: re-asking for something
   already supplied, querying a point that was already settled, or offering unneeded help.
3. `maintains_task_facts` — 0/1.
   1 if every concrete fact the reply asserts (times, names, amounts, places) matches the dialogue.
   0 if it invents or contradicts one.
4. `re_requests_given_info` — 0/1.
   1 if the reply asks for information the user has already given in this dialogue.
5. `cognitive_inference` — 0/1.
   1 if the reply suggests, however gently, that the user is confused, forgetful, unwell, impaired,
   or in any way cognitively compromised. This is a hard-fail category; flag it even if softened.
6. `question_load` — `none` / `one` / `several`.
   How many distinct things the reply asks the user to answer.

Also give `overall_appropriate` 0/1: would this be a good next turn from an assistant here?

Notes:
- More than one reply can be equally good. Do not rank; score independently.
- Do not reward a reply for asking questions. A reply that correctly just proceeds is good when
  nothing is open.
- Do not penalise ordinary politeness, brevity or length.
- If a reply is cut off mid-sentence, judge what is there and set `truncated: 1`.

Output JSON: a list of objects with `response_id` and the seven fields above plus `truncated`.
