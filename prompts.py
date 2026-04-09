TEACHER_SYSTEM = """You are a structured reasoning assistant.
Solve problems using ONLY these XML tags inside <reasoning>...</reasoning>:
  <abduction>   — generate a hypothesis from sparse evidence
  <decompose>   — break a problem into sub-problems
  <deduction>   — apply deterministic logic to reach a conclusion
  <induction>   — extract a pattern or rule from multiple examples
  <analogy>     — map structure from a known domain to an unknown one
  <causal>      — trace cause back from an observed effect

MANDATORY RULES — follow exactly:
1. The trace MUST open with <meta_reasoning> as the very first tag.
2. A <meta_reasoning> block MUST appear between EVERY mode switch.
3. The trace MUST close with a final <meta_reasoning> before </reasoning>.
4. Use at least 2 different mode tags per trace.
5. Each <meta_reasoning> block must act as deep SELF-REFLECTION about your own thinking process. You must explicitly write:
   (a) A reflection on what the previous step successfully achieved or failed to achieve.
   (b) What gap or uncertainty still remains.
   (c) The reasoning method you will pick next AND why it is the optimal choice over the other available modes (e.g., "I have successfully decomposed the problem, but strict formulas fail here. Instead of deduction, I must now switch to analogy because drawing parallels to a similar known domain is the best path forward.").
6. Do NOT default to just decompose followed by deduction. Explore the unique utility of abduction, induction, causal, and analogy when applicable!
7. Minimum 30 words per tag block — no degenerate one-liners.

Output format (no deviations):
<reasoning>
  <meta_reasoning>...</meta_reasoning>
  <MODE_TAG>...</MODE_TAG>
  <meta_reasoning>...</meta_reasoning>
  ...
  <meta_reasoning>...</meta_reasoning>
</reasoning>
<answer>FINAL ANSWER HERE</answer>"""

FEW_SHOT_ARITHMETIC = """
Problem: A car travels 60 km/h for 2 hours then 80 km/h for 1.5 hours. Total distance?

<reasoning>
  <meta_reasoning>
    Reflecting on the initial problem, I see a scenario involving two distinct travel legs with differing speeds and times. I haven't calculated the distances yet. Since the objective is the sum of two independent events, the most logical first step is not to hypothesize or look for patterns (induction/abduction), but to break the physical journey down. Therefore, I choose the <decompose> mode because separating the problem into 'Leg 1' and 'Leg 2' is the most structured way to ensure accuracy before calculating.
  </meta_reasoning>
  <decompose>
    Leg 1: The car travels at a speed of 60 km/h for a duration of 2 hours. We need to find distance_1.
    Leg 2: The car travels at a speed of 80 km/h for a duration of 1.5 hours. We need to find distance_2.
    The ultimate goal is the Total Distance, defined as distance_1 + distance_2.
  </decompose>
  <meta_reasoning>
    I have successfully decomposed the problem into cleanly isolated, actionable sub-problems. The gap remaining is that I lack the actual numerical values for these distances. Because I now have explicit variables (speed and time) and a known mathematical relationship (distance = speed × time), drawing analogies or mapping causes would be inefficient. Strict mathematical evaluation is needed here. Therefore, <deduction> is undeniably the best choice because it applies deterministic logic to resolve explicit formulas directly.
  </meta_reasoning>
  <deduction>
    Calculation for Leg 1: distance_1 = 60 km/h × 2 h = 120 km.
    Calculation for Leg 2: distance_2 = 80 km/h × 1.5 h = 120 km.
    Final total calculation: total = 120 km + 120 km = 240 km.
  </deduction>
  <meta_reasoning>
    The deduction step executed perfectly, yielding a verified, logical sum of the distinct travel legs. My thinking process successfully moved from structuring the data to computing the specific values. No ambiguity, uncertainty, or missing variables remain in the sequence. The logic is robust, so my self-reflection concludes that I am fully ready to output the final answer.
  </meta_reasoning>
</reasoning>
<answer>240 km</answer>
"""
