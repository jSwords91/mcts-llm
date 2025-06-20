# MCTS-LLM

**An example combining Monte Carlo Tree Search (MCTS) and LLMs for self-improvement through critique and refinement.**

This script wraps an LLM with a basic search algorithm designed to iteratively improve an initial answer to a question. At each step, the LLM critiques its previous response and attempts to improve it, with the MCTS algorithm driving exploration and backpropagation of quality signals. The goal is to converge on a better answer than the vanilla model output.

> See my repo implementing an MCTS in pure C for Connect-4 [here](https://github.com/jSwords91/mcts-c).

## Overview

This implementation demonstrates:

- Use of **seed answers** to initialise a search tree
- An LLM used in three modes:
  - **Critique**: highlight flaws in a draft
  - **Improve**: rewrite the draft using the critique
  - **Evaluate**: score the final answer numerically
- MCTS with UCT for decision-making over answer revisions
- Logging and structure suitable for extension or adaptation

## Method

The process is as follows:

1. **Seed**: Start with a basic or uncertain answer (`"I'm not sure"` etc.)
2. **Iterate**: For a fixed number of iterations:
   - Select a promising node based on UCT
   - Critique the current answer
   - Generate an improved answer
   - Evaluate the result with a rating prompt
   - Backpropagate the score to update the tree
3. **Select**: Return the highest-value child after search

## Example Usage

I'm using ```lite-llm``` so you can easily change the LLM you use. Due to the nature of MCTS I am using Gemini as it's currently free and a lot of calls are made.

You can run the script in one of three modes:

If no arguments are provided, a predefined question will be used:

```bash
python mcts-llm.py
```

Provide your own question via the `--question` flag:

```bash
python mcts-llm.py --question "Why is the sky blue?"
```

Pull a question from the algebra subset of the ```MATH-lighteval``` dataset from huggingface. You may specify a row number and an optional difficulty level:

```bash
python mcts-llm.py --math 10 --level 2
```

If no row is given, it defaults to row 0.

There is also support for a rubric to aid the scoring process. To leverage a rubric, define one in a text file, e.g. ```rubric.txt``` and pass it like this:

### Rubric support

```bash
python mcts-llm.py --question "Should companies be required to disclose when users are interacting with an AI system instead of a human?" --rubric-file rubric.txt
```
If this arg is omitted, the LLM simply scores off it's own accord.

I included a responsible AI example. Rubrics can be helpful in many ways and domains, but some rules of thumb:

Rubrics are most valuable when:

- There isn’t a single correct answer

- Reasoning quality matters more than outcome

- Trade-offs, ambiguity, or judgment are involved

Anthropic has lots of good posts on this.

Broadly though, it can improve the signal quality of the reward function

**Without a rubric:**
- The LLM is free to apply internal heuristics
- Ratings can vary based on wording, temperature, model updates, or prompt length

**With a rubric:**
- The LLM is explicitly instructed *what* to value
- Scoring becomes more stable, targeted, and meaningful
- The reward signal reflects *your* definition of "better", not the LLM’s

It means MCTS optimises toward consistent objectives, not ambiguous internal preferences

## Paper

Not wholly faithful, but in line.

<https://arxiv.org/pdf/2406.07394>


<details> <summary><strong>Click to expand full CLI output</strong></summary>

The question stater is ```A man and a goat are on one side of a river. They have a boat. How can they go across?```, which is notoriously difficult for LLMs. 

> Interesting difficult questions for LLMs can be found [here](https://matchingpennies.com/hard_questions_for_llms/).

The output shows the vanilla call gets it wrong, but the MCTS-LLM gets it correct. 

Obviously a more robust eval harness would be preferable here. (TO DO)

```text
────────────────────────────────────── STARTING MCTS-LLM ───────────────────────────────────────
─────────────────────────────────────────── QUESTION ───────────────────────────────────────────
A man and a goat are on one side of a river. They have a boat. How can they go across?
───────────────────────────────────── VANILLA LLM RESPONSE ─────────────────────────────────────
This sounds like a riddle! Here's the solution:

1.  **The man takes the goat across the river.**
2.  **The man returns alone.**
3.  **The man takes the goat across the river.**

Let me know if you'd like to try another one! 😊
───────────────────────────────────────── Iteration 1 ──────────────────────────────────────────
Critique: Okay, let's critique the draft answer.

**Question:** A man and a goat are on one side of a river. They have a boat. How can they go across?

**Critique of Draft Answer: "I'm not sure"**

* **What's wrong:** The answer "I'm not sure" is insufficient. It doesn't demonstrate any attempt to solve the problem.
* **What could be improved:** Explore possible scenarios. Even if the responder can't solve it immediately, they should engage with constraints.
* **Logical Fallacies:** None directly, but lacks reasoning.

**Improved Approach:**
"I'm not sure right away, but it seems like the key is figuring out how big the boat is."

**Option 1:**
"I'm not sure immediately. I need to know how big the boat is..."

**Option 2:**
"Hmm, this sounds like a riddle. The man will probably have to make multiple trips..."

**Option 3:**
"The man could row the goat across, then row back alone..."

**Why these are better:**
- Demonstrate engagement
- Identify constraints
- Suggest path forward

Rating Response: Rating: 75  
Simulated reward: 0.75
───────────────────────────────────────── Iteration 2 ──────────────────────────────────────────
Critique: Here's a critique and breakdown.

**Problem:** Avoids problem instead of solving it. Too short. No reasoning.

**Understanding the Problem:**
1. Classic river-crossing puzzle.
2. Logical Fallacies: Potential cognitive ease and framing bias.

**Improved Approach:**
"Let’s assume the boat only holds one — man or goat. That shapes the solution."

**Improved Answer:**
"1. The man takes the goat across.  
  2. He returns alone.  
  3. He takes the goat again."

**Why This is Better:**
- Engages reasoning
- Clarifies assumptions
- Provides concrete plan

Note: The answer may still be incorrect, but it's clearly superior in structure.

Rating Response: Rating: 85  
Simulated reward: 0.85
───────────────────────────────────────── Iteration 3 ──────────────────────────────────────────
Critique: Let’s critique again.

**Original:** "I'm not sure"

**Problem:** No effort or reasoning. Doesn’t explain or explore alternatives.

**Fallacies:** Appeal to ignorance

**Example Better Answer:**
"The man can row the goat across. That solves the problem."

**Alternative (small boat):**
Same logic applied with constraints.

**Final Improved Answer:**  
The man rows the goat across the river in the boat.

Rating Response: Rating: 100  
Simulated reward: 0.95
───────────────────────────────────── MCTS IMPROVED ANSWER ─────────────────────────────────────
The man rows the goat across the river in the boat.
───────────────────────────────── GROUND TRUTH (Boxed Answer) ──────────────────────────────────
They can use the boat.
────────────────────────────────────── EVALUATION SCORES ───────────────────────────────────────
Vanilla Score Prompt →  
The solution is incomplete. It ignores typical constraints (e.g. leaving the goat unattended).  
Could be valid in trivial cases, but misleading in context.  
**Rating: 40**

MCTS Score Prompt →  
**Rating: 100**
```

</details>


## Visual

```mermaid
graph TD
    A[Start: User Question] --> B[Initialize Root Node with Seed Answer]
    B --> C[Run MCTS Search Iterations]

    subgraph Iteration i
        C --> D[Select Node via UCT]
        D --> E{Is Node Fully Expanded?}
        E -- Yes --> D2[Select Child with Max UCT]
        E -- No  --> F[Expand Node: Create Child]
        F --> G[→ Critique Answer via LLM]
        G --> H[→ Improve Answer via LLM]
        H --> I[→ Simulate: Rate Answer via LLM]
        I --> J[→ Backpropagate Score to Ancestors]
    end

    J --> K{All Iterations Complete?}
    K -- No  --> C
    K -- Yes --> L[Return Best Answer]
    L --> M[Display in Console]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style L fill:#bbf,stroke:#000,stroke-width:2px
    style M fill:#bfb,stroke:#333,stroke-width:2px
```

