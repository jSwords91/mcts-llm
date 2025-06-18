import math
import random
import re
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LITELLM_DISABLE_SPEND_TRACKING"] = "true"  # makes cost calc a no-op
os.environ["LITELLM_LOG"] = "ERROR"  # keep the logger quiet

import pandas as pd
from datasets import load_dataset
from litellm import completion
from rich.console import Console

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
MODEL = "gemini/gemini-2.0-flash"
ITERATIONS = 3
MAX_CHILDREN = 3
SEED_ANSWERS = ["I don't know the answer", "I'm not sure", "I can't say"]

console = Console()

def query_llm(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        response = completion(model=MODEL, messages=messages)
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        console.print(f"[red]LLM Error:[/red] {e}")
        return ""

def get_critique_prompt(question: str, draft: str) -> str:
    return f"""Question: {question}\nDraft Answer: {draft}\nPlease critique the draft and explain what is wrong or could be improved. Re-state the question verbatim to ensure understanding, do not add any additional assumptions. Think from first-principles and highlight any logical fallacies."""

def get_improvement_prompt(question: str, draft: str, critique: str) -> str:
    return f"""Question: {question}\nDraft Answer: {draft}\nCritique: {critique}\nPlease rewrite the answer to improve it."""

def get_rating_prompt(question: str, answer: str) -> str:
    return f"""Question: {question}\nAnswer: {answer}\nEvaluate the answer and return a rating from 0 to 100. Format: Rating: <number>"""

class Node:
    def __init__(self, question: str, answer: str, parent=None):
        self.question = question
        self.answer = answer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self.children) >= MAX_CHILDREN

    def add_child(self, node: 'Node'):
        self.children.append(node)

class MCTS:
    def __init__(self, question: str, seed_answers: list[str]):
        self.question = question
        self.root = Node(question, random.choice(seed_answers))

    def search(self):
        for i in range(ITERATIONS):
            console.rule(f"[bold blue]Iteration {i+1}")
            node = self.select(self.root)
            if not node.is_fully_expanded():
                node = self.expand(node)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
            console.print(f"[green]Simulated reward:[/green] {reward:.2f}")
        best = max(self.root.children, key=lambda c: c.value / c.visits if c.visits > 0 else 0)
        return best.answer

    def select(self, node: Node) -> Node:
        while node.is_fully_expanded() and node.children:
            node = max(node.children, key=lambda c: self.uct(c, node.visits))
        return node

    def expand(self, node: Node) -> Node:
        child = Node(self.question, node.answer, parent=node)
        node.add_child(child)

        critique_prompt = get_critique_prompt(self.question, child.answer)
        critique = query_llm(critique_prompt)
        console.print("[yellow]Critique:[/yellow]", critique)

        improvement_prompt = get_improvement_prompt(self.question, child.answer, critique)
        improved = query_llm(improvement_prompt)
        console.print("[cyan]Improved Answer:[/cyan]", improved)

        child.answer = improved
        return child

    def simulate(self, node: Node) -> float:
        rating_prompt = get_rating_prompt(node.question, node.answer)
        result = query_llm(rating_prompt)
        console.print("[magenta]Rating Response:[/magenta]", result)
        try:
            score = int(re.findall(r"Rating:\s*(\d+)", result)[0])
            return min(score, 95) / 100
        except:
            return 0.0

    def backpropagate(self, node: Node, reward: float):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def uct(self, child: Node, parent_visits: int) -> float:
        if child.visits == 0:
            return float('inf')
        exploit = child.value / child.visits
        explore = math.sqrt((2 * math.log(parent_visits)) / child.visits)
        return exploit + 1.41 * explore

def extract_boxed_answer(text: str) -> str | None:
    matches = re.findall(r'\\boxed{((?:[^{}]|\{[^{}]*\})*)}', text)
    return matches[-1] if matches else None

def get_math_qa(row: int = 0, level: int | None = None):
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", "algebra", split='test[:100]')
    df = pd.DataFrame(ds)
    if level:
        string = "Level " + str(level)
        df = df[df['level'] == string]
        console.print(f"Level {level} has {len(df)} problems")
    prob, sol = df.iloc[row]['problem'], df.iloc[row]['solution']
    short = extract_boxed_answer(sol)
    return prob, sol, short

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MCTS-enhanced LLM response refinement")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--question", type=str, help="Custom question to refine")
    group.add_argument("--math", type=int, nargs="?", const=0, help="Use a math question from the dataset (optional row number, default 0)")
    parser.add_argument("--level", type=int, help="Optional level filter for math questions")
    args = parser.parse_args()

    console.rule("[bold green]STARTING MCTS-LLM")

    if args.math is not None:
        question, full_answer, short_answer = get_math_qa(row=args.math, level=args.level)
    elif args.question:
        question = args.question
        short_answer = None
    else:
        question = "A man and a goat are on one side of a river. They have a boat. How can they go across?"
        short_answer = "They can use the boat."

    console.rule("[bold green]QUESTION")
    console.print(question)

    console.rule("[bold red]VANILLA LLM RESPONSE")
    baseline = query_llm(question)
    console.print(baseline)

    mcts = MCTS(question, seed_answers=SEED_ANSWERS)
    best = mcts.search()

    console.rule("[bold blue]MCTS IMPROVED ANSWER")
    console.print(best)

    if short_answer:
        console.rule("[bold magenta]GROUND TRUTH (Boxed Answer)")
        console.print(short_answer)

    console.rule("[bold yellow]EVALUATION SCORES")
    for label, answer in [("Vanilla", baseline), ("MCTS", best)]:
        prompt = get_rating_prompt(question, answer)
        score_str = query_llm(prompt)
        console.print(f"{label} Score Prompt →", score_str)
