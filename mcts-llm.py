import os
import re
import math
import random
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from rich.console import Console
from litellm import completion

load_dotenv()
os.environ["LITELLM_DISABLE_SPEND_TRACKING"] = "true"
os.environ["LITELLM_LOG"] = "ERROR"
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

MODEL = "gemini/gemini-2.0-flash"
ITERATIONS = 3
MAX_CHILDREN = 3
SEED_ANSWERS = ["I'm not sure", "I can't say", "I don't know the answer"]

console = Console()

class LLMClient:
    def __init__(self, model: str, rubric: str | None = None):
        self.model = model
        self.system_message = self._build_system_message(rubric)

    def _build_system_message(self, rubric: str | None) -> dict:
        base = "You are an expert assistant focused on providing high-quality answers."
        if rubric:
            base += (
                f"\n\nIMPORTANT: All your responses should be evaluated against these criteria:\n{rubric}"
                "\n\nKeep these standards in mind for all tasks including critiques and improvements."
            )
            return {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": base,
                        "cache_control": {
                            "type": "ephemeral" # enables short-lived caching in Gemini
                        },
                    }
                ],
            }
        else:
            return {
                "role": "system",
                "content": [{"type": "text", "text": base}],
            }

    def query(self, prompt: str) -> str:
        messages = [
            self.system_message,
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]
        try:
            response = completion(model=self.model, messages=messages)
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            console.print(f"[red]LLM Error:[/red] {e}")
            return ""

    def critique(self, question: str, answer: str) -> str:
        console.rule("[cyan]CRITIQUE")
        prompt = self.get_critique_prompt(question, answer)
        console.print("[yellow]Critique Prompt:[/yellow]", prompt)
        critique = self.query(prompt)
        console.print("[yellow]Critique Response:[/yellow]", critique)
        return critique

    def improve(self, question: str, answer: str, critique: str) -> str:
        console.rule("[cyan]IMPROVEMENT")
        prompt = self.get_improvement_prompt(question, answer, critique)
        console.print("[cyan]Improvement Prompt:[/cyan]", prompt)
        improved = self.query(prompt)
        console.print("[cyan]Improved Answer:[/cyan]", improved)
        return improved

    def score(self, question: str, answer: str) -> float:
        console.rule("[magenta]SCORE")
        prompt = self.get_rating_prompt(question, answer)
        result = self.query(prompt)
        console.print("[magenta]Rating Response:[/magenta]", result)
        match = re.search(r"Rating:\s*(\d+)", result)
        if match:
            return min(int(match.group(1)), 95) / 100
        return 0.0

    @staticmethod
    def get_critique_prompt(q: str, a: str) -> str:
        return f"Question: {q}\nDraft Answer: {a}\nPlease critique this answer against the established criteria. What could be improved? Maintain high standards."

    @staticmethod
    def get_improvement_prompt(q: str, a: str, critique: str) -> str:
        return f"Question: {q}\nDraft Answer: {a}\nCritique: {critique}\nRewrite the answer to address these issues while meeting the quality standards."

    @staticmethod
    def get_rating_prompt(q: str, a: str) -> str:
        return f"Question: {q}\nAnswer: {a}\nRate this answer against the established criteria from 0 to 100.\nFormat: Rating: <number>"

class Node:
    def __init__(self, question: str, answer: str, parent=None):
        self.question = question
        self.answer = answer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.children) >= MAX_CHILDREN

    def add_child(self, child: 'Node'):
        self.children.append(child)


class MCTS:
    def __init__(self, question: str, seed_answers: list[str], llm: LLMClient):
        self.question = question
        self.root = Node(question, random.choice(seed_answers))
        self.llm = llm

    def search(self) -> str:
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
        critique = self.llm.critique(self.question, node.answer)
        improved = self.llm.improve(self.question, node.answer, critique)
        child = Node(self.question, improved, parent=node)
        node.add_child(child)
        return child

    def simulate(self, node: Node) -> float:
        return self.llm.score(node.question, node.answer)

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

def get_math_qa(row: int = 0, level: int | None = None) -> tuple[str, str, str | None]:
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
    parser.add_argument("--rubric-file", type=str, help="Optional file containing evaluation rubric text")
    args = parser.parse_args()

    console.rule("[bold green]STARTING MCTS-LLM")

    if args.math is not None:
        question, full_answer, short_answer = get_math_qa(row=args.math, level=args.level)
    elif args.question:
        question, short_answer = args.question, None
        short_answer = None
    else:
        question = "A man and a goat are on one side of a river. They have a boat. How can they go across?"
        short_answer = "They can use the boat."

    rubric = None
    if args.rubric_file:
        if os.path.exists(args.rubric_file):
            with open(args.rubric_file, "r", encoding="utf-8") as f:
                rubric = f.read()
            console.rule("[bold cyan]LOADED RUBRIC")
            console.print(rubric)
        else:
            console.print(f"[red]Rubric file not found:[/red] {args.rubric_file}")

    console.rule("[bold green]QUESTION")
    console.print(question)

    llm = LLMClient(MODEL, rubric)

    console.rule("[bold red]VANILLA LLM RESPONSE")
    baseline = llm.query(question)
    console.print(baseline)

    mcts = MCTS(question, seed_answers=SEED_ANSWERS, llm=llm)
    best = mcts.search()

    console.rule("[bold blue]MCTS IMPROVED ANSWER")
    console.print(best)
    if short_answer:
        console.rule("[bold magenta]GROUND TRUTH (Boxed Answer)")
        console.print(short_answer)

    console.rule("[bold yellow]EVALUATION SCORES")
    for label, answer in [("Vanilla", baseline), ("MCTS", best)]:
        score = llm.score(question, answer)
        console.print(f"{label} Score → {score * 100:.1f}/100")
