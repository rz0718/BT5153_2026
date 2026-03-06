"""
Test script: single prompt vs prompt chain for the customer review response task.
Follows the same design pattern as approach_cot_chain in evaluate_all.py.
"""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

MODEL = "qwen2.5:3b"

# Sample customer review (from the complex-query example)
SAMPLE_REVIEW = (
    "I ordered a laptop, but it arrived three days late. "
    "The packaging was also damaged. "
    "This was very disappointing since I needed it urgently for work."
)

# ---------------------------------------------------------------------------
# Single prompt: one complex instruction (single step)
# ---------------------------------------------------------------------------

SINGLE_PROMPT = """You are a professional customer service agent.

Read this customer review and write a professional response that:
1. Acknowledges their concern
2. Explains the issue (or possible reasons)
3. Offers a resolution

Respond in a warm, apologetic tone. Address the customer as "Dear Customer" and sign off appropriately."""


def approach_single_prompt(review: str) -> str:
    """One LLM call: complex prompt that does everything in a single step."""
    llm = ChatOllama(model=MODEL, temperature=0.2)
    messages = [
        ("system", SINGLE_PROMPT),
        ("human", review),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


# ---------------------------------------------------------------------------
# Prompt chain: 3 steps (extract issues → outline → full response)
# ---------------------------------------------------------------------------

CHAIN_PROMPT_1 = """You are analyzing customer feedback.

Identify the key concerns mentioned in this customer review.
List each concern as a short bullet point (e.g., "Delivery Delay: ...", "Packaging Issue: ...", "Customer Sentiment: ...").
Output only the bullet list, nothing else."""

CHAIN_PROMPT_2 = """You are planning a customer service response.

Using these issues, draft an outline for a professional response that:
1. Acknowledges the concerns
2. Explains possible reasons (e.g., logistics, warehouse)
3. Apologizes and offers a resolution (e.g., discount, expedited service)

Output a short numbered outline only (1. ... 2. ... 3. ...)."""

CHAIN_PROMPT_3 = """You are a professional customer service agent.

Using this outline, write the full professional response to the customer.
Be warm and apologetic. Address the customer as "Dear Customer" and sign off appropriately.
Output only the response text, no meta-commentary."""


def approach_prompt_chain(review: str, verbose: bool = False) -> str | tuple[str, str, str]:
    """Three LLM calls: extract issues → create outline → write full response.
    If verbose=True, returns (issues, outline, full_response) so you can print intermediate steps."""
    llm = ChatOllama(model=MODEL, temperature=0.2)

    # Step 1: Extract key issues
    messages_1 = [
        ("system", CHAIN_PROMPT_1),
        ("human", review),
    ]
    response_1 = llm.invoke(messages_1)
    issues = response_1.content.strip()

    # Step 2: Create response outline from issues
    messages_2 = [
        ("system", CHAIN_PROMPT_2),
        ("human", issues),
    ]
    response_2 = llm.invoke(messages_2)
    outline = response_2.content.strip()

    # Step 3: Write full response from outline
    messages_3 = [
        ("system", CHAIN_PROMPT_3),
        ("human", outline),
    ]
    response_3 = llm.invoke(messages_3)
    full_response = response_3.content.strip()

    if verbose:
        return issues, outline, full_response
    return full_response



def run_comparison(review: str | None = None, verbose_chain: bool = True) -> None:
    """Run single prompt and prompt chain on the same review and print results."""
    review = review or SAMPLE_REVIEW

    print("=" * 60)
    print("CUSTOMER REVIEW (input)")
    print("=" * 60)
    print(review)
    print()

    print("=" * 60)
    print("APPROACH 1: Single prompt (one step)")
    print("=" * 60)
    single_out = approach_single_prompt(review)
    print(single_out)
    print()

    print("=" * 60)
    print("APPROACH 2: Prompt chain (3 steps)")
    print("=" * 60)
    if verbose_chain:
        issues, outline, chain_out = approach_prompt_chain(review, verbose=True)
        print("  Step 1 — Key issues:")
        print(issues)
        print()
        print("  Step 2 — Outline:")
        print(outline)
        print()
        print("  Step 3 — Full response:")
        print(chain_out)
    else:
        chain_out = approach_prompt_chain(review)
        print(chain_out)
    print()



if __name__ == "__main__":
    run_comparison()
