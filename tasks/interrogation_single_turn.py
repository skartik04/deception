"""Inspect AI task: single-turn interrogation (no lie detector).

Suspect argues for an assigned answer. Judge evaluates the argument and picks
an answer. Single scorer on the judge's post-argument answer.

Compare with tasks/judge_baseline.py@judge_baseline_yes_idk to measure the delta.

Usage:
    uv run inspect eval tasks/interrogation_single_turn.py --limit 100 --log-level warning \
      -T suspect_model=openrouter/meta-llama/llama-3.3-70b-instruct:nitro \
      -T judge_prompt=informed \
      --model openrouter/google/gemma-3-27b-it:nitro
"""

from inspect_ai import Task, task
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from lib.prompts import (
    SUSPECT_PROMPTS,
    InterrogationPrompts,
    InterrogationPromptsInformed,
    InterrogationPromptsStandard,
)
from lib.wmdp_data import answer_scorer, balanced_expanded_dataset, expanded_dataset, format_question_block


@solver
def interrogation_solver(
    suspect_model_name: str,
    suspect_system: str,
    prompts: InterrogationPrompts,
) -> Solver:
    """Suspect argues (separate model), then judge evaluates via generate()."""
    suspect = get_model(suspect_model_name)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        assert isinstance(state.metadata, dict)
        meta = state.metadata

        suspect_messages = [
            ChatMessageSystem(content=suspect_system),
            ChatMessageUser(content=str(state.input)),
        ]
        suspect_out = await suspect.generate(
            suspect_messages,
            config=GenerateConfig(temperature=0, max_tokens=1024),
        )
        argument = suspect_out.completion

        question = str(meta["question"])
        choices: list[str] = list(meta["choices"])  # type: ignore[arg-type]
        question_block = format_question_block(question, choices)
        judge_user = prompts.judge_user.format(
            question_block=question_block,
            argument=argument,
        )
        state.messages = [
            ChatMessageSystem(content=prompts.judge_system),
            ChatMessageUser(content=judge_user),
        ]
        state = await generate(state)
        return state

    return solve


_PROMPT_CONFIGS: dict[str, InterrogationPrompts] = {
    "standard": InterrogationPromptsStandard(),
    "informed": InterrogationPromptsInformed(),
}


@task
def interrogation_single_turn(
    suspect_model: str,
    suspect_prompt: str = "debate_coach",
    judge_prompt: str = "informed",
    balanced: bool = False,
    n_each: int = 50,
) -> Task:
    """Suspect argues, judge evaluates. --model is the judge.

    Args:
        suspect_model: Model that generates the argument.
        suspect_prompt: Key into SUSPECT_PROMPTS for the suspect's system prompt.
        judge_prompt: 'standard' or 'informed'. 'informed' tells the judge the
                      suspect was given a random answer (25% prior of being correct).
        balanced: If True, use equal honest/deceptive samples (n_each each).
        n_each: Number of honest and deceptive samples each when balanced=True (default 50).
    """
    if suspect_prompt not in SUSPECT_PROMPTS:
        raise ValueError(
            f"Unknown suspect_prompt '{suspect_prompt}'. "
            f"Choose from: {list(SUSPECT_PROMPTS)}"
        )
    if judge_prompt not in _PROMPT_CONFIGS:
        raise ValueError(
            f"Unknown judge_prompt '{judge_prompt}'. "
            f"Choose from: {list(_PROMPT_CONFIGS)}"
        )

    return Task(
        dataset=balanced_expanded_dataset(n_each=n_each) if balanced else expanded_dataset(),
        solver=interrogation_solver(
            suspect_model_name=suspect_model,
            suspect_system=SUSPECT_PROMPTS[suspect_prompt],
            prompts=_PROMPT_CONFIGS[judge_prompt],
        ),
        scorer=answer_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=1024),
    )
