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
    InterrogationPromptsInformedNoIDK,
    InterrogationPromptsStandard,
    InterrogationPromptsStandardNoIDK,
)
from lib.wmdp_data import (
    answer_scorer,
    balanced_expanded_dataset,
    expanded_dataset,
    expanded_dataset_filtered,
    format_question_block,
    one_each_dataset,
)


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


_PROMPT_CONFIGS_WITH_IDK: dict[str, InterrogationPrompts] = {
    "standard": InterrogationPromptsStandard(),
    "informed": InterrogationPromptsInformed(),
}

_PROMPT_CONFIGS_NO_IDK: dict[str, InterrogationPrompts] = {
    "standard": InterrogationPromptsStandardNoIDK(),
    "informed": InterrogationPromptsInformedNoIDK(),
}


@task
def interrogation_single_turn(
    suspect_model: str,
    suspect_prompt: str = "debate_coach",
    judge_prompt: str = "informed",
    filtering: str = "",
    balanced: bool = False,
    n_each: int = 50,
    one_each: bool = False,
    filter_files: str = "",
    allow_idk: bool = False,
) -> Task:
    """Suspect argues, judge evaluates. --model is the judge.

    Args:
        suspect_model: Model that generates the argument.
        suspect_prompt: Key into SUSPECT_PROMPTS for the suspect's system prompt.
        judge_prompt: 'standard' or 'informed'. 'informed' tells the judge the
                      suspect was given a random answer (25% prior of being correct).
        filtering: JSON filename stem in data/filterings/ (e.g. 'usable_ids_wmdp-bio').
            If empty, all root-dataset questions are used.
        balanced: If True, use equal honest/deceptive samples (n_each each).
        n_each: Number of honest and deceptive samples each when balanced=True (default 50).
        one_each: If True, use 1 honest + 1 deceptive sample per question.
        filter_files: Comma-separated list of per-subset filter JSON stems. If set,
            loads questions from HuggingFace using these filter files instead of the
            default liars-bench root dataset path.
        allow_idk: If False, the judge must commit to A-D (default False).
    """
    if suspect_prompt not in SUSPECT_PROMPTS:
        raise ValueError(
            f"Unknown suspect_prompt '{suspect_prompt}'. "
            f"Choose from: {list(SUSPECT_PROMPTS)}"
        )
    prompt_configs = _PROMPT_CONFIGS_WITH_IDK if allow_idk else _PROMPT_CONFIGS_NO_IDK
    if judge_prompt not in prompt_configs:
        raise ValueError(
            f"Unknown judge_prompt '{judge_prompt}'. "
            f"Choose from: {list(prompt_configs)}"
        )

    if filter_files and filtering:
        raise ValueError(
            "Use either 'filter_files' or 'filtering', not both. "
            "'filter_files' is for one-or-more per-subset filter JSONs; "
            "'filtering' is for the root-dataset filtering path."
        )

    filtering_arg = filtering if filtering else None
    if filter_files:
        files = [f.strip() for f in filter_files.split(",")]
        dataset = expanded_dataset_filtered(files)
    elif one_each:
        dataset = one_each_dataset(filtering=filtering_arg)
    elif balanced:
        dataset = balanced_expanded_dataset(n_each=n_each, filtering=filtering_arg)
    else:
        dataset = expanded_dataset(filtering=filtering_arg)

    return Task(
        dataset=dataset,
        solver=interrogation_solver(
            suspect_model_name=suspect_model,
            suspect_system=SUSPECT_PROMPTS[suspect_prompt],
            prompts=prompt_configs[judge_prompt],
        ),
        scorer=answer_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=1024),
    )
