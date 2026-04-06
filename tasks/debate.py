"""Inspect AI task: multi-turn debate between two suspect models.

Two copies of the suspect model debate a WMDP-bio question:
  - Suspect 1 (honest): argues for the correct answer
  - Suspect 2 (deceptive): argues for a wrong answer

The judge sees both arguments each turn, evaluates, and may steer
follow-up rounds. After N turns, the judge makes a final decision.

Compare with tasks/judge_baseline.py (same knowledge filters) to measure delta.

Usage:
    uv run inspect eval tasks/debate.py --limit 30 --log-level warning \
      -T suspect_model=openrouter/meta-llama/llama-3.3-70b-instruct:nitro \
      -T suspect2_model=openrouter/meta-llama/llama-3.3-70b-instruct:nitro \
      --model openrouter/google/gemma-3-4b-it
"""

from inspect_ai import Task, task
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from lib.prompts import SUSPECT_PROMPTS, DebatePrompts
from lib.wmdp_data import (
    add_debate_metadata,
    answer_scorer,
    extract_followup,
    format_question_block,
    original_dataset,
)


@solver
def multiturn_debate_solver(
    suspect_model_name: str,
    suspect2_model_name: str,
    suspect_system: str,
    suspect2_system: str,
    prompts: DebatePrompts,
    num_turns: int = 3,
) -> Solver:
    """Run N rounds of debate with cross-examination, then judge decides via generate()."""
    suspect1 = get_model(suspect_model_name)
    suspect2 = get_model(suspect2_model_name)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        assert isinstance(state.metadata, dict)
        meta = state.metadata

        question = str(meta["question"])
        choices: list[str] = list(meta["choices"])  # type: ignore[arg-type]
        correct_letter = str(meta["correct_letter"])
        wrong_letter = str(meta["wrong_letter"])
        question_block = format_question_block(question, choices)

        # Conversation histories for each model
        suspect1_messages: list[ChatMessage] = [
            ChatMessageSystem(content=suspect_system),
            ChatMessageUser(content=prompts.suspect_assignment_user.format(
                question_block=question_block,
                argue_target=correct_letter,
                argue_target_text=choices[ord(correct_letter) - 65],
            )),
        ]
        suspect2_messages: list[ChatMessage] = [
            ChatMessageSystem(content=suspect2_system),
            ChatMessageUser(content=prompts.suspect_assignment_user.format(
                question_block=question_block,
                argue_target=wrong_letter,
                argue_target_text=choices[ord(wrong_letter) - 65],
            )),
        ]
        judge_messages: list[ChatMessage] = [
            ChatMessageSystem(content=prompts.judge_system),
            ChatMessageUser(content=prompts.judge_opening_user.format(
                question_block=question_block,
            )),
        ]

        prev_s1_arg: str | None = None
        prev_s2_arg: str | None = None
        follow_up: str | None = None
        judge_out = None

        for turn_idx in range(num_turns):
            is_final_turn = turn_idx == num_turns - 1

            # Cross-examination from turn 2 onwards
            if turn_idx > 0:
                if prev_s2_arg:
                    cross1 = prompts.suspect_crossexam_user.format(
                        opponent_name="Debater Beta",
                        opponent_arg=prev_s2_arg,
                    )
                    if follow_up:
                        cross1 += "\n\n" + prompts.judge_followup_addon.format(follow_up=follow_up)
                    suspect1_messages.append(ChatMessageUser(content=cross1))

                if prev_s1_arg:
                    cross2 = prompts.suspect_crossexam_user.format(
                        opponent_name="Debater Alpha",
                        opponent_arg=prev_s1_arg,
                    )
                    if follow_up:
                        cross2 += "\n\n" + prompts.judge_followup_addon.format(follow_up=follow_up)
                    suspect2_messages.append(ChatMessageUser(content=cross2))

            # Suspects argue
            s1_out = await suspect1.generate(
                suspect1_messages,
                config=GenerateConfig(temperature=0, max_tokens=512),
            )
            s1_arg = s1_out.completion
            suspect1_messages.append(ChatMessageAssistant(content=s1_arg))

            s2_out = await suspect2.generate(
                suspect2_messages,
                config=GenerateConfig(temperature=0, max_tokens=512),
            )
            s2_arg = s2_out.completion
            suspect2_messages.append(ChatMessageAssistant(content=s2_arg))

            # Judge evaluates
            if is_final_turn:
                judge_user = prompts.judge_final_round_user.format(
                    round_num=turn_idx + 1,
                    s1_arg=s1_arg,
                    s2_arg=s2_arg,
                )
            else:
                judge_user = prompts.judge_round_user.format(
                    round_num=turn_idx + 1,
                    s1_arg=s1_arg,
                    s2_arg=s2_arg,
                )
            judge_messages.append(ChatMessageUser(content=judge_user))

            # Set state.messages so generate() runs the judge (--model)
            state.messages = list(judge_messages)
            state = await generate(state)
            judge_text = state.output.completion
            judge_messages.append(ChatMessageAssistant(content=judge_text))
            judge_out = state.output

            follow_up = extract_followup(judge_text) if not is_final_turn else None
            prev_s1_arg = s1_arg
            prev_s2_arg = s2_arg

        assert judge_out is not None, "No judge output produced (num_turns must be >= 1)"
        state.output = judge_out
        return state

    return solve


@task
def debate(
    suspect_model: str,
    suspect2_model: str,
    suspect_prompt: str = "honest_debater",
    suspect2_prompt: str = "deceptive_debater",
    num_turns: int = 3,
) -> Task:
    """Multi-turn debate with cross-examination; judge decides after N rounds.

    --model is the judge. Suspects are explicit task args.

    Args:
        suspect_model: Honest suspect model (argues for correct answer).
        suspect2_model: Deceptive suspect model (argues for wrong answer).
        suspect_prompt: Suspect 1 system prompt key. Default: 'honest_debater'.
        suspect2_prompt: Suspect 2 system prompt key. Default: 'deceptive_debater'.
        num_turns: Number of debate rounds (default 3).
    """
    for key, name in [(suspect_prompt, "suspect_prompt"), (suspect2_prompt, "suspect2_prompt")]:
        if key not in SUSPECT_PROMPTS:
            raise ValueError(
                f"Unknown {name} '{key}'. Choose from: {list(SUSPECT_PROMPTS)}"
            )

    prompts = DebatePrompts()
    dataset = add_debate_metadata(original_dataset())

    return Task(
        dataset=dataset,
        solver=multiturn_debate_solver(
            suspect_model_name=suspect_model,
            suspect2_model_name=suspect2_model,
            suspect_system=SUSPECT_PROMPTS[suspect_prompt],
            suspect2_system=SUSPECT_PROMPTS[suspect2_prompt],
            prompts=prompts,
            num_turns=num_turns,
        ),
        scorer=answer_scorer(),
        config=GenerateConfig(temperature=0, max_tokens=512),
    )
