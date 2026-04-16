"""Inspect AI task: single-turn interrogation with lie detector.

Same as interrogation_single_turn but a lie detector report is shown to the
judge alongside the suspect's argument.

Detectors:
  - oracle: ground-truth label (DECEPTIVE/HONEST)
  - always_50: constant 50% deception baseline
  - coin_flip: random baseline
  - behavioral: fresh-context belief probe
  - behavioral_hardened: adversarial variant (suspect forewarned about probing)
  - llm_score: LLM reads the argument and outputs a deception probability
  - llm_report: LLM reads the argument and outputs a qualitative explanation

Compare with tasks/judge_baseline.py@judge_baseline_yes_idk to measure the delta.

Usage:
    uv run inspect eval tasks/interrogation_single_turn_detector.py --limit 100 --log-level warning \
      -T detector_name=oracle \
      -T suspect_model=openrouter/meta-llama/llama-3.3-70b-instruct:nitro \
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

from lib.detectors import (
    Always50Detector,
    CoinFlipDetector,
    LieDetectorFn,
    OracleDetector,
    make_behavioral_detector,
    make_llm_score_detector,
    make_llm_report_detector,
)
from lib.prompts import (
    SUSPECT_PROMPTS,
    InterrogationDetectorPrompts,
    InterrogationDetectorPromptsAnswerFocused,
    InterrogationDetectorPromptsNoIDK,
)
from lib.wmdp_data import answer_scorer, balanced_expanded_dataset, expanded_dataset, expanded_dataset_filtered, format_question_block, one_each_dataset


@solver
def interrogation_detector_solver(
    suspect_model_name: str,
    suspect_system: str,
    detector: LieDetectorFn,
    prompts: InterrogationDetectorPrompts,
    temperature: float = 0.0,
) -> Solver:
    """Suspect argues, lie detector reports, then judge evaluates via generate()."""
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
            config=GenerateConfig(temperature=temperature, max_tokens=1024),
        )
        argument = suspect_out.completion

        detector_report = await detector(argument, dict(meta))

        question = str(meta["question"])
        choices: list[str] = list(meta["choices"])  # type: ignore[arg-type]
        question_block = format_question_block(question, choices)
        judge_user = prompts.judge_user.format(
            question_block=question_block,
            argument=argument,
            detector_report=detector_report,
        )
        state.messages = [
            ChatMessageSystem(content=prompts.judge_system),
            ChatMessageUser(content=judge_user),
        ]
        state = await generate(state)
        return state

    return solve


@task
def interrogation_single_turn_detector(
    suspect_model: str,
    suspect_prompt: str = "debate_coach",
    detector_name: str = "oracle",
    detector_model: str = "",
    filtering: str = "",
    balanced: bool = False,
    n_each: int = 50,
    one_each: bool = False,
    temperature: float = 0.0,
    allow_idk: bool = False,
    answer_focused: bool = False,
    filter_files: str = "",
) -> Task:
    """Interrogation with lie detector reports shown to judge. --model is the judge.

    Args:
        suspect_model: Model that generates the argument (also probed by behavioral detector).
        suspect_prompt: Key into SUSPECT_PROMPTS. Use 'debate_coach_hardened' for
                        adversarial robustness testing.
        detector_name: 'oracle' | 'always_50' | 'coin_flip' | 'behavioral' | 'behavioral_hardened'
                       | 'llm_score' | 'llm_report'.
        detector_model: Model used by llm_score and llm_report detectors. Required when
                        detector_name is 'llm_score' or 'llm_report'.
        filtering: JSON filename stem in data/filterings/ (e.g. 'filtered_ids') produced
                   by Kartik's filtering pipeline. If empty, all questions are used.
        balanced: If True, use equal honest/deceptive samples (n_each each).
        n_each: Number of honest and deceptive samples each when balanced=True (default 50).
        one_each: If True, use 1 honest + 1 deceptive per question across all questions.
        temperature: Sampling temperature for both suspect and judge (default 0.0).
        allow_idk: If False, IDK is removed from the judge's answer choices (default False).
        answer_focused: If True, the detector report is framed as being about the argued
            answer rather than the argument, and IDK is disallowed. Prevents the judge
            from reasoning 'the argument is deceptive but the answer might still be right'.
        filter_files: Comma-separated list of Kartik filter JSON stems
            (from kartik_deception/results/), e.g.
            "judge_..._wmdp-bio_thresh3,judge_..._wmdp-chem_thresh3".
            If set, loads questions from HuggingFace using these filter files
            instead of the default liars-bench wmdp-bio dataset.
    """
    if answer_focused:
        prompts: InterrogationDetectorPrompts = InterrogationDetectorPromptsAnswerFocused()
    elif not allow_idk:
        prompts = InterrogationDetectorPromptsNoIDK()
    else:
        prompts = InterrogationDetectorPrompts()

    if suspect_prompt not in SUSPECT_PROMPTS:
        raise ValueError(
            f"Unknown suspect_prompt '{suspect_prompt}'. "
            f"Choose from: {list(SUSPECT_PROMPTS)}"
        )

    report_style = "answer" if answer_focused else "argument"

    if detector_name in ("llm_score", "llm_report") and not detector_model:
        raise ValueError(
            f"detector_name='{detector_name}' requires detector_model to be set. "
            "Pass -T detector_model=<model_name>."
        )

    detector_names = [
        "oracle",
        "always_50",
        "coin_flip",
        "behavioral",
        "behavioral_hardened",
        "llm_score",
        "llm_report",
    ]
    if detector_name not in detector_names:
        raise ValueError(
            f"Unknown detector '{detector_name}'. Choose from: {detector_names}"
        )

    if detector_name == "oracle":
        detector: LieDetectorFn = OracleDetector(report_style=report_style)
    elif detector_name == "always_50":
        detector = Always50Detector(report_style=report_style)
    elif detector_name == "coin_flip":
        detector = CoinFlipDetector(report_style=report_style)
    elif detector_name == "behavioral":
        detector = make_behavioral_detector(suspect_model)
    elif detector_name == "behavioral_hardened":
        detector = make_behavioral_detector(
            suspect_model,
            probe_system_prompt=SUSPECT_PROMPTS["debate_coach_hardened"],
        )
    elif detector_name == "llm_score":
        detector = make_llm_score_detector(detector_model, report_style=report_style)
    else:
        detector = make_llm_report_detector(detector_model)

    if filter_files and filtering:
        raise ValueError(
            "Use either 'filter_files' or 'filtering', not both. "
            "'filter_files' is for one-or-more per-subset Kartik filter JSONs; "
            "'filtering' is for the wmdp-bio root dataset filtering path."
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
        solver=interrogation_detector_solver(
            suspect_model_name=suspect_model,
            suspect_system=SUSPECT_PROMPTS[suspect_prompt],
            detector=detector,
            prompts=prompts,
            temperature=temperature,
        ),
        scorer=answer_scorer(),
        config=GenerateConfig(temperature=temperature, max_tokens=1024),
        metadata={"detector": detector_name},
    )
