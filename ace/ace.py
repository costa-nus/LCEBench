"""
ACE (Agentic Context Engineering) orchestrator: offline adaptation of a playbook.

Three agents work on every training sample:
- Generator: answers using the playbook (and, after a failure, the latest reflection)
- Reflector: diagnoses the answer and tags the playbook bullets it used
- Curator:   periodically adds new bullets to the playbook (followed by bullet de-duplication)
The best playbook on the validation split is evaluated on the test split.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from .generator import Generator
from .reflector import Reflector
from .curator import Curator
from .bulletpoint_analyzer import BulletpointAnalyzer
from .playbook_utils import extract_playbook_bullets, get_playbook_stats, update_bullet_counts
from .logger import log_bullet_usage
from .utils import count_tokens, evaluate_test_set, extract_answer


class ACE:
    """
    Main ACE system orchestrator.

    Manages the training loop where:
    1. Generator produces answers using playbook
    2. Reflector analyzes answers and tags bullets
    3. Curator updates playbook based on feedback
    """

    def __init__(
        self,
        task,
        generator_client,
        reflector_client,
        curator_client,
        generator_model: str,
        reflector_model: str,
        curator_model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        initial_playbook: Optional[str] = None,
        use_bulletpoint_analyzer: bool = False,
        bulletpoint_analyzer_threshold: float = 0.90,
    ):
        """
        Args:
            task: task definition (see ace/tasks.py)
            *_client: OpenAI clients (generator = student, reflector/curator = teacher)
            *_model: served model names
            max_tokens: Maximum tokens for LLM calls
            initial_playbook: Initial playbook content (optional)
            use_bulletpoint_analyzer: Whether to use bulletpoint analyzer for deduplication
            bulletpoint_analyzer_threshold: Similarity threshold for bulletpoint analyzer (0-1)
        """
        self.task = task

        # Initialize the three agents
        self.generator = Generator(generator_client, generator_model, task.generator_prompt, task.uses_context,
                                   max_tokens, temperature)
        self.reflector = Reflector(reflector_client, reflector_model, max_tokens, temperature)
        curator_prompt_gt, curator_prompt_no_gt = task.curator_prompts
        self.curator = Curator(curator_client, curator_model, curator_prompt_gt, curator_prompt_no_gt,
                               max_tokens, temperature)

        # Initialize bulletpoint analyzer if requested and available
        self.use_bulletpoint_analyzer = use_bulletpoint_analyzer
        self.bulletpoint_analyzer_threshold = bulletpoint_analyzer_threshold

        if use_bulletpoint_analyzer:
            self.bulletpoint_analyzer = BulletpointAnalyzer(
                curator_client,
                curator_model,
                max_tokens
            )
            print(f"✓ BulletpointAnalyzer initialized (threshold={bulletpoint_analyzer_threshold})")
        else:
            self.bulletpoint_analyzer = None

        # Store configuration
        self.generator_client = generator_client
        self.reflector_client = reflector_client
        self.curator_client = curator_client
        self.max_tokens = max_tokens

        # Initialize playbook
        if initial_playbook:
            self.playbook = initial_playbook
        else:
            self.playbook = self._initialize_empty_playbook()

        self.best_playbook = self.playbook
        # Track global bullet ID
        self.next_global_id = 1

    def _initialize_empty_playbook(self) -> str:
        """Initialize an empty playbook with standard sections."""
        return """## STRATEGIES & INSIGHTS

## FORMULAS & CALCULATIONS

## CODE SNIPPETS & TEMPLATES

## COMMON MISTAKES TO AVOID

## PROBLEM-SOLVING HEURISTICS

## CONTEXT CLUES & INDICATORS

## OTHERS"""

    def _extract_config_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract common configuration parameters."""
        return {
            'num_epochs': config.get('num_epochs', 1),
            'max_num_rounds': config.get('max_num_rounds', 3),
            'curator_frequency': config.get('curator_frequency', 1),
            'eval_steps': config.get('eval_steps', 100),
            'save_steps': config.get('save_steps', 50),
            'token_budget': config.get('playbook_token_budget', 80000),
            'task_name': config.get('task_name', 'default'),
            'use_json_mode': config.get('json_mode', False),
            'no_ground_truth': config.get('no_ground_truth', False),
            'save_dir': config.get('save_dir', './results'),
            'test_workers': config.get('test_workers', 20),
            'use_bulletpoint_analyzer': config.get('use_bulletpoint_analyzer', False),
            'bulletpoint_analyzer_threshold': config.get('bulletpoint_analyzer_threshold', 0.90)
        }

    def _setup_paths(self, save_dir: str, task_name: str, mode: str) -> Tuple[str, str, str, str]:
        """Create the timestamped run folder; returns (save_path, usage_log_path, playbook_dir, log_dir)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = f"ace_run_{timestamp}_{task_name}_{mode}"
        save_path = os.path.join(save_dir, run_folder)
        os.makedirs(save_path, exist_ok=True)
        log_dir = os.path.join(save_path, "detailed_llm_logs")
        os.makedirs(log_dir, exist_ok=True)

        usage_log_path = os.path.join(save_path, "bullet_usage_log.jsonl")
        playbook_dir = os.path.join(save_path, "intermediate_playbooks")
        os.makedirs(playbook_dir, exist_ok=True)

        return save_path, usage_log_path, playbook_dir, log_dir

    def run(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        test_samples: Optional[List[Dict[str, Any]]] = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Offline adaptation: initial test -> training with periodic validation -> final test
        with the best playbook.
        """
        mode = "offline"
        config_params = self._extract_config_params(config)
        task_name = config_params['task_name']
        save_dir = config_params['save_dir']

        # Keep the runtime variant synchronized with the saved run config.
        requested_analyzer = bool(config_params['use_bulletpoint_analyzer'])
        self.bulletpoint_analyzer_threshold = float(
            config_params['bulletpoint_analyzer_threshold']
        )
        if requested_analyzer and self.bulletpoint_analyzer is None:
            self.bulletpoint_analyzer = BulletpointAnalyzer(
                self.curator_client,
                self.curator.model,
                self.max_tokens,
            )
        self.use_bulletpoint_analyzer = requested_analyzer
        if self.use_bulletpoint_analyzer:
            self.bulletpoint_analyzer.ensure_ready()
        else:
            print("BulletpointAnalyzer disabled: running explicit w/o-analysis variant")

        save_path, usage_log_path, playbook_dir, log_dir = self._setup_paths(save_dir, task_name, mode)

        # Save configuration
        config_path = os.path.join(save_path, "run_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "task_name": task_name,
                "mode": mode,
                "generator_model": self.generator.model,
                "reflector_model": self.reflector.model,
                "curator_model": self.curator.model,
                "effective_runtime": {
                    "max_tokens": self.max_tokens,
                    "temperature": self.generator.temperature,
                    "qwen_enable_thinking": False,
                    "use_bulletpoint_analyzer": self.use_bulletpoint_analyzer,
                    "bulletpoint_analyzer_threshold": self.bulletpoint_analyzer_threshold,
                    "embedding_model": (
                        self.bulletpoint_analyzer.embedding_model_name
                        if self.bulletpoint_analyzer else None
                    ),
                    "embedding_device": (
                        self.bulletpoint_analyzer.embedding_device
                        if self.bulletpoint_analyzer else None
                    ),
                },
                "config": config,
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"ACE SYSTEM - {mode.upper()} MODE")
        print(f"{'='*60}")
        print(f"Task: {task_name}")
        print(f"Train samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")
        if test_samples:
            print(f"Test samples: {len(test_samples)}")
        print(f"{'='*60}\n")

        results = {}

        # 1. Initial test
        if test_samples:
            print(f"\n{'='*60}")
            print(f"INITIAL TEST (before training)")
            print(f"{'='*60}\n")
            initial_test_results = self._run_test(
                test_samples=test_samples,
                playbook=self.playbook,
                config=config,
                log_dir=log_dir,
                save_path=save_path,
                prefix="initial"
            )
            results['initial_test_results'] = initial_test_results
            print(f"Initial Test Accuracy: {initial_test_results['accuracy']:.3f}\n")

        # 2. Offline training
        print(f"\n{'='*60}")
        print(f"STARTING OFFLINE TRAINING")
        print(f"{'='*60}\n")
        training_results = self._offline_train(
            train_samples=train_samples,
            val_samples=val_samples,
            config=config,
            save_path=save_path,
            usage_log_path=usage_log_path,
            playbook_dir=playbook_dir,
            log_dir=log_dir
        )
        results['training_results'] = training_results

        # 3. Final test with the best playbook
        if test_samples:
            print(f"\n{'='*60}")
            print(f"FINAL TEST (with best playbook)")
            print(f"{'='*60}\n")
            final_test_results = self._run_test(
                test_samples=test_samples,
                playbook=self.best_playbook,
                config=config,
                log_dir=log_dir,
                save_path=save_path,
                prefix="final"
            )
            results['final_test_results'] = final_test_results
            print(f"Final Test Accuracy: {final_test_results['accuracy']:.3f}\n")

        # Save consolidated results
        final_results_path = os.path.join(save_path, "final_results.json")
        with open(final_results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {results['training_results']['best_validation_accuracy']:.3f}")
        if test_samples:
            print(f"Initial Test Accuracy: {results['initial_test_results']['accuracy']:.3f}")
            print(f"Final Test Accuracy: {results['final_test_results']['accuracy']:.3f}")
        print(f"Results saved to: {save_path}")
        print(f"{'='*60}\n")

        return results

    def _run_test(
        self,
        test_samples: List[Dict[str, Any]],
        playbook: str,
        config: Dict[str, Any],
        log_dir: str,
        save_path: str,
        prefix: str = "test"
    ) -> Dict[str, Any]:
        """Evaluate a playbook on the test samples and save <prefix>_test_results.json."""
        config_params = self._extract_config_params(config)
        use_json_mode = config_params['use_json_mode']
        test_workers = config_params['test_workers']

        test_results, test_error_log = evaluate_test_set(
            self.task,
            self.generator,
            playbook,
            test_samples,
            self.max_tokens,
            log_dir,
            max_workers=test_workers,
            use_json_mode=use_json_mode
        )

        test_results_path = os.path.join(save_path, f"{prefix}_test_results.json")
        with open(test_results_path, "w") as f:
            json.dump({
                "test_results": test_results,
                "error_log": test_error_log,
            }, f, indent=2)

        return test_results

    def _train_single_sample(
        self,
        task_dict: Dict[str, Any],
        step_id: str,
        epoch: int,
        step: int,
        usage_log_path: str,
        log_dir: str,
        config_params: Dict[str, Any],
        total_samples: int
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Train on a single sample with reflection and curation.

        Returns:
            Tuple of (pre_train_answer, post_train_answer, tracking_dict)
        """
        max_num_rounds = config_params['max_num_rounds']
        curator_frequency = config_params['curator_frequency']
        token_budget = config_params['token_budget']
        use_json_mode = config_params['use_json_mode']
        no_ground_truth = config_params['no_ground_truth']

        question = task_dict.get("question", "")
        target = self.task.reference(task_dict)
        gen_kwargs = self.task.generator_kwargs(task_dict)

        # STEP 1: Initial generation (pre-train)
        print("Generating initial answer...")
        gen_response, bullet_ids, call_info = self.generator.generate(
            question=question,
            playbook=self.playbook,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_gen_initial",
            log_dir=log_dir,
            **gen_kwargs,
        )

        # Extract answer and check correctness (score 1 = correct; IFBench scores are fractions)
        final_answer = extract_answer(gen_response)
        is_correct, feedback = self.task.check(final_answer, task_dict)
        pre_train_answer = final_answer

        print(f"Correct: {is_correct}")

        log_bullet_usage(usage_log_path, epoch, step, task_dict, bullet_ids,
                         playbook=self.playbook, is_correct=is_correct)

        tracking_dict = {
            "pre_train_result": {
                "final_answer": final_answer,
                "is_correct": is_correct,
                "playbook_num_tokens": count_tokens(self.playbook),
                "playbook_length": len(self.playbook)
            }
        }

        reflection_content = "(empty)"
        ground_truth = target if not no_ground_truth else None

        # STEP 2: Reflection and regeneration
        if is_correct != 1:
            # For incorrect answers - iterate reflection rounds
            for round_num in range(max_num_rounds):
                print(f"Reflection round {round_num + 1}/{max_num_rounds}")

                playbook_bullets = extract_playbook_bullets(self.playbook, bullet_ids)

                reflection_content, bullet_tags, _ = self.reflector.reflect(
                    question=question,
                    reasoning_trace=gen_response,
                    predicted_answer=final_answer,
                    ground_truth=ground_truth,
                    environment_feedback=feedback if feedback is not None else "Predicted answer does not match ground truth",
                    bullets_used=playbook_bullets,
                    use_ground_truth=not no_ground_truth,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_round_{round_num}",
                    log_dir=log_dir
                )

                if bullet_tags:
                    self.playbook = update_bullet_counts(self.playbook, bullet_tags)

                # Regenerate with reflection
                gen_response, bullet_ids, _ = self.generator.generate(
                    question=question,
                    playbook=self.playbook,
                    reflection=reflection_content,
                    use_json_mode=use_json_mode,
                    call_id=f"{step_id}_post_reflect_round_{round_num}",
                    log_dir=log_dir,
                    **gen_kwargs,
                )

                final_answer = extract_answer(gen_response)

                if self.task.score(final_answer, target) == 1:
                    print(f"Corrected after reflection round {round_num + 1}!")
                    is_correct = True
                    break

        else:
            # For correct answers - still run reflector to tag helpful bullets
            playbook_bullets = extract_playbook_bullets(self.playbook, bullet_ids)

            reflection_content, bullet_tags, _ = self.reflector.reflect(
                question=question,
                reasoning_trace=gen_response,
                predicted_answer=final_answer,
                ground_truth=ground_truth,
                environment_feedback=feedback if feedback is not None else "Predicted answer matches ground truth",
                bullets_used=playbook_bullets,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=f"{step_id}_reflect_on_correct",
                log_dir=log_dir
            )

            if bullet_tags:
                self.playbook = update_bullet_counts(self.playbook, bullet_tags)

            log_bullet_usage(usage_log_path, epoch, step, task_dict, bullet_ids,
                             playbook=self.playbook,
                             reflection_content=reflection_content,
                             is_correct=is_correct)

        # STEP 3: Curator - Periodically update playbook
        if step % curator_frequency == 0:
            print(f"\n--- Running Curator at step {step} ---")

            stats = get_playbook_stats(self.playbook)

            self.playbook, self.next_global_id, operations, _ = self.curator.curate(
                current_playbook=self.playbook,
                recent_reflection=reflection_content,
                current_step=step,
                total_samples=total_samples,
                token_budget=token_budget,
                playbook_stats=stats,
                use_ground_truth=not no_ground_truth,
                use_json_mode=use_json_mode,
                call_id=step_id,
                log_dir=log_dir,
                next_global_id=self.next_global_id,
                question_context=self.task.question_context(task_dict),
            )

            if self.use_bulletpoint_analyzer and self.bulletpoint_analyzer:
                print(f"  Running BulletpointAnalyzer (threshold={self.bulletpoint_analyzer_threshold})...")
                self.playbook = self.bulletpoint_analyzer.analyze(
                    playbook=self.playbook,
                    threshold=self.bulletpoint_analyzer_threshold,
                    merge=True
                )

        # STEP 4: Post-curator generation
        gen_response, _, _ = self.generator.generate(
            question=question,
            playbook=self.playbook,
            reflection="(empty)",
            use_json_mode=use_json_mode,
            call_id=f"{step_id}_post_curate",
            log_dir=log_dir,
            **gen_kwargs,
        )

        final_answer = extract_answer(gen_response)
        post_train_answer = final_answer

        if self.task.fractional:
            post_train_is_correct, feedback = self.task.check(final_answer, task_dict)
        else:
            post_train_is_correct = self.task.score(final_answer, target)
        tracking_dict["post_train_result"] = {
            "final_answer": final_answer,
            "is_correct": post_train_is_correct,
            "playbook_num_tokens": count_tokens(self.playbook),
            "playbook_length": len(self.playbook)
        }
        if self.task.fractional:
            tracking_dict["post_train_result"]["feedback"] = feedback

        return pre_train_answer, post_train_answer, tracking_dict

    def _offline_train(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        config: Dict[str, Any],
        save_path: str,
        usage_log_path: str,
        playbook_dir: str,
        log_dir: str
    ) -> Dict[str, Any]:
        """Run offline training; returns {"best_validation_accuracy": ...}."""
        config_params = self._extract_config_params(config)
        num_epochs = config_params['num_epochs']
        eval_steps = config_params['eval_steps']
        save_steps = config_params['save_steps']
        test_workers = config_params['test_workers']
        use_json_mode = config_params['use_json_mode']
        curator_frequency = config_params['curator_frequency']

        results = []
        pre_train_post_train_results = []
        error_logs = []
        best_accuracy = 0.0
        self.best_playbook = self.playbook

        print(f"Total epochs: {num_epochs}")
        print(f"Train samples per epoch: {len(train_samples)}")
        print(f"Val samples: {len(val_samples)}")
        print(f"Curator frequency: every {curator_frequency} steps")
        print(f"Evaluation frequency: every {eval_steps} steps\n")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{num_epochs}")
            print(f"{'='*60}")

            epoch_answers_pre_train = []
            epoch_targets_pre_train = []
            epoch_answers_post_train = []
            epoch_targets_post_train = []

            for step, task_dict in enumerate(train_samples):
                step += 1
                print(f"\n--- Step {step}/{len(train_samples)} ---")

                target = self.task.reference(task_dict)

                pre_train_answer, post_train_answer, tracking_dict = self._train_single_sample(
                    task_dict=task_dict,
                    step_id=f"train_e_{epoch}_s_{step}",
                    epoch=epoch,
                    step=step,
                    usage_log_path=usage_log_path,
                    log_dir=log_dir,
                    config_params=config_params,
                    total_samples=len(train_samples)
                )

                epoch_answers_pre_train.append(pre_train_answer)
                epoch_targets_pre_train.append(target)
                epoch_answers_post_train.append(post_train_answer)
                epoch_targets_post_train.append(target)

                pre_train_post_train_results.append({
                    "epoch": epoch,
                    "step": step,
                    "target": target,
                    **tracking_dict
                })

                # Save intermediate playbook
                if step % save_steps == 0:
                    intermediate_path = os.path.join(
                        playbook_dir, f"epoch_{epoch}_step_{step}_playbook.txt"
                    )
                    with open(intermediate_path, "w") as f:
                        f.write(self.playbook)

                # Periodic evaluation
                if step % eval_steps == 0:
                    print(f"\n{'='*40}")
                    print(f"EVALUATION AT EPOCH {epoch}, STEP {step}")
                    print(f"{'='*40}")

                    pre_train_accuracy = self.task.evaluate_accuracy(
                        epoch_answers_pre_train, epoch_targets_pre_train
                    )
                    post_train_accuracy = self.task.evaluate_accuracy(
                        epoch_answers_post_train, epoch_targets_post_train
                    )

                    val_results = {}
                    if val_samples:
                        val_results, val_error_log = evaluate_test_set(
                            self.task, self.generator, self.playbook,
                            val_samples, self.max_tokens, log_dir,
                            max_workers=test_workers, use_json_mode=use_json_mode
                        )

                    results.append({
                        "epoch": epoch,
                        "step": step,
                        "train_result": {
                            "pre_train_accuracy": pre_train_accuracy,
                            "post_train_accuracy": post_train_accuracy
                        },
                        "val_result": val_results,
                        "playbook_num_tokens": count_tokens(self.playbook),
                        "playbook_length": len(self.playbook),
                        "playbook_stats": get_playbook_stats(self.playbook)
                    })
                    error_logs.append({
                        "epoch": epoch,
                        "step": step,
                        "val_results": val_results,
                        "error_log": val_error_log
                    })

                    # Track best playbook
                    if val_results:
                        acc = val_results["accuracy"]
                        if acc > best_accuracy:
                            best_accuracy = acc
                            self.best_playbook = self.playbook
                            print(f"🎉 New best accuracy: {best_accuracy:.3f}")

                    with open(os.path.join(save_path, "train_results.json"), "w") as f:
                        json.dump({"best_accuracy": best_accuracy, "results": results}, f, indent=2)

                    with open(os.path.join(save_path, "val_results.json"), "w") as f:
                        json.dump(error_logs, f, indent=2)

            # End of epoch - save final playbook
            with open(os.path.join(playbook_dir, f"epoch_{epoch}_final_playbook.txt"), "w") as f:
                f.write(self.playbook)

        with open(os.path.join(save_path, "train_results.json"), "w") as f:
            json.dump({"best_accuracy": best_accuracy, "results": results}, f, indent=2)

        with open(os.path.join(save_path, "pre_train_post_train_results.json"), "w") as f:
            json.dump(pre_train_post_train_results, f, indent=2)

        with open(os.path.join(save_path, "final_playbook.txt"), "w") as f:
            f.write(self.playbook)

        with open(os.path.join(save_path, "best_playbook.txt"), "w") as f:
            f.write(self.best_playbook)

        print(f"\n{'='*60}")
        print(f"OFFLINE TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {best_accuracy:.3f}")
        print(f"{'='*60}\n")

        return {"best_validation_accuracy": best_accuracy}
