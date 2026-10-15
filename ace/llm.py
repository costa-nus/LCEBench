"""
LLM call wrapper for ACE: prompt truncation, logging and retries against a vLLM
OpenAI-compatible server.
"""
import time
import random
from datetime import datetime
import openai
from .logger import log_llm_call, log_problematic_request
from .utils import truncate_prompt

# Qwen3 chat template: answer directly, without a thinking block.
QWEN_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}

# Returned instead of a response when the server replies with empty content; it never matches an answer.
INCORRECT_DUE_TO_EMPTY_RESPONSE = "INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE, INCORRECT_DUE_TO_EMPTY_RESPONSE"


def timed_llm_call(client, model, prompt, role, call_id, max_tokens=4096, log_dir=None,
                   sleep_seconds=1, retries_on_timeout=6, attempt=1, use_json_mode=False, temperature=0.7,
                   playbook=None, prompt_builder=None):
    """
    Make a timed LLM call with error handling and retry logic.

    Empty responses in training ('train_*' call ids) and testing ('test_*') are returned as
    INCORRECT_DUE_TO_EMPTY_RESPONSE so the sample counts as wrong; timeouts, rate limits and
    server errors are retried up to ``retries_on_timeout`` times.

    Args:
        client: OpenAI client
        model: served model name
        prompt: text prompt to send
        role: role for logging (generator, reflector, curator)
        call_id: unique identifier for this call (format: {train|test}_{details})
        max_tokens: maximum tokens to generate
        log_dir: directory for detailed logging
        use_json_mode: whether to request a JSON object
        playbook: the only prompt content allowed to be truncated
        prompt_builder: callable that rebuilds the full prompt from a playbook

    Returns:
        tuple: (response_text, call_info_dict)
    """
    start_time = time.time()
    prompt_time = time.time()

    print(f"[{role.upper()}] Starting call {call_id}...")

    prompt_prepared = False

    while True:
        try:
            # Prepare once so retries reuse the exact same prompt. Prefer
            # shortening an explicitly supplied playbook; otherwise fall back
            # to truncating the full prompt from the end.
            if not prompt_prepared:
                prompt = truncate_prompt(
                    prompt,
                    max_output_tokens=max_tokens,
                    playbook=playbook,
                    prompt_builder=prompt_builder,
                )
                prompt_prepared = True

            api_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "seed": getattr(client, "sampling_seed", None),
                "max_completion_tokens": max_tokens,
                "extra_body": QWEN_EXTRA_BODY,
            }
            if use_json_mode:
                api_params["response_format"] = {"type": "json_object"}

            call_start = time.time()
            response = client.chat.completions.create(**api_params)
            call_end = time.time()

            # Check if response is valid
            if not response or not response.choices or len(response.choices) == 0:
                raise Exception("Empty response from API")

            response_time = time.time()
            total_time = response_time - start_time
            response_content = response.choices[0].message.content

            if response_content is None:
                raise Exception("API returned None content")

            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "response": response_content,
                "prompt_time": prompt_time - start_time,
                "response_time": response_time - prompt_time,
                "total_time": total_time,
                "call_time": call_end - call_start,
                "prompt_length": len(prompt),
                "response_length": len(response_content),
                "prompt_num_tokens": response.usage.prompt_tokens,
                "response_num_tokens": response.usage.completion_tokens,
            }

            print(f"[{role.upper()}] Call {call_id} completed in {total_time:.2f}s")

            if log_dir:
                log_llm_call(log_dir, call_info)

            return response_content, call_info

        except Exception as e:
            # Check for both timeout and rate limit errors
            is_timeout = any(k in str(e).lower() for k in ["timeout", "timed out", "connection"])
            is_rate_limit = any(k in str(e).lower() for k in ["rate limit", "429", "rate_limit_exceeded"])
            is_empty_response = "empty response" in str(e).lower() or "api returned none content" in str(e).lower()

            # Check for server errors (500, 502, 503, etc.) that should be retried
            is_server_error = False
            if hasattr(e, 'response'):
                try:
                    status_code = getattr(e.response, 'status_code', None)
                    if status_code and status_code >= 500:
                        is_server_error = True
                        print(f"[{role.upper()}] Server error detected: HTTP {status_code}")
                except:
                    pass

            # Also check for 500 errors in the error message itself
            if any(k in str(e).lower() for k in ["500 internal server error", "internal server error", "502 bad gateway", "503 service unavailable"]):
                is_server_error = True
                print(f"[{role.upper()}] Server error detected in message: {str(e)[:100]}...")

            if isinstance(e, openai.RateLimitError):
                is_rate_limit = True

            if isinstance(e, openai.InternalServerError):
                is_server_error = True
                print(f"[{role.upper()}] OpenAI InternalServerError detected")

            # Empty responses: count the sample as wrong
            if is_empty_response:
                print(f"\n🚨 Empty response for {call_id} ({type(e).__name__}: {e}); model={model}, "
                      f"json_mode={use_json_mode}, prompt length={len(prompt)}")
                log_problematic_request(call_id, prompt, model, api_params, e, log_dir)

                if call_id.startswith('train_') or call_id.startswith('test_'):
                    stage = "TRAINING" if call_id.startswith('train_') else "TEST"
                    print(f"[{role.upper()}] 🚨 Empty response - marking as INCORRECT for {call_id}")
                    error_time = time.time()
                    call_info = {
                        "role": role,
                        "call_id": call_id,
                        "model": model,
                        "prompt": prompt,
                        "error": f"{stage}_INCORRECT: " + str(e),
                        "total_time": error_time - start_time,
                        "prompt_length": len(prompt),
                        "response_length": 0,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3],
                        "datetime": datetime.now().isoformat(),
                        f"{stage.lower()}_marked_incorrect_due_to_empty_response": True
                    }
                    if log_dir:
                        log_llm_call(log_dir, call_info)
                    return INCORRECT_DUE_TO_EMPTY_RESPONSE, call_info

            # Retry logic for timeouts, rate limits, and server errors
            if (is_timeout or is_rate_limit or is_server_error) and attempt < retries_on_timeout:
                attempt += 1
                if is_rate_limit:
                    error_type = "rate limited"
                    base_sleep = sleep_seconds * 2
                elif is_server_error:
                    error_type = "server error (500+)"
                    base_sleep = sleep_seconds * 1.5  # Moderate delay for server errors
                else:
                    error_type = "timed out"
                    base_sleep = sleep_seconds
                jitter = random.uniform(0.5, 1.5)  # Add jitter to avoid thundering herd
                sleep_time = base_sleep * jitter
                print(f"[{role.upper()}] Call {call_id} {error_type}, sleeping {sleep_time:.1f}s then retrying "
                      f"({attempt}/{retries_on_timeout})...")
                time.sleep(sleep_time)
                continue

            error_time = time.time()
            call_info = {
                "role": role,
                "call_id": call_id,
                "model": model,
                "prompt": prompt,
                "error": str(e),
                "total_time": error_time - start_time,
                "prompt_length": len(prompt),
                "attempt": attempt,
            }

            print(f"[{role.upper()}] Call {call_id} failed after {error_time - start_time:.2f}s: {e}")

            if log_dir:
                log_llm_call(log_dir, call_info)

            raise e
