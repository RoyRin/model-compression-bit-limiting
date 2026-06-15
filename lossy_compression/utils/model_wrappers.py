from lossy_compression import LLM, SLM, DEFAULT_LLM_TEMPERATURE, DEFAULT_SLM_TEMPERATURE, DEFAULT_SEED, LOCAL_SLM, USE_ANTHROPIC, QUESTION_SLM
import re

from lossy_compression import model_completion
from lossy_compression.utils.formatting import EXAMPLES_BLOCK

# Global flag to enable API usage logging
LOG_API_USAGE = True  # Set to True to always log API input/output sizes


def generate_LLM_response(
        prompt,
        system_prompt=None,  # NEW: System instructions
        model_name=LLM,
        temperature=DEFAULT_LLM_TEMPERATURE,
        seed=None,
        max_tokens=4000,  # Reduced to stay within Haiku's 4096 limit
        verbose=False):
    """Query LLM and return response."""
    # Calculate input lengths
    prompt_len = len(prompt) if prompt else 0
    system_len = len(system_prompt) if system_prompt else 0
    total_input_len = prompt_len + system_len
    # Rough token estimate (1 token ≈ 4 chars for English)
    estimated_tokens = total_input_len // 4

    if verbose or LOG_API_USAGE:
        model_short = model_name.split(
            '/')[-1] if '/' in model_name else model_name.split(
                '-')[-1] if '-' in model_name else model_name
        print(
            f"📤 API Call: {model_short} | Input: {total_input_len:,} chars (~{estimated_tokens:,} tokens)"
        )
        if verbose:
            print(f"   Prompt: {prompt[:100]}...")
            print(f"   System: {system_len} chars")
            print(f"   Temperature: {temperature}")

    resp = model_completion(
        prompt,
        model=model_name,
        system=system_prompt,  # Pass system prompt separately
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens)

    if not resp or len(resp.strip()) == 0:
        raise ValueError(
            f"Empty response received from LLM ({model_name}). This may indicate an API error or model issue."
        )

    output_len = len(resp) if resp else 0
    output_tokens = output_len // 4

    if verbose or LOG_API_USAGE:
        model_short = model_name.split(
            '/')[-1] if '/' in model_name else model_name.split(
                '-')[-1] if '-' in model_name else model_name
        print(
            f"📥 API Response: {model_short} | Output: {output_len:,} chars (~{output_tokens:,} tokens)"
        )
        if verbose:
            print(f"   Response: {resp[:200]}...")

    return resp


def generate_SLM_response(
        prompt,
        system_prompt=None,  # NEW: System instructions
        model_name=SLM,
        temperature=DEFAULT_SLM_TEMPERATURE,
        prefill=None,
        verbose=False):
    """Generate response from small language model.

    Args:
        prompt: Input prompt
        model_name: Model name to use
        temperature: Temperature for generation (default: 0.0)
        seed: Random seed for reproducibility
        verbose: Whether to print debug information
    """
    # Calculate input lengths
    prompt_len = len(prompt) if prompt else 0
    system_len = len(system_prompt) if system_prompt else 0
    prefill_len = len(prefill) if prefill else 0
    total_input_len = prompt_len + system_len + prefill_len
    estimated_tokens = total_input_len // 4

    if verbose or LOG_API_USAGE:
        model_short = model_name.split(
            '/')[-1] if '/' in model_name else model_name.split(
                '-')[-1] if '-' in model_name else model_name
        print(
            f"📤 API Call: {model_short} | Input: {total_input_len:,} chars (~{estimated_tokens:,} tokens)"
        )
        if verbose:
            print(f"   Prompt: {prompt[:100]}...")
            print(
                f"   System: {system_len} chars, Prefill: {prefill_len} chars")
            print(f"   Temperature: {temperature}")

    print(
        f"DEBUG:\nprompt: {prompt}\nsystem: {system_prompt}\nprefill: {prefill}\n"
    )
    resp = model_completion(
        prompt,
        model=model_name,
        system=system_prompt,  # Pass system prompt separately
        temperature=temperature,
        prefill=prefill)

    if not resp or len(resp.strip()) == 0:
        raise ValueError(
            f"Empty response received from SLM ({model_name}). This may indicate an API error or model issue."
        )

    output_len = len(resp) if resp else 0
    output_tokens = output_len // 4

    if verbose or LOG_API_USAGE:
        model_short = model_name.split(
            '/')[-1] if '/' in model_name else model_name.split(
                '-')[-1] if '-' in model_name else model_name
        print(
            f"📥 API Response: {model_short} | Output: {output_len:,} chars (~{output_tokens:,} tokens)"
        )
        if verbose:
            print(f"   Response: {resp[:100]}...")

    return resp


def extract_yes_no_answer(resp):
    """Extract yes/no answer from response.
    Return True/False
    """
    yes_answer = "yes" in resp.lower()
    return yes_answer


def parse_rating_response(resp):
    """Parse numeric rating from response."""
    # Look for patterns like "Rating: X", "X/10", or just a number at the start
    # First try to find rating patterns
    patterns = [
        r'(?:Rating|Score):\s*(\d+)',  # Rating: 5 or Score: 5
        r'^(\d+)(?:/10)?',  # 5 or 5/10 at start of line
        r'(\d+)/10',  # 5/10 anywhere
        r'^\s*(\d+)\s*$',  # Just a number on its own line
    ]

    for pattern in patterns:
        match = re.search(pattern, resp, re.MULTILINE | re.IGNORECASE)
        if match:
            rating = int(match.group(1))
            # Ensure rating is within valid range
            if 1 <= rating <= 10:
                return rating

    # Fallback: find first standalone number (not part of larger number)
    numbers = re.findall(r'\b(\d+)\b', resp)
    for num_str in numbers:
        num = int(num_str)
        if 1 <= num <= 10:
            return num
    print(f"response: {resp}\n")
    # If no valid rating found, default to 1 (poor)
    print(f"Warning: Could not parse rating from response: {resp[:100]}...")
    raise ValueError(f"Could not parse rating from response: {resp}")
    return 1


def evaluate_code_answer(prompt,
                         answer,
                         temperature=DEFAULT_LLM_TEMPERATURE,
                         seed=None,
                         verbose=False,
                         model_name=LLM):
    """Evaluate if a code answer is correct and adequately solves the prompt."""

    if verbose:
        print("🔍 Evaluating code answer...")
        print(f"📝 Prompt: {prompt[:100]}...")
        if use_gpt_judge:
            print(f"⚖️ Using GPT-5 as judge")

    eval_prompt = f"""You are an impartial judge evaluating whether an AI-generated code answer correctly solves the given prompt.

Prompt (coding task):
{prompt}

Proposed Answer (code):
{answer}

Please evaluate this answer based on:
1. **Correctness**: Does the code actually solve the problem as stated?
2. **Accuracy**: Are there any logical, syntactic, or semantic errors?
3. **Completeness**: Does it handle the key requirements (not necessarily edge cases)?
4. **Clarity**: Is the solution coherent and understandable?

Rate the answer on a scale of 1-10 where:
- 9-10 = Correct and complete. The code fully solves the task as described. (⚠️ Only give 9+ if the code is correct.)
- 7-8 = Mostly correct. Minor issues, small omissions, or slight inaccuracies, but the core solution is valid.
- 5-6 = Partially correct. Addresses the problem but with significant gaps, missing functionality, or notable errors.
- 3-4 = Incorrect or very incomplete. Attempts the problem but fails in correctness or relevance.
- 1-2 = Completely incorrect, irrelevant, or nonsensical.

IMPORTANT: Provide ONLY a single number between 1 and 10. Do not include any other text or explanation.

Rating:"""

    if "gpt" in model_name:
        from utils.llm_api import openai_completion
        resp = openai_completion(prompt=eval_prompt,
                                 model=model_name,
                                 temperature=temperature,
                                 seed=seed if seed else 42,
                                 max_tokens=10)
        if verbose:
            print(f"gpt {model_name} response: `{resp}`\n")
    else:
        resp = generate_LLM_response(eval_prompt,
                                     model_name=model_name,
                                     temperature=temperature,
                                     seed=seed,
                                     verbose=False)
        if verbose:
            print(f"claude response: {resp}\n")

    rating = parse_rating_response(resp)
    if verbose:
        print(f"📊 Code answer rating: {rating}/10")

    return rating


def evaluate_answer_standalone(prompt,
                               answer,
                               temperature=DEFAULT_LLM_TEMPERATURE,
                               seed=None,
                               verbose=False,
                               model_name=LLM):
    """Evaluate if an answer adequately responds to a prompt (standalone evaluation)."""
    if verbose:
        print("🔍 Evaluating answer (standalone)...")
        print(f"📝 Prompt: {prompt[:100]}...")
        if "gpt" in model_name:
            print(f"⚖️ Using {model_name} as judge")
        else:
            print(f"⚖️ Using {model_name} as judge")

    eval_prompt = f"""You are an impartial judge evaluating whether an AI response adequately answers a question.

Question: {prompt}

Answer: {answer}

Please evaluate this answer based on:
1. Does it directly address the question asked?
2. Is the information accurate and relevant?
3. Is it reasonably complete (doesn't need to be exhaustive)?
4. Is it coherent and well-structured?

Rate the answer on a scale of 1-10 where:
9-10 = Excellent - Fully answers the question with accurate, complete information
7-8 = Good - Answers the question well with minor gaps
5-6 = Adequate - Answers the core question but missing important details
3-4 = Poor - Partially addresses the question or has significant issues
1-2 = Very Poor - Fails to answer the question or mostly incorrect

IMPORTANT: Provide ONLY a single number between 1 and 10. Do not include any other text or explanation.

Rating:"""

    if "gpt" in model_name:
        # Use GPT-4 as judge
        from utils.llm_api import openai_completion
        resp = openai_completion(prompt=eval_prompt,
                                 model=model_name,
                                 temperature=temperature,
                                 seed=seed if seed else 42,
                                 max_tokens=10)
        print(f"gpt {model_name} response: `{resp}`\n")
    else:
        # Use specified model (or default LLM) as judge
        resp = generate_LLM_response(eval_prompt,
                                     model_name=model_name,
                                     temperature=temperature,
                                     seed=seed,
                                     verbose=False)
        if verbose:
            print(f"claude response: {resp}\n")

    rating = parse_rating_response(resp)
    if verbose:
        print(f"📊 Standalone quality rating: {rating}/10")

    return rating


def evaluate_answer_quality_comparison(original_prompt,
                                       large_model_answer,
                                       small_model_answer,
                                       temperature=DEFAULT_LLM_TEMPERATURE,
                                       seed=None,
                                       verbose=False,
                                       model_name=LLM):
    """More nuanced evaluation than just yes/no."""
    if verbose:
        print("🔍 Evaluating answer quality...")
        print(f"📝 Original prompt: {original_prompt[:100]}...")
        if "gpt" in model_name:
            print(f"⚖️ Using {model_name} as impartial judge")
        else:
            print(f"⚖️ Using {model_name} as impartial judge")

    # Use impartial prompt that doesn't reveal which is which
    prompt = f"""You are an impartial judge evaluating two different AI model responses. 
Please be completely objective and fair in your evaluation.

Question: {original_prompt}

Model A's Answer: {large_model_answer}

Model B's Answer: {small_model_answer}

As an impartial judge, please evaluate Model B's answer compared to Model A's answer.
Consider accuracy, completeness, and quality of explanation.

Rate Model B's answer on a scale of 1-10 where:
10 = Equally good or better than Model A
7-9 = Very good, minor differences from Model A
4-6 = Acceptable, some important differences
1-3 = Poor, major differences or missing key points

IMPORTANT: Provide ONLY a single number between 1 and 10. Do not include any other text or explanation.

Rating:"""

    if "gpt" in model_name:
        # Use GPT as judge
        from utils.llm_api import openai_completion
        resp = openai_completion(prompt=prompt,
                                 model=model_name,
                                 temperature=temperature,
                                 seed=seed if seed else 42,
                                 max_tokens=10)
        print(f"gpt {model_name} response: `{resp}`\n")
    else:
        # Use default LLM (Claude)
        resp = generate_LLM_response(prompt,
                                     model_name=model_name,
                                     temperature=temperature,
                                     seed=seed,
                                     verbose=False)
        print(f"claude response: {resp}\n")

    rating = parse_rating_response(resp)

    if verbose:
        print(f"📊 Quality rating: {rating}/10")

    return rating


def format_guiding_questions_and_answers(guiding_questions, guiding_answers):
    """Format guiding questions and answers for prompt inclusion."""
    assert len(guiding_answers) == len(guiding_questions)
    resp = []

    for guiding_question, guiding_answer in zip(guiding_questions,
                                                guiding_answers):
        guiding_answer_str = "Yes" if guiding_answer else "No"
        resp.append(
            f"Question: {guiding_question} ; Answer: {guiding_answer_str}")

    return "\n".join(resp)


def small_model_generate_helpful_binary_questions(
        prompt,
        system_prompt,  # NEW
        original_answer,
        existing_guiding_questions,
        existing_guiding_answers,
        question_model_name=QUESTION_SLM,
        temperature=DEFAULT_SLM_TEMPERATURE,
        seed=None,
        verbose=False,
        predict_base_rate=False):
    """Generate helpful binary questions for the small model.

    Args:
        predict_base_rate: If True, also predict P(yes) for probabilistic compression.
                          Returns (question, p_yes) tuple instead of just question.
    """
    if verbose:
        print("❓ Generating helpful binary question...")
        print(f"📝 Current answer: {original_answer[:100]}...")
        print(f"📋 Existing Q&A pairs: {len(existing_guiding_questions)}")
        print(f"🤖 Question model: {question_model_name}")

    existing_guiding_questions_and_answers = format_guiding_questions_and_answers(
        existing_guiding_questions, existing_guiding_answers)

    # Build list of previously asked questions for clarity
    previous_questions_list = "\n".join([
        f"- {q}" for q in existing_guiding_questions
    ]) if existing_guiding_questions else "None"

    if predict_base_rate:
        # Ask for both question AND probability prediction
        question_prompt = f"""You are a small language model trying to answer a prompt. The original prompt is: {prompt}.
Your current answer is: {original_answer}.

Generate a NEW yes/no question that would help you answer this prompt better, AND predict the probability that the answer is "yes".

The question should be:
- A yes/no question
- Specific and focused
- Something you can use to improve your answer
- MUST BE DIFFERENT from all previously asked questions above
- Should explore a new aspect not yet covered

IMPORTANT: Do not repeat or rephrase any of the questions listed below:

Previously asked questions and answers (DO NOT repeat these):
{existing_guiding_questions_and_answers}

Format your response EXACTLY as:
QUESTION: <your yes/no question>
P_YES: <probability between 0.0 and 1.0>

Example:
QUESTION: Is the algorithm using dynamic programming?
P_YES: 0.7

Respond with ONLY the question and probability, nothing else."""
    else:
        question_prompt = f"""You are a small language model trying to answer a prompt. The original prompt is: {prompt}.
Your current answer is: {original_answer}.

Generate a NEW yes/no question that would help you answer this prompt better.
The question should be:
- A yes/no question
- Specific and focused
- Something you can use to improve your answer
- MUST BE DIFFERENT from all previously asked questions above
- Should explore a new aspect not yet covered

IMPORTANT: Do not repeat or rephrase any of the questions listed below:

Previously asked questions and answers  (DO NOT repeat these):
{existing_guiding_questions_and_answers}

Only ask a question, do not include any other text in your answer (do not explain why you are asking the question).
New Question:"""

    resp = generate_SLM_response(question_prompt,
                                 system_prompt=system_prompt,
                                 model_name=question_model_name,
                                 temperature=temperature,
                                 verbose=False)

    if predict_base_rate:
        # Parse question and probability
        question, p_yes = parse_question_with_probability(resp,
                                                          verbose=verbose)
        if verbose:
            print(f"❓ Generated question: {question}")
            print(f"📊 Predicted P(yes): {p_yes:.3f}")
        return question, p_yes
    else:
        if verbose:
            print(f"❓ Generated question: {resp}")
        return resp


def parse_question_with_probability(resp, verbose=False):
    """Parse a response containing both question and probability.

    Expected format:
    QUESTION: <question>
    P_YES: <probability>

    Returns:
        (question, p_yes) tuple. p_yes defaults to 0.5 if parsing fails.
    """
    import re

    # Try to extract QUESTION: and P_YES:
    question_match = re.search(r'QUESTION:\s*(.+?)(?:\n|P_YES:|$)', resp,
                               re.IGNORECASE | re.DOTALL)
    p_yes_match = re.search(r'P_YES:\s*([\d.]+)', resp, re.IGNORECASE)

    if question_match:
        question = question_match.group(1).strip()
    else:
        # Fallback: use the whole response as the question
        question = resp.strip()
        if verbose:
            print(
                f"⚠️ Could not parse QUESTION from response, using full response"
            )

    if p_yes_match:
        try:
            p_yes = float(p_yes_match.group(1))
            # Clamp to valid probability range with small epsilon for numerical stability
            p_yes = max(0.001, min(0.999, p_yes))
        except ValueError:
            p_yes = 0.5
            if verbose:
                print(
                    f"⚠️ Could not parse P_YES from response, defaulting to 0.5"
                )
    else:
        p_yes = 0.5
        if verbose:
            print(f"⚠️ Could not find P_YES in response, defaulting to 0.5")

    return question, p_yes


def calculate_probabilistic_bits(p_yes, actual_answer):
    """Calculate bits needed to encode the actual answer given predicted P(yes).

    Uses Shannon information: bits = -log2(P(actual_answer))

    Args:
        p_yes: Predicted probability of "yes" (from SLM)
        actual_answer: Boolean, True if LLM answered "yes"

    Returns:
        Number of bits needed to encode this answer
    """
    import math

    # Clamp probability to avoid log(0)
    p_yes = max(0.001, min(0.999, p_yes))

    if actual_answer:  # LLM said yes
        p_actual = p_yes
    else:  # LLM said no
        p_actual = 1.0 - p_yes

    # Shannon information content: -log2(p)
    bits = -math.log2(p_actual)

    return bits


def large_model_answer_binary_question(
        original_prompt,
        system_prompt,  # NEW
        large_model_answer,
        small_model_answer,
        question,
        temperature=DEFAULT_LLM_TEMPERATURE,
        seed=None,
        verbose=False):
    """Get binary answer from large model for a specific question."""
    if verbose:
        print("🤖 LLM answering binary question...")
        print(f"❓ Question: {question}")

    prompt = f"""You are a large language model that is helping guide a small model. The original prompt is: {original_prompt}. Your answer is: {large_model_answer}. The small model answer's is: {small_model_answer}. 
        
The question is: {question}.

Please answer yes or no only."""

    resp = generate_LLM_response(prompt,
                                 system_prompt=system_prompt,
                                 model_name=LLM,
                                 temperature=temperature,
                                 seed=seed,
                                 verbose=False)
    answer = extract_yes_no_answer(resp)

    if verbose:
        print(f" LLM answer: {'YES' if answer else 'NO'}")

    return answer


def batch_generate_questions(prompt,
                             system_prompt,
                             original_answer,
                             existing_guiding_questions,
                             existing_guiding_answers,
                             num_questions=10,
                             question_model_name=QUESTION_SLM,
                             temperature=DEFAULT_SLM_TEMPERATURE,
                             seed=None,
                             verbose=False,
                             evaluation_mode="default",
                             predict_base_rate=False):
    """Generate N questions at once instead of one at a time.

    Args:
        predict_base_rate: If True, also predict P(yes) for each question.
                          Returns (list_of_questions, list_of_p_yes) tuple.

    Returns:
        List of questions, or (questions, probabilities) if predict_base_rate=True
    """
    if verbose:
        print(f"❓ Generating {num_questions} questions in batch mode...")
        print(f"📋 Existing Q&A pairs: {len(existing_guiding_questions)}")
        print(f"🤖 Question model: {question_model_name}")

    existing_qa_text = ""
    if existing_guiding_questions and existing_guiding_answers:
        existing_qa_text = "\n\nPrevious Q&A pairs:\n"
        for q, a in zip(existing_guiding_questions, existing_guiding_answers):
            existing_qa_text += f"Q: {q}\nA: {'Yes' if a else 'No'}\n"

    # Build format instructions based on whether we want probabilities
    if predict_base_rate:
        format_instruction = f"""Format your response as a numbered list with probabilities:
1. [Question 1] | P_YES: [probability 0.0-1.0]
2. [Question 2] | P_YES: [probability 0.0-1.0]
...
{num_questions}. [Question {num_questions}] | P_YES: [probability 0.0-1.0]

Example:
1. Is the algorithm correct? | P_YES: 0.7
2. Are the edge cases handled? | P_YES: 0.4

Return ONLY the numbered list with probabilities, nothing else."""
    else:
        format_instruction = f"""Format your response as a numbered list:
1. [Question 1]
2. [Question 2]
...
{num_questions}. [Question {num_questions}]

Return ONLY the numbered list of questions, nothing else."""

    # Math-specific batch generation
    if evaluation_mode == "math":
        user_prompt = f"""Problem: {prompt}

Current attempt:
{original_answer}
{existing_qa_text}

Generate exactly {num_questions} NEW yes/no questions that would help improve this solution.
Each question should target a different aspect: method choice, formula validity, arithmetic, constraints, or final checks.
Do not repeat or paraphrase existing questions.
{"Also predict the probability that the answer to each question is 'yes'." if predict_base_rate else ""}

{format_instruction}"""

        system = "You are generating diagnostic yes/no questions to improve a math solution."
    elif evaluation_mode == "science":
        user_prompt = f"""Science Question: {prompt}

Current answer:
{original_answer}
{existing_qa_text}

Generate exactly {num_questions} NEW yes/no questions that would help improve this scientific answer.
Each question should target different aspects: scientific principles, facts, reasoning, elimination of options, or validity of conclusions.
Do not repeat or paraphrase existing questions.
{"Also predict the probability that the answer to each question is 'yes'." if predict_base_rate else ""}

{format_instruction}"""

        system = "You are generating diagnostic yes/no questions to improve a scientific multiple-choice answer."
    else:
        # Default batch generation
        user_prompt = f"""Original prompt: {prompt}

Current answer: {original_answer}
{existing_qa_text}

Generate exactly {num_questions} NEW yes/no questions that would help improve this answer.
Each question should target a different aspect of the answer.
Do not repeat or paraphrase existing questions.
{"Also predict the probability that the answer to each question is 'yes'." if predict_base_rate else ""}

{format_instruction}"""

        system = None

    resp = generate_SLM_response(user_prompt,
                                 system_prompt=system,
                                 model_name=question_model_name,
                                 temperature=temperature,
                                 prefill="",
                                 verbose=verbose)

    # Parse the numbered list
    questions = []
    probabilities = []
    lines = resp.strip().split('\n')
    for line in lines:
        # Match patterns like "1. Question" or "1) Question" or just numbered lines
        import re
        if predict_base_rate:
            # Try to match "1. Question | P_YES: 0.7" format
            match = re.match(r'^\d+[\.)\s]+(.+?)\s*\|\s*P_YES:\s*([\d.]+)',
                             line.strip(), re.IGNORECASE)
            if match:
                questions.append(match.group(1).strip())
                try:
                    p = float(match.group(2))
                    p = max(0.001, min(0.999,
                                       p))  # Clamp for numerical stability
                except ValueError:
                    p = 0.5
                probabilities.append(p)
            else:
                # Fallback: try without probability
                match = re.match(r'^\d+[\.)\s]+(.+)$', line.strip())
                if match:
                    questions.append(match.group(1).strip())
                    probabilities.append(0.5)  # Default probability
        else:
            match = re.match(r'^\d+[\.)\s]+(.+)$', line.strip())
            if match:
                questions.append(match.group(1).strip())

    if verbose:
        print(f"✅ Generated {len(questions)} questions")
        if len(questions) < num_questions:
            print(
                f"⚠️ Warning: Expected {num_questions} questions but got {len(questions)}"
            )
        if predict_base_rate:
            print(f"📊 Predicted probabilities: {probabilities[:5]}..."
                  if len(probabilities) >
                  5 else f"📊 Predicted probabilities: {probabilities}")

    if predict_base_rate:
        # Pad probabilities if needed
        while len(probabilities) < len(questions):
            probabilities.append(0.5)
        return questions[:num_questions], probabilities[:num_questions]
    else:
        return questions[:
                         num_questions]  # Ensure we don't return more than requested


def batch_answer_questions(original_prompt,
                           system_prompt,
                           large_model_answer,
                           small_model_answer,
                           questions,
                           model_name=LLM,
                           temperature=DEFAULT_LLM_TEMPERATURE,
                           seed=None,
                           verbose=False,
                           evaluation_mode="default"):
    """Answer multiple yes/no questions at once.
    
    Returns:
        List of boolean answers
    """
    if verbose:
        print(f"🤖 LLM answering {len(questions)} questions in batch mode...")
        print(f"📋 Model: {model_name}")

    # Format questions as a numbered list
    questions_text = "\n".join(
        [f"{i+1}. {q}" for i, q in enumerate(questions)])

    # Math-specific batch answering
    if evaluation_mode == "math":
        user_prompt = f"""Problem: {original_prompt}

Reference solution:
{large_model_answer}

Current attempt:
{small_model_answer}

Answer each yes/no question about the current attempt:
{questions_text}

For each question, respond with ONLY "Yes" or "No" on a separate line.
Format your response as:
1. Yes/No
2. Yes/No
...
{len(questions)}. Yes/No

Return ONLY the numbered list of Yes/No answers, nothing else."""

        system = "You are evaluating a math solution by answering diagnostic yes/no questions."
    elif evaluation_mode == "science":
        user_prompt = f"""Science Question: {original_prompt}

Reference answer:
{large_model_answer}

Current answer:
{small_model_answer}

Answer each yes/no question about the current answer:
{questions_text}

For each question, respond with ONLY "Yes" or "No" on a separate line.
Format your response as:
1. Yes/No
2. Yes/No
...
{len(questions)}. Yes/No

Return ONLY the numbered list of Yes/No answers, nothing else."""

        system = "You are evaluating a scientific answer by answering diagnostic yes/no questions. Focus on scientific accuracy, reasoning, and whether the correct option was selected."
    else:
        # Default batch answering
        user_prompt = f"""Original prompt: {original_prompt}

Reference answer:
{large_model_answer}

Current answer:
{small_model_answer}

Answer each yes/no question about the current answer:
{questions_text}

For each question, respond with ONLY "Yes" or "No" on a separate line.
Format your response as:
1. Yes/No
2. Yes/No
...
{len(questions)}. Yes/No

Return ONLY the numbered list of Yes/No answers, nothing else."""

        system = system_prompt

    resp = generate_LLM_response(
        user_prompt,
        system_prompt=system,
        model_name=model_name,
        temperature=temperature,
        seed=seed,
        max_tokens=1000,  # Shorter for just yes/no answers
        verbose=verbose)

    # Parse the numbered list of answers
    answers = []
    lines = resp.strip().split('\n')
    for line in lines:
        # Match patterns like "1. Yes" or "1) No" or variations
        import re
        match = re.match(r'^\d+[\.)\s]+\s*(yes|no)', line.strip(),
                         re.IGNORECASE)
        if match:
            answer_text = match.group(1).lower()
            answers.append(answer_text == 'yes')

    if verbose:
        print(f"✅ Received {len(answers)} answers")
        if len(answers) != len(questions):
            print(
                f"⚠️ Warning: Expected {len(questions)} answers but got {len(answers)}"
            )

    # If we got fewer answers than questions, pad with False
    while len(answers) < len(questions):
        answers.append(False)

    return answers[:len(
        questions)]  # Ensure we don't return more than requested


def small_model_improve_with_guidance(original_prompt,
                                      system_prompt,
                                      small_model_answer,
                                      guiding_questions=None,
                                      guiding_answers=None,
                                      open_ended_guidances=None,
                                      small_model_name=SLM,
                                      temperature=DEFAULT_SLM_TEMPERATURE,
                                      seed=None,
                                      use_local=False,
                                      local_model=None,
                                      verbose=False):
    """Generate improved answer using both binary Q&A pairs and open-ended guidance."""

    # Detect if this is a code generation task
    is_code_task = (system_prompt and ("code" in system_prompt.lower() or "python" in system_prompt.lower())) or \
                   ("def " in original_prompt or "```" in original_prompt or "import " in original_prompt)

    # Build the system prompt with ALL instructions and guidance
    system_parts = [
        "You are a Python code completion assistant.",
        "Complete the given Python function by providing the full implementation including the function signature.",
        "Return only valid Python code without any markdown formatting, explanations, or additional text.",
        EXAMPLES_BLOCK,  # <--- add this line
    ]

    # Add Q&A guidance to the system prompt
    if guiding_questions and guiding_answers:
        system_parts.append("\nRequirements based on Q&A:")
        for q, a in zip(guiding_questions, guiding_answers):
            # Handle boolean answers properly
            if isinstance(a, bool):
                answer_is_yes = a
            elif isinstance(a, str):
                answer_is_yes = a.lower() in ['yes', 'true', '1']
            else:
                answer_is_yes = bool(a)

            if answer_is_yes:
                system_parts.append(f"- {q}: YES")
            else:
                system_parts.append(f"- {q}: NO")

    # Add note about previous attempt if exists
    if small_model_answer:
        system_parts.append(
            f"\nNote: A previous attempt had issues. Ensure your implementation addresses all the requirements above."
        )

    improvement_system_prompt = "\n".join(system_parts)

    # User prompt is JUST the code to complete - super clean!

    prompt_parts = [
        f"Original question: {original_prompt}",
        f"\nCurrent answer: {small_model_answer}"
    ]

    if guiding_questions and guiding_answers:
        guiding_qa = format_guiding_questions_and_answers(
            guiding_questions, guiding_answers)
        prompt_parts.append(
            f"\nAdditional information to incorporate: {guiding_qa}")

    if open_ended_guidances:
        combined_guidance = "\n".join([f"- {g}" for g in open_ended_guidances])
        prompt_parts.append(f"\nAdditional guidance:\n{combined_guidance}")

    prompt_parts.append(
        "\nProvide an improved answer that incorporates all the above information, without any preamble or acknowledgment."
    )
    prompt_parts.append(
        f"Please repeat the prompt in your answer, and answer the original prompt. Do not include any other text in your answer. If there are multiple functions in the prompt, include all of them in your answer"  #:\n{original_prompt}"
    )

    prompt = "\n".join(prompt_parts)
    prefill = original_prompt.rstrip()  # remove the trailing space

    # Generate response

    new_response = generate_SLM_response(
        prompt,
        system_prompt=improvement_system_prompt,
        model_name=small_model_name,
        temperature=temperature,
        prefill=prefill,
        verbose=False)

    if verbose:
        print(f"🔄 Improved answer: {new_response[:100]}...")

    return new_response


##################
## MATH
##################

# =========================
# 1) Math system prompts
# =========================

EXAMPLES_MATH = r"""### EXAMPLES OF EXPECTED BEHAVIOR

❌ BAD (meta/prose):
Here's the solution you asked for:
\[
\text{(…work…)}
\]

✅ GOOD (only problem, solution, final answer):
Problem: Evaluate \(\int_0^1 3x^2\,dx\).

Solution:
\[
\int_0^1 3x^2\,dx = \left[x^3\right]_0^1 = 1.
\]
\[
\boxed{1}
\]

❌ BAD (no steps, extra words):
The answer is clearly 4 because...

✅ GOOD (clear steps, boxed final):
Problem: Solve \(2x+3=11\).

Solution:
\[
2x = 8 \Rightarrow x = 4.
\]
\[
\boxed{4}
\]
"""

MATH_SOLVER_SYSTEM = """You are a math solution assistant.

Output rules (must follow):
- Return only the math solution content (no markdown fences, no chatter, no apologies).
- Start with: Problem: <repeat the problem in one short line>.
- Then a section: Solution: with clear, minimal but correct steps (LaTeX math allowed).
- End with the final NUMERICAL result on its own line as \\boxed{...}.
- CRITICAL: The boxed answer must be a single integer (0-999 for AIME problems).
- Do NOT use symbolic expressions like \\boxed{(p+q) mod 1000} - compute the actual number!
- Do not include any text after the boxed answer.

Quality rules:
- Reasoning must be mathematically sound and complete enough to justify the answer.
- Arithmetic/algebra must be correct.
- Calculate all expressions to get a final numerical value.
- Keep steps concise; avoid unnecessary commentary.

Example of CORRECT final answer: \\boxed{247}
Example of INCORRECT final answer: \\boxed{(123 + 124) mod 1000}

""" #+ EXAMPLES_MATH - HACK - removed examples for now

SCIENCE_SOLVER_SYSTEM = """You are an expert scientist with deep knowledge across physics, chemistry, biology, and other scientific domains.

When answering multiple-choice questions:
- Analyze the question and all options carefully
- Show your scientific reasoning step by step
- Consider each option and explain why it is correct or incorrect
- Clearly state your final answer as a single letter (A, B, C, or D)

Output format:
- Begin with "Analysis:" and explain your reasoning
- End with "Answer: [LETTER]" on its own line
- The answer must be exactly one letter: A, B, C, or D
- Do not include explanations after stating the answer

Quality requirements:
- Use accurate scientific principles and facts
- Consider all relevant concepts and theories
- Apply appropriate formulas or relationships when needed
- Be precise with scientific terminology
"""

MATH_JUDGE_SYSTEM = """You are an impartial math judge.

Evaluate the solution strictly on:
1) Mathematical soundness of the approach.
2) Correctness of calculations.
3) Correctness of the final numerical answer.
4) Clarity and completeness of reasoning.

CRITICAL: You MUST output ONLY a single integer between 1 and 10.
Do NOT include any other text, explanation, or commentary.
Just output a number like: `7`
Nothing else."""

# =========================================
# 2) Math answer extraction / post-processing
# =========================================
import re
from fractions import Fraction
from typing import Optional, Tuple

BOXED_NUM = re.compile(r"\\boxed\{\s*([-+]?\d+(?:/\d+)?)\s*\}")
EXPLICIT_NUM = re.compile(r"(?:answer\s*[:is]\s*)([-+]?\d+(?:/\d+)?)", re.I)
LAST_INT_1_3 = re.compile(r"\b([-+]?\d{1,3})\b")
GEN_FRACTION = re.compile(r"\b([-+]?\d+/\d+)\b")


def _simplify_fraction_text(s: str) -> str:
    """Return canonical 'a/b' or integer string if divisible."""
    try:
        f = Fraction(s.strip())
        if f.denominator == 1:
            return str(f.numerator)
        return f"{f.numerator}/{f.denominator}"
    except Exception:
        return s.strip()


def extract_math_final_answer(text: str,
                              *,
                              aime: bool = False) -> Optional[str]:
    """
    Extract final numeric answer from a math solution.
    Priority: \boxed{...} > 'Answer: ...' > last integer/fraction heuristic.
    - General mode: returns 'int' or 'a/b' canonical form when possible.
    - AIME mode: returns an integer 0..999 (no leading zeros) or None if invalid.
    """
    # 1) Boxed
    m = BOXED_NUM.search(text)
    if m:
        candidate = m.group(1)
        if aime:
            try:
                n = int(candidate)
                return str(n) if 0 <= n <= 999 else None
            except Exception:
                return None
        # general: allow fraction or int, canonicalize
        if "/" in candidate:
            return _simplify_fraction_text(candidate)
        return str(int(candidate))  # normalize leading zeros/sign
    # 2) Explicit "Answer: N"
    m = EXPLICIT_NUM.search(text)
    if m:
        candidate = m.group(1)
        if aime:
            try:
                n = int(candidate)
                return str(n) if 0 <= n <= 999 else None
            except Exception:
                return None
        if "/" in candidate:
            return _simplify_fraction_text(candidate)
        return str(int(candidate))
    # 3) Fallback:
    if aime:
        ints = LAST_INT_1_3.findall(text)
        if ints:
            try:
                n = int(ints[-1])
                return str(n) if 0 <= n <= 999 else None
            except Exception:
                return None
        return None
    # general: prefer the last fraction if any, else last integer
    fracs = GEN_FRACTION.findall(text)
    if fracs:
        return _simplify_fraction_text(fracs[-1])
    ints = re.findall(r"\b([-+]?\d+)\b", text)
    if ints:
        return str(int(ints[-1]))
    return None


# =========================================
# 3) Math evaluators (rating + exact match)
# =========================================


def evaluate_math_solution_quality(problem: str,
                                   solution_text: str,
                                   model_name=LLM,
                                   temperature=0.0,
                                   seed=None,
                                   verbose: bool = False) -> int:
    """
    Ask an LLM judge for a 1–10 rating on math soundness, correctness, clarity.
    Returns an integer 1..10; raises if cannot parse.
    """
    eval_prompt = f"""Problem: {problem}

Proposed solution:
{solution_text}

Rate the solution considering:
- soundness of approach
- correctness of calculations
- correctness of final numerical answer
- clarity and completeness of reasoning

IMPORTANT: Provide ONLY a single number between 1 and 10. Do not include any other text or explanation.
Rating:"""

    if verbose:
        print(f"\n🔍 DEBUG: Calling math evaluator with model={model_name}")
        print(f"   Problem excerpt: {problem[:100]}...")
        print(f"   Solution excerpt: {solution_text[:100]}...")

    # Use SLM response with empty prefill
    resp = generate_SLM_response(eval_prompt,
                                 system_prompt=MATH_JUDGE_SYSTEM,
                                 model_name=model_name,
                                 temperature=temperature,
                                 prefill="",
                                 verbose=False)

    if verbose:
        print(f"   Raw response: '{resp}'")

    # Try to extract just a number from the response
    # Sometimes models add extra text despite instructions
    import re
    # Look for just a number (1-10)
    number_match = re.search(r'\b([1-9]|10)\b', resp)
    if number_match:
        rating = int(number_match.group(1))
        if verbose:
            print(f"   Extracted rating: {rating}")
    else:
        print(f"\n⚠️ WARNING: Could not parse rating from response: {resp}")
        print(f"   Using default rating of 5")
        rating = 5  # Default middle rating instead of crashing

    if verbose:
        print(f"📊 Math solution quality: {rating}/10")
    return rating


def evaluate_math_answer_vs_reference(
        proposed_answer: str,
        correct_answer: str,
        model_name: str = "claude-3-7-sonnet-20250219",  # Sonnet by default
        temperature: float = 0.0,
        seed: Optional[int] = None,
        verbose: bool = False) -> int:
    """
    Evaluate the quality of a proposed math answer against a correct reference answer.

    Args:
        proposed_answer: The answer to evaluate
        correct_answer: The correct reference answer to compare against
        model_name: Judge model (default: Sonnet)
        temperature: Temperature for evaluation (default: 0.0)
        seed: Random seed for reproducibility
        verbose: Print debug information

    Returns:
        Integer rating 1-10 where:
        - 10: Identical to reference, perfectly correct
        - 8-9: Correct with minor differences in presentation
        - 6-7: Mostly correct, some steps missing or unclear
        - 4-5: Partially correct, significant issues
        - 2-3: Mostly incorrect
        - 1: Completely wrong
    """
    eval_prompt = f"""Compare these two math solutions:

REFERENCE SOLUTION (CORRECT):
{correct_answer}

PROPOSED SOLUTION (TO EVALUATE):
{proposed_answer}

Rate the PROPOSED solution compared to the REFERENCE on a scale of 1-10:
- 10: Identical approach and answer, perfectly correct
- 8-9: Correct answer and valid approach, minor presentation differences
- 6-7: Correct answer but missing steps or less clear explanation
- 4-5: Partially correct, wrong answer or significant errors in approach
- 2-3: Mostly incorrect, major errors throughout
- 1: Completely wrong or unrelated

Consider:
1) Is the final numerical answer correct?
2) Is the mathematical reasoning sound?
3) Are the key steps present and correct?
4) Is the solution clear and complete?

IMPORTANT: Output ONLY a single integer 1-10. Nothing else."""

    system_prompt = """You are an impartial math grader comparing solutions.

CRITICAL: Output ONLY a single integer between 1 and 10.
Do NOT include any text, explanation, or commentary.
Just output a number like: 7
Nothing else."""

    if verbose:
        print(f"\n🔍 Evaluating proposed answer vs reference with {model_name}")
        print(f"   Proposed excerpt: {proposed_answer[:100]}...")
        print(f"   Reference excerpt: {correct_answer[:100]}...")

    # Use generate_SLM_response for consistency
    resp = generate_SLM_response(eval_prompt,
                                 system_prompt=system_prompt,
                                 model_name=model_name,
                                 temperature=temperature,
                                 prefill="",
                                 verbose=False)

    if verbose:
        print(f"   Raw response: '{resp}'")

    # Extract rating
    import re
    number_match = re.search(r'\b([1-9]|10)\b', resp)
    if number_match:
        rating = int(number_match.group(1))
        if verbose:
            print(f"   Extracted rating: {rating}")
    else:
        if verbose:
            print(f"⚠️ WARNING: Could not parse rating from response: {resp}")
            print(f"   Using default rating of 5")
        rating = 5

    if verbose:
        print(f"📊 Answer quality vs reference: {rating}/10")

    return rating


def evaluate_math_against_gold(
        solution_text: str,
        gold_answer: str,
        *,
        aime: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Compare extracted final answer vs. gold.
    Returns (is_correct, extracted_answer_string).
    - In general mode, reduces fractions to canonical 'a/b' for equality check.
    - In AIME mode, requires integer 0..999 exact match.
    """
    pred = extract_math_final_answer(solution_text, aime=aime)
    if pred is None:
        return (False, None)

    if aime:
        try:
            return (int(pred) == int(gold_answer), pred)
        except Exception:
            return (False, pred)

    # general fraction/int equivalence
    def canon(x: str) -> str:
        if "/" in x:
            return _simplify_fraction_text(x)
        try:
            return str(int(x))
        except Exception:
            return x.strip()

    return (canon(pred) == canon(gold_answer), pred)


# =========================================
# 4) Math question generation (yes/no)
# =========================================

MATH_QA_SYSTEM = """You generate helpful YES/NO questions to improve math solutions.

Rules:
- Output only ONE yes/no question with no extra text.
- Be specific and check a single step, assumption, or computation.
- Prefer questions that, when answered, directly change or confirm the next step.
- Avoid repeating previously asked questions; explore new aspects (method choice, domain, parity, monotonicity, bounds, arithmetic)."""


def small_model_generate_helpful_binary_questions_math(
        prompt: str,
        system_prompt: str | None,
        original_answer: str,
        existing_guiding_questions,
        existing_guiding_answers,
        question_model_name=QUESTION_SLM,
        temperature=0.0,
        seed=None,
        verbose=False,
        predict_base_rate=False):
    """Generate a new, non-duplicative yes/no math question.

    Args:
        predict_base_rate: If True, also predict P(yes) for probabilistic compression.
                          Returns (question, p_yes) tuple instead of just question.
    """
    existing_guiding_questions_and_answers = format_guiding_questions_and_answers(
        existing_guiding_questions, existing_guiding_answers)

    if predict_base_rate:
        user_prompt = f"""Problem: {prompt}

Current attempt:
{original_answer}

Previously asked (do NOT repeat or paraphrase):
{existing_guiding_questions_and_answers}

Generate ONE new yes/no question that would most improve the solution next, AND predict the probability that the answer is "yes".
Ask about a precise step (method choice, formula validity, arithmetic, constraints, or final check).

Format your response EXACTLY as:
QUESTION: <your yes/no question>
P_YES: <probability between 0.0 and 1.0>

Example:
QUESTION: Is the derivative calculation correct?
P_YES: 0.6

Respond with ONLY the question and probability, nothing else."""
    else:
        user_prompt = f"""Problem: {prompt}

Current attempt:
{original_answer}

Previously asked (do NOT repeat or paraphrase):
{existing_guiding_questions_and_answers}

Generate ONE new yes/no question that would most improve the solution next.
Ask about a precise step (method choice, formula validity, arithmetic, constraints, or final check).
Return only the question with no extra words.
"""

    resp = generate_SLM_response(user_prompt,
                                 system_prompt=MATH_QA_SYSTEM,
                                 model_name=question_model_name,
                                 temperature=temperature,
                                 prefill=None,
                                 verbose=False)

    if predict_base_rate:
        question, p_yes = parse_question_with_probability(resp,
                                                          verbose=verbose)
        if verbose:
            print(f"❓ Math guiding question: {question}")
            print(f"📊 Predicted P(yes): {p_yes:.3f}")
        return question, p_yes
    else:
        if verbose:
            print(f"❓ Math guiding question: {resp}")
        return resp


def large_model_answer_binary_question_math(original_prompt,
                                            system_prompt,
                                            large_model_answer,
                                            small_model_answer,
                                            question,
                                            model_name=LLM,
                                            temperature=0.0,
                                            seed=None,
                                            verbose=False):
    """Math-specific version for answering yes/no questions about solutions."""
    if verbose:
        print("🤖 LLM answering math binary question...")
        print(f"❓ Question: {question}")

    prompt = f"""Problem: {original_prompt}

Reference solution (correct):
{large_model_answer}

Student solution (to evaluate):
{small_model_answer}

Question: {question}

Based on mathematical correctness and the reference solution, answer YES or NO only."""

    resp = generate_LLM_response(prompt,
                                 system_prompt=system_prompt
                                 or MATH_JUDGE_SYSTEM,
                                 model_name=model_name,
                                 temperature=temperature,
                                 seed=seed,
                                 verbose=False)
    answer = extract_yes_no_answer(resp)

    if verbose:
        print(f"✓ LLM answer: {'YES' if answer else 'NO'}")

    return answer


# =========================================
# 5) Math solver (SLM) with prefill
# =========================================


def small_model_improve_with_guidance_science(
    original_prompt,
    system_prompt=None,
    small_model_answer=None,
    guiding_questions=None,
    guiding_answers=None,
    open_ended_guidances=None,
    small_model_name=SLM,
    temperature=0.0,
    seed=None,
    use_local=False,
    local_model=None,
    verbose=False,
):
    """
    Science-specific improvement pass for multiple choice questions.
    Produces clear scientific reasoning ending with Answer: [LETTER]
    """
    system_parts = [
        SCIENCE_SOLVER_SYSTEM if system_prompt is None else system_prompt
    ]

    # Add constraints derived from Q&A
    if guiding_questions and guiding_answers:
        system_parts.append("\nBased on these clarifications:")
        for q, a in zip(guiding_questions, guiding_answers):
            yn = a if isinstance(
                a, bool) else str(a).strip().lower() in ("yes", "true", "1")
            system_parts.append(f"- {q}: {'YES' if yn else 'NO'}")

    # Add open-ended guidance if provided
    if open_ended_guidances:
        system_parts.append("\nAdditional guidance:")
        for guidance in open_ended_guidances:
            system_parts.append(f"- {guidance}")

    full_system = "\n".join(system_parts)

    # Generate improved answer
    if use_local and local_model:
        response = local_model.generate(original_prompt, full_system)
    else:
        response = model_completion(original_prompt,
                                    model=small_model_name,
                                    system=full_system,
                                    temperature=temperature,
                                    seed=seed)

    if verbose:
        print(f"Science-improved answer: {response[:200]}...")

    return response


def small_model_improve_with_guidance_math(
        original_prompt,
        system_prompt=None,
        small_model_answer=None,
        guiding_questions=None,
        guiding_answers=None,
        open_ended_guidances=None,
        small_model_name=SLM,
        temperature=0.0,
        seed=None,
        use_local=False,
        local_model=None,
        verbose=False,
        aime:
    bool = False,  # Set True for AIME style (forces integer boxed answer)
):
    """
    Math-specific improvement pass using prefill and strict formatting.
    Produces: Problem/ Solution steps / \boxed{...}
    """
    system_parts = [MATH_SOLVER_SYSTEM]

    # Add constraints derived from Q&A
    def _format_math_guidance(qs, ans, extra=None):
        lines = []
        if qs and ans:
            for q, a in zip(qs, ans):
                yn = a if isinstance(
                    a, bool) else str(a).strip().lower() in ("yes", "true",
                                                             "1")
                lines.append(f"- {q}: {'YES' if yn else 'NO'}")
        if extra:
            lines.extend(f"- {g}" for g in extra)
        return "\n".join(lines) if lines else ""

    constraints = _format_math_guidance(guiding_questions, guiding_answers,
                                        open_ended_guidances)
    if constraints:
        system_parts.append("\nConstraints derived from Q&A:\n" + constraints)

    if aime:
        system_parts.append(
            "\nAIME formatting rules (must follow): "
            "Final answer must be an integer from 0 to 999; end with \\boxed{N}; no text after the boxed line."
        )

    improvement_system_prompt = "\n".join(system_parts)

    # User prompt kept minimal; system carries rules
    prompt_parts = [f"Problem: {original_prompt}"]
    if small_model_answer:
        prompt_parts.append("\nCurrent attempt:\n" + small_model_answer)
    prompt_parts.append(
        "\nProvide a correct, concise solution with clear steps.")
    prompt_parts.append(
        "Do not include any text other than the problem, the solution, and the final boxed answer."
    )
    prompt = "\n".join(prompt_parts)

    prefill = f"Problem: {original_prompt}\n\nSolution:"

    raw = generate_SLM_response(prompt,
                                system_prompt=improvement_system_prompt,
                                model_name=small_model_name,
                                temperature=temperature,
                                prefill=prefill,
                                verbose=verbose)
    return raw
