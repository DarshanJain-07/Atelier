import asyncio
import json
import os
from typing import Optional, Tuple

import aiohttp
import torch
from dotenv import load_dotenv
from pydantic import BaseModel

from schema import DIMENSIONS

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = "gemini-3-flash-preview"

# Simple in-memory cache to prevent redundant API calls
LLM_CACHE = {}

class LLMGenerationError(Exception):
    """Raised when the LLM API call fails or returns an unexpected format."""

    pass


class DimensionParseError(Exception):
    """Raised when the dimension extraction process fails after max retries."""

    pass


class WorldFrame(BaseModel):
    Wealth: float
    Physical_Safety: float
    Stability: float
    Reputation: float
    Fairness: float
    In_Group: float
    Innovation: float
    Freedom: float
    Sanctity: float
    Care: float
    Short_Term: float
    Long_Term: float


class WorldState(BaseModel):
    Reasoning: str
    Detected_Biases: list[str]
    Urgency: float
    Is_Personal: bool
    Backlash_Potential: float
    Official_Frame: WorldFrame
    Skeptical_Frame: WorldFrame

    model_config = {"extra": "forbid"}


def _frame_to_tensor(frame: WorldFrame) -> torch.Tensor:
    values = [max(-1.0, min(1.0, getattr(frame, d))) for d in DIMENSIONS]
    return torch.tensor([values], dtype=torch.float32)


def _estimate_text_backlash_prior(user_input: str) -> float:
    text = user_input.lower()

    cue_weights = {
        "finally": 0.22,
        "after": 0.08,
        "years": 0.12,
        "billion": 0.18,
        "billions": 0.18,
        "metaverse": 0.16,
        "avatar": 0.14,
        "avatars": 0.14,
        "announces": 0.05,
        "unveils": 0.05,
        "introduces": 0.05,
        "now has": 0.18,
        "adds": 0.08,
        "legs": 0.22,
    }

    score = 0.0
    for cue, weight in cue_weights.items():
        if cue in text:
            score += weight

    if "finally" in text and ("billion" in text or "years" in text):
        score += 0.18
    if ("adds" in text or "now has" in text) and ("basic" in text or "legs" in text):
        score += 0.15

    return max(0.0, min(1.0, score))


def _combine_backlash_signals(
    llm_backlash: float,
    user_input: str,
    official_tensor: torch.Tensor,
    skeptical_tensor: torch.Tensor,
) -> float:
    text_prior = _estimate_text_backlash_prior(user_input)
    frame_gap = torch.mean(torch.abs(official_tensor - skeptical_tensor)).item()
    combined = 0.55 * llm_backlash + 0.25 * text_prior + 0.20 * frame_gap
    return max(0.0, min(1.0, combined))


async def get_world_state(
    user_input: str,
) -> Tuple[torch.Tensor, torch.Tensor, float, bool, list[str], str, float]:
    """
    Analyzes the news event using the LLM.
    Returns:
        1. Official frame tensor (1, 12).
        2. Skeptical frame tensor (1, 12).
        3. Urgency score (float): 0.0 to 1.0.
        4. Personal flag (bool).
        5. Detected biases (list[str]).
        6. Reasoning (str).
        7. Backlash potential (float).
    """
    
    # Check Cache
    if user_input in LLM_CACHE:
        print(f"> Using Cached LLM Result for: '{user_input[:30]}...'")
        return LLM_CACHE[user_input]

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    dimension_props = {dimension: {"type": "number"} for dimension in DIMENSIONS}

    response_schema = {
        "type": "object",
        "properties": {
            "Reasoning": {
                "type": "string",
                "description": "Chain of thought analysis.",
            },
            "Detected_Biases": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of explicit biases, spins, or power dynamics detected in the text.",
            },
            "Urgency": {
                "type": "number",
                "description": "0.0 (Calm) to 1.0 (Panic/Immediate Action required).",
            },
            "Is_Personal": {
                "type": "boolean",
                "description": "True if the text uses 'My', 'I', 'Me' implying the user is directly involved.",
            },
            "Backlash_Potential": {
                "type": "number",
                "description": "0.0 to 1.0 probability of ridicule, meme backlash, or cynical narrative capture.",
            },
            "Official_Frame": {
                "type": "object",
                "properties": dimension_props,
                "required": DIMENSIONS,
            },
            "Skeptical_Frame": {
                "type": "object",
                "properties": dimension_props,
                "required": DIMENSIONS,
            },
        },
        "required": [
            "Reasoning",
            "Detected_Biases",
            "Urgency",
            "Is_Personal",
            "Backlash_Potential",
            "Official_Frame",
            "Skeptical_Frame",
        ],
    }

    # 2. The System Prompt
    system_instruction = """
    You are an objective "Divergent Perception Engine". Your task is to analyze the input event and predict both the intended public-relations framing and the likely skeptical backlash framing.

    ### 1. DUAL-FRAME ANALYSIS
    * Generate an "Official_Frame": how the actor wants the event to be perceived. This should capture PR spin, best-case framing, and the intended reputational upside.
    * Generate a "Skeptical_Frame": how critics, cynical users, or a meme-driven audience will perceive the same event. This should capture ridicule, cultural debt, tone-deafness, and delayed/basic functionality.
    * Generate "Backlash_Potential": estimate how likely it is that the skeptical frame captures the public narrative. High values are appropriate when the achievement is underwhelming relative to cost, timing, or prior hype.

    ### 2. PIERCE THE SPIN, DETECT BIAS, AND PREDICT OUTCOMES
    * Do NOT perform simple sentiment analysis.
    * Identify explicit biases or framing tactics in "Detected_Biases". Use labels like "Corporate Spin", "Political Framing", "Fearmongering", "Downplaying", "Tone-Deafness", or "Technical Debt Masking".
    * The Official frame can be positive even when the Skeptical frame is negative. For example, "Metaverse finally has legs" could have Official Innovation/Reputation gains while the Skeptical frame strongly penalizes Reputation and Fairness because the public sees it as a billion-dollar company shipping a basic feature years late.

    ### 3. CALIBRATION NOTES & THE MAGNITUDE RUBRIC
    You must be highly calibrated. Do NOT exaggerate minor events.
    Use this scale strictly:
    *   0.0: Neutral / No Relevance.
    *   ±0.1 - 0.2 (Routine/Minor): Local news, minor tech updates, standard political speeches, mild weather.
    *   ±0.3 - 0.5 (Significant): National protests, major corporate bankruptcy, hurricane making landfall, significant election results, widespread layoffs.
    *   ±0.6 - 0.8 (Crisis/Boom): 2008 Global Financial Crash, COVID-19 Pandemic lockdowns, outbreak of war between major nations.
    *   ±0.9 - 1.0 (Civilization Altering): Asteroid impact, Global Nuclear War, AGI Singularity, Alien Contact.

    *Crucial Notes on Dimensions:*
    - **Stability / Physical_Safety:** Do not give negative scores just because a situation is "tense." Only lower safety if physical harm is occurring or imminent.
    - **Wealth:** Consider the macro-economic scale unless it is a "Personal" event. A single CEO losing money is NOT -0.5 societal wealth. However, mass layoffs disguised as "restructuring" heavily impact societal wealth.
    - **Fairness / Care:** High values mean justice/empathy is being upheld. Negative values mean injustice/cruelty/exploitation is occurring.

    ### 4. URGENCY (Time Pressure)
    *   0.0: Historical event, slow-moving trend (e.g., demographic aging).
    *   0.5: Requires attention soon, but not an immediate emergency (e.g., upcoming election, new tax law next year).
    *   1.0: "Fight or Flight", immediate reaction required right now (e.g., active shooter, sudden massive earthquake, imminent missile strike).

    ### 5. PERSONAL RELEVANCE
    *   Set "Is_Personal" to true ONLY if the text explicitly uses first-person pronouns ("My", "I", "Me") indicating the user is directly experiencing the event (e.g., "I just got fired").
    *   Events happening to public figures, countries, or abstract groups (e.g., "Trump", "The Economy", "My country") are NOT personal.

    ### 6. EXAMPLES (Few-Shot Grounding)
    Input: "The Federal Reserve just announced a surprise 0.25% interest rate hike."
    Output sketch:
    Official_Frame: modest policy competence, small Stability benefit.
    Skeptical_Frame: mild Wealth drag and mild Stability drag.
    Backlash_Potential: low.

    Input: "Global pandemic declared. All international flights grounded and mandatory lockdowns initiated."
    Output sketch:
    Official_Frame and Skeptical_Frame should both be strongly negative on Physical_Safety, Wealth, Stability, and Freedom.
    Backlash_Potential: low to moderate because the frames do not diverge much.

    Input: "Meta finally adds legs to its metaverse avatars after years of investment."
    Output sketch:
    Official_Frame: positive Innovation and modest Reputation gain.
    Skeptical_Frame: negative Reputation, Fairness, and Innovation due to ridicule and cultural debt.
    Backlash_Potential: high.

    Return JSON only.
    """

    payload = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"parts": [{"text": user_input}], "role": "user"}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseJsonSchema": response_schema,
            "temperature": 0.0,  # Deterministic
        },
    }

    # 3. API Call
    max_retries = 3
    base_delay = 1
    result: Optional[WorldState] = None

    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            try:
                async with session.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=15,
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise LLMGenerationError(f"API Error {resp.status}: {text}")

                    data = await resp.json()

                    try:
                        text_response = data["candidates"][0]["content"]["parts"][0]["text"]
                    except (KeyError, IndexError) as e:
                        raise LLMGenerationError(
                            f"Unexpected API response format: {data}"
                        ) from e

                    parsed_json = json.loads(text_response)

                    result = WorldState(**parsed_json)

                    break

            except Exception as e:
                if attempt == max_retries - 1:
                    raise DimensionParseError(f"Dimension Retrieval Failure: {e}") from e

                print(f"Retrying after error: {e}")
                await asyncio.sleep(base_delay * (2**attempt))

    if result is None:
        raise RuntimeError("WorldState parsing failed after retries")

    # 4. Convert to Tensors
    official_tensor = _frame_to_tensor(result.Official_Frame)
    skeptical_tensor = _frame_to_tensor(result.Skeptical_Frame)
    urgency = max(0.0, min(1.0, result.Urgency))
    is_personal = result.Is_Personal
    detected_biases = result.Detected_Biases
    backlash_potential = _combine_backlash_signals(
        max(0.0, min(1.0, result.Backlash_Potential)),
        user_input,
        official_tensor,
        skeptical_tensor,
    )

    # Debug print
    print(f"> Detected Biases: {detected_biases}")
    print(f"> Backlash Potential: {backlash_potential:.2f}")
    print(f"> {result.Reasoning}\n--------------------")

    output = (
        official_tensor,
        skeptical_tensor,
        urgency,
        is_personal,
        detected_biases,
        result.Reasoning,
        backlash_potential,
    )
    LLM_CACHE[user_input] = output
    return output


if __name__ == "__main__":
    # Test
    async def main():
        news = "Meta finally announces avatars will have legs in the metaverse after a billion dollar investment"
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        off_tensor, skp_tensor, urg, pers, biases, reason, backlash = await get_world_state(news)
        print(f"Biases: {biases}")
        print(f"Reasoning: {reason}")
        print(f"Backlash: {backlash}")
        print(f"Official Tensor: {off_tensor}")
        print(f"Skeptical Tensor: {skp_tensor}")
        print(f"Urgency: {urg}")
        print(f"Personal: {pers}")

    asyncio.run(main())
