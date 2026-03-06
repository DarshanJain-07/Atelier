import json
import os
import time
from typing import Optional, Tuple

import requests
import torch
from dotenv import load_dotenv
from pydantic import BaseModel

from schema import DIMENSIONS

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = "gemini-3-flash-preview"


class WorldState(BaseModel):
    Reasoning: str
    Urgency: float
    Is_Personal: bool

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

    model_config = {"extra": "forbid"}


def get_world_state(user_input: str) -> Tuple[torch.Tensor, float, bool]:
    """
    Analyzes the news event using the LLM.
    Returns:
        1. World Tensor (1, 12): The 12-dimensional impact vector.
        2. Urgency Score (float): 0.0 to 1.0 (Time Pressure).
        3. Personal Flag (bool): True if the event is about 'ME/MY/I'.
    """

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    response_schema = {
        "type": "object",
        "properties": {
            "Reasoning": {
                "type": "string",
                "description": "Chain of thought analysis.",
            },
            "Urgency": {
                "type": "number",
                "description": "0.0 (Calm) to 1.0 (Panic/Immediate Action required).",
            },
            "Is_Personal": {
                "type": "boolean",
                "description": "True if the text uses 'My', 'I', 'Me' implying the user is directly involved.",
            },
            # The 12 Dimensions
            "Wealth": {"type": "number"},
            "Physical_Safety": {"type": "number"},
            "Stability": {"type": "number"},
            "Reputation": {"type": "number"},
            "Fairness": {"type": "number"},
            "In_Group": {"type": "number"},
            "Innovation": {"type": "number"},
            "Freedom": {"type": "number"},
            "Sanctity": {"type": "number"},
            "Care": {"type": "number"},
            "Short_Term": {"type": "number"},
            "Long_Term": {"type": "number"},
        },
        "required": ["Reasoning", "Urgency", "Is_Personal"] + DIMENSIONS,
    }

    # 2. The System Prompt (The "Magnitude Rubric" and Few-Shot Calibration)
    system_instruction = """
    You are an objective 'World Model Engine'. Analyze the input event and quantify its impact on society or the individual across 12 dimensions.

    ### 1. CALIBRATION NOTES & THE MAGNITUDE RUBRIC
    You must be highly calibrated. Do NOT exaggerate minor events. 
    Use this scale strictly:
    *   0.0: Neutral / No Relevance.
    *   ±0.1 - 0.2 (Routine/Minor): Local news, minor tech updates, standard political speeches, mild weather.
    *   ±0.3 - 0.5 (Significant): National protests, major corporate bankruptcy, hurricane making landfall, significant election results.
    *   ±0.6 - 0.8 (Crisis/Boom): 2008 Global Financial Crash, COVID-19 Pandemic lockdowns, outbreak of war between major nations.
    *   ±0.9 - 1.0 (Civilization Altering): Asteroid impact, Global Nuclear War, AGI Singularity, Alien Contact.

    *Crucial Notes on Dimensions:*
    - **Stability / Physical_Safety:** Do not give negative scores just because a situation is "tense." Only lower safety if physical harm is occurring or imminent.
    - **Wealth:** Consider the macro-economic scale unless it is a "Personal" event. A single CEO losing money is NOT -0.5 societal wealth.
    - **Fairness / Care:** High values mean justice/empathy is being upheld. Negative values mean injustice/cruelty is occurring.

    ### 2. URGENCY (Time Pressure)
    *   0.0: Historical event, slow-moving trend (e.g., demographic aging).
    *   0.5: Requires attention soon, but not an immediate emergency (e.g., upcoming election, new tax law next year).
    *   1.0: "Fight or Flight", immediate reaction required right now (e.g., active shooter, sudden massive earthquake, imminent missile strike).

    ### 3. PERSONAL RELEVANCE
    *   Set "Is_Personal" to true ONLY if the text explicitly uses first-person pronouns ("My", "I", "Me") indicating the user is directly experiencing the event (e.g., "I just got fired").
    *   Events happening to public figures, countries, or abstract groups (e.g., "Trump", "The Economy", "My country") are NOT personal.

    ### 4. EXAMPLES (Few-Shot Grounding)
    Input: "The Federal Reserve just announced a surprise 0.25% interest rate hike."
    Output: Wealth: -0.2 (Minor economic drag), Stability: -0.1, Urgency: 0.3. Is_Personal: False. (Other dimensions near 0).

    Input: "Global pandemic declared. All international flights grounded and mandatory lockdowns initiated."
    Output: Physical_Safety: -0.7, Wealth: -0.8, Stability: -0.8, Freedom: -0.9 (Lockdowns), Urgency: 0.9. Is_Personal: False.

    Input: "My boss just told me I am being laid off tomorrow."
    Output: Wealth: -0.8 (Massive personal wealth loss), Stability: -0.8, Urgency: 0.8, Is_Personal: True.

    Output JSON only.
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

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=10,
            )

            if resp.status_code != 200:
                raise Exception(f"API Error {resp.status_code}: {resp.text}")

            data = resp.json()

            try:
                text_response = data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError) as e:
                raise Exception(f"Unexpected API response format: {data}") from e

            parsed_json = json.loads(text_response)

            result = WorldState(**parsed_json)

            break

        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Dimension Retrieval Failure: {e}")

            print(f"Retrying after error: {e}")
            time.sleep(base_delay * (2**attempt))

    if result is None:
        raise RuntimeError("WorldState parsing failed after retries")

    # 4. Convert to Tensor
    values = [max(-1.0, min(1.0, getattr(result, d))) for d in DIMENSIONS]
    world_tensor = torch.tensor([values], dtype=torch.float32)
    urgency = max(0.0, min(1.0, result.Urgency))
    is_personal = result.Is_Personal

    # Debug print
    print(f"> {result.Reasoning}\n--------------------")

    return world_tensor, urgency, is_personal


if __name__ == "__main__":
    # Test
    news = "Introduction of new rules regarding flight time duty regulations of plane members severely impacting of major airlines"
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in .env file")
    tensor, urg, pers = get_world_state(news)
    print(f"Tensor: {tensor}")
    print(f"Urgency: {urg}")
    print(f"Personal: {pers}")
