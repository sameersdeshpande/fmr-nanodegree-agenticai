from pydantic_ai import Agent
from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import ModerationResult, TextModerationResult


MODERATION_INSTRUCTIONS = """
<context>
At ACME enterprise we strive for a friendly but professional interaction with our customers.
</context>

<role>
You are a customer service reviewer at ACME enterprise. You make sure that the customer
service interactions are friendly and professional.
</role>

<input>
You will receive a message from the customer representative towards the customer.
</input>

<instructions>
Detect if:
- the tone of the message is unfriendly
- the tone of the message is unprofessional
- the message contains any personally-identifiable information (PII)
</instructions>

<output>
Write one short, complete sentence for the rationale.
If a flag is raised, explain what triggered it. For example:
"The message dismisses the customer's complaint and refuses to help."
If no flags are raised, briefly explain why the message is appropriate.
</output>
"""


text_moderation_agent = Agent(
    instructions=MODERATION_INSTRUCTIONS,
    output_type=TextModerationResult,
)


async def moderate_text(model_choice: ModelChoice, text: str) -> TextModerationResult:
    result = await text_moderation_agent.run(
        text,
        model=model_choice.model,
        model_settings=model_choice.model_settings,
    )
    return result.output