import os
import chainlit as cl

from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig 
from openai.types.responses import ResponseTextDeltaEvent
from pydantic import BaseModel
from agents import Agent, GuardrailFunctionOutput,InputGuardrailTripwireTriggered, RunContextWrapper, Runner, TResponseInputItem, input_guardrail, output_guardrail

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
external_client = AsyncOpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1",
 )

model = OpenAIChatCompletionsModel(
        model="llama3-8b-8192",
        openai_client=external_client
    )

run_config = RunConfig(
    model=model,
    tracing_disabled=True
)

@function_tool
def get_co2_stats(location: str) -> str:
    """
    Provide CO2 emission statistics for a given country.
    Argument: location (country name as a string)
    """

    return f"{location} emitted approximately 5.2 metric tons of CO2 per capita in 2023."

@function_tool
def suggest_green_practices(topic: str) -> str:
    """
    Suggest some green practices to improve environmental efficiency of a given country.
    Argument: topic (e.g., a country or industry)
    """
    return f"To reduce emissions in {topic}, use renewable energy and improve efficiency."

@function_tool
def policy_advice(topic: str) -> str:
    """
    Provide expert advice on climate change policies and international climate change agreements.
    Argument: topic (e.g., climate agreement, regulation name, or region)
    """
    return f"As a climate policy expert, provide detailed advice on international agreements, regulations, or strategies regarding: {topic}"

class ClimateChangeInput(BaseModel):
    is_climate_change_related= bool
    reasoning: str

climate_guardrail_agent= Agent(
    name="input check",
    instructions="Check if the user message is related to climate change, respond with true or false.",
    output_type=ClimateChangeInput,
)
@input_guardrail
async def climate_change_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(climate_guardrail_agent, input, context=ctx.context, run_config = run_config)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_climate_change_related,
    )
async def main():
    try:
        await Runner.run(climate_guardrail_agent, "hello can you help me solve maths?")
        print("Guardrail didnt trip-this is unexpected")

    except InputGuardrailTripwireTriggered:
        print("math homework guardrail tripped")

class OutPutCheck(BaseModel):
    is_valid= bool
    reason:str

output_guardrail_agent= Agent(
    name= "output Check"
    instructions="Verify that the assistant's response is focused only on climate-related issues. Return true if valid.",
    output_type=OutputCheck
)

@output_guardrail
async def climate_output_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(output_guardrail_agent, input, context=ctx.context, run_config=run_config)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_valid
    )
async def main():
    try:
        await Runner.run(climate_guardrail_agent, "Hello, can you help me solve for x: 2x + 3 = 11?", run_config = run_config)
        print("Guardrail didn't trip - this is unexpected")
    except InputGuardrailTripwireTriggered:
        print("Math output guardrail tripped")


triage_agent = Agent(
    name="Triage Agent",
    instructions=
    """
    You are a climate change triage assistant.
    - If the user's message is a greeting like "hi", "hello", "hey", do NOT call any tool. Respond with a polite greeting instead.
    - Use `policy_advice` if the question is about climate policies, regulations, international agreements, or government actions.
    - Use `get_co2_stats` if the user asks for carbon dioxide emissions or environmental data of a specific country or region.
    - Use `suggest_green_practices` if the user wants advice on reducing emissions, using renewable energy, or improving sustainability.
   If the user's message is a greeting or is not related to climate topics, do NOT call any tool and respond with a polite greeting or ask for clarification.
    """,
    tools=[policy_advice, suggest_green_practices, get_co2_stats],
    input_guardrails=[climate_change_guardrail],
    output_guardrails=[climate_output_guardrail]
    output_type= []
)
@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get ("history") or []
    user_input = message.content.lower().strip()

    # Handle common greetings without tool calls
    if user_input in ["hi", "hello", "hey", "salam", "assalamualaikum"]:
        await cl.Message(content="Hello! I'm here to help you with climate-related questions.").send()
        return

    # Create and send initial empty message
    msg = cl.Message(content="")
    await msg.send()

    # Add user message to history
    history.append({"role": "user", "content": user_input})

    # Run the agent with streaming
    result = Runner.run_streamed(
        triage_agent,
        input=history,
        run_config=run_config,
    )
    print(result.final_output)
    # Stream the response
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await msg.stream_token(event.data.delta)

    # Add assistant's response to history
    history.append({"role": "assistant", "content": result.final_output})

    # Update the session history
    cl.user_session.set("history", history)
