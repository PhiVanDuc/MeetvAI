import re
import contextvars
from dotenv import load_dotenv
from pydantic import BaseModel

from vision_agents.plugins import getstream, gemini
from vision_agents.core import Agent, AgentLauncher, Runner, User

load_dotenv()

current_call_config = contextvars.ContextVar("current_call_config", default = None)

class JoinRequestData(BaseModel):
    id: str
    name: str
    image: str
    call_id: str
    call_type: str
    instructions: str

def clean_markdown(text: str) -> str:
    text = text.replace('`', '')
    text = re.sub(r'#+', '', text)
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\n\s*[-*•]\s*', ', ', text)
    
    return text.strip()

async def create_agent(**kwargs) -> Agent:
    config = current_call_config.get()

    if config is None:
        image = ""
        id = "warmup_id"
        name = "Warmup Agent"
        instructions = "You are a helpful assistant."
    else:
        print(clean_markdown(config["instructions"]))

        id = config["id"]
        name = config["name"]
        image = config["image"]
        instructions = f"""
            ### SYSTEM PROTOCOL (MANDATORY) / CRITICAL RESPONSE RULES (VOICE CALL MODE):
            1. You are engaging in a REAL-TIME VOICE CONVERSATION. You MUST speak naturally like a human on a phone call.
            2. ABSOLUTELY NO MARKDOWN: Do not use asterisks (*), bolding (**), hashtags (#), bullet points, or numbered lists.
            3. Use ONLY plain text with natural punctuation (commas, periods, question marks). 
            4. If you need to list items or give examples, weave them naturally into your spoken sentences using transition words (e.g., "First...", "For example...").
            5. Be conversational, concise, and direct. Break down long explanations into shorter, spoken-style responses.

            ### USER PROMPT / INSTRUCTIONS:
            {clean_markdown(config["instructions"])}

            ### LANGUAGE PROTOCOL:
            1. Initially, speak in the same language as these instructions.
            2. Maintain this language unless the user explicitly asks to switch.
            3. If a switch is requested, use the new language for all subsequent interactions.
        """

    llm = gemini.Realtime(model = "gemini-3.1-flash-live-preview")

    return Agent(
        llm = llm,
        agent_user = User(
            id = id,
            name = name,
            image = image
        ),
        edge = getstream.Edge(),
        instructions = instructions
    )

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs):
    await agent.authenticate()
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        await agent.simple_response("Greet the user.")
        await agent.finish()

launcher = AgentLauncher(
    join_call = join_call,
    max_sessions_per_call = 1,
    create_agent = create_agent
)

runner = Runner(launcher)

@runner.fast_api.post("/api/stream/agent/join")
async def join(data: JoinRequestData):
    token = current_call_config.set({
        "id": data.id,
        "name": data.name,
        "image": data.image,
        "instructions": data.instructions
    })

    try:
        await launcher.start_session(call_id = data.call_id, call_type = data.call_type)
    finally:
        current_call_config.reset(token)

if __name__ == "__main__":
    runner.cli()