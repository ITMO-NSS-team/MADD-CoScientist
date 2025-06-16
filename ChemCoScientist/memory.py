class AgentMemory:
    """Simple session-based memory handler for agents."""

    def __init__(self, state: dict):
        # Ensure agent_memory exists in state
        self.state = state
        if "agent_memory" not in self.state:
            self.state["agent_memory"] = {}

    def get_memory(self, agent_name: str) -> list:
        """Return memory list for agent, creating it if necessary."""
        return self.state["agent_memory"].setdefault(agent_name, [])

    def add(self, agent_name: str, message: str):
        """Add a message to agent memory."""
        self.get_memory(agent_name).append(message)

    def to_prompt(self, agent_name: str) -> str:
        """Return memory as joined string for prompt insertion."""
        mem = self.get_memory(agent_name)
        if not mem:
            return ""
        return "\n".join(mem)