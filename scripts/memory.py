class NPCMemory:
    def __init__(self):
        self.memory = []
        self.max_length = 50

    def add_conversation(self, user_input, npc_response):
        self.memory.append({"user": user_input, "npc": npc_response})
        if len(self.memory) > self.max_length:
            self._compress_memory()

    def _compress_memory(self):
        summary = " ".join([f"User: {c['user']} -> NPC: {c['npc']}" for c in self.memory[:20]])
        self.memory = [{"summary": summary}] + self.memory[20:]
