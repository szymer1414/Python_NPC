from scripts.memory import NPCMemory

def test_memory_compression():
    memory = NPCMemory()
    for i in range(55):
        memory.add_conversation(f"user_input_{i}", f"npc_response_{i}")
    assert len(memory.memory) == 31  # 1 summary + 30 conversations
