from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from scripts.memory import NPCMemory
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = FastAPI()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./models/npc_dwayne/checkpoint-3")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Initialize memory
memory = NPCMemory()
class ChatRequest(BaseModel):
    user_input: str

@app.get("/")
def read_root():
    return {"message": "Welcome to Dave's Chat API!"}

@app.post("/chat")
def chat(user_input: str):
#def chat(request: str):
    try:
        context = " ".join([f"User: {c['user']} -> NPC: {c['npc']}" for c in memory.memory])
        inputs = tokenizer(context + f"\nUser: {user_input}\nNPC:", return_tensors="pt")

    # Use max_new_tokens to control the output length
        outputs = model.generate(**inputs, max_new_tokens=50)  # Set the number of tokens to generate
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        memory.add_conversation(user_input, response)
        return {"response": response}

    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
