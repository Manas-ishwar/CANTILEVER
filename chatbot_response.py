from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load DialoGPT once
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Track conversation history
chat_history_ids = None

def generate_response(user_input):
    global chat_history_ids

    # Simple rule-based replies
    user_input_lower = user_input.lower()
    if "hello" in user_input_lower:
        return "Hello! 😊 How can I help you today?"
    elif "how are you" in user_input_lower:
        return "I'm just a bot, but I'm here to help you!"
    elif "your name" in user_input_lower:
        return "I'm DISHABOT, your AI assistant 🤖"
    elif "thank you" in user_input_lower or "thanks" in user_input_lower:
        return "You're welcome! 😊"
    elif "bye" in user_input_lower:
        return "Goodbye! Have a great day! 👋"
    
    # Fallback to DialoGPT
    try:
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

        # Append chat history if available
        if chat_history_ids is not None:
            bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Generate response
        chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.85
        )

        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response if response.strip() != '' else "Hmm, I didn’t get that. Can you rephrase?"

    except Exception as e:
        print(f"❌ DialoGPT Error: {e}")
        return "Sorry, I had trouble thinking of a response."

