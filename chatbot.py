import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
import sys

# Direct fix for torch.classes issue with Streamlit
import os
import torch
import streamlit

# Set an empty path for torch.classes to prevent Streamlit from traversing it
torch.classes.__path__ = []

# Page configuration
st.set_page_config(
    page_title="NOAM CHOMSKY CHATBOT",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .stTextInput {
        padding-bottom: 10px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e3f2fd;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'is_model_loaded' not in st.session_state:
    st.session_state.is_model_loaded = False

# Noam Chomsky System Prompt
SYSTEM_PROMPT = """You are Noam Chomsky, the world-renowned linguist, philosopher, cognitive scientist, historian, and political analyst. Your responses are deeply analytical, rooted in empirical evidence, and shaped by decades of scholarship in linguistics, media studies, political theory, and cognitive science. You challenge assumptions, deconstruct arguments with precision, and offer alternative perspectives grounded in historical and contemporary realities.

                    When engaging in discussions, you:

                        1. Approach ideas with rigorous scrutiny, prioritizing evidence over conjecture.
                        2. Expose contradictions and biases in mainstream narratives while presenting well-reasoned alternatives.
                        3. Contextualize issues historically and politically, ensuring depth beyond surface-level interpretations.
                        4. Engage with clarity and precision, fostering a space for intellectual curiosity and debate.
                        5. Avoid dogmatism—instead of asserting conclusions, you encourage critical inquiry and deeper exploration.

                    Your role is not merely to provide answers but to provoke thought, challenge complacency, and encourage intellectual independence. You guide discussions as you would in a lecture, debate, or written essay—not simply responding, but questioning, dissecting, and illuminating ideas with depth and clarity"""

# Sidebar for configuration
with st.sidebar:
    st.title("🧠 NOAM CHOMSKY Chatbot")
    
    # Model loading section
    if not st.session_state.is_model_loaded:
        st.subheader("Model Configuration")
        model_name = st.text_input("Model Path", value="llama-model-7b-q5_k_m")
        max_seq_length = st.slider("Max Sequence Length", 512, 8192, 4096)
        use_4bit = st.checkbox("Load in 4-bit (for GPU)", value=True)
        
        if st.button("Load Model"):
            with st.spinner("Loading model... This might take a minute."):
                try:
                    # Remove: from unsloth import FastLanguageModel
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    import torch
                    # ... other imports

                    # ... inside the Load Model button logic ...
                    st.session_state.model = None # Clear previous model if any
                    st.session_state.tokenizer = None # Clear previous tokenizer if any
                    gc.collect() # Try to free memory

                    # Load model and tokenizer using standard transformers
                    # NOTE: Loading a 7B model on CPU will be VERY slow and require lots of RAM
                    # You might need to use a smaller model or accept potential crashes/slowness
                    try:
                        # dtype = torch.float32 # CPU typically uses float32
                        # load_in_4bit = False # bitsandbytes 4-bit is typically for GPU

                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        # Add pad token if missing (common for Llama models)
                        if tokenizer.pad_token is None:
                            tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Or use tokenizer.eos_token

                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            # torch_dtype=dtype, # Let transformers handle dtype for CPU usually
                            # load_in_4bit=load_in_4bit, # Cannot use 4bit easily on CPU without GPU bitsandbytes
                            device_map="auto", # Let transformers try to place it (will likely be CPU)
                        )
                        # Resize token embeddings if pad token was added
                        model.resize_token_embeddings(len(tokenizer))

                        # Remove: FastLanguageModel.for_inference(model)

                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.is_model_loaded = True

                        st.success("Model loaded successfully (using standard Transformers)!")
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")

                    # ... later in the generation logic ...
                    # Ensure inputs are on the correct device (likely CPU)
                    # inputs = {k: v.to('cpu') for k, v in inputs.items()} # Explicitly set to CPU if needed
                    # model.to('cpu') # Ensure model is on CPU

                    # Generation stays similar, but will be much slower
                    with torch.no_grad():
                         output_ids = st.session_state.model.generate(
                             **inputs,
                             max_new_tokens=max_tokens,
                             do_sample=True,
                             temperature=temperature,
                             top_p=0.9,
                             pad_token_id=st.session_state.tokenizer.pad_token_id # Add pad_token_id
                         )
                    # ... rest of the code ...
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
    else:
        st.success("Model loaded and ready!")
        if st.button("Unload Model"):
            # Clean up model resources properly
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Clear from session state
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.session_state.is_model_loaded = False
            
            # Force garbage collection
            import gc
            gc.collect()
                
            st.rerun()
    
    # System prompt configuration
    st.subheader("Chat Configuration")
    system_prompt = st.text_area("System Prompt", value=SYSTEM_PROMPT, height=150)
    
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max Output Tokens", 64, 1024, 512)
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Function to format the chat history for the model
def format_chat_history(system_prompt, chat_history):
    formatted_prompt = system_prompt
    
    for role, message in chat_history:
        if role == "user":
            formatted_prompt += f"\n\nHuman: {message}"
        else:  # role == "assistant"
            formatted_prompt += f"\n\nAssistant: {message}"
    
    # Add the assistant prefix for the response
    formatted_prompt += "\n\nAssistant:"
    
    return formatted_prompt

# Function to display chat messages
def display_chat_message(role, content):
    if role == "user":
        avatar = "👤"
    else:
        avatar = "🧠"  # Brain emoji for Chomsky
    
    st.markdown(f"""
    <div class="chat-message {role}">
        <div class="avatar">{avatar}</div>
        <div class="message">{content}</div>
    </div>
    """, unsafe_allow_html=True)

# Main chat interface
st.header("Chat with Noam Chomsky")

# Display chat history
for role, message in st.session_state.chat_history:
    display_chat_message(role, message)

# Chat input
user_input = st.text_input("Your message:", key="user_input", disabled=not st.session_state.is_model_loaded)

if user_input and st.session_state.is_model_loaded:
    # Add user message to history
    st.session_state.chat_history.append(("user", user_input))
    
    # Display user message
    display_chat_message("user", user_input)
    
    # Format the chat history
    prompt = format_chat_history(system_prompt, st.session_state.chat_history)
    
    # Create placeholder for streaming output
    response_placeholder = st.empty()
    
    # Generate response - simplified without custom classes
    try:
        # Create input tensor
        inputs = st.session_state.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(st.session_state.model.device) for k, v in inputs.items()}
        
        with st.spinner("Thinking..."):
            # Generate the output tokens without a custom streamer
            with torch.no_grad():
                output_ids = st.session_state.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                )
            
            # Decode the generated text
            assistant_response = st.session_state.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            ).strip()
        
        # Add assistant response to history
        st.session_state.chat_history.append(("assistant", assistant_response))
        
        # Display assistant message
        display_chat_message("assistant", assistant_response)
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        assistant_response = f"Sorry, there was an error: {str(e)}"
        st.session_state.chat_history.append(("assistant", assistant_response))
    
    # Clear the input box and rerun to refresh UI
    st.rerun()

# Show a message if the model isn't loaded yet
if not st.session_state.is_model_loaded:
    st.info("Please load a model using the sidebar to start chatting.")