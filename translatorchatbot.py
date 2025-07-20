import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime # Added for timestamps in history

# Load environment variables from .env
load_dotenv()

# --- Streamlit Page Configuration ---
# Sets browser tab title, icon, and layout
st.set_page_config(
    page_title="🌐 Groq Language Translator",
    page_icon="🗣️", # Using a speaking head emoji as an icon
    layout="centered", # Can be "wide" or "centered"
    initial_sidebar_state="expanded" # Sidebar starts open
)

# --- Session State Initialization ---
# This is crucial for maintaining translation history across reruns
if "translations" not in st.session_state:
    st.session_state.translations = []

# --- Sidebar for Info and Settings ---
with st.sidebar:
    st.header("About This Translator 📚")
    st.write(
        """
        This application uses the blazing-fast **Groq API** and **Langchain** to provide instant language translations.
        Just type your text, select the target language, and hit 'Translate'!
        """
    )
    st.markdown("---") # Visual separator

    st.header("How to Use 🤔")
    st.markdown(
        """
        1.  **Enter Text:** Type or paste any text into the input box.
        2.  **Select Language:** Choose the language you want to translate your text *into*.
        3.  **Translate:** Click the '🚀 Translate!' button.
        4.  **Clear History:** Use the '🗑️ Clear All Translations' button below to reset.
        """
    )
    st.markdown("---")

    st.header("Theme Settings 🎨")
    st.info(
        """
        Streamlit generally follows your system's light/dark mode preference.
        To **manually switch** between Light and Dark mode within Streamlit:

        1.  Click the 'hamburger' menu (☰) in the **top-right corner** of the app.
        2.  Go to 'Settings'.
        3.  Under 'Theme', select your preferred mode (Light / Dark / Use system default).
        """
    )
    st.markdown("---") # Visual separator

    # Button to clear all past translations
    # Corrected Indentation: This 'if' statement is now correctly indented with 4 spaces
    if st.button("🗑️ Clear All Translations", use_container_width=True):
        # Corrected Indentation: These lines are now correctly indented with 8 spaces
        st.session_state.translations = [] # Reset the history list
        st.success("Translation history cleared!")
        st.rerun() # Correctly using st.rerun()

    st.markdown("---")
    st.caption("Developed with ❤️ using Streamlit, Langchain, and Groq.")


# --- LLM Setup (Groq Model) ---
# Initialize the ChatGroq model with your API key and chosen parameters
llm = ChatGroq(
    model_name="llama3-8b-8192", # Recommended for speed and quality
    groq_api_key=os.getenv("GROQ_API_KEY"), # Fetches API key from .env file
    temperature=0, # Low temperature for accurate, deterministic translation
    max_tokens=None, # Allow model to determine response length based on context
    timeout=None,
    max_retries=2,
)

# --- Langchain Components ---
# Define the prompt template for translation
# The system message guides the AI's behavior
# The human message defines the specific task and input variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and highly accurate language translator. Your sole purpose is to translate the provided text from any language to the target language. Provide only the translated text, do not add any extra explanations or conversational filler. If the source language is the same as the target language, simply return the input text as is."),
    ("human", "Translate the following text to {target_language}: {input}"),
])

# Output parser to get string response from LLM
output_parser = StrOutputParser()

# Combine prompt, LLM, and parser into a Langchain chain
chain = prompt | llm | output_parser

# --- Main UI Elements ---
st.header('⚡ Instant Language Translator with Groq')
st.write("Enter text below, select your target language, and get a quick translation!")

# Input area for text to be translated
input_text = st.text_area(
    "✍️ Enter text here:",
    height=150,
    placeholder="Type or paste your text to translate..."
)

# Language selection dropdown
languages = [
    "English", "Urdu", "German", "French", "Spanish", "Arabic",
    "Hindi", "Chinese", "Russian", "Turkish", "Japanese", "Italian", "Portuguese", "Korean", "Vietnamese", "Dutch", "Swedish" # Expanded list
]
languages.sort() # Sort alphabetically for better user experience
selected_language = st.selectbox(
    "🎯 Select language to translate to:",
    languages,
    index=languages.index("English") if "English" in languages else 0 # Default to English if available
)

# Buttons for actions
col1, col2 = st.columns([1, 1]) # Use columns for layout
with col1:
    translate_button = st.button("🚀 Translate!", use_container_width=True)
with col2:
    if st.button("🔄 Clear Input", use_container_width=True):
        input_text = "" # This will clear the text area on next rerun
        st.rerun() # Correctly using st.rerun()

# --- Translation Logic ---
if translate_button and input_text:
    with st.spinner(f"Translating to {selected_language}... ⏳"):
        try:
            # Invoke the Langchain chain with the user's input and selected language
            response = chain.invoke({
                "target_language": selected_language,
                "input": input_text
            })

            # Store the translation in session state for history display
            st.session_state.translations.insert(0, { # Insert at beginning to show newest first
                "input": input_text,
                "output": response,
                "target_lang": selected_language,
                "timestamp": datetime.now().strftime("%I:%M %p, %b %d, %Y") # Add detailed timestamp
            })
            st.success("Translation complete!")

        except Exception as e:
            st.error(f"An error occurred during translation: {e}")
            st.info("Please ensure your Groq API Key is valid and try again.")
elif translate_button and not input_text:
    st.warning("Please enter some text to translate before clicking 'Translate'!")

# --- Display Translation History ---
if st.session_state.translations:
    st.markdown("---")
    st.subheader("📚 Recent Translations")
    st.markdown("---") # Separator before history starts

    # Display each translation from history
    for i, entry in enumerate(st.session_state.translations):
        # Using Streamlit's chat message-like UI for better visual separation and engagement
        with st.container(border=True): # Use a container to group related history items
            st.markdown(f"**⏰ {entry['timestamp']}**")
            st.markdown(f"**🎯 Target Language:** {entry['target_lang']}")
            
            # Display original input using a 'user' bubble style
            with st.chat_message("user", avatar="💬"): # Custom emoji avatar
                st.markdown(f"**Original Text:**\n\n__{entry['input']}__")

            # Display translated output using an 'assistant' bubble style
            with st.chat_message("assistant", avatar="✨"): # Custom emoji avatar
                st.markdown(f"**Translated Text:**\n\n**{entry['output']}**")
            
            if i < len(st.session_state.translations) - 1:
                st.markdown("---") # Separator between individual history entries