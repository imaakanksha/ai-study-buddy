import streamlit as st
import google.generativeai as genai
import os
import re # Import regex for parsing

# --- Configuration ---
# Load API key from Streamlit secrets for secure deployment
try:
    # Attempt to retrieve the API key from Streamlit secrets
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=API_KEY)
    # Using gemini-1.5-flash for speed and free tier compatibility
    model = genai.GenerativeModel("gemini-1.5-flash")
    st.sidebar.success("Gemini API Key Loaded Successfully!") # Optional: Confirm key load
except KeyError:
    # Handle missing secret
    st.error("IMPORTANT: GOOGLE_API_KEY not found in Streamlit secrets.")
    st.error("Please go to your app settings on Streamlit Community Cloud, add a secret named GOOGLE_API_KEY, and paste your Gemini API key as the value.")
    st.stop() # Stop execution if API key secret is missing
except Exception as e:
    # Handle other potential configuration errors
    st.error(f"Error configuring Gemini API: {e}.")
    st.error("Please ensure your GOOGLE_API_KEY secret is correct and valid.")
    st.stop() # Stop execution on other errors

# --- Helper Functions (Gemini API Calls) ---
def generate_with_gemini(prompt):
    """Generic function to call Gemini API and handle potential errors."""
    try:
        # Add safety settings to reduce chances of blocking
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]
        response = model.generate_content(prompt, safety_settings=safety_settings)
        # Check for empty response or blocked content
        if not response.parts:
             if response.prompt_feedback and response.prompt_feedback.block_reason:
                 return f"Content generation blocked. Reason: {response.prompt_feedback.block_reason.name}"
             else:
                 # Sometimes empty parts list means blocked, check candidates
                 if response.candidates and response.candidates[0].finish_reason != 'STOP':
                     return f"Content generation stopped. Reason: {response.candidates[0].finish_reason.name}"
                 else:
                     return "Error: Received an empty response from the API."
        return response.text
    except Exception as e:
        return f"An error occurred during generation: {e}"

def generate_summary(text):
    """Generates a concise summary of the provided text using Gemini API."""
    if not text:
        return "Please provide some text to summarize."
    prompt = f"Please provide a concise summary of the following text, highlighting the key points and concepts suitable for studying:
\n---\n{text}
---"
    return generate_with_gemini(prompt)

def generate_flashcards(text):
    """Generates flashcards (Q/A pairs) from the provided text using Gemini API."""
    if not text:
        return "Please provide some text to generate flashcards from."
    prompt = f"Generate flashcards based on the key information in the following text. Format each flashcard strictly as:\nQuestion: [Your Question Here]\nAnswer: [Your Answer Here]\n\nEnsure there is a blank line between each flashcard.\n\n---\n{text}
---"
    flashcards_text = generate_with_gemini(prompt)
    if flashcards_text.startswith("An error occurred") or flashcards_text.startswith("Content generation") or flashcards_text.startswith("Error:"):
        return flashcards_text # Return error/block message directly

    pattern = re.compile(r"Question:\s*(.*?)\nAnswer:\s*(.*?)(?=\n\nQuestion:|\Z)", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(flashcards_text)
    flashcards = [(q.strip(), a.strip()) for q, a in matches]

    if not flashcards:
        if flashcards_text.strip(): return f"Could not parse flashcards from the response. Raw response:\n{flashcards_text}"
        else: return "The model did not generate any flashcards or the response was empty."
    return flashcards

def generate_quiz(text):
    """Generates a multiple-choice quiz from the provided text using Gemini API."""
    if not text:
        return "Please provide some text to generate a quiz from."
    prompt = f"Generate a multiple-choice quiz (around 3-5 questions) based on the key information in the following text. For each question, provide 4 options (A, B, C, D) and indicate the correct answer. Format each question strictly as:\nQuestion: [Your Question Here]\nA) [Option A]\nB) [Option B]\nC) [Option C]\nD) [Option D]\nAnswer: [Correct Option Letter]\n\nEnsure there is a blank line between each question block.\n\n---\n{text}
---"
    quiz_text = generate_with_gemini(prompt)
    if quiz_text.startswith("An error occurred") or quiz_text.startswith("Content generation") or quiz_text.startswith("Error:"):
        return quiz_text # Return error/block message directly

    pattern = re.compile(r"Question:\s*(.*?)\nA\)\s*(.*?)\nB\)\s*(.*?)\nC\)\s*(.*?)\nD\)\s*(.*?)\nAnswer:\s*([A-D])", re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(quiz_text)
    quiz_items = []
    for match in matches:
        question, opt_a, opt_b, opt_c, opt_d, answer = [m.strip() for m in match]
        options = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d}
        quiz_items.append({"question": question, "options": options, "answer": answer.upper()})

    if not quiz_items:
        if quiz_text.strip(): return f"Could not parse quiz questions from the response. Raw response:\n{quiz_text}"
        else: return "The model did not generate any quiz questions or the response was empty."
    return quiz_items

def generate_answer(context, question):
    """Answers a question based on the provided context using Gemini API."""
    if not context:
        return "Please provide study material first."
    if not question:
        return "Please enter a question."
    prompt = f"Based *only* on the following text, answer the question provided. If the answer cannot be found in the text, say 'The answer is not found in the provided text.'\n\nContext Text:\n---\n{context}
---\n\nQuestion: {question}"
    return generate_with_gemini(prompt)

# --- Application Structure ---
st.set_page_config(page_title="AI Study Buddy", layout="wide")

st.title("📚 AI Study Buddy")
st.caption("Upload your study material (text) and let AI help you learn!")

# Initialize session state variables
if 'flashcard_states' not in st.session_state: st.session_state.flashcard_states = {}
if 'quiz_answers' not in st.session_state: st.session_state.quiz_answers = {}
if 'show_quiz_results' not in st.session_state: st.session_state.show_quiz_results = False
if 'quiz_result_data' not in st.session_state: st.session_state.quiz_result_data = None # Store quiz data
if 'qa_answer' not in st.session_state: st.session_state.qa_answer = None
if 'prev_pasted_text' not in st.session_state: st.session_state.prev_pasted_text = ""
if 'prev_uploaded_filename' not in st.session_state: st.session_state.prev_uploaded_filename = None
if 'summary_result' not in st.session_state: st.session_state.summary_result = None
if 'flashcards_result' not in st.session_state: st.session_state.flashcards_result = None

# --- Sidebar for Input ---
st.sidebar.header("Input Material")
input_method = st.sidebar.radio("Choose input method:", ("Paste Text", "Upload Text File"), key="input_select")

study_material = ""
reset_results = False

if input_method == "Paste Text":
    study_material = st.sidebar.text_area("Paste your study material here:", height=300, key="pasted_text")
    if study_material != st.session_state.prev_pasted_text:
        reset_results = True
        st.session_state.prev_pasted_text = study_material
        st.session_state.prev_uploaded_filename = None # Clear uploaded file state
elif input_method == "Upload Text File":
    uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"], key="uploaded_file")
    if uploaded_file is not None:
        # Check if it's a new file upload
        if uploaded_file.name != st.session_state.get('prev_uploaded_filename'):
            try:
                study_material = uploaded_file.read().decode("utf-8")
                st.sidebar.success("File uploaded successfully!")
                reset_results = True
                st.session_state.prev_uploaded_filename = uploaded_file.name
                st.session_state.prev_pasted_text = "" # Clear pasted text state
            except Exception as e:
                st.sidebar.error(f"Error reading file: {e}")
                study_material = "" # Ensure material is empty on error
        else:
             # If same file is re-selected, try to read it again
             try:
                 uploaded_file.seek(0)
                 study_material = uploaded_file.read().decode("utf-8")
             except Exception as e:
                 st.sidebar.error(f"Error re-reading file: {e}")
                 study_material = ""

# Reset states if new material is provided
if reset_results:
    st.session_state.flashcard_states = {}
    st.session_state.quiz_answers = {}
    st.session_state.show_quiz_results = False
    st.session_state.quiz_result_data = None
    st.session_state.qa_answer = None
    st.session_state.summary_result = None
    st.session_state.flashcards_result = None
    # Force rerun to clear old outputs if new material is loaded
    st.rerun()

# --- Main Area for Features ---
st.header("Study Tools")

if study_material:
    st.subheader("Your Study Material:")
    st.text_area("Preview", study_material, height=150, disabled=True, key="material_preview")

    # --- Feature Tabs ---
    tab_summary, tab_flashcards, tab_quiz, tab_qa = st.tabs(["Summary", "Flashcards", "Quiz", "Ask Questions"])

    with tab_summary:
        if st.button("Generate Summary", key="summarize_btn"):
            with st.spinner("Generating summary..."):
                summary = generate_summary(study_material)
                st.session_state.summary_result = summary # Store result

        if st.session_state.summary_result:
            st.subheader("Summary:")
            result = st.session_state.summary_result
            if result.startswith("An error occurred") or result.startswith("Content generation") or result.startswith("Error:"):
                st.error(result)
            else:
                st.markdown(result)

    with tab_flashcards:
        if st.button("Generate Flashcards", key="flashcards_btn"):
            with st.spinner("Generating flashcards..."):
                flashcards_result = generate_flashcards(study_material)
                st.session_state.flashcards_result = flashcards_result # Store result
                st.session_state.flashcard_states = {} # Reset flip states

        if st.session_state.flashcards_result:
            st.subheader("Flashcards:")
            result = st.session_state.flashcards_result
            if isinstance(result, str): # Check if it's an error/block message
                st.error(result)
            elif result:
                for i, (question, answer) in enumerate(result):
                    card_key = f"flashcard_{i}"
                    is_flipped = st.session_state.flashcard_states.get(card_key, False)
                    with st.expander(f"**Flashcard {i+1}: {question}**", expanded=False):
                        if is_flipped:
                            st.write(f"**Answer:** {answer}")
                            if st.button("Hide Answer", key=f"hide_{card_key}"):
                                st.session_state.flashcard_states[card_key] = False
                                st.rerun()
                        else:
                            if st.button("Show Answer", key=f"show_{card_key}"):
                                st.session_state.flashcard_states[card_key] = True
                                st.rerun()
            else:
                st.warning("No flashcards were generated or the response was empty.")

    with tab_quiz:
        if st.button("Generate Quiz", key="quiz_btn"):
            with st.spinner("Generating quiz..."):
                quiz_result = generate_quiz(study_material)
                st.session_state.quiz_result_data = quiz_result # Store result
                st.session_state.quiz_answers = {} # Clear old answers
                st.session_state.show_quiz_results = False # Reset results view

        if st.session_state.quiz_result_data:
            st.subheader("Quiz Time!")
            quiz_data = st.session_state.quiz_result_data
            if isinstance(quiz_data, str): # Check if it's an error/block message
                st.error(quiz_data)
            elif quiz_data:
                # Display quiz form if results are not shown
                if not st.session_state.show_quiz_results:
                    for i, item in enumerate(quiz_data):
                        st.markdown(f"**Question {i+1}:** {item['question']}")
                        options = list(item['options'].items())
                        st.session_state.quiz_answers[i] = st.radio(
                            "Choose your answer:",
                            [f"{k}) {v}" for k, v in options],
                            key=f"quiz_{i}",
                            index=None # Default to no selection
                        )
                    if st.button("Submit Quiz", key="submit_quiz"):
                        st.session_state.show_quiz_results = True
                        st.rerun()
                # Display quiz results if submitted
                else:
                    st.subheader("Quiz Results")
                    score = 0
                    total = len(quiz_data)
                    for i, item in enumerate(quiz_data):
                        user_answer_full = st.session_state.quiz_answers.get(i)
                        user_answer_letter = user_answer_full.split(')')[0] if user_answer_full else None
                        correct_answer = item['answer']
                        st.markdown(f"**Question {i+1}:** {item['question']}")
                        st.write(f"Your answer: {user_answer_full if user_answer_full else 'Not answered'}")
                        st.write(f"Correct answer: {correct_answer}) {item['options'][correct_answer]}")
                        if user_answer_letter == correct_answer:
                            st.success("Correct!")
                            score += 1
                        elif user_answer_full:
                            st.error("Incorrect.")
                        else:
                            st.warning("Not answered.")
                        st.markdown("---")
                    st.markdown(f"**Your final score: {score}/{total}**")
                    if st.button("Retake Quiz / New Quiz", key="hide_results"):
                         st.session_state.show_quiz_results = False
                         st.session_state.quiz_answers = {} # Clear answers for retake
                         # Optionally clear quiz_result_data if you want 'Generate Quiz' to be pressed again
                         # st.session_state.quiz_result_data = None
                         st.rerun()
            else:
                st.warning("No quiz questions were generated or the response was empty.")

    with tab_qa:
        st.subheader("Ask a Question about the Material")
        user_question = st.text_input("Enter your question here:", key="qa_question")
        if st.button("Get Answer", key="qa_btn"):
            if user_question:
                with st.spinner("Thinking..."):
                    answer = generate_answer(study_material, user_question)
                    st.session_state.qa_answer = answer # Store answer
            else:
                st.warning("Please enter a question.")

        if st.session_state.qa_answer:
            st.markdown("**Answer:**")
            result = st.session_state.qa_answer
            if result.startswith("An error occurred") or result.startswith("Content generation") or result.startswith("Error:"):
                st.error(result)
            else:
                st.markdown(result)

else:
    st.info("Please provide study material using the sidebar to enable study tools.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with Streamlit and Google Gemini API")

