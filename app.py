import streamlit as st
import cohere


COHERE_API_KEY = "f9G37a19SxAsetxWEcb7WUxZJys2KJzXCtGikcHr"  # Replace with your actual key
co = cohere.Client(COHERE_API_KEY)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "bot_responses" not in st.session_state:
    st.session_state.bot_responses = []

if "bot_pairs" not in st.session_state:
    st.session_state.bot_pairs = []  # Stores (question, answer)

if "selected_response" not in st.session_state:
    st.session_state.selected_response = None

if "page" not in st.session_state:
    st.session_state.page = "main"

# --- Load model once ---
# --- Load and cache embeddings ---


# --- FAQs ---
faqs = [
    {"question": "What kind of vehicles does Toyota make?", "answer": "Toyota makes sedans, SUVs, trucks, hybrids, and electric vehicles."},
    {"question": "Where is Toyota headquartered?", "answer": "Toyota is headquartered in Toyota City, Japan, with U.S. operations based in Plano, Texas."},
    {"question": "How do I find a nearby Toyota dealership?", "answer": "You can search for dealerships on the Toyota website using your ZIP code."},
    {"question": "What is Toyota known for?", "answer": "Toyota is known for reliable, fuel-efficient vehicles and pioneering hybrid technology."},
    {"question": "How can I schedule vehicle service?", "answer": "Most Toyota dealers allow you to schedule service online or by calling directly."},
    {"question": "What is ToyotaCare?", "answer": "ToyotaCare is a no-cost maintenance plan included with new Toyota vehicles for a limited time."},
    {"question": "What warranty does Toyota offer?", "answer": "Toyota offers standard warranties for basic coverage and powertrain protection."},
    {"question": "Does Toyota sell electric or hybrid cars?", "answer": "Yes, Toyota offers several hybrid models and is expanding its electric vehicle lineup."},
    {"question": "Is Toyota a safe brand?", "answer": "Yes, Toyota vehicles often include standard safety features and receive high safety ratings."},
    {"question": "Can I test drive a Toyota vehicle?", "answer": "Yes, you can schedule a test drive at any participating Toyota dealership."},
    {"question": "Does Toyota offer financing?", "answer": "Yes, Toyota Financial Services provides loan and lease options through dealerships."},
    {"question": "How can I contact Toyota customer support?", "answer": "You can reach Toyota support via their official website or by phone."},
    {"question": "What is the Toyota app used for?", "answer": "The app helps manage your vehicle, schedule service, and access remote features (if supported)."},
    {"question": "What is a certified pre-owned Toyota?", "answer": "It’s a used Toyota that has passed a detailed inspection and comes with limited warranties."},
    {"question": "How long do Toyota cars typically last?", "answer": "With proper maintenance, many Toyota vehicles are known to last well over 200,000 miles."}
]

faq_questions = [faq["question"] for faq in faqs]

@st.cache_resource
def load_faq_embeddings():
    response = co.embed(texts=faq_questions, model="embed-english-v3.0", input_type="search_document")
    return response.embeddings

faq_embeddings = load_faq_embeddings()

# --- Cosine similarity without numpy/torch ---
def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot / (norm1 * norm2)


# --- App UI ---
st.set_page_config(page_title="Toyota Chatbot", layout="wide")

# Sidebar
with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSdgeUI4uPpTnu5OJ_OEMNc9bPfyUE9IYU8mg&s", )
    st.title("")
    st.markdown("Ask any question about Toyota's cars, service, or features.")
    st.header("🤖 Bot Responses")

    for i, (q, a) in enumerate(st.session_state.bot_pairs):
        if st.button(f"Response {i+1}"):
            st.session_state.selected_response = (q, a)
            st.session_state.page = "chat"
            st.experimental_rerun()

    if st.button("🧹 Clear History"):
        st.session_state.bot_pairs = []
        st.session_state.selected_response = None
        st.session_state.page = "main"
        st.experimental_rerun()

# Main UI
if st.session_state.page == "main":
    st.title("🚗 Welcome to Toyota Assistant")

    user_question = st.text_input("Ask me something about Toyota:")

    import re

    toyota_keywords = ["toyota", "corolla", "camry", "rav4", "highlander", "tacoma", "prius", "gr supra", "yaris", "4runner", "fortuner", "innova", "hybrid", "toyotacare", "dealership", "service", "warranty"]

    def is_toyota_related(text):
        return any(re.search(rf"\b{kw}\b", text.lower()) for kw in toyota_keywords)


    if user_question.strip():
        answer = ""
        user_embedding = co.embed(texts=[user_question], model="embed-english-v3.0", input_type="search_query").embeddings[0]
    
        scores = [cosine_similarity(user_embedding, faq_vec) for faq_vec in faq_embeddings]
        best_score = max(scores)
        best_index = scores.index(best_score)

    # --- ✅ Option 1: Similarity threshold filter ---
        threshold = 0.65
    

        if best_score >= threshold:
            st.success(faqs[best_index]['answer'])
            st.caption(f"✔️ Matched: **{faqs[best_index]['question']}**  \n🧠 Similarity Score: `{best_score:.2f}`")
        elif is_toyota_related(user_question):
        # 🔁 Fall back to Cohere LLM generate
            with st.spinner("🤖 Thinking..."):
                response = co.generate(
                model="command-r-plus",
                prompt=f"You are an expert assistant for Toyota customers. Answer this question briefly and clearly:\n\n{user_question}",
                max_tokens=150,
                temperature=0.7
            )
            answer = response.generations[0].text.strip()
            st.info(answer)
            st.caption("🤖 This answer was generated by AI.")
        else:
            st.warning("❌ Sorry, I can only answer questions related to Toyota vehicles, services, or products.") 
        st.session_state.bot_pairs.append((user_question, answer))
        if answer and (user_question, answer) not in st.session_state.bot_pairs:
            st.session_state.bot_pairs.append((user_question, answer))

elif st.session_state.page == "chat":
    st.title("📥 Selected Chat Response")
    q, a = st.session_state.selected_response
    st.markdown(f"**You asked:** {q}")
    st.info(a)

    if st.button("🔙 Back to Chat"):
        st.session_state.page = "main"
        st.experimental_rerun()
        