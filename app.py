import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. LOAD DATA & CLEAN NON-NUMERIC COLUMNS ---
@st.cache_data
def load_data():
    # Load raw data to keep titles, URLs, and topics for display
    df_raw = pd.read_csv("all-leetcode-questions.csv")
    
    # Create a working copy for numerical/boolean features
    df_numeric = df_raw.copy()
    
    # Drop known text/metadata columns from the similarity matrix calculation
    columns_to_remove = ['Title', 'URL', 'Description', 'Acceptance Rate', 'Difficulty', 'Acceptance', 'Topics', 'Tags']
    for col in columns_to_remove:
        if col in df_numeric.columns:
            df_numeric = df_numeric.drop(columns=[col])
            
    # Force convert ALL remaining columns to numbers (strips '%' signs, turns text to 0)
    for col in df_numeric.columns:
        df_numeric[col] = (
            df_numeric[col]
            .astype(str)
            .str.replace('%', '', regex=True)
            .str.strip()
        )
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(0)
            
    return df_raw, df_numeric

df_raw, df_cleaned = load_data()
all_titles = df_raw['Title'].tolist()
features = list(df_cleaned.columns)

# --- 2. RECOMMENDATION FUNCTION ---
def recommend_questions(user_input, df_r, df_c, feature_cols, solved_list=None, top_n=5):
    if solved_list is None:
        solved_list = []
    
    idx = None
    try:
        sr_no = int(user_input)
        if sr_no in df_r.index:
            idx = sr_no
    except ValueError:
        match = df_r[df_r['Title'].str.lower() == str(user_input).strip().lower()]
        if not match.empty:
            idx = match.index[0]
            
    if idx is None:
        return None
        
    selected_title = df_r.loc[idx, 'Title']
    
    # Cosine Similarity calculation on cleaned numeric/boolean features
    feature_matrix = df_c[feature_cols].values
    similarities = cosine_similarity(feature_matrix[idx].reshape(1, -1), feature_matrix).flatten()
    sorted_indices = similarities.argsort()[::-1]
    
    recommendations = []
    for i in sorted_indices:
        title = df_r.iloc[i]['Title']
        if i == idx or title in solved_list:
            continue
            
        # Safely extract URL and Topics/Tags if columns exist
        url = df_r.iloc[i]['URL'] if 'URL' in df_r.columns else "#"
        topics = (
            df_r.iloc[i]['Topics'] if 'Topics' in df_r.columns 
            else (df_r.iloc[i]['Tags'] if 'Tags' in df_r.columns else "N/A")
        )
        
        recommendations.append({
            "Sr. No.": i, 
            "Title": title, 
            "URL": url,
            "Topics": topics
        })
        
        if len(recommendations) == top_n:
            break
            
    return selected_title, recommendations

# --- 3. STREAMLIT FRONTEND DESIGN ---
st.set_page_config(page_title="LeetCode Recommender", page_icon="💡", layout="centered")

st.title("💡 LeetCode Smart Problem Recommender")
st.write("Select a problem you've practiced, track what you've solved, and get algorithmically similar questions with direct links!")

# Sidebar for tracking solved questions
st.sidebar.header("📊 Track Progress")
if "solved_questions" not in st.session_state:
    st.session_state.solved_questions = []

solved_input = st.sidebar.multiselect("Mark questions as solved:", all_titles, default=st.session_state.solved_questions)
st.session_state.solved_questions = solved_input

st.sidebar.markdown(f"**Total Solved:** {len(st.session_state.solved_questions)}")

# Main input section
st.markdown("### 🔍 Find Similar Questions")
user_query = st.selectbox(
    "Choose a question (or type to search):", 
    options=["-- Select a Question --"] + all_titles
)

# Slider restricted from 1 to 5 recommendations
top_n_choice = st.slider("Number of recommendations:", min_value=1, max_value=5, value=5)

if st.button("Get Recommendations 🚀", type="primary"):
    if user_query == "-- Select a Question --":
        st.warning("⚠️ Please select a valid question first!")
    else:
        result = recommend_questions(
            user_input=user_query, 
            df_r=df_raw,
            df_c=df_cleaned, 
            feature_cols=features, 
            solved_list=st.session_state.solved_questions, 
            top_n=top_n_choice
        )
        
        if result is None:
            st.error("Could not find that question. Try another name.")
        else:
            selected_title, recs = result
            st.success(f"Showing recommendations based on: **{selected_title}**")
            
            st.markdown("### Recommended Next Practice Problems:")
            for i, rec in enumerate(recs, 1):
                with st.container():
                    # Using HTML target='_blank' to open the link cleanly in a new tab
                    st.markdown(
                        f"""
                        **{i}.** <a href="{rec['URL']}" target="_blank" style="text-decoration: none; font-weight: bold;">{rec['Title']} ↗</a>  
                        *Sr. No.:* `{rec['Sr. No.']}` | *Topics:* {rec['Topics']}
                        """,
                        unsafe_allow_html=True
                    )
                    st.divider()
