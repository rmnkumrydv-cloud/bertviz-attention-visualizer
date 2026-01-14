import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from bertviz import model_view, head_view
import pandas as pd

# Page config
st.set_page_config(
    page_title="BERTviz - Attention Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔍 BERTviz - Attention Visualizer")
st.markdown("Visualize how BERT understands your text in real-time!")

# Sidebar
st.sidebar.title("⚙️ Settings")

# Model selection
models = {
    "⚡ DistilBERT (Fast)": "distilbert-base-uncased",
    "🧠 BERT (Accurate)": "bert-base-uncased",
    "💪 RoBERTa": "roberta-base",
    "🌍 Multilingual": "bert-base-multilingual-uncased"
}

selected_model_name = st.sidebar.selectbox(
    "Choose Model:",
    list(models.keys()),
    index=0
)
selected_model = models[selected_model_name]

# Visualization type
viz_type = st.sidebar.radio(
    "Visualization Type:",
    ["Model View (All Layers)", "Head View (Specific Layer)", "Token Analysis"]
)

# Layer and head selection for Head View
selected_layer = 5
if viz_type == "Head View (Specific Layer)":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        selected_layer = st.number_input("Layer:", 0, 11, 5)
    with col2:
        st.write("")
        st.write("")
        st.write("(0-11)")

# Load model with caching
@st.cache_resource
def load_model(model_name):
    """Load model and tokenizer with caching"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            output_attentions=True
        )
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Main content
st.subheader("📝 Enter Your Text")

# User input
user_text = st.text_area(
    "Type something:",
    value="The cat sat on the mat",
    height=100,
    placeholder="Enter any sentence here..."
)

# Quick examples
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("📌 Cat & Mat"):
        st.session_state.user_text = "The cat sat on the mat"
        st.rerun()
with col2:
    if st.button("🎯 Attention"):
        st.session_state.user_text = "Attention is all you need"
        st.rerun()
with col3:
    if st.button("🦊 Fox"):
        st.session_state.user_text = "The quick brown fox jumps over the lazy dog"
        st.rerun()
with col4:
    if st.button("🤖 ML"):
        st.session_state.user_text = "Machine learning is fascinating"
        st.rerun()

# Visualize button
if st.button("🚀 Visualize Attention", type="primary", use_container_width=True):
    
    if not user_text.strip():
        st.error("❌ Please enter some text!")
    else:
        with st.spinner(f"Loading {selected_model_name}..."):
            model, tokenizer = load_model(selected_model)
        
        if model and tokenizer:
            try:
                with st.spinner("Processing..."):
                    # Tokenize - Handle token limit
                    inputs = tokenizer.encode(user_text, return_tensors='pt', truncation=True, max_length=512)
                    
                    # Validate input length
                    if inputs.shape[1] < 2:
                        st.error("❌ Input too short! Use at least 2 tokens.")
                        st.stop()
                    
                    # Forward pass
                    with torch.no_grad():
                        outputs = model(inputs)
                    
                    # CRITICAL FIX: outputs.attentions is a tuple, keep it as tuple for bertviz
                    attention_tuple = outputs.attentions  # Keep as tuple for visualization
                    
                    # For Token Analysis, we need to stack into tensor
                    attention_stacked = torch.stack(attention_tuple)  # Convert to tensor for indexing
                    # Shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
                    
                    # Convert inputs to list for token conversion
                    token_ids = inputs[0].tolist()
                    tokens = tokenizer.convert_ids_to_tokens(token_ids)
                    
                    # Display tokens
                    st.success("✅ Done!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Tokens", len(tokens))
                    with col2:
                        st.metric("Model Layers", len(attention_tuple))
                    
                    st.subheader("📊 Tokens")
                    tokens_df = pd.DataFrame({
                        "Position": list(range(len(tokens))),
                        "Token": tokens
                    })
                    st.dataframe(tokens_df, use_container_width=True)
                    
                    # Visualization
                    st.subheader("🎨 Attention Pattern")
                    
                    if viz_type == "Model View (All Layers)":
                        st.info("Showing all layers and heads - may take a moment")
                        try:
                            # Model view expects attention as tuple of tensors
                            html_str = model_view(attention_tuple, tokens, html_action='return')
                            st.components.v1.html(html_str.data, height=800, scrolling=True)
                        except Exception as viz_error:
                            st.warning(f"⚠️ Visualization error: {viz_error}")
                            st.info("Try using Head View instead for faster results")
                    
                    elif viz_type == "Head View (Specific Layer)":
                        try:
                            # Validate layer selection
                            layer_idx = min(int(selected_layer), len(attention_tuple) - 1)
                            num_heads = attention_tuple[layer_idx].shape[1]
                            
                            # Head view also expects tuple format
                            html_str = head_view(
                                attention_tuple,
                                tokens,
                                layer=layer_idx,
                                heads=list(range(min(12, num_heads))),
                                html_action='return'
                            )
                            st.components.v1.html(html_str.data, height=600, scrolling=True)
                        except Exception as e:
                            st.error(f"Error in Head View: {e}")
                            st.info("Try Token Analysis mode instead")
                    
                    elif viz_type == "Token Analysis":
                        st.subheader("📈 Attention Score by Token")
                        
                        try:
                            # Use the stacked tensor for analysis
                            # attention_stacked shape: (layers, batch, heads, seq_len, seq_len)
                            avg_attention_per_token = []
                            
                            seq_len = attention_stacked.shape[3]
                            
                            for token_idx in range(seq_len):
                                # Get attention to this token from all other tokens
                                token_attention = attention_stacked[:, 0, :, :, token_idx]  # (layers, heads, seq_len)
                                avg_score = token_attention.mean().item()  # Convert to Python float
                                avg_attention_per_token.append(avg_score)
                            
                            attention_df = pd.DataFrame({
                                "Token": tokens,
                                "Attention Score": avg_attention_per_token
                            })
                            
                            st.bar_chart(attention_df.set_index("Token")["Attention Score"])
                            st.dataframe(attention_df, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error in Token Analysis: {e}")
                            st.info("Please try a shorter text (under 50 tokens)")
                    
                    # Download option
                    st.divider()
                    st.subheader("💾 Download Visualization")
                    
                    try:
                        # Use tuple format for download
                        html_download = model_view(attention_tuple, tokens, html_action='return')
                        st.download_button(
                            label="📥 Download as HTML",
                            data=html_download.data,
                            file_name=f"bertviz_{user_text[:20].replace(' ', '_')}.html",
                            mime="text/html"
                        )
                    except Exception as e:
                        st.warning(f"Download unavailable: {e}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try with shorter text or use Token Analysis mode")

# Footer
st.divider()
st.markdown("""
### 📚 About
- **BERTviz**: Interactive visualization for transformer attention
- **Models**: BERT, DistilBERT, RoBERTa, Multilingual BERT

### 💡 Tips
- DistilBERT is fastest, BERT is most accurate
- Shorter texts work better for Model View
- Try Token Analysis for quick results!
- Max 512 tokens per input

---
**Built with ❤️ for NLP enthusiasts**
""")