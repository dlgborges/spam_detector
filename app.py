import os
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "spam_model.h5"
TOKENIZER_PATH = "tokenizer_config.json"
MAX_WORDS = 10000
MAX_LEN = 100

# Página config
st.set_page_config(
    page_title="Detector de Spam",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="auto"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
    }
    .spam-box {
        background-color: #fff3f3;
        border: 2px solid #ff4b4b;
    }
    .ham-box {
        background-color: #f0fff4;
        border: 2px solid #28a745;
    }
    .confidence {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    .stTextArea textarea {
        font-size: 16px;
        min-height: 150px;
    }
    .divider {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 2rem 0;
    }
    .info-box {
        background-color: #f0f8ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1.5rem;
    }
    .st-emotion-cache-1kyxreq {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer. Train if not available."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        st.info("Modelo não encontrado. Treinando o modelo pela primeira vez...")
        from train import train_and_save_model
        train_and_save_model()
        st.rerun()
    
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load tokenizer with oov_token
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
    tokenizer.oov_token = "<OOV>"  # ADD THIS LINE
    
    return model, tokenizer


def predict_spam(text, model, tokenizer):
    """Predict if the given text is spam or not."""
    # Preprocess
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)[0][0]
    
    return prediction


def main():
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Header
    st.markdown('<div class="main-header">📧 Detector de Spam</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Utilize Inteligência Artificial para verificar se sua mensagem é spam ou não</div>', unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
    <div class="info-box">
        💡 <strong>Como usar:</strong> Cole o conteúdo da mensagem de e-mail no campo abaixo e clique em 
        <strong>"Verificar"</strong>. O sistema utiliza uma rede neural treinada com TensorFlow para 
        classificar a mensagem automaticamente.
    </div>
    """, unsafe_allow_html=True)
    
    # Input area
    st.markdown("### 📝 Cole a mensagem do e-mail aqui:")
    email_text = st.text_area(
        "",
        height=180,
        placeholder="Cole aqui o conteúdo da mensagem de e-mail que deseja verificar...",
        label_visibility="collapsed"
    )
    
    # Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        check_button = st.button("🔍 Verificar", use_container_width=True, type="primary")
    
    if check_button:
        if not email_text or not email_text.strip():
            st.warning("⚠️ Por favor, cole uma mensagem de e-mail para verificar.")
        else:
            # Predict
            with st.spinner("🤖 Analisando a mensagem com inteligência artificial..."):
                confidence = predict_spam(email_text, model, tokenizer)
            
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            
            # Display results
            st.markdown("### 📊 Resultado da Análise")
            
            if confidence >= 0.5:
                # SPAM
                spam_percentage = float(confidence) * 100
                st.markdown(f"""
                <div class="result-box spam-box">
                    <div style="font-size: 2.5rem;">🚨</div>
                    <div class="confidence" style="color: #FF4B4B;">⚠️ SPAM DETECTADO!</div>
                    <p style="color: #666; margin-top: 0.5rem;">
                        Esta mensagem foi classificada como <strong>spam</strong> com 
                        <strong>{spam_percentage:.1f}%</strong> de confiança.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.error("⛔ Esta mensagem é provavelmente um SPAM. Tome cuidado!")
                
                # Tips
                with st.expander("💡 Dicas de segurança"):
                    st.markdown("""
                    - **Não clique** em links suspeitos da mensagem
                    - **Não forneça** dados pessoais ou bancários
                    - **Não responda** a esta mensagem
                    - **Marque como spam** no seu cliente de e-mail
                    - Em caso de dúvida, entre em contato diretamente com a empresa mencionada
                    """)
                    
            else:
                # NOT SPAM (HAM)
                ham_percentage = (1.0 - float(confidence)) * 100
                st.markdown(f"""
                <div class="result-box ham-box">
                    <div style="font-size: 2.5rem;">✅</div>
                    <div class="confidence" style="color: #28a745;">✓ Mensagem Legítima!</div>
                    <p style="color: #666; margin-top: 0.5rem;">
                        Esta mensagem foi classificada como <strong>legítima</strong> com 
                        <strong>{ham_percentage:.1f}%</strong> de confiança.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("✅ Esta mensagem parece ser legítima e não representa spam!")
            
            # Show raw score
            with st.expander("📈 Detalhes técnicos"):
                st.write(f"**Score bruto do modelo:** {confidence:.6f}")
                st.write(f"**Probabilidade de SPAM:** {confidence*100:.2f}%")
                st.write(f"**Probabilidade de Legítimo:** {(1-confidence)*100:.2f}%")
                st.progress(float(confidence), text="Probabilidade de Spam")
    
    # Footer
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("""
    ### 🔧 Como funciona?
    
    Este detector utiliza uma **Rede Neural Artificial** construída com **TensorFlow/Keras** que foi 
    treinada com centenas de exemplos de mensagens spam e legítimas. O modelo utiliza:
    
    1. **Tokenização** — Converte o texto em sequências numéricas
    2. **Embedding** — Transforma tokens em vetores densos de significado
    3. **Global Average Pooling** — Reduz dimensionalidade mantendo informações relevantes
    4. **Camadas Dense** — Classifica a mensagem como spam (1) ou legítima (0)
    
    ---
    *Desenvolvido com ❤️ usando TensorFlow, Streamlit e Python*
    """)


if __name__ == "__main__":
    main()