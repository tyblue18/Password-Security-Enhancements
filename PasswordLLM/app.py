import streamlit as st
import sys
import os
import atexit
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from query import load_vector_stores, get_rag_response_streaming, detect_content_type, extract_password_from_query
import time

try:
    from password_analyzer import PasswordAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    PasswordAnalyzer = None

try:
    from query import is_password_generation_request
except ImportError:
    def is_password_generation_request(query: str) -> bool:
        """Check if the query is asking to generate/create a password"""
        query_lower = query.lower()
        generation_keywords = [
            r"generate.*password",
            r"create.*password",
            r"make.*password",
            r"new password",
            r"give me.*password",
            r"suggest.*password"
        ]
        for pattern in generation_keywords:
            if re.search(pattern, query_lower):
                return True
        return False


st.set_page_config(
    page_title="Password Security Enhancements",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1565a0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .example-query {
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 3px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .example-query:hover {
        background-color: #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_stores' not in st.session_state:
    st.session_state.vector_stores = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""


@st.cache_resource
def load_stores():
    """Load vector stores with caching"""
    try:
        return load_vector_stores()
    except Exception as e:
        st.error(f"Error loading vector stores: {e}")
        st.info("Make sure you've run `initialize_rag.py` first to create the vector stores.")
        return None


def initialize_app():
    """Initialize the application"""
    if not st.session_state.initialized:
        with st.spinner("Loading password security knowledge base..."):
            st.session_state.vector_stores = load_stores()
            if st.session_state.vector_stores:
                st.session_state.initialized = True
                return True
    return st.session_state.initialized


# Main content
st.markdown('<div class="main-header">Password Security Enhancements</div>', unsafe_allow_html=True)

if not initialize_app():
    st.error("Failed to initialize the application. Please check your vector stores.")
    st.stop()

# Query input
st.markdown("---")

query = st.text_input(
    "Enter your password security question:",
    value=st.session_state.query_input,
    key="query_text_input",
    placeholder="e.g., Is 'mypassword123' secure?",
    label_visibility="collapsed"
)

if 'query_input' in st.session_state:
    st.session_state.query_input = ""

submit_button = st.button("Ask", type="primary", use_container_width=True)

if submit_button and query.strip():
    if not st.session_state.vector_stores:
        st.error("Vector stores not loaded. Please check your setup.")
        st.stop()
    
    extracted_pwd = extract_password_from_query(query)
    if extracted_pwd and ANALYZER_AVAILABLE:
        try:
            analyzer = PasswordAnalyzer()
            quick_analysis = analyzer.analyze(extracted_pwd)
            
            with st.expander("Quick Technical Analysis", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Length", f"{quick_analysis['length']} characters")
                with col2:
                    st.metric("Entropy", f"{quick_analysis['entropy']} bits")
                with col3:
                    st.metric("Score", f"{quick_analysis['strength_score']}/10")
                with col4:
                    var = quick_analysis['character_variety']
                    st.metric("Unique Characters", var['unique_chars'])
                
                issues = []
                if quick_analysis['is_common']:
                    issues.append("Found in common password lists")
                if quick_analysis['has_keyboard_pattern']:
                    issues.append("Contains keyboard patterns")
                if quick_analysis['has_sequential_pattern']:
                    issues.append("Contains sequential patterns")
                if quick_analysis['has_dictionary_word']:
                    issues.append("Contains dictionary words")
                
                if issues:
                    st.warning("**Issues detected:** " + " | ".join(issues))
                else:
                    st.success("No obvious security issues detected")
        except Exception as e:
            st.debug(f"Quick analysis failed: {e}")
    
    content_type = detect_content_type(query)
    if content_type:
        st.info(f"Searching: {content_type.replace('_', ' ').title()} knowledge base")
    else:
        st.info("Searching all knowledge bases")
    
    response_placeholder = st.empty()
    full_response = ""
    
    try:
        with st.spinner("Generating detailed recommendations..."):
            response_stream = get_rag_response_streaming(query, st.session_state.vector_stores)
            
            connection_error = False
            for chunk in response_stream:
                if "Could not connect to Ollama" in chunk or "Connection refused" in chunk:
                    connection_error = True
                full_response += chunk
                if not connection_error:
                    response_placeholder.markdown(full_response)
                time.sleep(0.01)  
            
            is_connection_error = connection_error or ("Could not connect to Ollama" in full_response or "Connection refused" in full_response)
            
            is_gen_request = is_password_generation_request(query)
            
            llm_refused = full_response and any(phrase in full_response.lower() for phrase in [
                "cannot provide", "cannot analyze", "cannot help", "cannot generate", "illegal", "hacking", 
                "refuse", "unable to", "not able to", "i can't", "i cannot", "harmful activities"
            ])
            
            if is_gen_request:
                requested_length = None
                length_match = re.search(r'length\s+(?:of\s+)?(\d+)', query, re.IGNORECASE)
                if length_match:
                    requested_length = int(length_match.group(1))
                else:
                    length_match = re.search(r'(\d+)\s+characters?', query, re.IGNORECASE)
                    if length_match:
                        requested_length = int(length_match.group(1))
                    else:
                        length_match = re.search(r'(\d+)\s*-\s*(\d+)', query)
                        if length_match:
                            requested_length = int(length_match.group(2))  
                
                generated_pwd_from_llm = None
                if full_response and not llm_refused:
                    pwd_match = re.search(r'Generated Password:\s*([A-Za-z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?]{6,})', full_response)
                    if pwd_match:
                        generated_pwd_from_llm = pwd_match.group(1)
                    else:
                        pwd_patterns = [
                            r'Password:\s*([A-Za-z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?]{6,})',
                            r'password:\s*([A-Za-z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?]{6,})',
                            r'([A-Za-z0-9!@#$%^&*()_+\-=\[\]{}|;:,.<>?]{6,})',  
                        ]
                        for pattern in pwd_patterns:
                            matches = re.findall(pattern, full_response)
                            if matches:
                                if requested_length:
                                    for match in matches:
                                        if len(match) == requested_length and any(c.isupper() for c in match) and any(c.islower() for c in match):
                                            generated_pwd_from_llm = match
                                            break
                                    if not generated_pwd_from_llm:
                                        matches_by_length = sorted(matches, key=lambda x: abs(len(x) - requested_length))
                                        for match in matches_by_length:
                                            if any(c.isupper() for c in match) and any(c.islower() for c in match):
                                                generated_pwd_from_llm = match
                                                break
                                else:
                                    for match in sorted(matches, key=len, reverse=True):
                                        if len(match) >= 8 and any(c.isupper() for c in match) and any(c.islower() for c in match):
                                            generated_pwd_from_llm = match
                                            break
                                if generated_pwd_from_llm:
                                    break
                
                llm_length_mismatch = False
                if generated_pwd_from_llm and requested_length:
                    length_diff = abs(len(generated_pwd_from_llm) - requested_length)
                    if length_diff > 1:
                        # LLM didn't respect length requirement (off by more than 1), use programmatic generator
                        llm_length_mismatch = True
                        generated_pwd_from_llm = None
                    elif length_diff == 1:
                        llm_length_mismatch = True
                
                if generated_pwd_from_llm:
                    response_placeholder.empty()
                    if ANALYZER_AVAILABLE:
                        try:
                            analyzer = PasswordAnalyzer()
                            analysis = analyzer.analyze(generated_pwd_from_llm)
                            
                            if requested_length and abs(len(generated_pwd_from_llm) - requested_length) == 1:
                                st.warning(f"Note: Generated password is {len(generated_pwd_from_llm)} characters (requested {requested_length}). Close enough - using AI-generated password as required.")
                            
                            st.markdown("### Generated Password")
                            unique_id = abs(hash(generated_pwd_from_llm)) % 1000000
                            
                            try:
                                import streamlit.components.v1 as components
                                
                                password_html = f"""
                                <div style="margin: 1rem 0; padding: 1rem; background-color: #f8f9fa; border: 2px solid #1f77b4; border-radius: 8px;">
                                    <div style="display: flex; align-items: center; gap: 1rem;">
                                        <input type="text" 
                                               id="pwd_input_{unique_id}"
                                               value="{generated_pwd_from_llm.replace('"', '&quot;')}" 
                                               readonly
                                               onclick="this.select();"
                                               style="flex: 1; padding: 0.75rem; border: 2px solid #1f77b4; border-radius: 6px; font-family: 'Courier New', monospace; font-size: 1.1rem; font-weight: 600; background-color: #ffffff; color: #000000; letter-spacing: 1px; cursor: text;">
                                        <button onclick="copyPassword_{unique_id}()" 
                                                style="padding: 0.75rem 1.5rem; background: #1f77b4; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 1rem; font-weight: 600;">
                                            📋 Copy
                                        </button>
                                    </div>
                                    <div id="copy_msg_{unique_id}" style="margin-top: 0.5rem; color: #28a745; font-weight: 500; display: none;">
                                        ✓ Copied to clipboard!
                                    </div>
                                </div>
                                <script>
                                function copyPassword_{unique_id}() {{
                                    const input = document.getElementById('pwd_input_{unique_id}');
                                    const msg = document.getElementById('copy_msg_{unique_id}');
                                    const password = input.value;
                                    
                                    input.select();
                                    input.setSelectionRange(0, 99999);
                                    
                                    if (navigator.clipboard && navigator.clipboard.writeText) {{
                                        navigator.clipboard.writeText(password).then(function() {{
                                            if (msg) {{
                                                msg.style.display = 'block';
                                                setTimeout(function() {{ msg.style.display = 'none'; }}, 2000);
                                            }}
                                        }}).catch(function(err) {{
                                            // Fallback
                                            const textarea = document.createElement('textarea');
                                            textarea.value = password;
                                            textarea.style.position = 'fixed';
                                            textarea.style.opacity = '0';
                                            document.body.appendChild(textarea);
                                            textarea.select();
                                            document.execCommand('copy');
                                            document.body.removeChild(textarea);
                                            if (msg) {{
                                                msg.style.display = 'block';
                                                setTimeout(function() {{ msg.style.display = 'none'; }}, 2000);
                                            }}
                                        }});
                                    }} else {{
                                        // Fallback for older browsers
                                        const textarea = document.createElement('textarea');
                                        textarea.value = password;
                                        textarea.style.position = 'fixed';
                                        textarea.style.opacity = '0';
                                        document.body.appendChild(textarea);
                                        textarea.select();
                                        document.execCommand('copy');
                                        document.body.removeChild(textarea);
                                        if (msg) {{
                                            msg.style.display = 'block';
                                            setTimeout(function() {{ msg.style.display = 'none'; }}, 2000);
                                        }}
                                    }}
                                }}
                                </script>
                                """
                                components.html(password_html, height=120)
                            except ImportError:
                                st.code(generated_pwd_from_llm, language=None)
                                st.info("Click the copy button in the code block above to copy the password.")
                            
                            st.info("This password was generated by the AI model.")
                            st.stop()  
                        except Exception as e:
                            st.error(f"Failed to analyze generated password: {e}")
                
                # If LLM refused or didn't generate (or didn't respect length), use programmatic generator
                if llm_refused or not generated_pwd_from_llm:
                    response_placeholder.empty()
                    # Silently fall back to programmatic generator 
                    if ANALYZER_AVAILABLE:
                        try:
                            analyzer = PasswordAnalyzer()
                            length = requested_length
                            if length is None:
                                length_match = re.search(r'length\s+(?:of\s+)?(\d+)', query, re.IGNORECASE)
                                if length_match:
                                    length = int(length_match.group(1))
                                
                                if length is None:
                                    length_match = re.search(r'(\d+)\s+characters?', query, re.IGNORECASE)
                                    if length_match:
                                        length = int(length_match.group(1))
                                
                                if length is None:
                                    length_match = re.search(r'(\d+)\s*-\s*(\d+)', query)
                                    if length_match:
                                        min_len, max_len = int(length_match.group(1)), int(length_match.group(2))
                                        length = max_len  
                                
                                if length is None:
                                    length = 20
                            
                            generated_pwd = analyzer.generate_password(length=length, min_length=max(6, length), max_length=max(length, 24))
                            
                            st.markdown("### Generated Password")
                            unique_id = abs(hash(generated_pwd)) % 1000000
                            
                            try:
                                import streamlit.components.v1 as components
                                
                                password_html = f"""
                                <div style="margin: 1rem 0; padding: 1rem; background-color: #f8f9fa; border: 2px solid #1f77b4; border-radius: 8px;">
                                    <div style="display: flex; align-items: center; gap: 1rem;">
                                        <input type="text" 
                                               id="pwd_input_prog_{unique_id}"
                                               value="{generated_pwd.replace('"', '&quot;')}" 
                                               readonly
                                               onclick="this.select();"
                                               style="flex: 1; padding: 0.75rem; border: 2px solid #1f77b4; border-radius: 6px; font-family: 'Courier New', monospace; font-size: 1.1rem; font-weight: 600; background-color: #ffffff; color: #000000; letter-spacing: 1px; cursor: text;">
                                        <button onclick="copyPassword_prog_{unique_id}()" 
                                                style="padding: 0.75rem 1.5rem; background: #1f77b4; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 1rem; font-weight: 600;">
                                            📋 Copy
                                        </button>
                                    </div>
                                    <div id="copy_msg_prog_{unique_id}" style="margin-top: 0.5rem; color: #28a745; font-weight: 500; display: none;">
                                        ✓ Copied to clipboard!
                                    </div>
                                </div>
                                <script>
                                function copyPassword_prog_{unique_id}() {{
                                    const input = document.getElementById('pwd_input_prog_{unique_id}');
                                    const msg = document.getElementById('copy_msg_prog_{unique_id}');
                                    const password = input.value;
                                    
                                    input.select();
                                    input.setSelectionRange(0, 99999);
                                    
                                    if (navigator.clipboard && navigator.clipboard.writeText) {{
                                        navigator.clipboard.writeText(password).then(function() {{
                                            if (msg) {{
                                                msg.style.display = 'block';
                                                setTimeout(function() {{ msg.style.display = 'none'; }}, 2000);
                                            }}
                                        }}).catch(function(err) {{
                                            // Fallback
                                            const textarea = document.createElement('textarea');
                                            textarea.value = password;
                                            textarea.style.position = 'fixed';
                                            textarea.style.opacity = '0';
                                            document.body.appendChild(textarea);
                                            textarea.select();
                                            document.execCommand('copy');
                                            document.body.removeChild(textarea);
                                            if (msg) {{
                                                msg.style.display = 'block';
                                                setTimeout(function() {{ msg.style.display = 'none'; }}, 2000);
                                            }}
                                        }});
                                    }} else {{
                                        // Fallback for older browsers
                                        const textarea = document.createElement('textarea');
                                        textarea.value = password;
                                        textarea.style.position = 'fixed';
                                        textarea.style.opacity = '0';
                                        document.body.appendChild(textarea);
                                        textarea.select();
                                        document.execCommand('copy');
                                        document.body.removeChild(textarea);
                                        if (msg) {{
                                            msg.style.display = 'block';
                                            setTimeout(function() {{ msg.style.display = 'none'; }}, 2000);
                                        }}
                                    }}
                                }}
                                </script>
                                """
                                components.html(password_html, height=120)
                            except ImportError:
                                # Fallback to st.code which has built-in copy
                                st.code(generated_pwd, language=None)
                                st.info("Click the copy button in the code block above to copy the password.")
                            
                            st.info("This password was generated using cryptographically secure random generation.")
                        except Exception as e:
                            st.error(f"Failed to generate password programmatically: {e}")
                    st.stop()  
            
            if extracted_pwd and ANALYZER_AVAILABLE and not is_connection_error:
                try:
                    analyzer = PasswordAnalyzer()
                    analysis = analyzer.analyze(extracted_pwd)
                    char_variety = analysis.get('character_variety', {})
                    
                    correct_uppercase = char_variety.get('uppercase', 0)
                    correct_lowercase = char_variety.get('lowercase', 0)
                    correct_digits = char_variety.get('digits', 0)
                    correct_special = char_variety.get('special', 0)
                    correct_unique = char_variety.get('unique_chars', 0)
                    length = analysis.get('length', 0)
                    
                    patterns_to_fix = [
                        (r'Uppercase:\s*\d+\s*(?:characters?|chars?)?', f"Uppercase: {correct_uppercase} characters", re.IGNORECASE),
                        (r'Lowercase:\s*\d+\s*(?:characters?|chars?)?', f"Lowercase: {correct_lowercase} characters", re.IGNORECASE),
                        (r'Digits?:\s*\d+\s*(?:characters?|chars?)?', f"Digits: {correct_digits} characters", re.IGNORECASE),
                        (r'Special:\s*\d+\s*(?:characters?|chars?)?', f"Special: {correct_special} characters", re.IGNORECASE),
                        (r'Unique\s+characters?:\s*\d+', f"Unique characters: {correct_unique}", re.IGNORECASE),
                    ]
                    
                    for pattern, replacement, flags in patterns_to_fix:
                        full_response = re.sub(pattern, replacement, full_response, flags=flags)
                    
                    if correct_unique > 1:
                        full_response = re.sub(r'only\s+one\s+unique\s+character[^.]*\.', '', full_response, flags=re.IGNORECASE)
                        full_response = re.sub(r'lack\s+of\s+uniqueness[^.]*\.', '', full_response, flags=re.IGNORECASE)
                        uniqueness_ratio = correct_unique / length if length > 0 else 0
                        if uniqueness_ratio > 0.8:
                            full_response = re.sub(r'[^.]*uniqueness[^.]*\.', '', full_response, flags=re.IGNORECASE)
                    
                    lines = full_response.split('\n')
                    cleaned_lines = []
                    for line in lines:
                        is_contradictory = False
                        if 'only one' in line.lower():
                            if (('uppercase' in line.lower() and correct_uppercase > 1) or
                                ('lowercase' in line.lower() and correct_lowercase > 1) or
                                ('digit' in line.lower() and correct_digits > 1) or
                                ('special' in line.lower() and correct_special > 1) or
                                ('unique' in line.lower() and correct_unique > 1)):
                                is_contradictory = True
                        if not is_contradictory:
                            cleaned_lines.append(line)
                    full_response = '\n'.join(cleaned_lines)
                    
                    response_placeholder.markdown(full_response)
                except Exception as e:
                    print(f"[WARNING] Could not post-process LLM response: {e}")
            
            if is_connection_error:
                response_placeholder.empty()
                st.warning("**Could not connect to Ollama.** Make sure Ollama is running (run `ollama serve` in terminal).")
                st.info("Showing technical analysis only (works without Ollama).")
                if extracted_pwd and ANALYZER_AVAILABLE:
                    try:
                        analyzer = PasswordAnalyzer()
                        analysis = analyzer.analyze(extracted_pwd)
                        report = analyzer.format_analysis_report(analysis)
                        st.markdown("### Technical Analysis")
                        st.text(report)
                    except:
                        pass
                st.stop()
        
    except Exception as e:
        error_msg = str(e)
        is_conn_error = ("Connection refused" in error_msg or "Errno 61" in error_msg)
        if is_conn_error:
            st.warning("**Could not connect to Ollama.** Make sure Ollama is running.")
            if extracted_pwd and ANALYZER_AVAILABLE:
                try:
                    analyzer = PasswordAnalyzer()
                    analysis = analyzer.analyze(extracted_pwd)
                    report = analyzer.format_analysis_report(analysis)
                    st.markdown("### Technical Analysis")
                    st.text(report)
                except:
                    pass
        else:
            st.error(f"Error generating response: {error_msg}")

# Cleanup function
def cleanup():
    try:
        if 'vector_stores' in st.session_state and st.session_state.vector_stores:
            for store in st.session_state.vector_stores.values():
                if hasattr(store, '_client'):
                    try:
                        store._client.clear_system_cache()
                    except:
                        pass
    except:
        pass

atexit.register(cleanup)

