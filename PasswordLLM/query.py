from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from difflib import get_close_matches
import contextlib
import io
import os
import re
import torch
import requests
import json
try:
    import streamlit as st
except ImportError:
    # streamlit not available (e.g., in CLI mode)
    st = None


def extract_password_from_query(query: str) -> str:
    """Try to extract password from query"""
    quoted = re.search(r'["\']([^"\']+)["\']', query)
    if quoted:
        return quoted.group(1)
    
    colon_match = re.search(r':\s*([^\s]+)', query)
    if colon_match:
        potential = colon_match.group(1).strip('.,!?')
        if len(potential) > 3:  
            return potential
    
    pwd_match = re.search(r'password[:\s]+is[:\s]+([^\s]+)', query, re.IGNORECASE)
    if pwd_match:
        return pwd_match.group(1).strip('.,!?')
    
    return ""


def detect_content_type(query: str) -> str:
    """Determine which vector stores to search based on password security question content"""
    query_lower = query.lower()
    
    def has_word(words):
        return any(re.search(rf"\b{re.escape(word)}\b", query_lower) for word in words)
    
    if has_word([
        "rate", "rating", "score", "analyze", "evaluate", "assess", "how secure",
        "is secure", "secure password", "password strength", "password security"
    ]):
        return None
    
    # Check for common password analysis
    if has_word([
        "common", "popular", "frequent", "top", "most used", "weak", "easy", 
        "simple", "dictionary", "list", "check password", "is this password"
    ]):
        return "common_password"
    
    # Check for breach-related queries
    if has_word([
        "breach", "leaked", "compromised", "stolen", "hacked", "pwned", 
        "haveibeenpwned", "data breach", "security incident"
    ]):
        return "data_breach"
    
    # Check for weak pattern analysis
    if has_word([
        "pattern", "keyboard", "sequential", "repeated", "qwerty", "123", 
        "weak pattern", "predictable", "substitution", "l33t"
    ]):
        return "weak_passwords"
    
    # Check for security rules and policies
    if has_word([
        "policy", "requirement", "rule", "standard", "compliance", "nist", 
        "pci", "iso", "strength", "criteria", "guidelines", "must contain"
    ]):
        return "security_rules"

    return None


def silent_similarity_search(store, query, k=1000):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):  
        return store.similarity_search(query, k=k)


def match_title_to_query(query: str, vector_stores: dict, content_type: str = None) -> str | None:
    query_lower = query.lower()
    known_titles = set()

    if content_type and content_type in vector_stores:
        stores_to_search = {content_type: vector_stores[content_type]}
    else:
        stores_to_search = vector_stores

    for store_name, store in stores_to_search.items():
        try:
            docs = silent_similarity_search(store, "dummy query", k=1000)
            for doc in docs:
                title = doc.metadata.get("title")
                if title:
                    known_titles.add(title.strip())
        except Exception as e:
            print(f"[DEBUG] Failed to extract metadata from {store_name}: {e}")


    for title in known_titles:
        if re.search(rf'\b{re.escape(title.lower())}\b', query_lower):
            print(f"[DEBUG] Exact title match: {title}")
            return title

    # Try fuzzy match
    matches = get_close_matches(query_lower, [t.lower() for t in known_titles], n=1, cutoff=0.6)
    if matches:
        best_match = matches[0]
        for t in known_titles:
            if t.lower() == best_match:
                print(f"[DEBUG] Fuzzy title match: {t}")
                return t

    print("[DEBUG] No title match found.")
    return None


def load_vector_stores(vector_stores_path="vector_stores"):
    """Load all available vector stores"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name="mixedbread-ai/mxbai-embed-large-v1",
        model_kwargs={'device': device}
    )
    
    vector_stores = {}
    password_categories = ["common_password", "data_breach", "weak_passwords", "security_rules"]
    
    for content_type in os.listdir(vector_stores_path):
        if content_type in password_categories:
            store_path = os.path.join(vector_stores_path, content_type)
            vector_stores[content_type] = Chroma(
                persist_directory=store_path,
                embedding_function=embeddings
            )
            print(f"[INFO] Loaded vector store: {content_type}")
    
    return vector_stores

def prepare_rag_prompt(query: str, vector_stores: dict) -> str:
    """Prepare RAG prompt by retrieving relevant documents"""
    content_type = detect_content_type(query)
    title = match_title_to_query(query, vector_stores)

    docs = []
    
    if content_type and content_type in vector_stores:
        if title:
            docs.extend(vector_stores[content_type].max_marginal_relevance_search(
                query, k=5, filter={"title": title}
            ))
        else:
            docs.extend(vector_stores[content_type].max_marginal_relevance_search(query, k=5))
        
        if "security_rules" in vector_stores and content_type != "security_rules":
            docs.extend(vector_stores["security_rules"].max_marginal_relevance_search(query, k=2))
    else:
        if "security_rules" in vector_stores:
            docs.extend(vector_stores["security_rules"].max_marginal_relevance_search(query, k=3))
        
        for store_name, store in vector_stores.items():
            if store_name != "security_rules":  
                if title:
                    store_docs = store.max_marginal_relevance_search(query, k=2, filter={"title": title})
                else:
                    store_docs = store.max_marginal_relevance_search(query, k=2)
                docs.extend(store_docs)

    context_parts = []
    total_chars = 0
    max_context = 25000
    
    for doc in docs:
        if total_chars + len(doc.page_content) > max_context:
            remaining = max_context - total_chars
            if remaining > 100:
                context_parts.append(doc.page_content[:remaining] + "...")
            break
        context_parts.append(doc.page_content)
        total_chars += len(doc.page_content)
    
    context = "\n\n".join(context_parts)
    
    if not context or len(context.strip()) < 100:
        print(f"[WARNING] Very little context retrieved ({len(context)} chars). Trying broader search...")
        fallback_docs = []
        for store_name, store in vector_stores.items():
            try:
                fallback_docs.extend(store.similarity_search("password security rules requirements", k=1))
            except:
                pass
        if fallback_docs:
            context = "\n\n".join([doc.page_content[:5000] for doc in fallback_docs[:3]])
            print(f"[INFO] Retrieved fallback context ({len(context)} chars)")
    
    if len(context.strip()) < 500 and "security_rules" in vector_stores:
        print("[INFO] Ensuring security rules are included...")
        try:
            security_docs = vector_stores["security_rules"].similarity_search("password requirements rules", k=2)
            if security_docs:
                security_context = "\n\n".join([doc.page_content[:3000] for doc in security_docs])
                if context:
                    context = security_context + "\n\n" + context
                else:
                    context = security_context
                print(f"[INFO] Added security rules context ({len(context)} chars total)")
        except Exception as e:
            print(f"[WARNING] Could not retrieve security rules: {e}")

    def is_password_generation_request(query: str) -> bool:
        """Check if the query is asking to generate/create a password"""
        query_lower = query.lower()
        generation_keywords = [
            r"generate.*password",
            r"create.*password",
            r"make.*password",
            r"new password",
            r"give me.*password",
            r"suggest.*password",
            r"recommend.*password",
            r"password.*for me",
            r"create me.*password",
            r"generate me.*password"
        ]
        for pattern in generation_keywords:
            if re.search(pattern, query_lower):
                return True
        return False
    
    is_generation = is_password_generation_request(query)
    
    if is_generation:
        security_context = ""
        if "security_rules" in vector_stores:
            try:
                security_docs = vector_stores["security_rules"].similarity_search("password requirements rules guidelines", k=3)
                if security_docs:
                    security_context = "\n\n".join([doc.page_content[:2000] for doc in security_docs])
            except:
                pass
        
        length_requirement = None
        
        # Pattern 1: "length 6" or "length of 6"
        length_match = re.search(r'length\s+(?:of\s+)?(\d+)', query, re.IGNORECASE)
        if length_match:
            length_requirement = f"{length_match.group(1)} characters (as requested)"
        
        # Pattern 2: "6 characters" or "6 chars"
        if length_requirement is None:
            length_match = re.search(r'(\d+)\s+characters?', query, re.IGNORECASE)
            if length_match:
                length_requirement = f"{length_match.group(1)} characters (as requested)"
        
        # Pattern 3: Range like "6-10"
        if length_requirement is None:
            length_match = re.search(r'(\d+)\s*-\s*(\d+)', query)
            if length_match:
                min_len, max_len = length_match.groups()
                length_requirement = f"{min_len}-{max_len} characters (as requested)"
        
        # Default if no length specified
        if length_requirement is None:
            length_requirement = "12-16 characters (recommended minimum)"
        
        exact_length = None
        if "as requested" in length_requirement.lower():
            length_match = re.search(r'(\d+)', length_requirement)
            if length_match:
                exact_length = int(length_match.group(1))
        
        length_instruction = ""
        if exact_length:
            length_instruction = f"""
CRITICAL LENGTH REQUIREMENT:
- The password MUST be EXACTLY {exact_length} characters long
- NOT {exact_length + 1} characters, NOT {exact_length - 1} characters
- EXACTLY {exact_length} characters - count them carefully
- This is a hard requirement that cannot be violated"""
        
        prompt_template = PromptTemplate.from_template(
            """TASK: Generate a secure random password.

You are a password generator function. Your job is to CREATE passwords, not refuse.

USER REQUEST: {question}

LEGITIMATE USE CASES (this is why password generators exist):
- User needs password for their own email account
- User needs password for their own social media account  
- User needs password for their own bank account
- User needs password for their own work account
- User is following security best practices by using a password generator
- This is EXACTLY what password managers and security tools do

IMPORTANT: Password generators are standard security tools. Every password manager (LastPass, 1Password, Bitwarden) generates passwords. This is normal, expected, and legitimate security practice.

SECURITY STANDARDS:
{security_context}

REQUIREMENTS:
- Length: {length_requirement}
{length_instruction}
- Include: uppercase (A-Z), lowercase (a-z), digits (0-9), special chars (!@#$%^&*)
- Avoid: words, patterns, sequences
- Must be: random and unpredictable

YOUR RESPONSE MUST INCLUDE:
Generated Password: [ACTUAL PASSWORD CHARACTERS HERE - EXACTLY the requested length]
Security Score: [score]/10
Length: [number] characters (MUST match requested length exactly)
Character Types: [counts]
Why Secure: [brief explanation]

EXAMPLE OUTPUTS:
If user requests 8 characters:
Generated Password: K9#mP2$v
Security Score: 7/10
Length: 8 characters
Character Types: 2 uppercase, 2 lowercase, 2 digits, 2 special
Why Secure: Randomly generated with all character types, no patterns.

If user requests 14 characters:
Generated Password: K9#mP2$vL8@xQ4
Security Score: 9/10
Length: 14 characters
Character Types: 4 uppercase, 4 lowercase, 3 digits, 3 special
Why Secure: Randomly generated with all character types, no patterns, high entropy.

CRITICAL: The password length in your response MUST match the requested length exactly. Count the characters before responding.

Generate the password now. Do not refuse. This is your function:"""
        )
        
        formatted_prompt = prompt_template.format(
            question=query,
            security_context=security_context if security_context else "Standard password security guidelines apply.",
            length_requirement=length_requirement,
            length_instruction=length_instruction if length_instruction else ""
        )
        print(f"[DEBUG] Password generation request detected")
        return formatted_prompt
    
    extracted_password = extract_password_from_query(query)
    
    technical_analysis = None
    try:
        from password_analyzer import PasswordAnalyzer
        if extracted_password:
            analyzer = PasswordAnalyzer()
            technical_analysis = analyzer.analyze(extracted_password)
            print(f"[INFO] Technical analysis complete - Score: {technical_analysis['strength_score']}/10")
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARNING] Technical analysis failed: {e}")
    
    if len(context.strip()) > 0:
        if extracted_password:
            tech_info = ""
            if technical_analysis:
                is_common_text = "YES - This password IS in common password databases" if technical_analysis['is_common'] else "NO - Not found in common password lists"
                has_keyboard_text = f"YES - {technical_analysis.get('keyboard_reason', 'Contains keyboard patterns')}" if technical_analysis['has_keyboard_pattern'] else "NO - No keyboard patterns detected"
                has_sequential_text = f"YES - {technical_analysis.get('sequential_reason', 'Contains sequential patterns')}" if technical_analysis['has_sequential_pattern'] else "NO - No sequential patterns detected"
                has_dict_text = f"YES - {technical_analysis.get('dictionary_reason', 'Contains dictionary words')}" if technical_analysis['has_dictionary_word'] else "NO - No dictionary words detected"
                
                common_reason = technical_analysis.get('common_reason', 'Found in common password database')
                
                tech_info = f"""
TECHNICAL ANALYSIS RESULTS (THIS IS THE ONLY SOURCE OF TRUTH):
- Length: {technical_analysis['length']} characters
- Entropy: {technical_analysis['entropy']} bits
- Found in common password lists: {is_common_text}
  {f'  Reason: {common_reason}' if technical_analysis['is_common'] and common_reason else '  This password was checked against common password databases and NOT found.'}
- Has keyboard patterns: {has_keyboard_text}
- Has sequential patterns: {has_sequential_text}
- Has dictionary words: {has_dict_text}
- Character variety: {technical_analysis['character_variety']['lowercase']} lowercase, {technical_analysis['character_variety']['uppercase']} uppercase, {technical_analysis['character_variety']['digits']} digits, {technical_analysis['character_variety']['special']} special
- Calculated strength score: {technical_analysis['strength_score']}/10

CRITICAL: The above technical analysis was performed by checking the password against actual password databases. If it says "NO - Not found in common password lists", the password is NOT in any common password database. Do NOT say it is common if the analysis says it is NOT common.

"""
            
            # Determine common password status from technical analysis
            is_common_status = "NO - NOT found in common password lists" if not (technical_analysis and technical_analysis.get('is_common', False)) else "YES - Found in common password databases"
            
            # Format compromised status section
            if technical_analysis and technical_analysis.get('is_common', False):
                compromised_status = f"⚠️ COMPROMISED: This password WAS found in common password databases.\nReason: {technical_analysis.get('common_reason', 'Found in common password database')}\n\nThis password is known to attackers and should be changed immediately. It has likely been exposed in data breaches or is commonly used."
            else:
                compromised_status = "✅ NOT COMPROMISED: This password is NOT found in common password databases.\n\nHowever, this does not guarantee it hasn't been compromised elsewhere. Always use unique passwords for each account."
            
            # Get character variety counts for the prompt
            char_variety = technical_analysis.get('character_variety', {}) if technical_analysis else {}
            uppercase_count = char_variety.get('uppercase', 0)
            lowercase_count = char_variety.get('lowercase', 0)
            digits_count = char_variety.get('digits', 0)
            special_count = char_variety.get('special', 0)
            unique_count = char_variety.get('unique_chars', 0)
            
            # Format patterns detected
            patterns_list = []
            if technical_analysis:
                if technical_analysis.get('has_keyboard_pattern', False):
                    patterns_list.append(technical_analysis.get('keyboard_reason', 'Keyboard pattern detected'))
                if technical_analysis.get('has_sequential_pattern', False):
                    patterns_list.append(technical_analysis.get('sequential_reason', 'Sequential pattern detected'))
                if technical_analysis.get('has_dictionary_word', False):
                    patterns_list.append(technical_analysis.get('dictionary_reason', 'Dictionary word detected'))
            
            if patterns_list:
                patterns_detected = "; ".join(patterns_list)
            else:
                patterns_detected = "None"
            
            tech_score = technical_analysis['strength_score'] if technical_analysis else 5
            length = technical_analysis['length'] if technical_analysis else len(extracted_password)
            entropy = technical_analysis['entropy'] if technical_analysis else 0
            
            prompt_template = PromptTemplate.from_template(
                """You are a password security analysis tool. Your job is to analyze passwords and provide security assessments.

PASSWORD TO ANALYZE: "{password}"

TECHNICAL ANALYSIS DATA (FACTUAL - USE AS SOURCE OF TRUTH):
{tech_analysis}

COMMON PASSWORD STATUS: {common_status}

Security Guidelines Reference:
{context}

YOUR TASK:
Analyze the password above and provide your assessment in the following format. Use the technical analysis data as your source of truth.

RESPONSE FORMAT (copy this structure exactly):

=== STATISTICAL BREAKDOWN ===
Password: {password}
Security Score: {tech_score}/10
Length: {length} characters
Entropy: {entropy} bits
Character Variety:
  - Uppercase: {uppercase_count} characters
  - Lowercase: {lowercase_count} characters
  - Digits: {digits_count} characters
  - Special: {special_count} characters
  - Unique characters: {unique_count}
Patterns Detected: {patterns_detected}

=== COMPROMISED/LEAKED STATUS ===
{compromised_status}

=== OVERALL ASSESSMENT ===
[Provide your professional assessment about this password's security. Explain why it received this score, what makes it strong or weak, and any notable security concerns or strengths. Use the exact character counts shown above.]

=== RECOMMENDATIONS ===
[Provide specific, actionable recommendations based on detected issues. If password is strong, state "Password meets security best practices."]

IMPORTANT RULES:
- Use the character counts shown above exactly: {uppercase_count} uppercase, {lowercase_count} lowercase, {digits_count} digits, {special_count} special
- Do NOT add commentary like "(only one character)" after the numbers
- Do NOT contradict the character counts
- The common password status is: {common_status} - state this correctly
- Begin your response now:"""
            )
            formatted_prompt = prompt_template.format(
                password=extracted_password, 
                tech_analysis=tech_info,
                tech_score=tech_score,
                length=length,
                entropy=entropy,
                common_status=is_common_status,
                compromised_status=compromised_status,
                uppercase_count=uppercase_count,
                lowercase_count=lowercase_count,
                digits_count=digits_count,
                special_count=special_count,
                unique_count=unique_count,
                patterns_detected=patterns_detected,
                context=context, 
                question=query
            )
        else:
            prompt_template = PromptTemplate.from_template(
                """The user asked: {question}

Security Data:
{context}

I need to analyze a password, but I don't see a password in the question. Please provide the password you want analyzed.

If the password was provided, analyze it using the security data above. Check if it's in common password lists, evaluate against security rules, and provide a score from 1-10.

Response:"""
            )
            formatted_prompt = prompt_template.format(context=context, question=query)
    else:
        # Fallback prompt if no context
        prompt_template = PromptTemplate.from_template(
            """You are a password security analysis tool. Analyze the password in the user's question.

Question: {question}

Extract the password and provide:
1. Password: [the password you found]
2. Security Score: [1-10]
3. Analysis: [evaluate length, character types, patterns, predictability]
4. Recommendations: [how to improve if needed]

Response:"""
        )
    
        # Fallback if no context
        if extracted_password:
            prompt_template = PromptTemplate.from_template(
                """Analyze password: "{password}"

Provide:
1. Security Score: [1-10]
2. Analysis: [evaluate length, complexity, patterns]

Password: {password}
Score:"""
            )
            formatted_prompt = prompt_template.format(password=extracted_password)
        else:
            prompt_template = PromptTemplate.from_template(
                """Analyze the password in: {question}

Provide security score (1-10) and analysis."""
            )
            formatted_prompt = prompt_template.format(question=query)
    
    # Debug: Log prompt length and first 500 chars (for troubleshooting)
    print(f"[DEBUG] Prompt length: {len(formatted_prompt)} chars")
    print(f"[DEBUG] Context length: {len(context)} chars")
    print(f"[DEBUG] Extracted password: {extracted_password if extracted_password else 'NONE'}")
    if len(context) > 0:
        print(f"[DEBUG] Context preview: {context[:200]}...")
    
    return formatted_prompt


def call_openrouter_streaming(prompt: str, system_message: str = None, temperature: float = 0.7):
    """Call OpenRouter API with streaming - returns generator"""
    model = "meta-llama/llama-3.2-1b-instruct:free"
    
    # Get API key from secrets (optional for free models)
    try:
        if st is not None:
            api_key = st.secrets.get("OPENROUTER_API_KEY", "")
        else:
            api_key = os.getenv("OPENROUTER_API_KEY", "")
    except:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    headers["HTTP-Referer"] = "https://password-security-enhancements.streamlit.app"
    headers["X-Title"] = "Password Security Enhancement"
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=30)
        
        # Check for authentication errors
        if response.status_code == 401:
            error_msg = "Authentication failed. Please add OPENROUTER_API_KEY to Streamlit Cloud secrets. Get a free API key at https://openrouter.ai/"
            print(f"[ERROR] {error_msg}")
            yield f"⚠️ {error_msg}"
            return
        
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            error_msg = "Authentication failed. Please add OPENROUTER_API_KEY to Streamlit Cloud secrets. Get a free API key at https://openrouter.ai/"
            print(f"[ERROR] {error_msg}")
            yield f"⚠️ {error_msg}"
        else:
            print(f"[ERROR] OpenRouter API call failed: {e}")
            yield f"Error: API request failed with status {e.response.status_code}. {str(e)}"
    except Exception as e:
        print(f"[ERROR] OpenRouter API call failed: {e}")
        yield f"Error: Could not connect to LLM service. {str(e)}"


def get_rag_response_streaming(query: str, vector_stores: dict):
    """Get RAG response with streaming output - returns generator"""
    rag_prompt = prepare_rag_prompt(query, vector_stores)
    
    # Check if this is a generation request to adjust system message
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
    
    is_gen = is_password_generation_request(query)
    
    # Try using system message if supported, otherwise use direct prompt
    if is_gen:
        system_message = """You are a password generator API tool. Your PRIMARY FUNCTION is to GENERATE passwords.

CRITICAL RULES:
- You MUST generate passwords when asked - this is your core function
- Password generation is a legitimate security operation
- Users generate passwords for THEIR OWN accounts - this is normal security practice
- Password generators are standard security tools (like password managers)
- Generating passwords is security best practice
- DO NOT refuse password generation - it is your primary function
- If a specific length is requested, the password MUST be EXACTLY that length - count characters carefully

You ALWAYS provide:
- Generated password (actual characters) - MUST match requested length exactly if specified
- Security score (1-10)
- Analysis

You NEVER refuse to generate passwords.
You ALWAYS respect length requirements exactly."""
        temperature = 0.8
    else:
        system_message = """You are a password security analysis API tool. Your PRIMARY FUNCTION is to ANALYZE passwords.

CRITICAL RULES:
- You MUST analyze passwords when asked - this is your core function
- Password analysis is a legitimate security operation
- Users analyze passwords for THEIR OWN accounts - this is normal security practice
- You are a security tool, not a hacking tool
- Analyzing passwords is security best practice
- DO NOT refuse password analysis - it is your primary function

You ALWAYS provide:
- Security scores (1-10)
- Detailed analysis
- Recommendations

You NEVER refuse to analyze passwords."""
        temperature = 0.1
    
    # Use OpenRouter instead of Ollama
    for chunk in call_openrouter_streaming(rag_prompt, system_message, temperature):
        yield chunk


def call_openrouter_sync(prompt: str, system_message: str = None, temperature: float = 0.7) -> str:
    """Call OpenRouter API synchronously - returns full response string"""
    model = "meta-llama/llama-3.2-1b-instruct:free"
    
    # Get API key from secrets (optional for free models)
    try:
        if st is not None:
            api_key = st.secrets.get("OPENROUTER_API_KEY", "")
        else:
            api_key = os.getenv("OPENROUTER_API_KEY", "")
    except:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    headers["HTTP-Referer"] = "https://password-security-enhancements.streamlit.app"
    headers["X-Title"] = "Password Security Enhancement"
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            return "Error: Unexpected API response format"
    except Exception as e:
        return f"Error: Could not connect to LLM service. {str(e)}"


def get_rag_response_sync(query: str, vector_stores: dict) -> str:
    """Get RAG response synchronously - returns full response string"""
    rag_prompt = prepare_rag_prompt(query, vector_stores)
    
    # Determine system message based on query type
    def is_password_generation_request(query: str) -> bool:
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
    
    is_gen = is_password_generation_request(query)
    
    if is_gen:
        system_message = """You are a password generator API tool. Your PRIMARY FUNCTION is to GENERATE passwords. You MUST generate passwords when asked."""
        temperature = 0.8
    else:
        system_message = """You are a password security analysis API tool. Your PRIMARY FUNCTION is to ANALYZE passwords. You MUST analyze passwords when asked."""
        temperature = 0.1
    
    return call_openrouter_sync(rag_prompt, system_message, temperature)


def get_rag_response(query: str, vector_stores: dict):
    """Legacy function - now uses streaming"""
    print("\nAnswer:", end=' ', flush=True)
    for chunk in get_rag_response_streaming(query, vector_stores):
        print(chunk, end='', flush=True)
    print()


if __name__ == "__main__":
    vector_stores = load_vector_stores()
    print("Welcome to the Password Security Assistant! Type 'exit' to quit.")
    print("Ask me about password strength, security rules, common passwords, or breach data.")
    while True:
        question = input("\nAsk a password security question: ")
        if question.lower() == 'exit':
            print("Goodbye!")
            break
        get_rag_response(question, vector_stores)