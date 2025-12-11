# How to Get Free OpenRouter API Key

## Quick Steps

1. **Go to OpenRouter**: https://openrouter.ai/
2. **Sign up** (free account)
3. **Go to Keys page**: https://openrouter.ai/keys
4. **Create a new key** (it's free!)
5. **Copy the API key**

## Add to Streamlit Cloud

1. Go to your Streamlit Cloud app settings
2. Click **"Secrets"** (in Advanced settings)
3. Add this line:

```toml
OPENROUTER_API_KEY = "sk-or-v1-YOUR_ACTUAL_KEY_HERE"
```

4. Click **"Save"**
5. Streamlit will auto-redeploy

## Why You Need It

Even though the model is free, OpenRouter requires an API key for:
- Rate limiting
- Usage tracking
- Security

**Don't worry** - the free model (`meta-llama/llama-3.2-1b-instruct:free`) won't charge you anything. The API key is just for authentication.

## After Adding the Key

Your app will work immediately! The "unauthorized" error will be gone.

