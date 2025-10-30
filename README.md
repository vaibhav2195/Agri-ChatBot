

# Multilingual AI Farmer's Assistant (WhatsApp Bot)

This is a sophisticated, multilingual WhatsApp bot designed to assist farmers. It integrates multiple AI services to provide crucial information like market prices, weather forecasts, and crop disease diagnosis, all in the user's native language and accessible via both text and AI-generated voice notes.

The bot is built with **FastAPI** and integrates with **Google Gemini** for translation and vision, **Sarvam AI** for text-to-speech, and various data APIs.

## ✨ Features

  * **Multilingual Support:** Automatically detects the user's language from their first message (e.g., "नमस्ते", "வணக்கம்", "Hello") and translates all subsequent interactions.
  * **Dual Response (Text + Voice):** Responds to all queries with both a formatted text message and an AI-generated voice note (using Sarvam AI) in the user's language.
  * **Conversational AI:** Uses an in-memory state machine to guide users through multi-step queries (e.g., asking for State, then District, then Commodity).
  * **Core Farmer Services:**
      * **1. Mandi Price Checker:** Fetches the latest commodity prices for a specific state and district from the [data.gov.in](https://data.gov.in/) API.
      * **2. Weather Forecast:** Provides a 5-day weather forecast for any city using the OpenWeather API.
      * **3. AI Crop Disease Diagnosis:** Uses the **Gemini Vision** model to analyze user-uploaded photos of crops, identify diseases, and suggest treatment and prevention steps.
  * **Fuzzy Matching:** Intelligently corrects user spelling for Indian state and district names to improve query success.
  * **Caching:** Caches translations to reduce API calls and improve response speed.

## 🤖 Core Services Used

  * **Web Server:** **FastAPI**
  * **Messaging:** **Meta (WhatsApp) Graph API**
  * **Translation:** **Google Gemini 1.5 Flash** (Text Model)
  * **Vision AI:** **Google Gemini 1.5 Flash** (Vision Model)
  * **Text-to-Speech (TTS):** **Sarvam AI**
  * **Audio Conversion:** **Pydub** (converts Sarvam's WAV to MP3 for WhatsApp)
  * **Weather Data:** **OpenWeather API**
  * **Market Data:** **Mandi Price API** (data.gov.in)
  * **Data Handling:** **Pandas** (for loading and matching district/state names)

## ⚙️ Setup and Installation

### 1\. Prerequisites

  * **Python 3.8+**
  * **FFmpeg:** This is a **crucial dependency** for `pydub` to convert audio files.
      * **On Ubuntu/Debian:** `sudo apt-get install ffmpeg`
      * **On macOS (via Homebrew):** `brew install ffmpeg`
      * **On Windows:** Download the binaries from the official FFmpeg website and add them to your system's PATH.

### 2\. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 3\. Install Python Dependencies

Create a `requirements.txt` file with the following contents:

```txt
fastapi
uvicorn[standard]
requests
pandas
python-dotenv
sarvamai
pydub
```

Then, install them:

```bash
pip install -r requirements.txt
```

### 4\. Download Data File

This bot relies on a CSV file for fuzzy matching state and district names. Make sure `district_level_latest.csv` is present in the same directory.

### 5\. Configure Environment Variables

Create a file named `.env` in the root directory and add the following keys with your API credentials:

```ini
# --- Meta / WhatsApp ---
# Your verification token (any random string) for the webhook setup
VERIFY_TOKEN="YOUR_WHATSAPP_VERIFY_TOKEN"
# Your permanent WhatsApp Access Token from the Meta Dev Dashboard
WHATSAPP_TOKEN="YOUR_WHATSAPP_ACCESS_TOKEN"
# Your app's Phone Number ID from the Meta Dev Dashboard
PHONE_NUMBER_ID="YOUR_WHATSAPP_PHONE_NUMBER_ID"

# --- APIs ---
# Your OpenWeather API key
OPENWEATHER_API="YOUR_OPENWEATHER_API_KEY"
# Your data.gov.in Mandi API key
MANDI_API_KEY="YOUR_MANDI_API_KEY"
# Your Google AI Studio API key for Gemini
GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# Your Sarvam AI API key
SARVAM_API_KEY="YOUR_SARVAM_AI_API_KEY"

# --- API Endpoints (Already in script, but good to have here) ---
MANDI_API="https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
```

### 6\. Run the Server

```bash
uvicorn combined_bot:app --host "0.0.0.0" --port 8000
```

### 7\. Configure the WhatsApp Webhook

Your server is now running locally on port 8000. You must expose this to the internet to receive webhooks from WhatsApp.

1.  **Use a Tunneling Service:** Use a tool like **ngrok** to create a secure public URL.
    ```bash
    ngrok http 8000
    ```
2.  **Copy the URL:** ngrok will give you a public URL like `https://<random-string>.ngrok.io`.
3.  **Set up Meta Webhook:**
      * Go to your app in the [Meta for Developers](https://developers.facebook.com/) dashboard.
      * Navigate to **WhatsApp \> Configuration**.
      * Click "Edit" for the Webhook section.
      * **Callback URL:** Enter your ngrok URL, followed by `/webhook`. (e.g., `https://<random-string>.ngrok.io/webhook`)
      * **Verify token:** Enter the *same* `VERIFY_TOKEN` you set in your `.env` file.
      * **Subscribe to Webhook Fields:** Click "Manage" and subscribe to `messages`.

Your bot is now live and ready to receive messages.

-----

## 📂 Code Structure Explained

| Function / Variable | Purpose |
| :--- | :--- |
| **FastAPI Endpoints** | |
| `@app.get("/webhook")` | Handles the one-time verification challenge from Meta/WhatsApp. |
| `@app.post("/webhook")` | The main endpoint. Receives all incoming user messages (text, images, etc.). |
| `@app.get("/")` | A simple root endpoint to check if the bot is running. |
| **Core Logic** | |
| `handle_webhook()` | The primary function that processes incoming messages, manages state, and routes logic. |
| `user_state {}` | A Python dictionary that acts as an in-memory database to track each user's conversation step (e.g., `awaiting_district`). |
| `translate_cache {}` | Caches translations to avoid redundant API calls to Gemini. |
| **Translation & AI** | |
| `call_gemini()` | A low-level helper to make HTTP POST requests to the Gemini API. |
| `translate_incoming_to_english()` | Uses Gemini to detect the source language and translate the user's text to English. |
| `translate_outgoing_from_english()` | Uses Gemini to translate the bot's English response back into the user's detected language. |
| `detect_crop_disease_from_base64()` | Sends a base64-encoded image and a specialized prompt to the Gemini Vision API. |
| **Sarvam AI (TTS)** | |
| `convert_text_to_speech()` | Takes text and a language code, calls the Sarvam AI SDK, and returns raw WAV audio bytes. |
| `send_whatsapp_audio()` | **1.** Converts the WAV audio bytes from Sarvam into an MP3 file in memory using `pydub`. <br> **2.** Uploads the MP3 to WhatsApp's media servers. <br> **3.** Sends the uploaded media to the user via its Media ID. |
| **Feature APIs** | |
| `get_mandi_price()` | Queries the data.gov.in API for market prices. |
| `get_weather_forecast()` | Queries the OpenWeather API for a 5-day forecast. |
| **Utility Functions** | |
| `get_media_url()` / `download_media_bytes()` | Helpers to download images sent by the user from WhatsApp servers. |
| `correct_state_name()` / `correct_district_name()` | Uses `pandas` and `difflib` to find the closest match for a user-typed state/district. |

-----

## 🔄 How It Works: A User's Journey

1.  A user (e.g., a Hindi speaker) sends **"नमस्ते"** to the bot's WhatsApp number.
2.  `handle_webhook` receives the message.
3.  `detect_greeting_lang()` identifies the language as **"hi"** (Hindi) and stores it in `user_state` for that user.
4.  The bot translates the English menu (`EN_TEXTS["menu"]`) into Hindi using `translate_outgoing_from_english()`.
5.  `send_translated()` sends the Hindi menu text to the user.
6.  The user replies with **"3"** to select "Crop Disease Diagnosis".
7.  `handle_webhook` translates "3" to English ("3"). The logic routes to the "crop disease" flow.
8.  The bot sets the user's state to `step = "awaiting_crop_image"`.
9.  It translates `EN_TEXTS["send_image_prompt"]` to Hindi and sends it ("Please send a photo...").
10. The user sends an **image** of a diseased plant.
11. `handle_webhook` detects `msg_type == "image"` and `current_step == "awaiting_crop_image"`.
12. The bot sends a "Please wait..." message.
13. `get_media_url()` and `download_media_bytes()` download the user's image.
14. `detect_crop_disease_from_base64()` sends the image to Gemini Vision with a prompt asking for the diagnosis *in Hindi*.
15. Gemini Vision returns a structured text response (Diagnosis, Cause, Treatment) **already in Hindi**.
16. `send_whatsapp_message_raw()` sends this Hindi text response to the user.
17. The *same Hindi text* is passed to `convert_text_to_speech()`.
18. Sarvam AI generates Hindi audio (WAV bytes).
19. `send_whatsapp_audio()` converts the WAV to MP3 and sends the voice note.
20. The user's state is cleared, and the conversation is complete.
