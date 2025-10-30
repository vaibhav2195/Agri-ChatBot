# combined_bot.py (Final Version)

"""
WhatsApp bot:
- Translates incoming user messages -> English via Gemini before calling APIs
- Processes Weather / Mandi / Crop detection using English
- Translates outgoing responses -> user's original language via Gemini
- Uses Sarvam AI for Text-to-Speech voice notes for final responses
- Uses Gemini Vision for crop disease detection (image inline_data)
- Uses in-memory per-user state and translation cache
"""

from fastapi import FastAPI, Request, Response
import requests
import os
import pandas as pd
from difflib import get_close_matches
from dotenv import load_dotenv
from datetime import datetime, timedelta
import base64
import uvicorn
import traceback
import time
import json
import io

# --- New Imports for Sarvam AI ---
from sarvamai import SarvamAI
from pydub import AudioSegment

load_dotenv()

# -------------------------
# ENV / CONFIG
# -------------------------
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
OPENWEATHER_API = os.getenv("OPENWEATHER_API")
MANDI_API = os.getenv("MANDI_API")
MANDI_API_KEY = os.getenv("MANDI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

GEMINI_TEXT_MODEL = "gemini-1.5-flash"
GEMINI_VISION_MODEL = "gemini-1.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_TEXT_MODEL}:generateContent"
GEMINI_VISION_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_VISION_MODEL}:generateContent"

# --- Sarvam AI Client Initialization ---
sarvam_client = None
if SARVAM_API_KEY:
    sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
else:
    print("WARNING: SARVAM_API_KEY not set. TTS will not function.")


# -------------------------
# Load district CSV for fuzzy match
# -------------------------
district_mapping = pd.read_csv("district_level_latest.csv", usecols=["State", "District"], dtype=str)
district_mapping["State"] = district_mapping["State"].str.strip()
district_mapping["District"] = district_mapping["District"].str.strip()
district_mapping = district_mapping.dropna(subset=["State", "District"])
STATE_LIST = sorted(district_mapping["State"].unique())
DISTRICT_LIST = sorted(district_mapping["District"].unique())

# -------------------------
# App and state
# -------------------------
app = FastAPI()
user_state = {}
translate_cache = {}

# -------------------------
# Greetings & Language Maps
# -------------------------
GREETINGS = { "en": ["hi", "hello", "hey"], "hi": ["नमस्ते", "नमस्कार", "हाय", "हेलो"], "ta": ["வணக்கம்", "ஹாய்"], "te": ["హాయ్", "నమస్తే"], "bn": ["হাই", "নমস্কার"], "pa": ["ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "ਹੈਲੋ"], "mr": ["नमस्कार"], "gu": ["નમસ્તે"], "ml": ["ഹലോ", "നമസ്കാരം"], "kn": ["ನಮಸ್ಕಾರ", "ಹಾಯ್"] }
LANG_NAME = { "en": "English", "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali", "pa": "Punjabi", "mr": "Marathi", "gu": "Gujarati", "ml": "Malayalam", "kn": "Kannada" }
SARVAM_LANGUAGE_MAP = { "en": "en-IN", "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "bn": "bn-IN", "pa": "pa-IN", "mr": "mr-IN", "gu": "gu-IN", "ml": "ml-IN", "kn": "kn-IN" }


def detect_greeting_lang(text: str):
    txt = (text or "").strip().lower()
    for lang, words in GREETINGS.items():
        if txt in [w.lower() for w in words]:
            return lang
    return None

# -------------------------
# Gemini helper (low-level)
# -------------------------
def call_gemini(url: str, payload: dict, max_retries: int = 2, timeout: int = 25):
    headers = { "Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY }
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"Gemini API request failed on attempt {attempt+1}: {e}")
            traceback.print_exc()
            time.sleep(1 + attempt)
    return None

# -------------------------
# Translation helpers
# -------------------------
def translate_incoming_to_english(text: str):
    if not text: return "", "en"
    cache_key = ("in->en", text)
    if cache_key in translate_cache: return translate_cache[cache_key]
    prompt = ( "Detect the language of the following text and translate it into English. " "Return a single valid JSON object and NOTHING else with keys 'lang_code' and 'translation'.\n" f"Text: {text}" )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = call_gemini(GEMINI_API_URL, payload)
    if not resp: return text, "en"
    try:
        content = resp["candidates"][0]["content"]["parts"][0]["text"]
        json_start, json_end = content.find('{'), content.rfind('}') + 1
        parsed = json.loads(content[json_start:json_end])
        lang, translation = parsed.get("lang_code", "en"), (parsed.get("translation", "").strip() or text)
        translate_cache[cache_key] = (translation, lang)
        return translation, lang
    except (KeyError, IndexError, json.JSONDecodeError): return text, "en"

def translate_outgoing_from_english(text: str, target_lang_code: str):
    if not text or not target_lang_code or target_lang_code == "en": return text
    cache_key = (f"en->{target_lang_code}", text)
    if cache_key in translate_cache: return translate_cache[cache_key]
    target_name = LANG_NAME.get(target_lang_code, target_lang_code)
    prompt = ( f"Translate the following English text into {target_name}. " "Keep the translation concise and friendly for a farmer. Return ONLY the translated text.\n\n" f"English Text: \"{text}\"" )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = call_gemini(GEMINI_API_URL, payload)
    if not resp: return text
    try:
        translated = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
        if translated:
            translate_cache[cache_key] = translated
            return translated
        return text
    except (KeyError, IndexError): return text

# -------------------------
# WhatsApp send (raw)
# -------------------------
def send_whatsapp_message_raw(to: str, message: str):
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "text": {"body": message}}
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        print(f"WhatsApp send to {to}: {r.status_code}")
    except Exception:
        traceback.print_exc()

def send_translated(to: str, english_text: str, user_lang: str):
    text_to_send = translate_outgoing_from_english(english_text, user_lang)
    send_whatsapp_message_raw(to, text_to_send)

# -------------------------
# Sarvam AI TTS Implementation (USER PROVIDED LOGIC)
# -------------------------
def convert_text_to_speech(text: str, lang_code: str) -> bytes:
    """
    Calls the Sarvam AI Text-to-Speech API to convert text into audio data.
    """
    if not sarvam_client:
        print("Error: Sarvam AI client is not initialized.")
        return None

    print(f"Calling Sarvam AI to convert text: '{text}'")
    target_language = SARVAM_LANGUAGE_MAP.get(lang_code, "en-IN")

    try:
        # Make the actual API call to Sarvam AI
        response = sarvam_client.text_to_speech.convert(
            text=text,
            target_language_code=target_language,
            speaker="anushka",
            model="bulbul:v2"
        )

        # Handle different possible response formats from the SDK
        if hasattr(response, 'audio_content'): # Check for raw bytes
            return response.audio_content
        elif hasattr(response, 'audios') and response.audios: # Check for object with 'audios' list
            base64_audio = response.audios[0]
            return base64.b64decode(base64_audio)
        elif isinstance(response, dict) and "audios" in response and response["audios"]: # Fallback for dict
            base64_audio = response["audios"][0]
            return base64.b64decode(base64_audio)
        else:
            print(f"Unexpected API response format: {response}")
            return None

    except Exception as e:
        print(f"Error calling Sarvam AI API: {e}")
        return None


def send_whatsapp_audio(audio_data: bytes, recipient_phone: str):
    """
    Converts raw WAV audio data to MP3, uploads it to WhatsApp's
    servers, and sends it as a message.
    """
    if not audio_data:
        print("Audio data is empty. Cannot send message.")
        return

    try:
        # Load the WAV audio data from the in-memory bytes
        wav_audio = AudioSegment.from_file(io.BytesIO(audio_data), format="wav")

        # Export the audio to an in-memory MP3 file
        mp3_buffer = io.BytesIO()
        wav_audio.export(mp3_buffer, format="mp3")
        mp3_buffer.seek(0) # Rewind the buffer to the beginning

    except Exception as e:
        print(f"Error converting audio to MP3: {e}")
        print("Please ensure you have FFmpeg installed on your system (e.g., `sudo apt-get install ffmpeg` or `brew install ffmpeg`).")
        return

    media_url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/media"
    headers = {'Authorization': f'Bearer {WHATSAPP_TOKEN}'}
    files = {'file': ('speech.mp3', mp3_buffer, 'audio/mpeg')}
    form_data = {'messaging_product': 'whatsapp'}

    try:
        upload_response = requests.post(media_url, headers=headers, data=form_data, files=files, timeout=20)
        upload_response.raise_for_status()
        media_id = upload_response.json().get('id')

        if not media_id:
            print(f"Failed to get media ID. Response: {upload_response.text}")
            return

        print(f"Successfully uploaded audio. Media ID: {media_id}")

        messages_url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
        message_payload = {
            "messaging_product": "whatsapp",
            "to": recipient_phone,
            "type": "audio",
            "audio": {"id": media_id}
        }

        send_response = requests.post(messages_url, headers=headers, json=message_payload, timeout=15)
        send_response.raise_for_status()

        print(f"Successfully sent audio message to {recipient_phone}")

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with WhatsApp API: {e}")
        response_text = "No response body."
        if e.response is not None:
            try:
                response_text = e.response.json()
            except ValueError:
                response_text = e.response.text
        print(f"Response Body: {response_text}")

# -------------------------
# Fuzzy helpers
# -------------------------
def correct_state_name(user_input: str):
    matches = get_close_matches(user_input.lower(), [s.lower() for s in STATE_LIST], n=1, cutoff=0.7)
    return STATE_LIST[[s.lower() for s in STATE_LIST].index(matches[0])] if matches else None

def correct_district_name(user_input: str, state: str = None):
    target_districts = district_mapping
    if state:
        target_districts = district_mapping[district_mapping["State"].str.lower() == state.lower()]
    if target_districts.empty: return None
    
    district_list_lower = [d.lower() for d in target_districts["District"]]
    matches = get_close_matches(user_input.lower(), district_list_lower, n=1, cutoff=0.7)
    if matches:
        original_case_district = target_districts[target_districts["District"].str.lower() == matches[0]]["District"].iloc[0]
        return original_case_district
    return None

# -------------------------
# Mandi price logic
# -------------------------
def get_mandi_price(state, district, commodity=None):
    try:
        results = []
        today = datetime.now().date()
        for i in range(10):
            check_date = today - timedelta(days=i)
            date_str = check_date.strftime("%d/%m/%Y")
            url = (
                f"{MANDI_API}?api-key={MANDI_API_KEY}&format=json"
                f"&filters[State]={state}"
                f"&filters[District]={district}"
                f"&filters[Arrival_Date]={date_str}"
            )
            if commodity and commodity.lower() != "all":
                url += f"&filters[Commodity]={commodity}"

            r = requests.get(url, timeout=10)
            data = r.json().get("records", [])
            if data:
                results.extend(data)

        if not results:
            return f"⚠️ No mandi prices found for the last 10 days in {district}, {state}."

        latest_date = max(datetime.strptime(item["Arrival_Date"], "%d/%m/%Y").date() for item in results)
        latest_date_str = latest_date.strftime("%d/%m/%Y")
        latest_records = [item for item in results if datetime.strptime(item["Arrival_Date"], "%d/%m/%Y").date() == latest_date]

        result = f"📊 Mandi Prices in {district}, {state} (Latest: {latest_date_str}):\n"
        for item in latest_records[:5]:
            result += (
                f"\n🏛 Market: {item['Market']}\n"
                f"📦 Commodity: {item['Commodity']}\n"
                f"🌾 Variety: {item.get('Variety', 'N/A')}\n"
                f"💰 Price: ₹{item.get('Modal_Price', '-')}/qtl\n"
            )
        return result
    except Exception as e:
        traceback.print_exc()
        return "⚠️ Could not fetch mandi prices right now."

# -------------------------
# Weather forecast
# -------------------------
def get_weather_forecast(city_country: str):
    try:
        city_country = city_country.strip() + ",IN"
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city_country}&appid={OPENWEATHER_API}&units=metric"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return f"⚠️ Could not find weather for '{city_country.split(',')[0]}'."

        data = r.json()
        city_name = data["city"]["name"]
        daily = {}
        for entry in data["list"]:
            date = datetime.fromtimestamp(entry["dt"]).strftime("%Y-%m-%d")
            if date not in daily: daily[date] = {"temps": [], "conds": set()}
            daily[date]["temps"].append(entry["main"]["temp"])
            daily[date]["conds"].add(entry["weather"][0]["description"].capitalize())

        out = f"📅 Weather forecast for {city_name}:\n"
        for date, info in list(daily.items())[:5]:
            avg_temp = sum(info["temps"]) / len(info["temps"])
            conditions = ", ".join(info["conds"])
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            out += f"\n{date_obj.strftime('%a, %d %b')}: {avg_temp:.1f}°C, {conditions}"
        return out
    except Exception:
        traceback.print_exc()
        return "⚠️ Could not fetch weather forecast."

# -------------------------
# WhatsApp media helpers
# -------------------------
def get_media_url(media_id: str):
    try:
        url = f"https://graph.facebook.com/v19.0/{media_id}"
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
        r = requests.get(url, headers=headers, timeout=10)
        return r.json().get("url") if r.status_code == 200 else None
    except Exception:
        return None

def download_media_bytes(media_url: str):
    try:
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
        r = requests.get(media_url, headers=headers, timeout=15)
        return r.content if r.status_code == 200 else None
    except Exception:
        return None

# -------------------------
# Gemini vision: crop disease
# -------------------------
def detect_crop_disease_from_base64(image_b64: str, target_lang_code: str = "en"):
    target_lang_name = LANG_NAME.get(target_lang_code, "English")
    prompt_text = (
        f"You are an expert plant pathologist. Analyze the attached image of a crop. "
        f"Your entire response must be in {target_lang_name} and be easy for a farmer to understand. "
        f"Respond ONLY in the following format, keeping the English keywords (like 'Diagnosis') but translating the values:\n\n"
        "Diagnosis: <Your diagnosis in {target_lang_name}>\n"
        "Cause: <A brief, simple cause in {target_lang_name}>\n"
        "Treatment:\n- <Step 1 in {target_lang_name}>\n- <Step 2 in {target_lang_name}>\n"
        "Prevention:\n- <Step 1 in {target_lang_name}>\n- <Step 2 in {target_lang_name}>\n"
        "Confidence: <A percentage value, e.g., 95%>"
    )
    payload = { "contents": [{"parts": [{"text": prompt_text}, {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}]}]}
    try:
        resp = call_gemini(GEMINI_VISION_URL, payload, timeout=40)
        if not resp:
            return f"⚠️ Gemini Vision API call failed."
        text_out = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text_out if text_out else "⚠️ Gemini did not return a diagnosis."
    except (KeyError, IndexError, Exception) as e:
        traceback.print_exc()
        return f"⚠️ An error occurred during image analysis: {e}"

# -------------------------
# English templates
# -------------------------
EN_TEXTS = {
    "menu": "👋 Hi! How can I help you today? Reply with:\n1️⃣ Mandi Price\n2️⃣ Weather Forecast\n3️⃣ Crop Disease Diagnosis",
    "ask_state": "Please enter your State name.",
    "ask_district": "✅ State: {state}. Now, what is your District?",
    "ask_commodity": "✅ District: {district}. Now enter the Crop/Commodity name (or type 'all' to see everything).",
    "ask_city": "Please enter your city name for the weather forecast.",
    "send_image_prompt": "Please send a clear photo of the affected plant part (leaf, stem, etc.).",
    "analysis_wait": "🔎 Analyzing your photo... This may take a moment.",
    "unknown_command": "Sorry, I didn't understand. Please type 'Hi' to see the menu again.",
    "cant_find_state": "❌ Sorry, I couldn't find that state. Please check the spelling and try again.",
    "cant_find_district": "❌ Sorry, I couldn't find that district in the selected state. Please try again.",
    "no_media_id": "⚠️ Could not get the image from your message. Please try sending it again.",
    "failed_media_url": "⚠️ Failed to process the image URL from WhatsApp.",
    "failed_download": "⚠️ Failed to download the image for analysis.",
}

# -------------------------
# Webhook endpoints
# -------------------------
@app.get("/webhook")
async def verify_webhook(request: Request):
    if (request.query_params.get("hub.mode") == "subscribe" and 
        request.query_params.get("hub.verify_token") == VERIFY_TOKEN):
        return Response(content=request.query_params.get("hub.challenge"), media_type="text/plain")
    return Response(content="Verification token mismatch", status_code=403)

@app.post("/webhook")
async def handle_webhook(request: Request):
    body = await request.json()
    try:
        if not (body.get("object") and body.get("entry") and body["entry"][0].get("changes") and body["entry"][0]["changes"][0].get("value") and body["entry"][0]["changes"][0]["value"].get("messages") and body["entry"][0]["changes"][0]["value"]["messages"][0]):
            return Response(status_code=200)
            
        message = body["entry"][0]["changes"][0]["value"]["messages"][0]
        from_number = message["from"]
        raw_text, msg_type = "", message.get("type")
        if msg_type == "text":
            raw_text = message["text"].get("body", "").strip()
        elif msg_type == "interactive" and message["interactive"]["type"] == "button_reply":
            raw_text = message["interactive"]["button_reply"].get("title", "").strip()

        state = user_state.get(from_number, {})
        user_lang = state.get("lang")

        if greeting_lang := detect_greeting_lang(raw_text):
            user_state[from_number] = {"lang": greeting_lang}
            send_translated(from_number, EN_TEXTS["menu"], greeting_lang)
            return Response(status_code=200)
            
        if raw_text and not user_lang:
            _, detected_lang = translate_incoming_to_english(raw_text)
            user_lang = detected_lang
            state["lang"] = user_lang
            user_state[from_number] = state
        
        incoming_en = raw_text
        if raw_text and user_lang and user_lang != 'en':
             incoming_en, _ = translate_incoming_to_english(raw_text)
        text_for_logic = incoming_en.lower()
        current_step = state.get("step")

        if text_for_logic in ["hi", "hello", "menu", "start"] or any(g in text_for_logic for g in ["namaste", "vanakkam"]):
            user_state[from_number] = {"lang": user_lang or "en"}
            send_translated(from_number, EN_TEXTS["menu"], user_lang or "en")
            return Response(status_code=200)

        if text_for_logic in ["1", "mandi price"]:
            state["step"] = "awaiting_state"
            send_translated(from_number, EN_TEXTS["ask_state"], user_lang)
        elif text_for_logic in ["2", "weather forecast"]:
            state["step"] = "awaiting_city"
            send_translated(from_number, EN_TEXTS["ask_city"], user_lang)
        elif text_for_logic in ["3", "crop disease diagnosis"]:
            state["step"] = "awaiting_crop_image"
            send_translated(from_number, EN_TEXTS["send_image_prompt"], user_lang)
        
        elif current_step == "awaiting_state":
            if corrected_state := correct_state_name(text_for_logic.title()):
                state.update({"State": corrected_state, "step": "awaiting_district"})
                send_translated(from_number, EN_TEXTS["ask_district"].format(state=corrected_state), user_lang)
            else:
                send_translated(from_number, EN_TEXTS["cant_find_state"], user_lang)
        
        elif current_step == "awaiting_district":
            if corrected_district := correct_district_name(text_for_logic.title(), state.get("State")):
                state.update({"District": corrected_district, "step": "awaiting_commodity"})
                send_translated(from_number, EN_TEXTS["ask_commodity"].format(district=corrected_district), user_lang)
            else:
                send_translated(from_number, EN_TEXTS["cant_find_district"], user_lang)

        elif current_step == "awaiting_commodity":
            commodity = None if text_for_logic == "all" else text_for_logic.title()
            res_text = get_mandi_price(state["State"], state["District"], commodity)
            translated_text = translate_outgoing_from_english(res_text, user_lang)
            send_whatsapp_message_raw(from_number, translated_text)
            
            # --- New Audio Logic ---
            audio_data = convert_text_to_speech(translated_text, user_lang)
            if audio_data:
                send_whatsapp_audio(audio_data, from_number)
            
            user_state.pop(from_number, None)

        elif current_step == "awaiting_city":
            res_text = get_weather_forecast(text_for_logic)
            translated_text = translate_outgoing_from_english(res_text, user_lang)
            send_whatsapp_message_raw(from_number, translated_text)

            # --- New Audio Logic ---
            audio_data = convert_text_to_speech(translated_text, user_lang)
            if audio_data:
                send_whatsapp_audio(audio_data, from_number)

            user_state.pop(from_number, None)
            
        elif msg_type == "image" and current_step == "awaiting_crop_image":
            media_id = message["image"].get("id")
            if not media_id:
                send_translated(from_number, EN_TEXTS["no_media_id"], user_lang)
            else:
                send_translated(from_number, EN_TEXTS["analysis_wait"], user_lang)
                media_url = get_media_url(media_id)
                img_bytes = download_media_bytes(media_url) if media_url else None
                if not img_bytes:
                    send_translated(from_number, EN_TEXTS["failed_download"], user_lang)
                else:
                    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                    diagnosis_result = detect_crop_disease_from_base64(img_b64, user_lang)
                    send_whatsapp_message_raw(from_number, diagnosis_result)
                    
                    # --- New Audio Logic ---
                    audio_data = convert_text_to_speech(diagnosis_result, user_lang)
                    if audio_data:
                        send_whatsapp_audio(audio_data, from_number)

            user_state.pop(from_number, None)

        else:
            if raw_text:
                send_translated(from_number, EN_TEXTS["unknown_command"], user_lang)
        
        if from_number in user_state:
            user_state[from_number] = state
        return Response(status_code=200)

    except Exception:
        traceback.print_exc()
        return Response(content="Error processing request", status_code=500)

@app.get("/")
async def home():
    return {"status": "Bot is running with Sarvam AI TTS"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
