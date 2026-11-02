from dotenv import load_dotenv
from google import genai

load_dotenv()              # reads .env and makes GEMINI_API_KEY available
client = genai.Client()    # automatically finds GEMINI_API_KEY

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="In one sentence, what is FastAPI?",
)
print(response.text)
