from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str
    GOOGLE_API_KEY: str
    GROQ_API_KEY: str
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "zora-s1-s2"
    LANGCHAIN_TRACING_V2: str = "true"
    UPLOAD_DIR: str = "./uploads"
    OUTPUT_DIR: str = "./outputs"
    TWILIO_ACCOUNT_SID: str = ""
    TWILIO_AUTH_TOKEN: str = ""
    TWILIO_SMS_FROM: str = ""
    TWILIO_WHATSAPP_FROM: str = ""
    DEFAULT_DOCTOR_WHATSAPP_TO: str = ""
    DOCTOR_APPROVAL_BASE_URL: str = "http://localhost:3000"
    CLOUD_TTS_API_KEY: str = ""
    CLOUD_TTS_VOICE_EN: str = "en-IN-Neural2-A"
    CLOUD_TTS_VOICE_HI: str = "hi-IN-Neural2-A"
    CLOUD_TTS_SPEAKING_RATE: float = 0.96
    HUGGINGFACE_API_KEY: str = ""   # Optional, for future ESMFold integration

    class Config:
        env_file = ".env"


settings = Settings()
