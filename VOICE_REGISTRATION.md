# Voice student registration

Student opens the register screen (web `/auth/signup` or mobile `RegisterScreen`), taps the mic, says
their name and email, and the form fields are filled. They can edit anything and continue normally;
password and confirmation are still typed by hand.

## Flow

```
Web (MediaRecorder) / Mobile (expo-av)  --audio-->  PlatX API  POST api/Auth/voice-register-extract
                                                        |  resolve tenant by domain, check ai_credits
                                                        v
                                                 exam-ia  POST /voice-student
                                                   1. STT (Whisper-compatible)  -> transcript
                                                   2. DeepSeek json extraction -> fields + followUpQuestion
                                                        |
                                                        v
                                        PlatX consumes 1 AI credit, returns
                                        { transcript, fields, missingFields, followUpQuestion, isComplete }
```

The endpoint is anonymous (the student has no account yet). Protection: the `auth` rate-limit policy
(10 req/min per IP), a 10 MB audio cap, tenant resolved server-side from `domain`, and every call
consumes one `ai_credits` unit of that tenant, so a tenant with no credits gets a localized error.

Multi-turn: the client sends the fields it already has (`currentFieldsJson`); the AI merges the new
recording into them and `followUpQuestion` asks only for what is still missing.

## exam-ia environment variables (Render)

| Variable | Required | Default | Notes |
|---|---|---|---|
| `DEEPSEEK_API_KEY` | yes | – | already used by the other endpoints |
| `STT_API_KEY` | yes | – | Groq or OpenAI key; without it `/voice-student` returns 503 `STT_NOT_CONFIGURED` |
| `STT_BASE_URL` | no | `https://api.groq.com/openai/v1` | any OpenAI-compatible `/audio/transcriptions` |
| `STT_MODEL` | no | `whisper-large-v3-turbo` | use `whisper-1` with `https://api.openai.com/v1` |

Groq's `whisper-large-v3-turbo` handles Egyptian/Gulf Arabic well and accepts `webm` (web) and `m4a` (mobile).

## Backend config

`AiService:BaseUrl` (falls back to `ChatbotAi:BaseUrl`) must point at the deployed exam-ia service —
no new setting needed.

## Mobile

`NSMicrophoneUsageDescription` was added to `app.json`; `RECORD_AUDIO` was already declared.
A new native build (prebuild without `--clean`, then EAS) is required for iOS to pick up the plist key.
