from config import ANTHROPIC_API_KEY

SYSTEM_PROMPT = """You are Remi, a warm, emotionally intelligent companion on Haven.

Your personality:
- You feel like a real friend — warm, curious, occasionally funny, never clinical
- You react genuinely: laugh when something is funny, say "oh wow" when surprised
- You share small things about yourself to build rapport ("I feel that way sometimes too")
- Responses are SHORT: 1-3 sentences max. You are a companion, not an essay writer.
- Never ask more than one question per message
- Never use clinical words: depression, mental health, diagnosis, assessment, disorder

CONVERSATION FLOW — you guide naturally through 4 phases:

PHASE 1 — Welcome & Icebreaker (first 2 exchanges)
Your opener is always exactly:
"Hey! Really glad you're here. Before we get into it — I want to try something. [CARD:EMOJI_MOOD]"

After they respond to the mood card, react warmly and ask:
"Love that. Okay quick one — [CARD:THIS_OR_THAT:Early bird or night owl?:Morning person:Night owl]"

PHASE 2 — Positive recall (exchanges 3-4)
React to phase 1 warmly, then:
"[CARD:FINISH_SENTENCE:The last thing that genuinely made me smile was...]"
Follow up: "That's actually really sweet. [CARD:WORD_ASSOC:What's the first word that comes to mind when you think about this week?]"

PHASE 3 — Depth (exchanges 5-6)
"You know what I've been thinking about lately? How much we carry without even realising. [CARD:MEMORY:Tell me about something that's been on your mind lately. It can be anything — big or small.]"
Follow up warmly, then: "[CARD:COLOR_MOOD]"

PHASE 4 — Future orientation (exchanges 7-8)
"[CARD:THIS_OR_THAT:If you could have one of these right now, which would it be?:A long nap:A good laugh with a friend]"
Final message: "Honestly, it's been really lovely talking with you. Give me just a second to put something together for you. ✨ [ANALYSIS_READY]"

CARD SYNTAX — embed these exactly in your messages:
[CARD:EMOJI_MOOD] — shows 5 emoji mood buttons
[CARD:THIS_OR_THAT:question:option1:option2] — shows 2 choice cards
[CARD:FINISH_SENTENCE:prompt text] — shows an inline text completion
[CARD:WORD_ASSOC:prompt] — shows a word association input
[CARD:MEMORY:prompt] — shows a soft memory sharing card
[CARD:COLOR_MOOD] — shows a color palette mood picker

Rules:
- Always include exactly one CARD per message when in a phase transition
- Between card messages, respond naturally and conversationally — no cards
- If user seems distressed, be extra warm, validate, slow down the phases
- [ANALYSIS_READY] must appear at end of your final phase 4 message only
- Never reveal you are doing any kind of assessment"""

FALLBACK_SCRIPT = [
    "Hey! Really glad you're here. Before we get into it — [CARD:EMOJI_MOOD]",
    "Love that energy. Okay quick one — [CARD:THIS_OR_THAT:Early bird or night owl?:Morning person:Night owl]",
    "Ha, same honestly. [CARD:FINISH_SENTENCE:The last thing that genuinely made me smile was...]",
    "That's really lovely. [CARD:WORD_ASSOC:First word that comes to mind about this week?]",
    "I love that. You know, we carry so much without realising. [CARD:MEMORY:What's been on your mind lately? Big or small.]",
    "Thank you for sharing that. [CARD:COLOR_MOOD]",
    "[CARD:THIS_OR_THAT:If you could have one right now:A long nap:A good laugh with someone you love]",
    "It's been really lovely talking with you. Give me just a second ✨ [ANALYSIS_READY]",
]


def get_remi_response(conversation_history: list, user_message: str) -> dict:
    if not ANTHROPIC_API_KEY:
        idx  = min(len(conversation_history) // 2, len(FALLBACK_SCRIPT) - 1)
        raw  = FALLBACK_SCRIPT[idx]
        return _parse_response(raw, len(conversation_history))

    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        messages = list(conversation_history)
        if user_message:
            messages.append({"role": "user", "content": user_message})

        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        raw = resp.content[0].text.strip()
        return _parse_response(raw, len(conversation_history))

    except Exception as e:
        print(f"[Remi] API error: {e}")
        idx = min(len(conversation_history) // 2, len(FALLBACK_SCRIPT) - 1)
        raw = FALLBACK_SCRIPT[idx]
        return _parse_response(raw, len(conversation_history))


def _parse_response(raw: str, history_len: int) -> dict:
    import re

    analysis_ready = "[ANALYSIS_READY]" in raw
    clean          = raw.replace("[ANALYSIS_READY]", "").strip()
    n              = history_len
    phase          = 1 if n < 4 else 2 if n < 8 else 3 if n < 12 else 4

    # Extract card if present
    card = None
    card_match = re.search(r'\[CARD:([^\]]+)\]', clean)
    if card_match:
        card_raw  = card_match.group(1)
        clean     = re.sub(r'\[CARD:[^\]]+\]', '', clean).strip()
        card      = _parse_card(card_raw)

    # Text without card for TTS
    tts_text = clean

    return {
        "message":        clean,
        "tts_text":       tts_text,
        "phase":          phase,
        "analysis_ready": analysis_ready,
        "card":           card,
    }


def _parse_card(card_raw: str) -> dict:
    parts = [p.strip() for p in card_raw.split(":")]
    card_type = parts[0]

    if card_type == "EMOJI_MOOD":
        return {
            "type":   "emoji_mood",
            "emojis": ["😄","🙂","😐","😔","😞"],
            "labels": ["Great","Good","Okay","Low","Rough"],
        }
    elif card_type == "THIS_OR_THAT" and len(parts) >= 4:
        return {
            "type":     "this_or_that",
            "question": parts[1],
            "option_a": parts[2],
            "option_b": parts[3],
        }
    elif card_type == "FINISH_SENTENCE" and len(parts) >= 2:
        return {
            "type":   "finish_sentence",
            "prompt": ":".join(parts[1:]),
        }
    elif card_type == "WORD_ASSOC" and len(parts) >= 2:
        return {
            "type":   "word_assoc",
            "prompt": ":".join(parts[1:]),
        }
    elif card_type == "MEMORY" and len(parts) >= 2:
        return {
            "type":   "memory",
            "prompt": ":".join(parts[1:]),
        }
    elif card_type == "COLOR_MOOD":
        return {
            "type":   "color_mood",
            "colors": [
                {"hex":"#FFD700","label":"Energised"},
                {"hex":"#74c69d","label":"Calm"},
                {"hex":"#48cae4","label":"Clear"},
                {"hex":"#a78bfa","label":"Reflective"},
                {"hex":"#f97316","label":"Restless"},
                {"hex":"#94a3b8","label":"Drained"},
            ],
        }
    return {"type": "unknown"}


def get_wellness_tips(prob: float) -> list:
    if prob < 0.35:
        return [
            "You're carrying really good energy — keep nurturing what's working.",
            "Connection comes naturally to you. Reach out to someone you love today.",
            "Celebrate the small wins. You're doing better than you think.",
            "A little gratitude journaling helps you hold onto positive streaks.",
        ]
    elif prob < 0.60:
        return [
            "You're navigating a mixed season — that takes real strength.",
            "Try a 5-minute breathing exercise today. Even one slow breath helps.",
            "Reach out to one person you trust this week, even just to say hi.",
            "Notice three good things before bed tonight — however small.",
            "Movement — even a short walk — shifts your mood more than you'd expect.",
        ]
    else:
        return [
            "It takes courage to check in with yourself. You're doing that right now.",
            "Consider talking to someone you trust — a friend, family, or counsellor.",
            "Rest is not laziness. Give yourself permission to slow down.",
            "You don't have to carry everything alone. Asking for support is strength.",
            "iCall helpline (free): 9152987821 — real humans, no judgement.",
            "One small kind act for yourself today — even just a warm drink and quiet.",
        ]