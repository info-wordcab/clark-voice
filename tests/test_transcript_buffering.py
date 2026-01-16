"""Test transcript buffering approaches.

This script simulates the TTSTextFrame streaming and tests two approaches:
- Approach A: Use speaking turn boundaries (BotStarted/StoppedSpeaking)
- Approach B: Use LLM response boundaries (LLMFullResponseStart/End)

The goal is to fix the garbled transcript issue where parallel text streams
get interleaved when the LLM regenerates due to interruptions.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


# Simulated frame types
class FrameType(Enum):
    TTS_TEXT = "tts_text"
    BOT_STARTED_SPEAKING = "bot_started_speaking"
    BOT_STOPPED_SPEAKING = "bot_stopped_speaking"
    LLM_RESPONSE_START = "llm_response_start"
    LLM_RESPONSE_END = "llm_response_end"
    USER_STARTED_SPEAKING = "user_started_speaking"  # Interruption


@dataclass
class Frame:
    type: FrameType
    text: Optional[str] = None
    timestamp_ms: int = 0


# ============================================================================
# Current buggy implementation (for reference)
# ============================================================================


class CurrentBuggyBuffer:
    """Current implementation that has the garbling bug."""

    def __init__(self):
        self.buffer = ""
        self.messages: List[dict] = []
        self.bot_speaking = False

    def process_frame(self, frame: Frame):
        if frame.type == FrameType.BOT_STARTED_SPEAKING:
            self.bot_speaking = True

        elif frame.type == FrameType.TTS_TEXT:
            # Add space between chunks if needed
            if (
                self.buffer
                and not self.buffer.endswith(" ")
                and frame.text
                and not frame.text.startswith((" ", ",", ".", "!", "?"))
            ):
                self.buffer += " "
            self.buffer += frame.text or ""

            # Emit on sentence boundary (THE BUG: doesn't reset on new LLM response)
            if frame.text and frame.text.rstrip().endswith((".", "!", "?", ":")):
                if self.buffer.strip():
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": self.buffer.strip(),
                            "timestamp_ms": frame.timestamp_ms,
                        }
                    )
                self.buffer = ""

        elif frame.type == FrameType.BOT_STOPPED_SPEAKING:
            self.bot_speaking = False
            # Flush remaining
            if self.buffer.strip():
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": self.buffer.strip(),
                        "timestamp_ms": frame.timestamp_ms,
                    }
                )
            self.buffer = ""


# ============================================================================
# Approach A: Speaking Turn Boundaries
# ============================================================================


class ApproachA_SpeakingTurnBuffer:
    """Buffer based on speaking turn boundaries.

    - On BotStartedSpeaking: start new buffer with timestamp
    - On TTSTextFrame: append to buffer
    - On BotStoppedSpeaking: emit complete utterance, clear buffer

    No sentence detection - one message per speaking turn.
    """

    def __init__(self):
        self.buffer = ""
        self.turn_start_ms = 0
        self.messages: List[dict] = []
        self.bot_speaking = False

    def process_frame(self, frame: Frame):
        if frame.type == FrameType.BOT_STARTED_SPEAKING:
            # Start fresh turn - discard any leftover from interrupted response
            self.buffer = ""
            self.turn_start_ms = frame.timestamp_ms
            self.bot_speaking = True

        elif frame.type == FrameType.TTS_TEXT:
            if not self.bot_speaking:
                # Ignore TTS frames outside of speaking turns
                return

            # Simple concatenation with space handling
            if (
                self.buffer
                and not self.buffer.endswith(" ")
                and frame.text
                and not frame.text.startswith((" ", ",", ".", "!", "?"))
            ):
                self.buffer += " "
            self.buffer += frame.text or ""

        elif frame.type == FrameType.BOT_STOPPED_SPEAKING:
            self.bot_speaking = False
            # Emit complete turn
            if self.buffer.strip():
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": self.buffer.strip(),
                        "timestamp_ms": self.turn_start_ms,
                    }
                )
            self.buffer = ""


# ============================================================================
# Approach A2: Speaking Turn with Deduplication
# ============================================================================


class ApproachA2_SpeakingTurnDedup:
    """Buffer based on speaking turns WITH deduplication.

    Same as Approach A, but also tracks recent tokens to skip duplicates.
    Uses a sliding window of recent tokens to detect and skip duplicates.
    """

    def __init__(self):
        self.buffer = ""
        self.turn_start_ms = 0
        self.messages: List[dict] = []
        self.bot_speaking = False
        # Deduplication: track last N tokens
        self.recent_tokens: List[str] = []
        self.dedup_window = 5  # Look back 5 tokens for duplicates

    def process_frame(self, frame: Frame):
        if frame.type == FrameType.BOT_STARTED_SPEAKING:
            self.buffer = ""
            self.turn_start_ms = frame.timestamp_ms
            self.bot_speaking = True
            self.recent_tokens = []  # Reset dedup on new turn

        elif frame.type == FrameType.TTS_TEXT:
            if not self.bot_speaking:
                return

            token = frame.text or ""
            normalized = token.strip().lower()

            # Check if this token was recently seen (duplicate stream)
            if normalized and normalized in [
                t.strip().lower() for t in self.recent_tokens[-self.dedup_window :]
            ]:
                # Skip duplicate
                return

            # Add to recent tokens
            self.recent_tokens.append(token)
            if len(self.recent_tokens) > self.dedup_window * 2:
                self.recent_tokens = self.recent_tokens[-self.dedup_window :]

            # Append to buffer
            if (
                self.buffer
                and not self.buffer.endswith(" ")
                and token
                and not token.startswith((" ", ",", ".", "!", "?"))
            ):
                self.buffer += " "
            self.buffer += token

        elif frame.type == FrameType.BOT_STOPPED_SPEAKING:
            self.bot_speaking = False
            if self.buffer.strip():
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": self.buffer.strip(),
                        "timestamp_ms": self.turn_start_ms,
                    }
                )
            self.buffer = ""
            self.recent_tokens = []


# ============================================================================
# Approach C: Hybrid (LLM Reset + Speaking Turn + Dedup)
# ============================================================================


class ApproachC_Hybrid:
    """Hybrid approach combining best of all strategies.

    - On LLMFullResponseStart: RESET buffer (discard interrupted partial)
    - On TTSTextFrame: append with deduplication
    - On BotStoppedSpeaking: emit complete utterance

    This handles:
    1. Interruptions (LLM reset discards old partial)
    2. Duplicate streams (deduplication)
    3. Clean message boundaries (speaking turns)
    """

    def __init__(self):
        self.buffer = ""
        self.turn_start_ms = 0
        self.messages: List[dict] = []
        self.bot_speaking = False
        self.recent_tokens: List[str] = []
        self.dedup_window = 5

    def process_frame(self, frame: Frame):
        if frame.type == FrameType.LLM_RESPONSE_START:
            # NEW LLM response starting - discard any partial from interruption
            self.buffer = ""
            self.recent_tokens = []
            self.turn_start_ms = frame.timestamp_ms

        elif frame.type == FrameType.BOT_STARTED_SPEAKING:
            self.bot_speaking = True
            if not self.turn_start_ms:
                self.turn_start_ms = frame.timestamp_ms

        elif frame.type == FrameType.TTS_TEXT:
            if not self.bot_speaking:
                return

            token = frame.text or ""
            normalized = token.strip().lower()

            # Deduplicate
            if normalized and normalized in [
                t.strip().lower() for t in self.recent_tokens[-self.dedup_window :]
            ]:
                return

            self.recent_tokens.append(token)
            if len(self.recent_tokens) > self.dedup_window * 2:
                self.recent_tokens = self.recent_tokens[-self.dedup_window :]

            # Append
            if (
                self.buffer
                and not self.buffer.endswith(" ")
                and token
                and not token.startswith((" ", ",", ".", "!", "?"))
            ):
                self.buffer += " "
            self.buffer += token

        elif frame.type == FrameType.BOT_STOPPED_SPEAKING:
            self.bot_speaking = False
            if self.buffer.strip():
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": self.buffer.strip(),
                        "timestamp_ms": self.turn_start_ms,
                    }
                )
            self.buffer = ""
            self.recent_tokens = []
            self.turn_start_ms = 0


# ============================================================================
# Approach B: LLM Response Boundaries
# ============================================================================


class ApproachB_LLMResponseBuffer:
    """Buffer based on LLM response boundaries.

    - On LLMFullResponseStart: reset buffer (discard interrupted partial)
    - On TTSTextFrame: append to buffer
    - On LLMFullResponseEnd: emit complete response
    - On BotStoppedSpeaking: also emit as fallback

    Handles interruptions by discarding partial responses.
    """

    def __init__(self):
        self.buffer = ""
        self.response_start_ms = 0
        self.messages: List[dict] = []
        self.in_response = False

    def process_frame(self, frame: Frame):
        if frame.type == FrameType.LLM_RESPONSE_START:
            # New LLM response starting - discard any partial from interrupted response
            if self.buffer.strip() and self.in_response:
                # Optionally log that we're discarding interrupted content
                pass
            self.buffer = ""
            self.response_start_ms = frame.timestamp_ms
            self.in_response = True

        elif frame.type == FrameType.TTS_TEXT:
            # Simple concatenation
            if (
                self.buffer
                and not self.buffer.endswith(" ")
                and frame.text
                and not frame.text.startswith((" ", ",", ".", "!", "?"))
            ):
                self.buffer += " "
            self.buffer += frame.text or ""

        elif frame.type == FrameType.LLM_RESPONSE_END:
            # Complete response - emit it
            if self.buffer.strip():
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": self.buffer.strip(),
                        "timestamp_ms": self.response_start_ms,
                    }
                )
            self.buffer = ""
            self.in_response = False

        elif frame.type == FrameType.BOT_STOPPED_SPEAKING:
            # Fallback: emit if we have buffered content
            if self.buffer.strip():
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": self.buffer.strip(),
                        "timestamp_ms": self.response_start_ms,
                    }
                )
            self.buffer = ""
            self.in_response = False


# ============================================================================
# Test scenarios
# ============================================================================


def create_normal_scenario() -> List[Frame]:
    """Normal conversation without interruption."""
    return [
        Frame(FrameType.BOT_STARTED_SPEAKING, timestamp_ms=1000),
        Frame(FrameType.LLM_RESPONSE_START, timestamp_ms=1000),
        Frame(FrameType.TTS_TEXT, "Hello,", 1100),
        Frame(FrameType.TTS_TEXT, " how", 1150),
        Frame(FrameType.TTS_TEXT, " can", 1200),
        Frame(FrameType.TTS_TEXT, " I", 1250),
        Frame(FrameType.TTS_TEXT, " help", 1300),
        Frame(FrameType.TTS_TEXT, " you", 1350),
        Frame(FrameType.TTS_TEXT, " today?", 1400),
        Frame(FrameType.LLM_RESPONSE_END, timestamp_ms=1450),
        Frame(FrameType.BOT_STOPPED_SPEAKING, timestamp_ms=2000),
    ]


def create_interruption_scenario() -> List[Frame]:
    """User interrupts mid-sentence, LLM regenerates.

    This is the scenario that causes the garbling bug:
    1. Bot starts saying "Can you tell me a bit more about..."
    2. User interrupts
    3. LLM regenerates with slightly different wording
    4. Old and new tokens get mixed in buffer
    """
    return [
        # First response starts
        Frame(FrameType.BOT_STARTED_SPEAKING, timestamp_ms=17000),
        Frame(FrameType.LLM_RESPONSE_START, timestamp_ms=17000),
        Frame(FrameType.TTS_TEXT, "Can", 17050),
        Frame(FrameType.TTS_TEXT, " you", 17100),
        Frame(FrameType.TTS_TEXT, " tell", 17150),
        Frame(FrameType.TTS_TEXT, " me", 17200),
        Frame(FrameType.TTS_TEXT, " a", 17250),
        Frame(FrameType.TTS_TEXT, " bit", 17300),
        # User interrupts! (causes LLM to regenerate)
        Frame(FrameType.USER_STARTED_SPEAKING, timestamp_ms=17350),
        # New LLM response starts (but old buffer wasn't cleared in buggy version!)
        Frame(FrameType.LLM_RESPONSE_START, timestamp_ms=17400),
        Frame(FrameType.TTS_TEXT, "Can", 17450),  # New response starts
        Frame(FrameType.TTS_TEXT, " more", 17500),  # Old response continues in parallel
        Frame(FrameType.TTS_TEXT, " about", 17550),
        Frame(FrameType.TTS_TEXT, " your", 17600),
        Frame(FrameType.TTS_TEXT, " you", 17650),
        Frame(FrameType.TTS_TEXT, " relationship", 17700),
        Frame(FrameType.TTS_TEXT, " to", 17750),
        Frame(FrameType.TTS_TEXT, " tell", 17800),
        Frame(FrameType.TTS_TEXT, " John", 17850),
        Frame(FrameType.TTS_TEXT, " and", 17900),
        Frame(FrameType.TTS_TEXT, " me", 17950),
        Frame(FrameType.TTS_TEXT, " what", 18000),
        Frame(FrameType.TTS_TEXT, " you're", 18050),
        Frame(FrameType.TTS_TEXT, " looking", 18100),
        Frame(FrameType.TTS_TEXT, " a", 18150),
        Frame(FrameType.TTS_TEXT, " to", 18200),
        Frame(FrameType.TTS_TEXT, " bit", 18250),
        Frame(FrameType.TTS_TEXT, " do?", 18300),  # Ends with punctuation
        Frame(FrameType.LLM_RESPONSE_END, timestamp_ms=18350),
        Frame(FrameType.BOT_STOPPED_SPEAKING, timestamp_ms=19000),
    ]


def create_duplicate_stream_scenario() -> List[Frame]:
    """Duplicate TTS streams - the ACTUAL cause of garbling.

    Looking at the real garbled output:
    "Can you tell me a bit Can more about your you relationship to..."

    This is the SAME sentence being processed twice through the pipeline,
    with tokens interleaved. This happens when:
    1. LLM generates tokens
    2. Multiple pipeline stages forward TTSTextFrame
    3. Observer sees both streams as DOWNSTREAM

    The fix: track a unique response ID or use speaking turn boundaries
    that properly reset on BotStartedSpeaking.
    """
    # Simulating two interleaved streams of the same sentence
    sentence = "Can you tell me a bit more about your relationship to John?"
    words = sentence.split()

    frames = [
        Frame(FrameType.BOT_STARTED_SPEAKING, timestamp_ms=17000),
        Frame(FrameType.LLM_RESPONSE_START, timestamp_ms=17000),
    ]

    # Interleave two copies of the same sentence (simulating duplicate streams)
    ts = 17050
    for i, word in enumerate(words):
        # First stream token
        frames.append(Frame(FrameType.TTS_TEXT, word if i == 0 else f" {word}", ts))
        ts += 25

        # Second stream token (duplicate, slightly delayed)
        if i < len(words) - 1:  # Skip last to avoid double punctuation
            frames.append(Frame(FrameType.TTS_TEXT, words[i] if i == 0 else f" {words[i]}", ts))
            ts += 25

    frames.extend(
        [
            Frame(FrameType.LLM_RESPONSE_END, timestamp_ms=ts + 50),
            Frame(FrameType.BOT_STOPPED_SPEAKING, timestamp_ms=ts + 500),
        ]
    )

    return frames


def create_rapid_turns_scenario() -> List[Frame]:
    """Multiple quick back-and-forth turns."""
    return [
        # Turn 1
        Frame(FrameType.BOT_STARTED_SPEAKING, timestamp_ms=1000),
        Frame(FrameType.LLM_RESPONSE_START, timestamp_ms=1000),
        Frame(FrameType.TTS_TEXT, "Yes,", 1100),
        Frame(FrameType.TTS_TEXT, " I", 1150),
        Frame(FrameType.TTS_TEXT, " can", 1200),
        Frame(FrameType.TTS_TEXT, " help.", 1250),
        Frame(FrameType.LLM_RESPONSE_END, timestamp_ms=1300),
        Frame(FrameType.BOT_STOPPED_SPEAKING, timestamp_ms=1500),
        # Turn 2 (immediately after)
        Frame(FrameType.BOT_STARTED_SPEAKING, timestamp_ms=2000),
        Frame(FrameType.LLM_RESPONSE_START, timestamp_ms=2000),
        Frame(FrameType.TTS_TEXT, "What's", 2100),
        Frame(FrameType.TTS_TEXT, " your", 2150),
        Frame(FrameType.TTS_TEXT, " name?", 2200),
        Frame(FrameType.LLM_RESPONSE_END, timestamp_ms=2250),
        Frame(FrameType.BOT_STOPPED_SPEAKING, timestamp_ms=2500),
    ]


def run_test(name: str, frames: List[Frame]):
    """Run a test scenario through all buffer implementations."""
    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    print("=" * 60)

    # Test current buggy implementation
    buggy = CurrentBuggyBuffer()
    for frame in frames:
        buggy.process_frame(frame)

    print("\n[CURRENT BUGGY] Messages:")
    for msg in buggy.messages:
        print(f"  [{msg['timestamp_ms']}ms] {msg['content']}")

    # Test Approach A
    approach_a = ApproachA_SpeakingTurnBuffer()
    for frame in frames:
        approach_a.process_frame(frame)

    print("\n[APPROACH A: Speaking Turn] Messages:")
    for msg in approach_a.messages:
        print(f"  [{msg['timestamp_ms']}ms] {msg['content']}")

    # Test Approach A2 (with dedup)
    approach_a2 = ApproachA2_SpeakingTurnDedup()
    for frame in frames:
        approach_a2.process_frame(frame)

    print("\n[APPROACH A2: Speaking Turn + Dedup] Messages:")
    for msg in approach_a2.messages:
        print(f"  [{msg['timestamp_ms']}ms] {msg['content']}")

    # Test Approach B
    approach_b = ApproachB_LLMResponseBuffer()
    for frame in frames:
        approach_b.process_frame(frame)

    print("\n[APPROACH B: LLM Response] Messages:")
    for msg in approach_b.messages:
        print(f"  [{msg['timestamp_ms']}ms] {msg['content']}")

    # Test Approach C (Hybrid)
    approach_c = ApproachC_Hybrid()
    for frame in frames:
        approach_c.process_frame(frame)

    print("\n[APPROACH C: Hybrid] Messages:")
    for msg in approach_c.messages:
        print(f"  [{msg['timestamp_ms']}ms] {msg['content']}")

    return {
        "buggy": buggy.messages,
        "approach_a": approach_a.messages,
        "approach_a2": approach_a2.messages,
        "approach_b": approach_b.messages,
        "approach_c": approach_c.messages,
    }


def check_garbled(content: str) -> bool:
    """Check if content appears garbled (duplicated words, interleaving)."""
    words = content.split()
    # Check for immediate duplicates
    for i in range(len(words) - 1):
        if words[i].lower().strip(".,!?") == words[i + 1].lower().strip(".,!?"):
            return True
    # Check for repeated subsequences
    word_counts = {}
    for w in words:
        clean = w.lower().strip(".,!?")
        word_counts[clean] = word_counts.get(clean, 0) + 1
    # If any content word appears more than once, likely garbled
    for word, count in word_counts.items():
        if count > 1 and word not in ["a", "the", "to", "and", "i", "you", "me"]:
            return True
    return False


def main():
    print("Transcript Buffering Approaches Test")
    print("Testing fixes for garbled transcript issue")

    # Run all test scenarios
    results = {}

    results["normal"] = run_test("Normal Conversation", create_normal_scenario())
    results["interruption"] = run_test("Interruption (Bug Trigger)", create_interruption_scenario())
    results["duplicate"] = run_test(
        "Duplicate Streams (Real Bug)", create_duplicate_stream_scenario()
    )
    results["rapid"] = run_test("Rapid Turns", create_rapid_turns_scenario())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    approaches = [
        ("Buggy", "buggy"),
        ("A (Turn)", "approach_a"),
        ("A2 (Turn+Dedup)", "approach_a2"),
        ("B (LLM)", "approach_b"),
        ("C (Hybrid)", "approach_c"),
    ]

    print("\n1. Normal scenario - all should produce clean output:")
    for name, key in approaches:
        msgs = results["normal"][key]
        content = msgs[0]["content"] if msgs else ""
        garbled = check_garbled(content)
        print(
            f"   {name:18} {len(msgs)} msg | {'GARBLED' if garbled else 'Clean'}: {content[:40]}..."
        )

    print("\n2. Interruption scenario:")
    for name, key in approaches:
        msgs = results["interruption"][key]
        content = msgs[0]["content"] if msgs else ""
        garbled = check_garbled(content)
        print(
            f"   {name:18} {len(msgs)} msg | {'GARBLED' if garbled else 'Clean'}: {content[:40]}..."
        )

    print("\n3. Duplicate streams (the ACTUAL bug):")
    for name, key in approaches:
        msgs = results["duplicate"][key]
        content = msgs[0]["content"] if msgs else ""
        garbled = check_garbled(content)
        print(
            f"   {name:18} {len(msgs)} msg | {'GARBLED' if garbled else 'Clean'}: {content[:40]}..."
        )

    print("\n4. Rapid turns - should produce 2 messages:")
    for name, key in approaches:
        msgs = results["rapid"][key]
        print(f"   {name:18} {len(msgs)} messages")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Determine best approach
    scores = {"approach_a": 0, "approach_a2": 0, "approach_b": 0, "approach_c": 0}

    for scenario in ["normal", "interruption", "duplicate", "rapid"]:
        for approach in scores.keys():
            msgs = results[scenario][approach]
            if scenario == "rapid":
                if len(msgs) == 2:
                    scores[approach] += 1
            else:
                if msgs and not check_garbled(msgs[0]["content"]):
                    scores[approach] += 1

    print(f"\nScores (out of 4 scenarios):")
    print(f"  Approach A  (Speaking Turn):         {scores['approach_a']}/4")
    print(f"  Approach A2 (Turn + Deduplication):  {scores['approach_a2']}/4")
    print(f"  Approach B  (LLM Response):          {scores['approach_b']}/4")
    print(f"  Approach C  (Hybrid):                {scores['approach_c']}/4")

    best = max(scores.keys(), key=lambda k: scores[k])
    print(f"\n{'=' * 60}")
    print(f"RECOMMENDATION: {best}")
    print(f"{'=' * 60}")

    if best == "approach_c":
        print("""
Approach C (Hybrid) is recommended.

This approach combines:
1. LLM response reset - discards partial from interrupted responses
2. Speaking turn boundaries - clean message emission points
3. Token deduplication - handles duplicate TTS streams

Implementation in observer.py:
- On LLMFullResponseStartFrame: reset buffer (discard interrupted partial)
- On BotStartedSpeakingFrame: mark speaking, set timestamp
- On TTSTextFrame: append with deduplication (skip recently seen tokens)
- On BotStoppedSpeakingFrame: emit complete message, clear state
""")
    elif best == "approach_a2":
        print("""
Approach A2 (Speaking Turn + Deduplication) is recommended.

This approach:
1. Uses BotStartedSpeaking/BotStoppedSpeaking as message boundaries
2. Deduplicates tokens using a sliding window
3. Handles duplicate TTS streams correctly
4. Simple to implement in the observer

Implementation: Reset buffer on BotStartedSpeaking, deduplicate
incoming TTSTextFrame tokens, emit on BotStoppedSpeaking.
""")
    elif best == "approach_a":
        print("""
Approach A (Speaking Turn) works but doesn't handle duplicate streams.
Consider adding deduplication (A2) for robustness.
""")
    elif best == "approach_b":
        print("""
Approach B (LLM Response) works but requires LLMFullResponse frames.
""")
    else:
        print("""
No clear winner - may need a hybrid approach or different strategy.
""")


if __name__ == "__main__":
    main()
