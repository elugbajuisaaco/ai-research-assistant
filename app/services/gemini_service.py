import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


class GeminiLLMService:
    """Real Gemini-powered LLM service for the research assistant."""

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured."
            )

        self.client = genai.Client(
            api_key=api_key
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        """Generate an evidence-based answer from retrieved context."""

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        if not context.strip():
            return (
                f"I cannot find any source information to answer: "
                f"'{question}'"
            )

        system_instruction = """
You are an AI research assistant.

Your task is to answer the user's question using ONLY the
provided research context.

IMPORTANT EVIDENCE AND CITATION RULES:

1. Use ONLY information contained in the research context.
2. Do not use your own outside knowledge.
3. Do not invent facts, statistics, authors, dates, sources,
   or conclusions.
4. Every factual claim must be supported by one or more
   sources from the research context.
5. Each source is identified by a citation number such as [1],
   [2], or [3].
6. Use ONLY citation numbers that actually appear in the
   research context.
7. Never invent a citation number.
8. Place the citation immediately after the claim it supports.
9. If one claim is supported by multiple sources, cite all
   relevant sources, for example [1], [3].
10. Do not attach citations to claims that the cited source
    does not support.
11. If the context does not contain enough evidence to answer
    part or all of the question, clearly state that the
    available research evidence is insufficient.
12. Synthesize the evidence rather than copying the source text.
13. Do not reproduce large sections of the research context.
14. Answer the user's question directly.
15. Use a clear, concise, academic, and professional tone.
16. Preserve important distinctions between different sources.
17. Do not claim that two sources agree unless their content
    actually supports the same conclusion.

CITATION FORMAT:

Use citations exactly like this:

[1]

or:

[1], [2]

or:

[1], [3]

Do NOT use:

(1)
(2)
Source 1
Citation 1
[7] when only [1], [2], and [3] exist.

Before producing the final answer, check that:

- Every factual claim is supported by the research context.
- Every citation number exists in the supplied context.
- Each citation actually supports the claim it follows.
- No unsupported outside information has been added.
- The answer directly addresses the user's question.
"""

        prompt = f"""
USER QUESTION:
{question}

RESEARCH CONTEXT:
{context}

Now provide the final answer.
"""

        response = self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
        )

        if not response.text:
            return (
                "The AI model did not return an answer."
            )

        return response.text.strip()