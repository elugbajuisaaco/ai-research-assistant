class LLMService:
    def generate(self, question: str, context: str) -> str:
        """Generates a text answer using the provided research context block."""
        if not question.strip():
            return "Please provide a valid question."
            
        if not context.strip():
            return f"I cannot find any source information to answer: '{question}'"

        # Predictable mock engine response format for robust pipeline testing
        return f"Based on the provided records, the answer to '{question}' is found within context: {context}"
