import unittest
import os
import sys
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from paper_rag_explorer import (
    build_or_load_index,
    generate_answer,
    MINISTRAL_PATH,
    ROCKET_PATH,
    EMBED_MODEL_NAME
)

# Check the environment variable set in PyCharm
FAST_MODE = os.getenv("FAST_MODEL") == "1"

class TestRAGExplorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_pdf_dir = "docs"

        if not os.path.exists(cls.test_pdf_dir):
            raise FileNotFoundError(
                f"Test directory '{cls.test_pdf_dir}' not found. Please create it and add the target PDF."
            )

        print("\n--- Setting up Test Environment (Loading Models & Index) ---")

        cls.index_data = build_or_load_index(cls.test_pdf_dir)
        cls.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        # Select model based on FAST_MODE flag
        model_path = ROCKET_PATH if FAST_MODE else MINISTRAL_PATH
        cls.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=os.cpu_count() or 4,
            verbose=False
        )
        print("--- Setup Complete ---\n")


    def test_algorithm_abbreviation_extraction_detailed(self):
        query = (
            "Task: Extract the exact algorithm abbreviation from the text below.\n"
            "Constraint: Output ONLY the abbreviation and nothing else.\n\n"
            "Paper: An Order-based Algorithm for Minimum Dominating Set with Application in Graph Mining\n"
            "Question: What is the abbreviation of the order-based algorithm?"
        )
        answer = generate_answer(query, self.index_data, self.embed_model, self.llm)
        print("Answer:", answer)
        self.assertIn(
            "RLSo",
            answer,
            f"Expected substring 'RLSo' was not found in the generated answer.\nActual Answer:\n{answer}"
        )

    def test_algorithm_abbreviation_extraction(self):
        query = (
            "Can you tell me the abbreviation of the order based algorithm "
            "introduced in paper \"An Order-based Algorithm for Minimum Dominating "
            "Set with Application in Graph Mining\"?"
        )
        answer = generate_answer(query, self.index_data, self.embed_model, self.llm)
        print("Answer:", answer)
        self.assertIn(
            "RLSo",
            answer,
            f"Expected substring 'RLSo' was not found in the generated answer.\nActual Answer:\n{answer}"
        )

if __name__ == '__main__':
    unittest.main()
