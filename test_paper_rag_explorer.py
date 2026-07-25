import unittest
import os
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# Import functions and constants from your RAG script
from paper_rag_explorer import (
    build_or_load_index,
    generate_answer,
    MINISTRAL_PATH,
    EMBED_MODEL_NAME
)


class TestRAGExplorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the models and index once for all tests to save time."""
        # Define where the test will look for PDFs.
        cls.test_pdf_dir = "docs"

        if not os.path.exists(cls.test_pdf_dir):
            raise FileNotFoundError(
                f"Test directory '{cls.test_pdf_dir}' not found. Please create it and add the target PDF.")

        print("\n--- Setting up Test Environment (Loading Models & Index) ---")

        # 1. Build or load the index
        cls.index_data = build_or_load_index(cls.test_pdf_dir)

        # 2. Load embed model
        cls.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        # 3. Load LLM (Using default Ministral, switch to ROCKET_PATH if needed)
        cls.llm = Llama(
            model_path=MINISTRAL_PATH,
            n_ctx=4096,
            n_threads=os.cpu_count() or 4,
            verbose=False  # Suppress llama.cpp C-level logging during tests if desired
        )
        print("--- Setup Complete ---\n")

    def test_algorithm_abbreviation_extraction(self):
        """Tests if the RAG can successfully extract the specific algorithm abbreviation."""
        query = (
            "Can you tell me the abbreviation of the order based algorithm "
            "introduced in paper \"An Order-based Algorithm for Minimum Dominating "
            "Set with Application in Graph Mining\"?"
        )

        # Run the RAG pipeline
        answer = generate_answer(query, self.index_data, self.embed_model, self.llm)

        print("Answer:", answer)

        # Assert that the required substring is present in the final output
        self.assertIn(
            "RLSo",
            answer,
            f"Expected substring 'RLSo' was not found in the generated answer.\nActual Answer:\n{answer}"
        )


if __name__ == '__main__':
    unittest.main()
