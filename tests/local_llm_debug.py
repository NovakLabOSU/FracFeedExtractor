"""Debug wrapper for local_llm.py - saves all intermediate outputs.

This patches the extraction functions to save debug output without modifying local_llm.py
"""

import sys
import json
from pathlib import Path

debug_dir = None
call_counter = {"metadata": 0, "stomach": 0}


def save_debug(filename, content, description=""):
    """Save debug content to file."""
    if debug_dir:
        filepath = debug_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            if description:
                f.write(f"# {description}\n")
                f.write(f"# {'=' * 60}\n\n")
            f.write(content)
        print(f"[DEBUG] Saved: {filepath}", file=sys.__stderr__)


# Monkey-patch the chat function to intercept prompts and responses
from ollama import chat as original_chat


def patched_chat(messages, model, format=None):
    """Intercept chat calls to save prompts and responses."""
    prompt = messages[0]['content'] if messages else ""

    # Determine which pass this is
    if "species_name" in str(format):
        pass_type = "metadata"
        call_counter["metadata"] += 1
        call_num = call_counter["metadata"]

        # Save prompt
        save_debug(f"02_metadata_prompt_{call_num}.txt", prompt, f"Metadata extraction prompt (call {call_num})")
    elif "num_empty_stomachs" in str(format):
        pass_type = "stomach"
        call_counter["stomach"] += 1
        call_num = call_counter["stomach"]

        # Save prompt
        save_debug(f"05_stomach_data_prompt_{call_num}.txt", prompt, f"Stomach data extraction prompt (call {call_num})")
    else:
        pass_type = "unknown"
        call_num = 0

    # Call original
    response = original_chat(messages, model, format)

    # Save response
    if pass_type == "metadata":
        save_debug(f"03_metadata_response_{call_num}.json", response.message.content, f"Metadata extraction response (call {call_num})")
    elif pass_type == "stomach":
        save_debug(f"06_stomach_data_response_{call_num}.json", response.message.content, f"Stomach data extraction response (call {call_num})")

    return response


# Apply patches
import ollama

ollama.chat = patched_chat


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Debug wrapper for local_llm.py")
    parser.add_argument("pdf", type=str, help="Path to PDF")
    parser.add_argument("--model", type=str, default="llama3.1:8b")
    parser.add_argument("--output-dir", type=str, default="data/results")
    parser.add_argument("--debug-dir", type=str, default="debug_output", help="Where to save debug files")

    args = parser.parse_args()

    # Setup debug directory
    global debug_dir
    pdf_name = Path(args.pdf).stem
    debug_dir = Path(args.debug_dir) / pdf_name
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] Debug output will be saved to: {debug_dir}", file=sys.__stderr__)

    # Add src directory to path
    # Script is in tests/local_llm_debug.py
    # Get project root (parent of tests/) then add src/
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / "src"

    if not src_path.exists():
        print(f"[ERROR] src directory not found at: {src_path}", file=sys.__stderr__)
        sys.exit(1)

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    print(f"[DEBUG] Added to path: {src_path}", file=sys.__stderr__)

    # Import the module correctly
    from llm import llm_client

    # Store original functions
    original_extract_metadata = llm_client.extract_metadata_with_search
    original_extract_stomach = llm_client.extract_stomach_data_with_search
    original_extract_counts = llm_client.extract_stomach_counts_from_text

    # Patch extract_metadata_with_search
    def patched_extract_metadata(text, model):
        # Save the context being analyzed
        context = text[:12000]
        save_debug("01_metadata_context.txt", context, "Context sent to LLM for metadata extraction (first 12000 chars)")

        # Call original
        result = original_extract_metadata(text, model)

        # Save result
        save_debug("03_metadata_result.json", json.dumps(result, indent=2), "Final metadata extraction result")

        return result

    # Patch extract_stomach_data_with_search
    def patched_extract_stomach(text, model):
        # Save full text first
        save_debug("00_full_text.txt", text, f"Full extracted text ({len(text)} characters)")

        # Call regex extraction
        regex_result = original_extract_counts(text)
        save_debug("04_regex_extraction.json", json.dumps(regex_result, indent=2), "Regex-based stomach count extraction (ground truth)")

        # Find stomach-related passages
        import re

        stomach_pattern = r'.{0,800}(?:stomachs?|empty|sample\s+size|n\s*=).{0,800}'
        matches = re.findall(stomach_pattern, text, re.IGNORECASE)
        context = '\n\n---\n\n'.join(matches[:15])

        if not context:
            context = text[:15000]

        save_debug("04_stomach_data_context.txt", context, "Context sent to LLM for stomach data extraction")

        # Call original (which will call our patched chat function)
        result = original_extract_stomach(text, model)

        # Save final result
        save_debug("06_stomach_data_result.json", json.dumps(result, indent=2), "Final stomach data extraction result")

        return result

    # Patch extract_stomach_counts_from_text to log what it finds
    def patched_extract_counts(text):
        result = original_extract_counts(text)

        # Save what regex patterns found
        import re

        debug_info = {"result": result, "patterns_found": []}

        # Check what each pattern matched
        empty_pattern = r'(\d+)\s+stomachs?\s+(?:were\s+)?(?:found\s+)?empty'
        empty_matches = re.findall(empty_pattern, text, re.IGNORECASE)
        if empty_matches:
            debug_info["patterns_found"].append({"pattern": "empty stomachs", "matches": empty_matches})

        total_pattern = r'(?:total\s+of\s+|^|\.\s+)(\d+)\s+stomachs'
        total_matches = re.findall(total_pattern, text, re.IGNORECASE | re.MULTILINE)
        if total_matches:
            debug_info["patterns_found"].append({"pattern": "total stomachs", "matches": total_matches})

        save_debug("04_regex_patterns_detail.json", json.dumps(debug_info, indent=2), "Detailed regex pattern matching results")

        return result

    # Apply all patches
    llm_client.extract_metadata_with_search = patched_extract_metadata
    llm_client.extract_stomach_data_with_search = patched_extract_stomach
    llm_client.extract_stomach_counts_from_text = patched_extract_counts

    # Set up args for local_llm
    sys.argv = ["local_llm.py", args.pdf, "--model", args.model, "--output-dir", args.output_dir]

    # Run it
    try:
        llm_client.main()
    except SystemExit:
        pass

    print(f"\n{'=' * 60}", file=sys.__stderr__)
    print(f"[DEBUG] All debug files saved to: {debug_dir}", file=sys.__stderr__)
    print(f"{'=' * 60}\n", file=sys.__stderr__)
    print("[DEBUG] Review these files to diagnose issues:", file=sys.__stderr__)
    print(f"  1. {debug_dir}/00_full_text.txt - Full extracted PDF text", file=sys.__stderr__)
    print(f"  2. {debug_dir}/01_metadata_context.txt - Context for metadata extraction", file=sys.__stderr__)
    print(f"  3. {debug_dir}/02_metadata_prompt_1.txt - Prompt sent to LLM", file=sys.__stderr__)
    print(f"  4. {debug_dir}/03_metadata_response_1.json - LLM response", file=sys.__stderr__)
    print(f"  5. {debug_dir}/04_regex_extraction.json - What regex found", file=sys.__stderr__)
    print(f"  6. {debug_dir}/04_stomach_data_context.txt - Context for stomach data", file=sys.__stderr__)
    print(f"  7. {debug_dir}/05_stomach_data_prompt_1.txt - Prompt sent to LLM", file=sys.__stderr__)
    print(f"  8. {debug_dir}/06_stomach_data_response_1.json - LLM response\n", file=sys.__stderr__)


if __name__ == "__main__":
    main()
