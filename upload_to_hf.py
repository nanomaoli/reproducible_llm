import argparse
from huggingface_hub import HfApi, login
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to the .zip file to upload")
    args = parser.parse_args()

    hf_token = "replace this text with the token, keep the quotation marks"
    
    api = HfApi()
    api.upload_file(
        path_or_fileobj=args.file,
        path_in_repo=f"followup_exp_outputs/{os.path.basename(args.file)}",
        repo_id="cane-sugar-soda/reproducible_llm_exp_results",
        repo_type="dataset", 
        token=hf_token
    )

if __name__ == "__main__":
    main()