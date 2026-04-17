import hashlib
import os
import torch
from proj_head_package import ProjectionHead
import sys


# LFS refactor E2E verification: compute + print sha256 of the LFS-tracked
# weights.bin on startup so the container's runtime output can be verified
# against the local sha. Prints BEFORE the input-file check so the signal
# surfaces even when no real input is provided for the test run.
_LFS_PROBE = "/app/weights.bin"
if os.path.exists(_LFS_PROBE):
    _h = hashlib.sha256()
    with open(_LFS_PROBE, "rb") as _f:
        for _chunk in iter(lambda: _f.read(1 << 20), b""):
            _h.update(_chunk)
    print(f"LFS_PROBE weights.bin size={os.path.getsize(_LFS_PROBE)} sha256={_h.hexdigest()}", flush=True)
else:
    print(f"LFS_PROBE weights.bin MISSING at {_LFS_PROBE}", flush=True)


#Get environment variables
job_name = os.getenv("JobName", "CL_Embedding_Refinement")  #Fallback if JobName is not provided
input_file = "inputs/raw_embeddings.pt"  #Input file path (TamarindBio standard)
output_dir = "out/"  #Output directory (TamarindBio standard)
output_file = os.path.join(output_dir, "output_vectors.pt")  #Output file path

#makes sure output dir exists
os.makedirs(output_dir, exist_ok=True)

 #rename model dict variables
def load_compatible_state_dict(model, model_path):
    # Load the saved state_dict
    saved_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    
    # Adjust the keys to remove the "projection_head." prefix
    adjusted_state_dict = {}
    for key, value in saved_state_dict.items():
        if key.startswith("projection_head."):
            new_key = key.replace("projection_head.", "")  # Remove the prefix
        else:
            new_key = key
        adjusted_state_dict[new_key] = value
    
    # Load the adjusted state_dict into the model
    model.load_state_dict(adjusted_state_dict, strict=True)
    return model

def load_model():
    """Load the trained model."""

    model_path = "proj_head_package/model_weights/best_multi_contrastive_model.pth"
    model = ProjectionHead(input_dim = 1280, output_dim = 256)
    model = load_compatible_state_dict(model, model_path)
    model.eval()
    return model

def process_input_and_run_model(model):
    """Load input dictionary, mean-pool embeddings, and run them through the model."""
    #Load the input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    input_dict = torch.load(input_file)["per_residue_representations"]

    if not isinstance(input_dict, dict):
        raise ValueError("Input file must contain a dictionary with sequence names as keys.")
    
    output_dict = {}
    for seq_name, embeddings in input_dict.items():
        if embeddings.dim() != 2 or embeddings.size(1) != 1280:
            raise ValueError(f"Embedding for key `{seq_name}` must have shape [N, 1280].")

        # Mean-pooling over the first dimension (N)
        mean_pooled = embeddings.mean(dim=0, keepdim=True)  # Shape: [1, 1280]
        
        with torch.no_grad():
            # Run the mean-pooled embedding through the model
            processed_embedding = model(mean_pooled).squeeze(0)  # Shape: [256]
        
        # Save the processed embedding to the output dictionary
        output_dict[seq_name] = processed_embedding
    
    return output_dict

def save_output(output_dict):
    """Save the output dictionary to the specified file."""
    torch.save(output_dict, output_file)
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    try:
        # Load the model
        model = load_model()
        # Process input and run the model
        output_dict = process_input_and_run_model(model)
        # Save the output
        save_output(output_dict)
    except Exception as e:
        print(f"Error during processing: {e}")
# refactor E2E test 1: non-lfs change 1776388816
