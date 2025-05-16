import sys
import os
import torch
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict


def is_deepspeed_checkpoint(checkpoint_dir):
    """
    Check if the given directory/file is a DeepSpeed checkpoint
    Args:
        checkpoint_dir (str): Path to checkpoint directory/file
    Returns:
        bool: True if it's a DeepSpeed checkpoint, False otherwise
    """
    # Check if it's a directory and contains DeepSpeed-specific files
    if os.path.isdir(checkpoint_dir):
        return True
    return False


def convert_deepspeed_to_pytorch(deepspeed_checkpoint_dir, output_path):
    """
    Convert a DeepSpeed checkpoint to a single PyTorch .pt file
    Args:
        deepspeed_checkpoint_dir (str): Directory containing DeepSpeed checkpoint files
        output_path (str): Desired output path for the .pt file
    """
    # Create temporary directory for intermediate files
    temp_dir = output_path + "_temp"
    os.makedirs(temp_dir, exist_ok=True)
    try:
        # First convert to intermediate format
        print(
            f"Converting DeepSpeed checkpoint from directory: {deepspeed_checkpoint_dir}"
        )
        state_dict = convert_zero_checkpoint_to_fp32_state_dict(
            deepspeed_checkpoint_dir, temp_dir
        )
        # Load all parts if multiple files were created
        if os.path.isfile(os.path.join(temp_dir, "pytorch_model.bin")):
            # Single file case
            state_dict = torch.load(os.path.join(temp_dir, "pytorch_model.bin"))
        else:
            # Multiple file case
            state_dict = {}
            for filename in sorted(os.listdir(temp_dir)):
                if filename.startswith("pytorch_model-") and filename.endswith(".bin"):
                    part_dict = torch.load(os.path.join(temp_dir, filename))
                    state_dict.update(part_dict)
        # Save as single .pt file
        print(f"Saving converted checkpoint to: {output_path}")
        torch.save(state_dict, output_path)
        print("Conversion complete!")
    finally:
        # Cleanup temporary directory
        import shutil

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def convert_checkpoint(input_path, output_path):
    """
    Convert either a DeepSpeed checkpoint or regular PyTorch checkpoint to a .pt file
    Args:
        input_path (str): Path to input checkpoint (directory for DeepSpeed, file for regular)
        output_path (str): Desired output path for the .pt file
    """
    if is_deepspeed_checkpoint(input_path):
        print("Detected DeepSpeed checkpoint. Converting...")
        convert_deepspeed_to_pytorch(input_path, output_path)
    else:
        print("Detected regular PyTorch checkpoint. Converting...")
        try:
            # Load the checkpoint
            checkpoint = torch.load(input_path)

            # If it's a state dict, save directly
            if isinstance(checkpoint, dict) and any(
                isinstance(v, torch.Tensor) for v in checkpoint.values()
            ):
                state_dict = checkpoint
            # If it's a module or model checkpoint, extract state dict
            elif hasattr(checkpoint, "state_dict"):
                state_dict = checkpoint.state_dict()
            # If it has a 'model_state_dict' key (common format)
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                print(checkpoint.keys())
                raise ValueError("Unable to extract state dict from checkpoint")

            # Save as .pt file
            print(f"Saving converted checkpoint to: {output_path}")
            torch.save(state_dict, output_path)
            print("Conversion complete!")
        except Exception as e:
            print(f"Error converting checkpoint: {str(e)}")
            sys.exit(1)


if __name__ == "__main__":
    # Ensure two arguments are passed (the checkpoint directory/file and output filename)
    if len(sys.argv) != 3:
        print(
            "Usage: python deepspeed_to_pytorch.py <checkpoint_path> <output_filename>"
        )
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Error: The checkpoint path '{input_path}' does not exist.")
        sys.exit(1)

    # Run the conversion function
    convert_checkpoint(input_path, output_path)
