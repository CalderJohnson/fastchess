"""Concatenate all dataset chunks into a single file."""
import os
import torch
from tqdm import tqdm
import hyperparams as hp


def concatenate_chunks():
    """Load all chunks and save as single file."""
    chunk_dir = hp.PT_DATASET_PATH + "_chunks"
    output_path = hp.PT_DATASET_PATH + "all_data.pt"
    
    if not os.path.exists(chunk_dir):
        print(f"Chunk directory not found: {chunk_dir}")
        return
    
    # Get all chunk files
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.pt')])
    
    if not chunk_files:
        print("No chunk files found!")
        return
    
    print(f"Found {len(chunk_files)} chunks")
    print(f"Loading chunks from: {chunk_dir}")
    
    # Load and concatenate all chunks
    all_positions = []
    total_size_mb = 0
    
    for chunk_file in tqdm(chunk_files[:25], desc="Loading chunks"):
        chunk_path = os.path.join(chunk_dir, chunk_file)
        chunk_data = torch.load(chunk_path)
        all_positions.extend(chunk_data)
        
        # Track file size
        chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
        total_size_mb += chunk_size_mb
    
    print(f"\nTotal positions: {len(all_positions):,}")
    print(f"Total chunk size on disk: {total_size_mb:.2f} MB")
    
    # Estimate memory usage
    if all_positions:
        # Sample estimate based on first position
        import sys
        sample_pos = all_positions[0]
        sample_size = (
            sys.getsizeof(sample_pos['fen']) +
            sys.getsizeof(sample_pos['move_idx']) +
            sys.getsizeof(sample_pos['value'])
        )
        estimated_mb = (sample_size * len(all_positions)) / (1024 * 1024)
        print(f"Estimated in-memory size: {estimated_mb:.2f} MB")
    
    # Save concatenated file
    print(f"\nSaving to: {output_path}")
    torch.save(all_positions, output_path)
    
    # Check actual saved file size
    saved_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved file size: {saved_size_mb:.2f} MB")
    
    print(f"\n✓ Successfully concatenated {len(chunk_files)} chunks into single file")
    print(f"  Positions: {len(all_positions):,}")
    print(f"  File size: {saved_size_mb:.2f} MB")


if __name__ == "__main__":
    concatenate_chunks()