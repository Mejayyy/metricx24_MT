"""
Random Baseline MSE Calculator

This script calculates the expected Mean Squared Error (MSE) when predicting random
integers in the range [0, 25] for translation quality scores. It runs 100 iterations
to generate a generalized baseline that can be used to evaluate the performance of
trained models.

The script reads scores from all_data.jsonl and saves statistical analysis to
baseline_mse_summary.txt.
"""

import json
import random
from datetime import datetime
import sys
import statistics


def load_scores(filepath):
    """
    Load scores from JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        Tuple of (scores list, total entries, valid entries)
    """
    print(f"Loading scores from {filepath}...")
    
    try:
        scores = []
        total_entries = 0
        invalid_count = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                total_entries += 1
                if total_entries % 2000 == 0:
                    print(f"  Processed {total_entries} entries...")
                
                try:
                    data = json.loads(line)
                    if 'score' not in data:
                        invalid_count += 1
                        continue
                    
                    score = float(data['score'])
                    scores.append(score)
                except (json.JSONDecodeError, ValueError, TypeError):
                    invalid_count += 1
                    continue
        
        valid_entries = len(scores)
        
        if invalid_count > 0:
            print(f"Warning: {invalid_count} entries with invalid scores were skipped")
        
        return scores, total_entries, valid_entries
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading file: {str(e)}")


def calculate_mse(predictions, actual):
    """
    Calculate Mean Squared Error.
    
    Args:
        predictions: List of predicted values
        actual: List of actual values
        
    Returns:
        MSE value
    """
    if len(predictions) != len(actual):
        raise ValueError("Predictions and actual values must have same length")
    
    squared_errors = [(p - a) ** 2 for p, a in zip(predictions, actual)]
    return sum(squared_errors) / len(squared_errors)


def run_baseline_iterations(scores, num_iterations=100):
    """
    Run baseline MSE calculations with random predictions.
    
    Args:
        scores: List of actual scores
        num_iterations: Number of iterations to run (default 100)
        
    Returns:
        List of MSE values from all iterations
    """
    mse_values = []
    num_samples = len(scores)
    
    print(f"\nRunning {num_iterations} iterations of random baseline...")
    
    for i in range(num_iterations):
        # Generate random integers in range [0, 25]
        random_predictions = [random.randint(0, 25) for _ in range(num_samples)]
        
        # Calculate MSE
        mse = calculate_mse(random_predictions, scores)
        mse_values.append(mse)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_iterations} iterations")
    
    return mse_values


def calculate_percentile(data, percentile):
    """
    Calculate percentile value.
    
    Args:
        data: List of values
        percentile: Percentile to calculate (0-100)
        
    Returns:
        Percentile value
    """
    sorted_data = sorted(data)
    index = (percentile / 100) * (len(sorted_data) - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(sorted_data) - 1)
    
    if lower_index == upper_index:
        return sorted_data[lower_index]
    
    lower_value = sorted_data[lower_index]
    upper_value = sorted_data[upper_index]
    fraction = index - lower_index
    
    return lower_value + fraction * (upper_value - lower_value)


def calculate_statistics(scores, mse_values):
    """
    Calculate comprehensive statistics.
    
    Args:
        scores: List of actual scores
        mse_values: List of MSE values from iterations
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'dataset_size': len(scores),
        'score_min': min(scores),
        'score_max': max(scores),
        'score_mean': statistics.mean(scores),
        'score_std': statistics.stdev(scores) if len(scores) > 1 else 0,
        'mse_mean': statistics.mean(mse_values),
        'mse_std': statistics.stdev(mse_values) if len(mse_values) > 1 else 0,
        'mse_min': min(mse_values),
        'mse_max': max(mse_values),
        'mse_median': statistics.median(mse_values),
        'mse_p25': calculate_percentile(mse_values, 25),
        'mse_p75': calculate_percentile(mse_values, 75),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    return stats


def save_summary(stats, output_filepath):
    """
    Save statistical summary to text file.
    
    Args:
        stats: Dictionary of statistics
        output_filepath: Path to save the summary file
    """
    print(f"\nSaving summary to {output_filepath}...")
    
    with open(output_filepath, 'w') as f:
        f.write("Random Baseline MSE Analysis\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset: all_data.jsonl\n")
        f.write(f"Total Entries: {stats['dataset_size']}\n")
        f.write(f"Date: {stats['timestamp']}\n")
        f.write("\n")
        
        f.write("Baseline MSE Statistics (100 iterations):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean MSE:           {stats['mse_mean']:.6f}\n")
        f.write(f"Std Deviation:      {stats['mse_std']:.6f}\n")
        f.write(f"Min MSE:            {stats['mse_min']:.6f}\n")
        f.write(f"Max MSE:            {stats['mse_max']:.6f}\n")
        f.write(f"Median MSE:         {stats['mse_median']:.6f}\n")
        f.write(f"25th Percentile:    {stats['mse_p25']:.6f}\n")
        f.write(f"75th Percentile:    {stats['mse_p75']:.6f}\n")
        f.write("\n")
        
        f.write("Dataset Score Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Score Range:        [{stats['score_min']:.2f}, {stats['score_max']:.2f}]\n")
        f.write(f"Mean Score:         {stats['score_mean']:.6f}\n")
        f.write(f"Std Deviation:      {stats['score_std']:.6f}\n")
        f.write("\n")
        
        f.write("Interpretation:\n")
        f.write("-" * 80 + "\n")
        f.write("This baseline represents the expected MSE when predicting random\n")
        f.write("integers in the range [0, 25] for each translation quality score.\n")
        f.write("Any trained model should achieve MSE significantly lower than this\n")
        f.write("baseline to demonstrate meaningful learning from the data.\n")
        f.write("\n")
        f.write(f"Target MSE: < {stats['mse_mean'] * 0.5:.6f} (50% of baseline)\n")
        f.write(f"Good MSE:   < {stats['mse_mean'] * 0.25:.6f} (25% of baseline)\n")
        f.write(f"Excellent:  < {stats['mse_mean'] * 0.1:.6f}  (10% of baseline)\n")


def main():
    """Main execution function."""
    try:
        # File paths
        input_file = "all_data.jsonl"
        output_file = "baseline_mse_summary.txt"
        
        # Load scores
        scores, total_entries, valid_entries = load_scores(input_file)
        
        if len(scores) == 0:
            raise ValueError("No valid scores found in the dataset")
        
        print(f"Successfully loaded {valid_entries} valid scores from {total_entries} total entries")
        
        # Run baseline iterations
        mse_values = run_baseline_iterations(scores, num_iterations=100)
        
        # Calculate statistics
        stats = calculate_statistics(scores, mse_values)
        
        # Save summary
        save_summary(stats, output_file)
        
        print("\nBaseline MSE Calculation Complete!")
        print(f"Mean Baseline MSE: {stats['mse_mean']:.6f} ± {stats['mse_std']:.6f}")
        print(f"Summary saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
