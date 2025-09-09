import time
import torch
import json
import os
from typing import Dict, List, Optional
import psutil


class TrainingProfiler:
    """Utility class to profile training performance across different training modes."""
    
    def __init__(self):
        self.step_times = []
        self.memory_usage = []
        self.step_start_time = None
        self.total_start_time = None
        
    def start_timing(self):
        """Start timing the overall training."""
        self.total_start_time = time.time()
        
    def start_step(self):
        """Start timing a training step."""
        self.step_start_time = time.time()
        
        # Record memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            self.memory_usage.append(gpu_memory)
        else:
            self.memory_usage.append(psutil.virtual_memory().percent / 100 * psutil.virtual_memory().total / 1024**3)
            
    def end_step(self):
        """End timing a training step."""
        if self.step_start_time:
            step_time = time.time() - self.step_start_time
            self.step_times.append(step_time)
            self.step_start_time = None
            
    def get_stats(self) -> Dict:
        """Get profiling statistics."""
        if not self.step_times:
            return {}
            
        total_time = time.time() - self.total_start_time if self.total_start_time else 0
        
        return {
            'avg_step_time': sum(self.step_times) / len(self.step_times),
            'min_step_time': min(self.step_times),
            'max_step_time': max(self.step_times),
            'total_steps': len(self.step_times),
            'total_training_time': total_time,
            'steps_per_second': len(self.step_times) / sum(self.step_times) if self.step_times else 0,
            'avg_memory_usage_gb': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            'max_memory_usage_gb': max(self.memory_usage) if self.memory_usage else 0,
        }
        
    def save_stats(self, filepath: str, additional_info: Optional[Dict] = None):
        """Save profiling stats to JSON file."""
        stats = self.get_stats()
        if additional_info:
            stats.update(additional_info)
            
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
            
    def print_stats(self, training_mode: str = "Unknown"):
        """Print profiling statistics."""
        stats = self.get_stats()
        if not stats:
            print("No profiling data available")
            return
            
        print(f"\n{'='*50}")
        print(f"TRAINING PROFILING RESULTS - {training_mode.upper()}")
        print(f"{'='*50}")
        print(f"Total Training Time: {stats['total_training_time']:.2f}s")
        print(f"Total Steps: {stats['total_steps']}")
        print(f"Average Step Time: {stats['avg_step_time']:.4f}s")
        print(f"Steps per Second: {stats['steps_per_second']:.2f}")
        print(f"Average Memory Usage: {stats['avg_memory_usage_gb']:.2f} GB")
        print(f"Peak Memory Usage: {stats['max_memory_usage_gb']:.2f} GB")
        print(f"{'='*50}")


def count_parameters(model) -> Dict[str, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_efficiency_pct': (trainable_params / total_params) * 100 if total_params > 0 else 0
    }


def compare_training_efficiency(model_stats: List[Dict], save_path: Optional[str] = None):
    """Compare efficiency metrics across different training configurations."""
    
    print(f"\n{'='*80}")
    print("TRAINING EFFICIENCY COMPARISON")
    print(f"{'='*80}")
    
    # Print header
    print(f"{'Mode':<20} {'Total Params':<15} {'Trainable':<15} {'Efficiency%':<12} {'Step/s':<10} {'Memory GB':<12}")
    print("-" * 80)
    
    for stats in model_stats:
        mode = stats.get('training_mode', 'Unknown')
        total_p = stats.get('total_parameters', 0)
        train_p = stats.get('trainable_parameters', 0)
        efficiency = stats.get('parameter_efficiency_pct', 0)
        steps_per_sec = stats.get('steps_per_second', 0)
        memory = stats.get('avg_memory_usage_gb', 0)
        
        print(f"{mode:<20} {total_p:<15,} {train_p:<15,} {efficiency:<12.2f} {steps_per_sec:<10.2f} {memory:<12.2f}")
    
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(model_stats, f, indent=2)
        print(f"\nComparison data saved to: {save_path}")


def block_expansion(ckpt, split, original_layers):

    layer_cnt = 0
    selected_layers = []
    output = {}

    for i in range(original_layers):
        for k in ckpt:
            if ('layer.' + str(i) + '.') in k:
                output[k.replace(('layer.' + str(i) + '.'), ('layer.' + str(layer_cnt) + '.'))] = ckpt[k]
        layer_cnt += 1
        if (i+1) % split == 0:
            for k in ckpt:
                if ('layer.' + str(i) + '.') in k:
                    if 'attention.output' in k or str(i)+'.output' in k:
                        output[k.replace(('layer.' + str(i) + '.'), ('layer.' + str(layer_cnt) + '.'))] = torch.zeros_like(ckpt[k])
                        selected_layers.append(layer_cnt)
                    else:
                        output[k.replace(('layer.' + str(i) + '.'), ('layer.' + str(layer_cnt) + '.'))] = ckpt[k]
            layer_cnt += 1

    for k in ckpt:
        if not 'layer' in k:
            output[k] = ckpt[k]
        elif k == "vit.layernorm.weight" or k == "vit.layernorm.bias" or k == "dinov2.layernorm.bias" or k == "dinov2.layernorm.weight":
            output[k] = ckpt[k]
    
    selected_layers = list(set(selected_layers))

    return output, selected_layers
