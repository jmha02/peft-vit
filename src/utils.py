import time
import torch
import json
import os
from typing import Dict, List, Optional
import psutil
import subprocess
import sys
import argparse
import datetime
from collections import defaultdict, deque


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


def bool_flag(s):
    """Parse boolean arguments from the command line."""
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def get_rank():
    """Get rank of current process."""
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_sha():
    """Get current git commit SHA."""
    try:
        cwd = os.path.dirname(os.path.abspath(__file__))
        def _run(command):
            return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
        sha = 'N/A'
        diff = "clean"
        branch = 'N/A'
        try:
            sha = _run(['git', 'rev-parse', 'HEAD'])
            subprocess.check_output(['git', 'diff'], cwd=cwd)
            diff = _run(['git', 'diff-index', 'HEAD'])
            diff = "has uncommitted changes" if diff else "clean"
            branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        except Exception:
            pass
        message = f"sha: {sha}, status: {diff}, branch: {branch}"
        return message
    except Exception:
        return "N/A"


def init_distributed_mode(args):
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def multi_scale(samples, model):
    """Multi-scale feature extraction (placeholder implementation)."""
    # Simple implementation - just return the features from the model
    # In a real implementation, you would apply multiple scales and aggregate
    return model(samples).logits
