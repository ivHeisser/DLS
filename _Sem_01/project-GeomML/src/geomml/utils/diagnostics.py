from collections import Counter, defaultdict
import torch

def inspect_dataset(dataset):
    key_counter = Counter()
    task_stats = defaultdict(list)

    for data in dataset:
        # какие поля существуют
        key_counter[tuple(sorted(data.keys()))] += 1

        # статистика target
        if hasattr(data, "task_id") and hasattr(data, "y"):
            task_stats[int(data.task_id)].append(data.y.view(-1))

    print("=== Different schemas ===")
    for k, v in key_counter.items():
        print(v, k)

    print("\n=== Target statistics ===")
    for task, values in task_stats.items():
        x = torch.cat(values)
        print(
            f"task {task}:",
            f"mean={x.mean():.4f}",
            f"std={x.std():.4f}",
            f"min={x.min():.4f}",
            f"max={x.max():.4f}",
        )




def inspect_loader(loader):
    stats = defaultdict(list)

    for batch in loader:
        for key in batch.keys():
            value = getattr(batch, key)
            if torch.is_tensor(value) and value.dtype.is_floating_point:
                stats[key].append(value.cpu())

    print("\n=== Loader statistics ===")
    for key, values in stats.items():
        x = torch.cat([v.reshape(-1) for v in values])
        print(
            key,
            f"mean={x.mean():.4f}",
            f"std={x.std():.4f}",
            f"min={x.min():.4f}",
            f"max={x.max():.4f}",
        )