# Dataloader的使用

## Dataloader需要配合dataset来使用

dataset按照文件中的《Datasets的使用》进行构建，dataloader即配合datasets来进行循环调用即可

## loader调用

```python
CLASStorch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=2, persistent_workers=False, pin_memory_device='')
```

参数说明:

- **shuffle** ([*bool*](https://docs.python.org/3/library/functions.html#bool)*,* *optional*) – set to `True` to have the data reshuffled**(打乱)** at every epoch (default: `False`).

- **num_workers** ([*int*](https://docs.python.org/3/library/functions.html#int)*,* *optional*) – how many subprocesses to use for data loading. `0` means that the data will be loaded in the main process. (default: `0`)

- 其他参数不常用不予以说明。具体说明：[torch.utils.data — PyTorch 1.12 documentation](https://pytorch.org/docs/stable/data.html?highlight=dataloader#torch.utils.data.DataLoader)

  

  调用

```
for batch_ndx, sample in enumerate(loader):
```

