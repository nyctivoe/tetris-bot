import inspect

import model


def test_validation_iterator_resets_each_epoch():
    lines = inspect.getsource(model.main).splitlines()
    epoch_idx = next(i for i, line in enumerate(lines) if "for epoch in range" in line)
    val_idx = next(i for i, line in enumerate(lines) if "val_iter = iter(val_loader)" in line)

    epoch_indent = len(lines[epoch_idx]) - len(lines[epoch_idx].lstrip())
    val_indent = len(lines[val_idx]) - len(lines[val_idx].lstrip())

    assert val_idx > epoch_idx
    assert val_indent > epoch_indent
