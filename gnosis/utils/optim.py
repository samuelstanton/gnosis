def get_decay_fn(initial_val, final_val, start, stop):
    """
    Returns function handle to use in torch.optim.lr_scheduler.LambdaLR.
    The returned function supplies the multiplier to decay a value linearly.
    """
    assert stop > start

    def decay_fn(counter):
        if counter <= start:
            return 1
        if counter >= stop:
            return final_val / initial_val
        time_range = stop - start
        return 1 - (counter - start) * (1 - final_val / initial_val) / time_range

    assert decay_fn(start) * initial_val == initial_val
    assert decay_fn(stop) * initial_val == final_val
    return decay_fn
