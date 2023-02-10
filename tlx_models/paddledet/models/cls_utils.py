def _get_class_default_kwargs(cls, *args, **kwargs):
    """
    Get default arguments of a class in dict format, if args and
    kwargs is specified, it will replace default arguments
    """
    varnames = cls.__init__.__code__.co_varnames
    argcount = cls.__init__.__code__.co_argcount
    keys = varnames[:argcount]
    assert [
    keys[0] == 'self']
    keys = \
    keys[1:]
    values = list(cls.__init__.__defaults__)
    assert len(values) == len(keys)
    if len(args) > 0:
        for i, arg in enumerate(args):
            values[i] = arg
    default_kwargs = dict(zip(keys, values))
    if len(kwargs) > 0:
        for k, v in kwargs.items():
            default_kwargs[k] = v
    return default_kwargs
