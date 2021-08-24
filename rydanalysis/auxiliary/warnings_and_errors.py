def conditional_waning(message, ignore_warning=True):
    if ignore_warning:
        pass
    else:
        raise Warning(message)
