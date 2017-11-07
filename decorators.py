def instanciate_globals():
    global timing_dict
    timing_dict = {}

def infinity_handler():
    def decorator(func):
        def wrapper(*args, **kwargs):
            try: return func(*args, **kwargs)
            except OverflowError: return 0.0  # Threshold for when the number gets to low.
        return wrapper
    return decorator

# Takes the time of a function or method
def timer(key):
    def decorator(func):
        def wrapper(*args,**kwargs):
            from time import time
            start = time()
            func(*args, **kwargs)

            try: timing_dict[key] += [time() - start]
            except KeyError:
                timing_dict[key] = []
                timing_dict[key] += [time() - start]

        return wrapper
    return decorator

def print_time_averages():
    if len(timing_dict.items()) > 0:
        print("\nAverage runtime for function:")
        for key, value in timing_dict.items():
            if len(value) > 1:
                pretty_print_time("\t"+key+" (%d calls)" %len(value), sum(value)/len(value))

def print_time_totals():
    if len(timing_dict.items()) > 0:
        print("\nTotal runtime for function:")
        for key, value in timing_dict.items():
            pretty_print_time("\t"+key+" (%d calls)" %len(value), sum(value))

MAX_LENGTH_OF_TEXT = 50
def pretty_print_time(text, time, spaces = MAX_LENGTH_OF_TEXT):
    print(text + "%s%f seconds" % (" " * (spaces - len(text)), round(time, 4)))

def table_print_time_dict(d=None):
    def sh(text, length):
        return " " * ( length - len(text))
    if not d: d = timing_dict
    txt_leng = max(len(key) for key in d.keys())
    num_leng = max(len(str(value)) for value in d.values())
    decimals = 6
    headers = ["Average", "Total"]
    template = "| %s | %s | %s |"
    output = template % (" "*txt_leng, sh(headers[0], num_leng) + headers[0], sh(headers[1], num_leng) + headers[1])
    output += "\n"+template %("-"*txt_leng, "-"*num_leng, "-"*num_leng)
    for text, values in timing_dict.items():
        tot = str(sum(values))
        avg = str(sum(values)/len(values))
        output += "\n"+template % (
            text + sh(text, txt_leng),
            sh(avg, num_leng) + avg,
            sh(tot, num_leng) + tot
        )

    print(output)
