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
def timer(text):
    def decorator(func):
        def wrapper(*args,**kwargs):
            from time import time
            start = time()
            func(*args, **kwargs)
            pretty_print_time(text, time()-start)
        return wrapper
    return decorator

def average_runtime(key):
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
            pretty_print_time("\t"+key+" (%d calls)" %len(value), sum(value)/len(value))

length_of_text = 25
def pretty_print_time(text, time):
    print(text + " " * (length_of_text - len(text)) + str(round(time, 4)) + " seconds")
