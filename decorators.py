
length_of_text = 25

# Takes the time of a function or method
def timer(text):
    def decorator(func):
        def wrapper(*args,**kwargs):
            from time import time
            start = time()
            func(*args, **kwargs)
            print(text + " "*(length_of_text-len(text)) + str(round(time() - start, 4)) + " seconds")
        return wrapper
    return decorator
