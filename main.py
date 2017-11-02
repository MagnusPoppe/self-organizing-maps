from Trainer import Trainer
from configuration import Configuration
from decorators import timer


@timer("Total time: ")
def run():
    config = Configuration("datasets/TSP/1.txt")
    trainer = Trainer( config )
    try:
        trainer.train()
    except KeyboardInterrupt as e:
        print("User cancelled the program.")
    finally:
        if config.visuals:
            import matplotlib.pyplot as plt
            plt.close("all")

if __name__ == '__main__':
    run()
