from Trainer import Trainer
from configuration import Configuration
from decorators import timer


@timer("Total time: ")
def run():
    trainer = Trainer( Configuration("datasets/TSP/1.txt") )
    trainer.train()

if __name__ == '__main__':
    run()
