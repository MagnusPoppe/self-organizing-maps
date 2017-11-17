from features.decorators import print_time_totals, print_time_averages
from kohonen_network.Trainer import Trainer
from kohonen_network.configuration import Configuration
from shell_commands.input_commands import *


def listen( controller=None ):
    # Configuration:
    run = True
    config  = controller.config if controller else None
    trainer = controller        if controller else None

    # Listeining:
    while run:
        x = input(">>> ").strip()

        # System:
        if x in ["q", "quit", "exit"]:
            run = False
        elif x in ["new network"]:
            config = Configuration("configurations/" + input_json_file())
            trainer = Trainer(config)
        elif trainer:
            # Visuals:
            if x == "train more":
                trainer.train_more(input_number("epochs="))

            elif x in ["run tests"]:
                trainer.run_all_tests()

            elif x in ["stats"]:
                print_time_totals()
                print_time_averages()

            elif x in ["close windows", "close"]:
                if config.visuals:
                    import matplotlib.pyplot as plt
                    plt.close("all")
            else: print("Unknown command.")
        else:
            print("Unknown command.")

