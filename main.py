from game import *

import random

# plt.ion()
def main():
    random.seed(RANDOM_SEED)
    game = Game()
    while game.running:
        game.reset()
        game.run()

if __name__ == '__main__':
    main()
