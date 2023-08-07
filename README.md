# FCLappyBird: Adaptive Agent Behavior Control Based on Feedforward Close-Loop Learning

![Game Benchmark Enviroment Interface](./Results/Game%20interfce.png)

#### (For futher details, please check the codes, which are well commented)

## Prerequisites
Python Coding enviroment with the libraries: pygame and feedforward-closedloop-learning (see below)
```
pip install pygame

pip install feedforward-closedloop-learning

```

# How to run:
## 1. Run the game benchmark enviroment
```
python main.py
```
## 2. Game control

Ctrtl + P: pause the game

Ctrl + 1: sets the frame rate of the game to the default value.

Ctrl + 2: sets the frame rate of the game to half of the default value.

Ctrl + 3: sets the frame rate of the game to 10 times of the default value.

Ctrl + 4: creates a human player controlled bird (if you really want to complete with the agent bird).

Space: makes the bird jump for human player controlled bird.

# AI Agents which can be used to play the game
(For futher details, please check the codes, which are well commented)
THe enviroment can be used to test different AI agents. The following agents are available:
## 1. FCLNet driven by position errors
The error signal is defined as the error between the bird's height and the height of the front pipe's center
and the enviroment informations used to train the FCLNet are the errors of nearest front 4 pipes.   

if you want to use this agent, please uncomment
```
# from fcl import * 
```
in the game.py file and comment
```
from fcl2 import * 
```
and also ,change the following code in the game.py file
```
    def _agent_action(self):
        # in all birds, if a bird is not controlled by the human player,
        # then try to make the bird jump. This part is the operation part of the AI bird.
        for bird in self.birds:
            if bird is not self._human_bird:
                self.fcl2_flap(bird)# fcl_flap, fcl2_flap, fclappy_vision_flap
```
to
```
    def _agent_action(self):
        # in all birds, if a bird is not controlled by the human player,
        # then try to make the bird jump. This part is the operation part of the AI bird.
        for bird in self.birds:
            if bird is not self._human_bird:
                self.fcl_flap(bird)# fcl_flap, fcl2_flap, fclappy_vision_flap
```
and the learing rate should also be changed, which can be found in the config.py file


## 2. FCLNet Driven by abstract error
The error signal is defined as to maximize the score which bird can get. 
The enviroment informations used to train the FCLNet are the errors of nearest front 4 pipes.

## 3. FCLNet Driven by abstract error and vision enviroment input (under development)
Now facing some problems, maybe the vision input is too large for the FCLNet to learn. Convolutional layer may be needed to reduce the input size and extract the features of the vision input.

# Results
## 1. FCLNet driven by position errors
![FCLNet driven by position errors](./Results/FCL.png)

## 2. FCLNet Driven by abstract error
![FCLNet Driven by abstract error](./Results/FCL2.png)

## 3. FCLNet Driven by abstract error and vision enviroment input (under development)

## 4. Bird Trajectory comparison between FCLNet driven by position errors and FCLNet Driven by abstract error and simply error-based control 
![Bird Trajectory comparison between FCLNet driven by position errors and FCLNet Driven by abstract error and simply error-based control](./Results/Trajectory.png)


