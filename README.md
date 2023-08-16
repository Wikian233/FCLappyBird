# FCLappyBird: Adaptive Agent Behavior Control Based on Feedforward Close-Loop Learning

![Game Benchmark Enviroment Interface](./Results/Game%20interfce.png)

#### (For futher details, please check the codes, which are well commented)

## DeepFCL with Convolutional layers
![DeepFCL](./Results/DeepFCL.png)
The working principle of DeepFCL is as follows: Firstly, the image signals are passed into the convolutional layers, where the parameters are in a preset state. As the depth of the convolutional layers increases, features from the image signals are extracted for the first time. Subsequently, the extracted features are fed as inputs into the dense layer formed by the FCL network, resulting in corresponding actions to interact with the environment. As a result of this interaction, error signals are transmitted back through the feedback loop into the network. As can be seen from the attached structural diagram, the error signal that reflects the system's state after the agent interacts with the environment, not only passes to the FCL dense layer as error input but also gets converted into the system's post-interaction state through the 'error history reference' mechanism, and then is relayed to the convolutional layers to serve as a reference for adjusting filter parameters. The mechanism behind the 'error history reference' is that when the error expects a reduction due to the output of the FCL dense layer, it takes the current system state as the target state for the convolutional layer, passing it for learning. This way, it can better extract features of the system state when the error gets optimized. 

In this particular case, the approach involves storing the system state in a sequence when the error signal is below a certain value. Whenever a new state is saved, the oldest saved state is popped out. As a result, we obtain a continuously updated experience sequence related to error optimization (the length of this sequence correlates with the output of our convolutional layer), which guides the parameter updates of the convolutional layer itself. 

If you're interested, please check out the DeepFCL branch at https://github.com/Wikian233/FCLappyBird.git for more information.

## Result of DeepFCL with Convolutional layers
![Feature1](./Results/Feature1.png)


Features extracted by the convolutional layers of DeepFCL driven by closed-loop error-based experience history sequence.

It can be seen that the convolutional layers can successfully extract the features of the bird and the pipes' end points.

And the features of the bird and the pipes' end points are the most important features for the bird to make decisions.

(Especially considering how do we define the error signal,it is highly related to the bird's height and the pipes' end points)

![Feature2](./Results/Feature2.png)
Another example of features extracted by the convolutional layers of DeepFCL driven by closed-loop error-based experience history sequence.

![DeepFCL Result in first 3000 learning steps ](./Results/DeepFCLresult.png)

DeepFCL Result in first 3000 learning steps

Due to the reseaon that the time for submission is limited, the DeepFCL is not fully trained, but it can still be seen that the DeepFCL can learn to play the game.

The further work is to slecet the appropriate hyperparameters and train the DeepFCL for a longer time. And also try to code it with GPU to speed up the training process.




