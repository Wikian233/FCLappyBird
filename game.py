'''
the following code is the main program
'''

import random
import pygame as pg
from gameobjects import *

import os
import os.path
import pandas as pd
# from fcl import * 
from fcl2 import *

# from FCLappyVision import *

import numpy as np

import torch
import torch.nn as nn


class Game:
    def __init__(self):
        os.environ['SDL_VIDEO_WINDOW_POS'] = '0, 0'
        # Setting Pygame window position. 
        # '0, 0' represents the coordinate position of the upper left corner of the window relative to the upper left corner of the screen, 
        # i.e., x=0, y=0.
        pg.init()
        #initialize pygame modules    
        self._screen = pg.display.set_mode((MAP_WIDTH, MAP_HEIGHT))
        # Create a window with a size of MAP_WIDTH * MAP_HEIGHT.
        pg.display.set_caption(TITLE)
        # set the title of the window to TITLE

        self.all_sprites = pg.sprite.LayeredUpdates()
        # create a sprite group to store all sprites, this sprite group is a LayeredUpdates object, 
        # which is a subclass of pygame.sprite.Sprite class and can manage sprites in layers. 
        # The principle of layer management is that each sprite has a _layer attribute, 
        # the larger the value of this attribute, the closer the sprite is to the front of the screen, and vice versa.
        self.birds = pg.sprite.Group()
        self.pipes = pg.sprite.Group()
        # create two sprite groups to store bird sprites and pipe sprites respectively

        self._bird_image = None
        self._pipe_images = None
        self._background_image = None
        # to store the images of the bird, the pipes and the background
        self._load_images()
        # load images

        self.running = True
        self.playing = False
        # to control the game state, if self.running is False, the game main loop will stop.
        self._is_paused = False
        # a flag to indicate whether the game is paused
        self._clock = pg.time.Clock()
        # create a clock object to control the frame rate of the game
        self._fps = FPS
        # the frame rate of the game

        # self._frame = self._get_frame()

        self._human_bird = None
        # to store the bird controlled by the human player
        self._agent_bird_1 = None
        # to store the bird controlled by the AI player
        self._very_front_pipe = None  
        # the closest pipe to the bird

        self.FCLNet = FCLNet(LEARNING_RATE)
        # initialize the FCL network to control the bird agent

        self._max_score = 0
        self._max_score_so_far = 0 
        self._iteration = 0
        # max score so far


    def reset(self):
        self._max_score = 0
        for s in self.all_sprites:
            s.kill()
        # kill all sprites in the all_sprites group

        # self._add_human_player()

        # uncomment the following line to use the fcl2 network
        if self._max_score_so_far > 2500:
            self.FCLNet.loadBestModel()
        # self.FCLNet.loadBestModel()
        self._create_agent_player(brain = 1)

        # print(self._frame.shape)

        self._generate_pipe(80) 
        while self._very_front_pipe.rect.x < MAP_WIDTH:
            self._generate_pipe()
        # This is used to create the first pipe. The parameter is the initial x-coordinate of the pipe, 
        # and the default value of this parameter is 80, indicating that the initial x-coordinate of the pipe is 80.
        # Then, the function will keep creating pipes until they extend beyond the right side of the screen.
        # The purpose of doing this is to provide the bird with enough space to jump when the game starts, so it won't immediately hit a pipe.
        
        # Create a background object
        Background(self, self._background_image)
        self._iteration += 1

    def _load_images(self):
        def _load_one_image(file_name):
            return pg.image.load(os.path.join(IMG_DIR, file_name)).convert_alpha()
        self._bird_image = _load_one_image('bird.png')
        self._pipe_images = [_load_one_image(name) for name in ['pipetop.png', 'pipebottom.png']]
        self._background_image = _load_one_image('background.png')
        self._blue_bird_image = _load_one_image('bluebird.png')

    def _generate_pipe(self, front_x = None):
        '''
        In the Flappy Bird game, the player needs to control the bird to pass through pairs of pipes that appear continuously. 
        This function is used to generate new pairs of pipes in front.
        Each pair of pipes consists of a top pipe and a bottom pipe, with a gap in between, and the bird needs to pass through this gap.
        This code first calculates the x coordinates of the new pair of pipes (that is, their position on the screen), 
        then randomly generates the size of the gap between the pipes and the length of the top pipe, 
        and then creates the new pair of pipes based on these parameters.
        '''
        if front_x is None:
            front_x = self._very_front_pipe.rect.x 
            # if front_x is None, use the x coordinate of the very front pipe

        pipe_space = random.randint(MIN_PIPE_SPACE, MAX_PIPE_SPACE)
        # is the distance between the new pair of pipes and the current front pipe.
        # the distance is randomly selected between MIN_PIPE_SPACE and MAX_PIPE_SPACE.
        new_pair_x = front_x + pipe_space
        # the x coordinate of the new pair of pipes is the x coordinate of the current front pipe plus the distance between the two pipes.
        d_gap = MAX_PIPE_GAP - MIN_PIPE_GAP
        d_space = MAX_PIPE_SPACE - MIN_PIPE_SPACE

        # Calculate the parameters for the gap size of the pipe pair and the length of the top pipe
        if pipe_space > (MIN_PIPE_SPACE + MAX_PIPE_SPACE) / 2:
            gap = random.randint(MIN_PIPE_GAP, MAX_PIPE_GAP)
        else:
            gap = random.randint(int(MAX_PIPE_GAP - d_gap * (pipe_space - MIN_PIPE_SPACE) / d_space),
                                 MAX_PIPE_GAP) + 8      
        
        if pipe_space - MIN_PIPE_GAP < d_space // 3:
            top_length = self._very_front_pipe.length + random.randint(-50, 50)
        else:
            top_length = random.randint(MIN_PIPE_LENGTH, MAP_HEIGHT - gap - MIN_PIPE_LENGTH)

        if self._very_front_pipe is not None:
            gap += abs(top_length - self._very_front_pipe.length) // 10
        
        bottom_length = MAP_HEIGHT - gap - top_length

        # GENERATE NEW PIPE PAIR
        top_pipe = Pipe(self, self._pipe_images[0], new_pair_x, top_length, PipeType.TOP)
        bottom_pipe = Pipe(self, self._pipe_images[1], new_pair_x, bottom_length, PipeType.BOTTOM)
        # UPDATE VERY FRONT PIPE
        self._very_front_pipe = top_pipe
        top_pipe.gap = gap
        bottom_pipe.gap = gap

    def run(self):
        self.playing = True
        while self.playing:
            self._agent_action()
            self._handle_events()
            self._update()
            self._draw()
            self._clock.tick(self._fps)
            # self._save_FCL_parameters()
            # self._save_Bird_Trajectory()
        if not self.running:
            return

    def _pause(self):
        """
        Pause the game (ctrl + p to continue)
        """
        while True:
            for event in pg.event.get():
                # If the event type is pg.QUIT, it sets the two flags, self.playing and self.running, to False,
                # and then exits the function using return. This is done to respond to the user closing the game.
                if event.type == pg.QUIT:
                    self.playing = False
                    self.running = False
                    return
                # if the event type is pg.KEYDOWN, it checks whether a key is pressed.
                # In particular, if the Ctrl key and the P key are pressed (indicating the key combination Ctrl + P),
                # it sets self._is_paused to False and exits the function, which means that the game will continue to run.
                if event.type == pg.KEYDOWN:
                    pressed = pg.key.get_pressed()
                    ctrl_held = pressed[pg.K_LCTRL] or pressed[pg.K_RCTRL]
                    if ctrl_held and event.key == pg.K_p:
                        self._is_paused = False
                        return
                # if the game is paused, it will display a "Paused" on the screen.
                # Then update the display. At the same time, it will pause the program for 50 milliseconds,
                # which is done to reduce the CPU usage, because in the paused state, we don't need to run the game at full speed.
                self._draw_text("Paused", x=MAP_WIDTH // 2 - 50, y=MAP_HEIGHT // 2 - 10,
                                color=WHITE, size=2 * FONT_SIZE)
                pg.display.update()
                pg.time.wait(50)
    
    def _add_human_player(self):
        # find the x coordinate to place the human player's bird
        # It does this by iterating through all the pipes and finding the right boundary position of all the pipes on the left half of the screen, 
        # which is saved in the x_left_pipes list.
        x_left_pipes = [p.rect.right for p in self.pipes if p.rect.right < MAP_WIDTH // 2]

        if len(x_left_pipes) > 0:
            x = max(x_left_pipes) + 20
        # then, if the list is not empty, 
        # it will choose the right boundary of the rightmost pipe and place the bird 20 pixels to the right of it as the initial x coordinate of the bird.
        # This is to ensure that the bird is on the right side of the pipe and does not hit the pipe immediately.
        else:
            x = MAP_WIDTH // 2 - 100
        # if there is no pipe on the left half of the screen, 
        # it will set the initial x coordinate of the bird to the position 100 pixels to the left of the middle of the screen.
        y = MAP_HEIGHT // 2
        # For the initial y coordinate of the bird, the function chooses the midpoint of the screen.
        self._human_bird = Bird(self, self._blue_bird_image, x, y)

    def _create_agent_player(self, brain):

        x = MAP_WIDTH // 5
        # because the AI player's bird wiil be initiated at the start stage of the game, so we need not to consider the position of the pipes.
        y = MAP_HEIGHT // 2
        self._agent_bird_1 = AgentBird(self, self._bird_image, x, y, brain)

    def _agent_action(self):
        # in all birds, if a bird is not controlled by the human player,
        # then try to make the bird jump. This part is the operation part of the AI bird.
        for bird in self.birds:
            if bird is not self._human_bird:
                self.fcl2_flap(bird)# fcl_flap, fcl2_flap, fclappy_vision_flap

    def _handle_events(self):
        for event in pg.event.get():
            # close the game if the event type is pg.QUIT
            if event.type == pg.QUIT:
                self.playing = False
                self.running = False
            # if the event type is pg.KEYDOWN, it checks whether a key is pressed.
            elif event.type == pg.KEYDOWN:
                pressed = pg.key.get_pressed()
                ctrl_held = pressed[pg.K_LCTRL] or pressed[pg.K_RCTRL]
                # if the Ctrl key is pressed, combined with other keys,
                if ctrl_held:
                    if event.key == pg.K_p:  # if Ctrl + P are pressed, it pauses the game.
                        self._is_paused = True
                        self._pause()
                    elif event.key == pg.K_1:  # if Ctrl + 1 are pressed, it sets the frame rate of the game to the default value.
                        self._fps = FPS
                    elif event.key == pg.K_2:  # if Ctrl + 2 are pressed, it sets the frame rate of the game to half of the default value.
                        self._fps = 0.5 * FPS
                    elif event.key == pg.K_3:  # if Ctrl + 3 are pressed, it sets the frame rate of the game to 10 times of the default value.
                        self._fps = 20 * FPS
                    elif event.key == pg.K_4:  # if Ctrl + h are pressed, it creates a human player controlled bird (if you really want to complete with the agent bird).
                        if not self._human_bird or not self._human_bird.alive():
                            self._add_human_player()
                elif event.key == pg.K_SPACE or event.key == pg.K_UP:   
                    # if the space key or the up arrow key is pressed, it makes the human player's bird jump.
                    if self._human_bird is not None and self._human_bird.alive():
                        self._human_bird.flap()        

    def _get_pipe(self, bird):
        def get_pipe_x(pipe):
            return pipe.rect.x
        front_bottom_pipe = min((p for p in self.pipes if p.type == PipeType.BOTTOM \
                                 and p.rect.right >= bird.rect.left), key=get_pipe_x)
        return front_bottom_pipe
    
    def _get_error(self, bird):
        # calculate the error between the bird's height and the height of the front pipe's center
        # the error is used to train the FCL network,
        # which equantion is: error = the height of the bird - the height of the front pipe's center
        front_bottom_pipe = self._get_pipe(bird)
        error = bird.rect.y - (front_bottom_pipe.rect.y - (front_bottom_pipe.gap // 2))
        return error
    
    def _get_error2(self, bird):
        # error definition function to maximize the bird's score by using sigmoid function 
        # to abstract the error of the input error of the FCL2 network
        errorGain = 20
        # error = errorGain / 1 + np.arctan(bird.score)
        if bird.score < 700:
            error = errorGain / (1 + np.e ** (bird.score)) 
        else:
            error = 0
        return error
    
    def _get_error3(self, bird):
        errorGain = 20
        if bird.score < 700:
            error = errorGain / (1 + np.e ** (bird.score - bird.flapTimes))
        else:
            error = 0
        return error
    
    def _get_netinput(self, bird):
        def get_pipe_x(pipe):
            return pipe.rect.x
        first_front_bottom_pipe = sorted((p for p in self.pipes if p.type == PipeType.BOTTOM \
                                 and p.rect.right >= bird.rect.left), key=get_pipe_x)[0]
        second_front_bottom_pipe = sorted((p for p in self.pipes if p.type == PipeType.BOTTOM \
                                 and p.rect.right >= bird.rect.left), key=get_pipe_x)[1]
        third_front_bottom_pipe = sorted((p for p in self.pipes if p.type == PipeType.BOTTOM \
                                 and p.rect.right >= bird.rect.left), key=get_pipe_x)[2]
        fourth_front_bottom_pipe = sorted((p for p in self.pipes if p.type == PipeType.BOTTOM \
                                    and p.rect.right >= bird.rect.left), key=get_pipe_x)[3]
        input0 = bird.rect.y - (first_front_bottom_pipe.rect.y - (first_front_bottom_pipe.gap // 2))
        input1 = bird.rect.y - (second_front_bottom_pipe.rect.y - (second_front_bottom_pipe.gap // 2))
        input2 = bird.rect.y - (third_front_bottom_pipe.rect.y - (third_front_bottom_pipe.gap // 2))
        input3 = bird.rect.y - (fourth_front_bottom_pipe.rect.y - (fourth_front_bottom_pipe.gap // 2))
        # by getting the height error between the bird and the center of the nearest 4 pipes in front,
        # to use as the input of the FCL network, so as to realize the utilization of future information,
        # to judge whether the bird should fly up, and generate predictive control signals
        return [input0, input1, input2, input3]
    
    # def _get_frame(self):
    #     frame = pg.surfarray.array3d(pg.display.get_surface())
    #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     frame = cv2.Canny(frame, threshold1 = 200, threshold2=300)
    #     return frame
    
    def error_flap(self, bird):
        # use error
        error = self._get_error(bird)
        bird.brain = error
        if bird.brain > 0:
            bird.flap()

    def fcl_flap(self, bird):
        # use fcl
        inputs = self._get_netinput(bird)
        error = self._get_error(bird)    
        bird.brain = self.FCLNet.train(inputs, error)
        # print("bird.brain: ", bird.brain)
        if bird.brain > 0:
            bird.flap()   

    
    def fcl2_flap(self, bird):
        # use fcl2
        inputs = self._get_netinput(bird)
        error = self._get_error2(bird)
        if error == 0 and bird.score > 2500 and bird.score % 1000 == 0:
            if not os.path.exists("Models"):
                os.mkdir("Models")
            self.FCLNet.saveModel("Models/fcl2-" + str(bird.score) + ".txt")
            file = open("Models/checkpoint", 'w')
            file.write("Models/fcl2-" + str(bird.score) + ".txt")
            file.close()
        bird.brain = self.FCLNet.train(inputs, error)
        if bird.brain > 0:
            bird.flap()
    
    def fcl3_flap(self, bird):
        # use fcl2
        inputs = self._get_netinput(bird)
        error = self._get_error3(bird)
        if error == 0 and bird.score > 2500 and bird.score % 1000 == 0:
            if not os.path.exists("Models"):
                os.mkdir("Models")
            self.FCLNet.saveModel("Models/fcl2-" + str(bird.score) + ".txt")
            file = open("Models/checkpoint", 'w')
            file.write("Models/fcl2-" + str(bird.score) + ".txt")
            file.close()
        bird.brain = self.FCLNet.train(inputs, error)
        if bird.brain > 0:
            bird.flap()
            bird.flapTimes += 1

    # def fclappy_vision_flap(self, bird):
    #     # use fclappy_vision
    #     inputs = self._get_frame()
    #     error = self._get_error(bird)
    #     if error == 0 and bird.score > 2500 and bird.score % 1000 == 0:
    #         if not os.path.exists("Models"):
    #             os.mkdir("Models")
    #         self.FCLNet.saveModel("Models/fcl2-" + str(bird.score) + ".txt")
    #         file = open("Models/checkpoint", 'w')
    #         file.write("Models/fcl2-" + str(bird.score) + ".txt")
    #         file.close()
    #     bird.brain = self.FCLNet.train(inputs, error)
    #     if bird.brain > 0:
    #         bird.flap() 

    def _update(self):
        # call the update function of all sprites to update their status.
        # The specific update operation depends on the definition of the update function of each sprite.

        self.all_sprites.update()
        if not self.birds:
            self.playing = False
            return
        
        # the movement of the bird and the pipe: in the game, the bird looks like it is flying forward, 
        # but in fact it is the background and the pipe that are moving backward.
        # Here, first find the bird at the very front. If its position is within 1/3 of the screen width,
        # then all birds move forward; otherwise, all pipes move backward.
        # In addition, if the pipe has completely moved out of the screen (its x coordinate is less than -50),
        # then the pipe will be removed.
        def get_pipe_x(pipe):
            return pipe.rect.x         
        if max(self.birds, key=get_pipe_x).rect.x < MAP_WIDTH / 4:
            for bird in self.birds:
                bird.moveby(dx=BIRD_X_SPEED)
        else:
            for pipe in self.pipes:
                pipe.moveby(dx=-BIRD_X_SPEED)
                if pipe.rect.x < -55:
                    pipe.kill()
        
        # calculate the score: bird will get one point for each frame it is alive.
        for bird in self.birds:
            bird.score += 1
        self._max_score += 1
        self._max_score_so_far = max(self._max_score_so_far, self._max_score)
        
        # generate new pipes:
        # if the position of the front pipe (self._very_front_pipe) is not enough to cover the screen,
        # then generate a new pipe.
        while self._very_front_pipe.rect.x < MAP_WIDTH:
            self._generate_pipe()

    def _save_FCL_parameters(self):
        # layer 0
        a = np.zeros((1,4))
        a = np.array([self.FCLNet.getLayer(0).getNeuron(0).getWeight(0), self.FCLNet.getLayer(0).getNeuron(1).getWeight(0), self.FCLNet.getLayer(0).getNeuron(2).getWeight(0), self.FCLNet.getLayer(0).getNeuron(3).getWeight(0)])
        a = a.reshape(1,4)
        df = pd.DataFrame(a)
        df.to_csv('layer1.csv', mode='a', header=False, index=False)


        # layer 1
        b = np.zeros((1,8))
        b = np.array([self.FCLNet.getLayer(1).getNeuron(0).getWeight(0), self.FCLNet.getLayer(1).getNeuron(1).getWeight(0), self.FCLNet.getLayer(1).getNeuron(2).getWeight(0), self.FCLNet.getLayer(1).getNeuron(3).getWeight(0), self.FCLNet.getLayer(1).getNeuron(4).getWeight(0), self.FCLNet.getLayer(1).getNeuron(5).getWeight(0), self.FCLNet.getLayer(1).getNeuron(6).getWeight(0), self.FCLNet.getLayer(1).getNeuron(7).getWeight(0)])
        b = b.reshape(1,8)
        df = pd.DataFrame(b)
        df.to_csv('layer2.csv', mode='a', header=False, index=False)

        # layer 2
        c = np.zeros((1,4))
        c = np.array([self.FCLNet.getLayer(2).getNeuron(0).getWeight(0), self.FCLNet.getLayer(2).getNeuron(1).getWeight(0), self.FCLNet.getLayer(2).getNeuron(2).getWeight(0), self.FCLNet.getLayer(2).getNeuron(3).getWeight(0)])
        c = c.reshape(1,4)
        df = pd.DataFrame(c)
        df.to_csv('layer3.csv', mode='a', header=False, index=False)

        # error
        d = np.zeros((1,1))
        d = np.array(self._get_error2(self._agent_bird_1))
        d = d.reshape(1,1)
        df = pd.DataFrame(d)
        df.to_csv('error.csv', mode='a', header=False, index=False)

    def _save_Bird_Trajectory(self):
        a = np.zeros((1,1))
        a = np.array(self._agent_bird_1.rect.y)
        a = a.reshape(1,1)
        df = pd.DataFrame(a)
        df.to_csv('bird_trajectory4.csv', mode='a', header=False, index=False)

    def _draw(self):

        self.all_sprites.draw(self._screen)
        # show score
        self._draw_text('Score: {}'.format(self._max_score), 10, 10)
        self._draw_text('Max score so far: {}'.format(self._max_score_so_far), 10, 10 + FONT_SIZE + 2)
        self._draw_text('Iteration: {}'.format(self._iteration), 10, 10 + 2 * (FONT_SIZE + 2))

        pg.display.update()

    def _draw_text(self, text, x, y, color=WHITE, font=FONT_NAME, size=FONT_SIZE):
        font = pg.font.SysFont(font, size)
        text_surface = font.render(text, True, color)
        self._screen.blit(text_surface, (x, y))
