'''
Game objects which are used in the game
1, Bird
2, Pipe
3, Background

'''
import enum
import pygame as pg
from config import *
from os import path

class Background(pg.sprite.Sprite):
    """
    Seamless background class.
    """
    def __init__(self, game, image):
        self._layer = 0

        # this attribute is required by pygame.sprite.LayeredUpdates to determine the order of different sprites when drawing.
        # The sprite with the smallest number will be drawn first, which means that they will appear below all other sprites.
        # Here, the background is set to the bottom layer (0), which means that all other sprites will be drawn on top of the background.

        super().__init__(game.all_sprites)
        # if the width of the given image < screen width, then repeat it until we get a wide enough one
        if image.get_width() < MAP_WIDTH:
            w = image.get_width()
            repeats = MAP_WIDTH // w + 1
            self.image = pg.Surface((w * repeats, image.get_height()))
            for i in range(repeats):
                self.image.blit(image, (i * w, 0))
        else:
            self.image = image
        self.rect = self.image.get_rect()

class MovableObject(pg.sprite.Sprite):
    def __init__(self, *groups):
        super().__init__(*groups)
        self.rect = None
    
    def moveto(self, x=0, y=0):
        self.rect.x = x
        self.rect.y = y
    
    def moveby(self, dx=0, dy=0):
        self.rect.move_ip(dx, dy)
    
class PipeType(enum.Enum): 
    TOP = 0
    BOTTOM = 1

class Pipe(MovableObject):
    def __init__(self, game, image, x, length, type_):
        self._layer = 1
        super().__init__(game.all_sprites, game.pipes)
        self._game = game
        self.type = type_
        
        self.image = pg.Surface((image.get_width(), length))
        if type_ == PipeType.TOP: #顶部
            self.image.blit(image, (0, 0), (0, image.get_height() - length, image.get_width(), length))
            # image.blit(source, dest, area=None, special_flags = 0) -> Rect
        else:
            self.image.blit(image, (0, 0), (0, 0, image.get_width(), length))
        # position and region
        self.rect = self.image.get_rect(centerx = x) # get_rect()returns a Rect object that represents the rectangular area of ​​the Surface object
        # gap between the top and bottom pipes
        if type_ == PipeType.TOP:
            self.rect.top = 0
        else:
            self.rect.bottom = MAP_HEIGHT # note that the origin is at the top left corner, so here is bottom, which is the bottom of the screen
        self.gap = 0
        self.length = length

class Bird(MovableObject):
    def __init__(self, game, image: pg.Surface, x, y):
        self._layer = 2 
        super().__init__(game.all_sprites, game.birds) # add to all_sprites and birds group
        self._game = game
        self.image = image
        self.original_image = self.image
        self.rect = image.get_rect(x=x, y=y)
        self._vel_y = 0 # vertical velocity
        self.score = 0

    def update(self, *args):

        if self.rect.top > MAP_HEIGHT or self.rect.bottom < 0 or pg.sprite.spritecollideany(self, self._game.pipes):
            self.kill()
        
        self._vel_y = min(self._vel_y + GRAVITY_ACC, BIRD_MAX_Y_SPEED)
        self.rect.y += self._vel_y

        angle = 40 - (self._vel_y + 4) / 8 * 80
        angle = min(30, max(angle, -30))
        self.image = pg.transform.rotate(self.original_image, angle)
        self.rect = self.image.get_rect(center=self.rect.center)

    def flap(self):
        self._vel_y = JUMP_SPEED
    
    @property
    def vel_y(self):
        return self._vel_y


class AgentBird(Bird): # agent bird
    def __init__(self, game, image: pg.Surface, x, y, brain): # add a brain parameter to load the relevant control logic
        super().__init__(game, image, x, y)
        self.brain = brain

    def kill(self):
        super().kill()



