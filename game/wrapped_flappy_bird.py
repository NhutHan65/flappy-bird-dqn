import numpy as np
import sys
import random
import pygame
import flappy_bird_utils
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle

FPS         = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN   = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, SOUNDS, HITMASKS = flappy_bird_utils.load()

PIPEGAPSIZE      = 100
BASEY            = SCREENHEIGHT * 0.79
PLAYER_WIDTH     = IMAGES['player'][0].get_width()
PLAYER_HEIGHT    = IMAGES['player'][0].get_height()
PIPE_WIDTH       = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT      = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


class GameState:
    """
    Flappy Bird environment.

    Parameters
    ----------
    render : bool
        True  → display is updated and FPS is capped (use for visual testing).
        False → headless fast mode for training (no FPS cap, no display update).
    """

    def __init__(self, render: bool = True, visual_mode: str = 'rgb'):
        self.render_game  = render
        self.visual_mode  = visual_mode   # 'rgb' | 'grayscale' | 'threshold'
        self.last_score   = 0
        self._reset()

    def _reset(self):
        self.score       = 0
        self.playerIndex = 0
        self.loopIter    = 0
        self.playerx     = int(SCREENWIDTH * 0.2)
        self.playery     = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex       = 0
        self.baseShift   = IMAGES['base'].get_width() - BACKGROUND_WIDTH

        newPipe1 = getRandomPipe()
        newPipe2 = getRandomPipe()
        self.upperPipes = [
            {'x': SCREENWIDTH,                   'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + SCREENWIDTH // 2, 'y': newPipe2[0]['y']},
        ]
        self.lowerPipes = [
            {'x': SCREENWIDTH,                   'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + SCREENWIDTH // 2, 'y': newPipe2[1]['y']},
        ]

        self.pipeVelX      = -4
        self.playerVelY    =  0
        self.playerMaxVelY =  10
        self.playerMinVelY = -8
        self.playerAccY    =  1
        self.playerFlapAcc = -9
        self.playerFlapped = False

    def frame_step(
        self,
        input_actions,
        step_reward:  float = 0.1,
        pipe_reward:  float = 0.0,
        crash_reward: float = -1.0,
    ):
        """
        Advance the game by one frame.

        Returns
        -------
        image_data : np.ndarray  shape (288, 512, 3)
        reward     : float
        terminal   : bool
        """
        pygame.event.pump()

        if sum(input_actions) != 1:
            raise ValueError('Exactly one input action must be 1.')

        reward   = step_reward
        terminal = False

        if input_actions[1] == 1 and self.playery > -2 * PLAYER_HEIGHT:
            self.playerVelY    = self.playerFlapAcc
            self.playerFlapped = True

        # Score: bird passes through a pipe gap
        playerMid = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMid = pipe['x'] + PIPE_WIDTH / 2
            if pipeMid <= playerMid < pipeMid + 4:
                self.score += 1
                reward = step_reward + pipe_reward

        # Player animation
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex    = -((-self.basex + 100) % self.baseShift)

        # Physics
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # Move pipes
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # Spawn new pipe
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # Remove off-screen pipe
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # Collision check
        if checkCrash(
            {'x': self.playerx, 'y': self.playery, 'index': self.playerIndex},
            self.upperPipes, self.lowerPipes,
        ):
            terminal        = True
            self.last_score = self.score   # capture before reset
            reward          = crash_reward
            self._reset()

        # Render
        SCREEN.blit(IMAGES['background'], (0, 0))
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))
        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        SCREEN.blit(IMAGES['player'][self.playerIndex], (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        if self.render_game:
            if self.visual_mode != 'rgb':
                px = pygame.surfarray.pixels3d(SCREEN)
                gray = (0.299 * px[:,:,0] + 0.587 * px[:,:,1] + 0.114 * px[:,:,2]).astype(np.uint8)
                if self.visual_mode == 'threshold':
                    gray = ((gray > 127) * 255).astype(np.uint8)
                px[:,:,0] = gray
                px[:,:,1] = gray
                px[:,:,2] = gray
                del px  # release surface lock before update
            pygame.display.update()
            FPSCLOCK.tick(FPS)

        return image_data, reward, terminal


# ── Helpers ───────────────────────────────────────────────────────────────────

def getRandomPipe():
    gapYs  = [20, 30, 40, 50, 60, 70, 80, 90]
    gapY   = random.choice(gapYs) + int(BASEY * 0.2)
    pipeX  = SCREENWIDTH + 10
    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},
    ]


def checkCrash(player, upperPipes, lowerPipes):
    pi        = player['index']
    player_w  = IMAGES['player'][0].get_width()
    player_h  = IMAGES['player'][0].get_height()

    if player['y'] + player_h >= BASEY - 1:
        return True

    playerRect = pygame.Rect(player['x'], player['y'], player_w, player_h)
    for uPipe, lPipe in zip(upperPipes, lowerPipes):
        uRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
        lRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
        if (pixelCollision(playerRect, uRect, HITMASKS['player'][pi], HITMASKS['pipe'][0])
                or pixelCollision(playerRect, lRect, HITMASKS['player'][pi], HITMASKS['pipe'][1])):
            return True
    return False


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)
    if rect.width == 0 or rect.height == 0:
        return False
    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y
    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False
