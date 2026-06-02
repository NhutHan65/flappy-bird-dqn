import pygame
import sys


def load():
    PLAYER_PATH = (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    )
    BACKGROUND_PATH = 'assets/sprites/background-black.png'
    PIPE_PATH       = 'assets/sprites/pipe-green.png'

    IMAGES, SOUNDS, HITMASKS = {}, {}, {}

    IMAGES['numbers'] = tuple(
        pygame.image.load(f'assets/sprites/{i}.png').convert_alpha()
        for i in range(10)
    )
    IMAGES['base']       = pygame.image.load('assets/sprites/base.png').convert_alpha()
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()
    IMAGES['player']     = tuple(
        pygame.image.load(PLAYER_PATH[i]).convert_alpha() for i in range(3)
    )
    IMAGES['pipe'] = (
        pygame.transform.rotate(pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    # Sound loading is best-effort; fails silently under dummy audio driver
    soundExt = '.wav' if 'win' in sys.platform else '.ogg'
    for name in ('die', 'hit', 'point', 'swoosh', 'wing'):
        try:
            SOUNDS[name] = pygame.mixer.Sound(f'assets/audio/{name}{soundExt}')
        except Exception:
            SOUNDS[name] = None

    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )
    HITMASKS['player'] = tuple(getHitmask(IMAGES['player'][i]) for i in range(3))

    return IMAGES, SOUNDS, HITMASKS


def getHitmask(image):
    mask = []
    for x in range(image.get_width()):
        mask.append([bool(image.get_at((x, y))[3]) for y in range(image.get_height())])
    return mask
