import pygame
import os

alarm_loaded = False

def init_alarm(file_path):
    global alarm_loaded
    try:
        pygame.mixer.init()
        if os.path.exists(file_path):
            pygame.mixer.music.load(file_path)
            alarm_loaded = True
        else:
            print("Alarm file not found")
    except pygame.error as e:
        print(f"Pygame error: {e}")


def play_alarm():
    if alarm_loaded and not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(loops=-1)


def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
