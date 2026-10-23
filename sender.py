import json, socket, time
import pygame

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
DEST = ("192.168.137.2", 6767)

pygame.init()
pygame.joystick.init()

def wait_for_joystick():
    print("Waiting for controller...")
    while True:
        pygame.event.pump()
        pygame.joystick.quit()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            js = pygame.joystick.Joystick(0)
            js.init()
            print(f"Streaming {js.get_name()} ...")
            return js
        time.sleep(0.5)

joystick = wait_for_joystick()
clock = pygame.time.Clock()

try:
    while True:
        pygame.event.pump()
        if pygame.joystick.get_count() == 0:   # controller dropped
            joystick = wait_for_joystick()
            continue
        axes = [round(joystick.get_axis(i), 3) for i in range(joystick.get_numaxes())]
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]
        sock.sendto(json.dumps({"axes": axes, "buttons": buttons}).encode(), DEST)
        clock.tick(50)
except KeyboardInterrupt:
    sock.close()