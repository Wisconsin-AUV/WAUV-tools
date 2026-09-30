import json
import socket
import pygame

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # sock_dgram for UDP, AF_inet for ipv4
DEST = ("192.168.10.2", 6767)

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller found!")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

clock = pygame.time.Clock()

print(f"Streaming {joystick.get_name()} ...")

try:
    while True:
        pygame.event.pump()
        a = []
        # for i in range(joystick.get_numbuttons()):
        #     a.append(joystick.get_button(i))
        # print(a)
        joy_msg = {
            "axes": [round(joystick.get_axis(i), 3) for i in range(joystick.get_numaxes())]
        }
        
        sock.sendto(json.dumps(joy_msg).encode('utf-8'), DEST)
        
        clock.tick(50) # 50hz
except KeyboardInterrupt:
    sock.close()
    pass