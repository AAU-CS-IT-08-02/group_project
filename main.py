import pygame
import numpy as np

pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Basic APF Pursuit-Evasion")
clock = pygame.time.Clock()

evader_pos = np.array([100.0, 100.0])
pursuer_pos = np.array([100.0, 500.0])
goal_pos = np.array([700.0, 500.0])

# Tuning Parameters
evader_speed = 3.0
pursuer_speed = 3.2
repulsion_radius = 200.0

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #Attractive Force Calculation
    vec_to_goal = goal_pos - evader_pos
    dist_to_goal = np.linalg.norm(vec_to_goal)
    if dist_to_goal > 0:
        f_attract = (vec_to_goal / dist_to_goal) * evader_speed
    else:
        f_attract = np.zeros(2)

    #Repulsive Force Calculation
    vec_from_pursuer = evader_pos - pursuer_pos
    dist_from_pursuer = np.linalg.norm(vec_from_pursuer)
    f_repulse = np.zeros(2)

    if 0 < dist_from_pursuer < repulsion_radius:

        if dist_to_goal < dist_from_pursuer:
            push_strength = (repulsion_radius / dist_from_pursuer) * (evader_speed * 0.1)
        else:
            push_strength = (repulsion_radius / dist_from_pursuer) * evader_speed

        f_repulse = (vec_from_pursuer / dist_from_pursuer) * push_strength

    total_force = f_attract + f_repulse
    speed_magnitude = np.linalg.norm(total_force)

    if speed_magnitude > evader_speed:
        actual_velocity = (total_force / speed_magnitude) * evader_speed
    else:
        actual_velocity = total_force

    evader_pos += actual_velocity

    vec_to_evader = evader_pos - pursuer_pos
    dist_to_evader = np.linalg.norm(vec_to_evader)
    if dist_to_evader > 0:
        f_pursuit = (vec_to_evader / dist_to_evader) * pursuer_speed
        pursuer_pos += f_pursuit

    screen.fill((220, 220, 220))

    #goal
    pygame.draw.circle(screen, (0, 200, 0), goal_pos.astype(int), 15)
    #evader
    pygame.draw.circle(screen, (0, 0, 255), evader_pos.astype(int), 10)
    #pursuer
    pygame.draw.circle(screen, (255, 0, 0), pursuer_pos.astype(int), 10)
    #repulsion radius
    pygame.draw.circle(screen, (255, 100, 100), evader_pos.astype(int), int(repulsion_radius), 1)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()