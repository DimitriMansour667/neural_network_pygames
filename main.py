import pygame
import random
from neural_network import neural_network

# Initialize neural network (6 inputs: ball_x, closest_wall_x, closest_wall_y, hole_left_x, hole_right_x, distance_to_wall)
nn = neural_network(100, 5, 6, 0.005)
nn.populate()

# Game Constants
WIDTH, HEIGHT = 500, 600
BALL_SIZE = 30
BALL_SPEED = 10
WALL_HEIGHT = 40
WALL_SPEED = 5
HOLE_WIDTH = 100
SPAWN_INTERVAL = 50  # Frames between wall spawns
MIN_FPS = 5
MAX_FPS = 240
DEFAULT_FPS = 30

# Colors
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Initialize PyGame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ça Passe ou Ça Casse - AI Training")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

class GameAgent:
    def __init__(self, nn_agent):
        self.nn_agent = nn_agent
        self.x = WIDTH // 2
        self.alive = True
        self.score = 0

def reset_game():
    return {
        "walls": [],
        "frame_count": 0
    }

def get_closest_wall(walls, ball_y):
    closest_wall = None
    min_distance = float('inf')
    for wall in walls:
        if wall["y"] < ball_y:  # Only consider walls above the ball
            distance = ball_y - wall["y"]
            if distance < min_distance:
                min_distance = distance
                closest_wall = wall
    return closest_wall

def get_inputs(agent_x, walls, ball_y):
    closest_wall = get_closest_wall(walls, ball_y)
    inputs = [0] * 6
    
    # Normalize inputs between 0 and 1
    inputs[0] = agent_x / WIDTH  # Ball position
    
    if closest_wall:
        inputs[1] = closest_wall["x"] / WIDTH  # Wall x position
        inputs[2] = closest_wall["y"] / HEIGHT  # Wall y position
        inputs[3] = closest_wall["x"] / WIDTH  # Hole left x
        inputs[4] = (closest_wall["x"] + HOLE_WIDTH) / WIDTH  # Hole right x
        inputs[5] = (ball_y - closest_wall["y"]) / HEIGHT  # Distance to wall
    
    return inputs

# Game Loop
generation = 1
best_score = 0
running = True
current_fps = DEFAULT_FPS

while running:
    game_state = reset_game()
    
    # Create game agents for each neural network agent
    game_agents = [GameAgent(agent) for agent in nn.agents]
    
    while running:
        screen.fill(WHITE)
        
        # Handle Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    current_fps = min(MAX_FPS, current_fps * 2)
                elif event.key == pygame.K_DOWN:
                    current_fps = max(MIN_FPS, current_fps // 2)

        # Spawn Walls
        if game_state["frame_count"] % SPAWN_INTERVAL == 0:
            hole_x = random.randint(50, WIDTH - HOLE_WIDTH - 50)
            game_state["walls"].append({"x": hole_x, "y": 0})

        # Move and Draw Walls
        new_walls = []
        for wall in game_state["walls"]:
            wall["y"] += WALL_SPEED
            pygame.draw.rect(screen, BLACK, (0, wall["y"], wall["x"], WALL_HEIGHT))
            pygame.draw.rect(screen, BLACK, (wall["x"] + HOLE_WIDTH, wall["y"], 
                           WIDTH - wall["x"] - HOLE_WIDTH, WALL_HEIGHT))
            
            # Keep walls that are still on screen
            if wall["y"] < HEIGHT:
                new_walls.append(wall)
        
        game_state["walls"] = new_walls

        # Update and Draw Agents
        alive_count = 0
        for game_agent in game_agents:
            if game_agent.alive:
                alive_count += 1
                # Get inputs for neural network
                inputs = get_inputs(game_agent.x, game_state["walls"], HEIGHT - 50)
                # Get decision from neural network
                decision = game_agent.nn_agent.think(inputs)
                
                # Move agent based on neural network output
                if decision > 0.66 and game_agent.x < WIDTH - BALL_SIZE:
                    game_agent.x += BALL_SPEED
                elif decision <= 0.33 and game_agent.x > 0:
                    game_agent.x -= BALL_SPEED
                else:
                    game_agent.x += 0

                # Draw agent
                color = BLUE if game_agent.alive else RED
                pygame.draw.circle(screen, color, (game_agent.x + BALL_SIZE // 2, HEIGHT - 50 + BALL_SIZE // 2), 
                                 BALL_SIZE // 2)

                # Collision Detection
                for wall in game_state["walls"]:
                    if wall["y"] + WALL_HEIGHT >= HEIGHT - 50:
                        if not (wall["x"] < game_agent.x < wall["x"] + HOLE_WIDTH - BALL_SIZE):
                            game_agent.alive = False
                
                # Update score for surviving agents
                game_agent.score += 1

        # Check if all agents are dead
        if alive_count == 0:
            # Update neural network scores before evolution
            for game_agent, nn_agent in zip(game_agents, nn.agents):
                nn_agent.score = game_agent.score
            
            # Create next generation
            nn.next_generation()

            generation += 1
            break

        # Display Info
        best_agent = max(game_agents, key=lambda x: x.score)
        best_score = max(best_score, best_agent.score)
        
        info_text = [
            f"Generation: {generation}",
            f"Alive Agents: {alive_count}",
            f"Current Best: {best_agent.score}",
            f"All-time Best: {best_score}",
            f"Speed: {current_fps} FPS"
        ]
        
        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (10, 10 + i * 30))

        pygame.display.flip()
        clock.tick(current_fps)
        game_state["frame_count"] += 1

pygame.quit()
