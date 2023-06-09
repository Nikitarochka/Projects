from livewires import games, colour
import random
games.init(screen_width = 640, screen_hignt = 480, fps = 50)
class Pan(games.Sprite):
    image = games.load_image("korzina.png")
    def __init__(self):
        super(Pan, self).__init__(image = Pan.image,
                                  x = games.mouse.x,
                                  bottom = games.screen.height)
        self.score = games.Text(value = 0, size = 25, colour = colour.black,
                                top = 5, right = games.screen.width - 10)
        games.screen.add(self.score)
    def update(self):
        self.x = games.mouse.x
        if self.left < 0:
            self.left = 0
        if self.right > games.screen.width:
            self.right = games.screen.width
        self.check_catch()
    def check_catch(self):
        for work in self.overlapping_sprites:
            self.score.value += 10
            self.score.right = games.screen.width - 10
            work.handle_caught()
class Work(games.Sprite):
    image = games.load_image("work.jpg")
    speed = 1
    def __init__(self, x, y = 90):
        super(Work, self).__init__(image = Work.image,
                                     x = x,
                                     y = y,
                                     dy = Work.speed)
    def update(self):
        if self.bottom > games.screen.height:
            self.end_game()
            self.destroy()
    def handle_caught(self):
        self.destroy()
    def end_game(self):
        end_message = games.Message(value = "Game Over",
                                    size = 90,
                                    color = colour.red,
                                    x = games.screen.width/2,
                                    y = games.screen.height/2,
                                    lifetime = 5 * games.screen.fps,
                                    after_death = games.screen.quit)
        games.screen.add(end_message)
class Chef(games.Sprite):
    image = games.load_image("ignatev.jpg")
    def __init__(self, y = 55, speed = 2, odds_change = 200):
        super(Chef, self).__init__(image = Chef.image,
                                   x = games. screen.width/2,
                                   y = y,
                                   dx = speed)
        self.odds_change = odds_change
        self.time_til_drop = 0
    def update(self):
        if self.left < 0 or self.right > games.screen.width:
            self.dx = -self.dx
        elif random.randrange(self.odds_change) == 0:
            self.dx = -self.dx
        self.check_drop()
    def check_drop(self):
        if self.time_til_drop > 0:
            self.time_til_drop -= 1
        else:
            new_work = Work(x = self.x)
            games.screen.add(new_work)
            self.time_til_drop = int(new_work.height * 1.3 / Work.speed) + 1
def main():
    wall_image = games.load_image("gornii.jpg",transparent = False)
    games.screen.background = wall_image
    the_chef = Chef()
    games.screen.add(the_chef)
    the_pan = Pan()
    games.screen.add(the_pan)
    games.mouse.is_visible = False
    games.screen.event_grab = True
    games.screen.mainloop()
main()
