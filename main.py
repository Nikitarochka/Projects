'''import random
k = 0
t = 0
i=0
coin = ["Орёл", "Решка"]
while i < 100:
    m = random.choice(coin)
    if m == "Орёл":
        k += 1
    else:
        t += 1
    i += 1
print("Количество выпавших орлов: " and k)
print("Количество выпавших решек: " and t)'''
'''import random
def display_instruct():
    print("Загадайте любое число в заданном диапазоне.\nНа отгадывание 10 попыток")
def main(w,l):
    k = 50
    if k > w and k < l:
        display_instruct()
        t = 0
        while t < 10:
            print(k)
            k1 = k
            t += 1
            m = input("Это загаданное вами число? ")
            if m == "Да":
                print("Я победил вас, попробуйте снова. Количество затраченных попыток: ", t)
                break
            elif m == "Нет":
                u = input("Оно больше или меньше данного числа? ")
                if u == "Больше":
                    h = input("Очень горячо(от 1 до 5), горячо(от 6 до 15), холодно(от 16 до 40)или очень холодно?:)")
                    if h == "Очень горячо":
                        k = random.randint(k1 + 1, k1 + 5)
                    elif h == "Горячо":
                        k = random.randint(k1 + 6, k1 + 15)
                    elif h == "Холодно":
                        k = random.randint(k1 + 16, k1 + 40)
                    elif h == "Очень холодно":
                        k = random.randint(k1 + 41, l)
                elif u == "Меньше":
                    h = input("Очень горячо(от 1 до 5), горячо(от 6 до 15), холодно(от 16 до 40)или очень холодно?:)")
                    if h == "Очень горячо":
                        k = random.randint(k1 -1, k1 - 5)
                    elif h == "Горячо":
                        k = random.randint(k1 - 6, k1 - 15)
                    elif h == "Холодно":
                        k = random.randint(k1 - 16, k1 - 40)
                    elif h == "Очень холодно":
                        k = random.randint(k1 - 41, w)
        else:
            print("Вы ввели неправильные данные, повторите попытку!")
    if t >= 10:
        print("Вы победили меня, поздравляю!:)")
main(0, 100)'''
'''import random
WORDS = ("яблоко", "ананас", "базилик", "носорог", "горный", "девушка")
word = random.choice(WORDS)
print("Угадайте загаданное мною слово, у вас 5 попыток")
print("В данном слове", len(word), "букв")
t = 0
while t < 5:
    k = input("Ваше предположение? ")
    t += 1
    if word[:] == k:
        print("Поздравляю! Вы отгадали слово.")
        break
    elif word[0]  == k:
        print("Да")
    elif word[1]  == k:
        print("Да")
    elif word[2]  == k:
        print("Да")
    elif word[3]  == k:
        print("Да")
    elif word[4]  == k:
        print("Да")
    elif word[5]  == k:
        print("Да")
    elif word[6]  == k:
        print("Да")
    else:
        print("Нет")
if t == 5:
    print("К сожалению,вы не отгадали слово:(")'''
'''import random
WORDS = ("Хлеб","Батон","Каравай","Булка","Рогалик","Круассан","Торт")
for i in range(6):
    if WORDS[i]!=WORDS[i-1]!=WORDS[i-2]!=WORDS[i-3]!=WORDS[i-4]!=WORDS[i-5]!=WORDS[i-6]:
        k = random.choice(WORDS)
    print(k)'''
'''Characters = {"Сила": 0, "Здоровье": 0, "Ловкость": 0, "Мудрость": 0}
point = 30
print("Здравствуйте, это симулятор характеристик героя.\nВам доступно 30 очков, распределите их на ваше усмотрение.")
print("Сейчас ваши характеристики таковы:, ",Characters)
ol = 0
while ol != 1:
    hero = input("Выберите нужный параметр: ")
    key = list(Characters)
    if hero == key[0]:
        print( "Введите то, что нужно сделать с параметром:")
        print("Введите 1, если хотите прибавить очки к параметру;")
        print("Введите 2, если хотите отнять очки от параметра.")
        parametrs = input("Введите нужную цифру: ")
        if parametrs == "1":
            k = int(input("Введите количество очков: "))
            if point >= 0:
                point -= k
                m = Characters["Сила"]
                Characters["Сила"] = m + k
            else:
                print("Превышено максимально допустимое значение очков.")
        elif parametrs == "2":
            k = int(input("Введите количество очков: "))
            point += k
            Characters[0] -= k
        else:
            print("Введены неправильные данные, повторите попытку.")
        print("Оставшееся количество очков: ", point)
    elif hero == key[1]:
        print( "Введите то, что нужно сделать с параметром:")
        print("Введите 1, если хотите прибавить очки к параметру;")
        print("Введите 2, если хотите отнять очки от параметра.")
        parametrs = input("Введите нужную цифру: ")
        if parametrs == "1":
            k = int(input("Введите количество очков: "))
            if point >= 0:
                point -= k
                m = Characters["Здоровье"]
                Characters["Здоровье"] = m + k
            else:
                print("Превышено максимально допустимое значение очков.")
        elif parametrs == "2":
            k = int(input("Введите количество очков: "))
            point += k
            Characters[0] -= k
        else:
            print("Введены неправильные данные, повторите попытку.")
        print("Оставшееся количество очков: ", point)
    elif hero == key[2]:
        print( "Введите то, что нужно сделать с параметром:")
        print("Введите 1, если хотите прибавить очки к параметру;")
        print("Введите 2, если хотите отнять очки от параметра.")
        parametrs = input("Введите нужную цифру: ")
        if parametrs == "1":
            k = int(input("Введите количество очков: "))
            if point >= 0:
                point -= k
                m = Characters["Ловкость"]
                Characters["Ловкость"] = m + k
            else:
                print("Превышено максимально допустимое значение очков.")
        elif parametrs == "2":
            k = int(input("Введите количество очков: "))
            point += k
            Characters[0] -= k
        else:
            print("Введены неправильные данные, повторите попытку.")
        print("Оставшееся количество очков: ", point)
    elif hero == key[3]:
        print( "Введите то, что нужно сделать с параметром:")
        print("Введите 1, если хотите прибавить очки к параметру;")
        print("Введите 2, если хотите отнять очки от параметра.")
        parametrs = input("Введите нужную цифру: ")
        if parametrs == "1":
            k = int(input("Введите количество очков: "))
            if point >= 0:
                point -= k
                m = Characters["Мудрость"]
                Characters["Мудрость"] = m + k
            else:
                print("Превышено максимально допустимое значение очков.")
        elif parametrs == "2":
            k = int(input("Введите количество очков: "))
            point += k
            Characters[0] -= k
        else:
            print("Введены неправильные данные, повторите попытку.")
        print("Оставшееся количество очков: ", point)
    else:
        print("Вы ввели несуществующий параметр.")
    print("Сейчас ваши характеристики таковы:, ",Characters)
    z = input("Хотите продолжить распределение очков? (Да или Нет) ")
    if z == "Нет":
        ol = 1
        print("Распределение характеристик закончено.\n" + "Все непотраченные очки сохранятся.")
    elif z == "Да":
        print("Давайте продолжим.")
    else:
        print("Введено что-то неверное...")'''
'''parents = {"Кирилл":"Семён","Игорь":"Костантин"}
i = int(input("Введите номер человека. ")) - 1
print ("У ",  list(parents.keys())[i], "отец - это ", list(parents.values())[i])'''
'''import sys
def open_file(file_name, mode):
    try:
        the_file = open(file_name, mode, encoding='utf-8')
    except IOError as e:
        print("Невозможно открыть файл", file_name, ". Работа программы будет завершена.\n", e)
        input("\n\nНажмите Enter, чтобы выйти.")
        sys.exit()
    else:
        return the_file
def next_line(the_file):
    line = the_file.readline()
    line = line.replace("/", "\n")
    return line
def next_block(the_file):
    category = next_line(the_file)
    question = next_line(the_file)
    answers = []
    for i in range(4):
        answers.append(next_line(the_file))
        correct = next_line(the_file)
        if correct:
            correct = correct[0]
            explanation = next_line(the_file)
    return category, question, answers, correct, explanation
def welcome(title):
    print("\t\tДобро пожаловать в игру 'Викторина'!\n")
    print("\t\t", title, "\n")
def main():
    trivia_file = open_file("", "r")
    title = next_line(trivia_file)
    welcome(title)
    score = 0
    while category:
        print(category)
        print(question)
        for i in range(4):
            print("\t", i + 1, "-", answers[i])
            answers = input("Ваш ответ: ")
            if answer == correct:
                print("\nДа!", end = " ")
                score += 1
            else:
                print("\nНет.", end=" " )
            print(explanation)
            print("Счёт:", score, "\n\n")
        category, question, answers, correct, explanation = next_block(trivia_file)
    trivia_file.close()
    print("Это был последний вопрос!")
    print("На вашем счету: ", score)
main()
input("\n\nНажмите Enter, чтобы выйти.")'''
"""class Zooferma(object):
    def __init__(self, name, hunger = 0, boredom = 0):
        print("Появилась на свет новая зверюшка!")
        self.__name = name
        self.__hunger = hunger
        self.__boredom = boredom
    def __pass_time(self):
        self.__hunger += 1
        self.__boredom += 1
    @property
    def mood(self):
        unhappiness = self.__hunger + self.__boredom
        if unhappiness < 5:
            m = "Прекрасно"
        elif 5 <= unhappiness <= 10:
            m = "Неплохо"
        elif 11 <= unhappiness <= 15:
            m = "Не сказать чтобы хорошо"
        else:
            m = "Ужасно"
        return m
    @property
    def name(self):
        return self.__name
    @name.setter
    def name(self, new_name):
        if new_name == "":
            print("Имя зверюшки не может быть пустой строкой.")
        else:
            self.__name = new_name
            print("Имя успешно изменено.")
    def talk(self):
        print("Меня зовут", self.__name, ", и сейчас чувствую себя", self.mood )
        self.__pass_time()
    def eat(self, food = 4):
        print("Мррр... Спасибо!")
        self.__hunger -= food
        if self.__hunger < 0:
            self.__hunger = 0
        self.__pass_time()
    def play(self, fun = 4):
        print("Уиии!")
        self.__boredom -= fun
        if self.__boredom < 0:
            self.__boredom = 0
        self.__pass_time()
def main():
    crit_name = input("Как вы назовёте свою зверюшку? ")
    crit = Zooferma(crit_name)
    choice = None
    while choice != "0":
        print \
        ('''
        Моя зверюшка
        0 - выйти
        1 - Узнать о самочувствии зверюшки
        2 - Покормить зверюшку
        3 - Поиграть с зверюшкой
        4 - Поменять имя зверюшки
        ''')
        choice = input("Ваш выбор: ")
        print()
        if choice == "0":
            print("До свидания.")
        elif choice == "1":
            crit.talk()
        elif choice == "2":
            crit.eat()
        elif choice == "3":
            crit.play()
        else:
            print("Извините, в меню нет пункта", choice)
main()
input("\n\nНажмите Enter, чтобы выйти. ")"""
'''import cards, games
class BJ_Card(cards.Card):
    AGE_VALUE = 1
    @property
    def value(self):
        if self.is_face_up:
            v = BJ_Card.RANKS.index(self.rank) + 1
            if v > 10:
                v = 10
        else:
            v = None
        return v
class BJ_Deck(cards.Deck):
    def populate(self):
        for suit in BJ_Card.SUITS:
            for rank in BJ_Card.RANKS:
                self.cards.append(BJ_Card(rank, suit))
class BJ_Hand(cards.Hand):
    def __init__(self, name):
        super(BJ_Hand, self).__init__()
        self.name = name
    def __str__(self):
        rep = self.name + ":\t" + super(BJ_Hand, self).__str__()
        if self.total:
            rep += "(" + str(self.total) + ")"
        return rep
    @property
    def total(self):
        for card in self.cards:
            if not card.value:
                return None
        t = 0
        for card in self.cards:
            t += card.value
        contains_ace = False
        for card in self.cards:
            if card.value == BJ_Card.ACE_VALUE:
                contains_ace = True
        if contains_ace and t <= 11:
            t += 10
        return t
    def is_busted(self):
        return self.total > 21
class BJ_Player(BJ_Hand):
    def is_hitting(self):
        response = games.ask_yes_no("\n" + self.name + ", будете брать ещё карты? (Y/N): ")
        return response == "y"
    def bust(self):
        print(self.name, "перебрал.")
        self.lose()
    def lose(self):
        print(self.name, "проиграл.")
    def win(self):
        print(self.name, "выиграл.")
    def push(self):
        print(self.name, "сыграл с компьютером вничью.")
class BJ_Dealer(BJ_Hand):
    def is_hitting(self):
        return self.total < 17
    def bust(self):
        print(self.name, "перебрал.")
    def flip_first_card(self):
        first_card = self.cards[0]
        first_card.flip()
class BJ_Game(object):
    def __init__(self, names):
        self.players = []
        for name in names:
            player = BJ_Player(name)
            self.players.append(player)
        self.dealer = BJ_Dealer("Dealer")
        self.deck = BJ_Deck()
        self.deck.populate()
        self.deck.shuffle()
    @property
    def still_playing(self):
        sp = []
        for player in self.players:
            if not player in self.players:
                sp.append(player)
        return sp
    def __additional_cards(self, player):
        while not player.is_busted() and player.is_hitting():
            self.deck.deal([player])
            print(player)
            if player.is_busted():
                player.bust()
    def play(self):
        self.deck.deal(self.players + [self.dealer], per_hand = 2)
        self.dealer.flip_first_card()
        for player in self.players:
            print(player)
        print(self.dealer)
        for player in self.players:
            self.__additional_cards(player)
        self.dealer.flip_first_card()
        if not self.still_playing:
            print(self.dealer)
        else:
            print(self.dealer)
            self.__additional_cards(self.dealer)
            if self.dealer.is_busted():
                for player in self.still_playing:
                    player.win()
            else:
                for player in self.still_playing:
                    if player.total > self.dealer.total:
                        player.win()
                    elif player.total < self.dealer.total:
                        player.lose()
                    else:
                        player.push()
        for player in self.players:
            player.clear()
        self.dealer.clear()
def main():
    print("\t\tДобро пожаловать за игровой стол Блэк Джэка!\n")
    names = []
    number = games.ask_number("Сколько всего игроков(1-7): ", low = 1, high = 8)
    for i in range(number):
        name = input("Введите имя игрока: ")
        names.append(name)
        print()
    game = BJ_Game(names)
    again = None
    while again != "n":
        game.play()
        again = games.ask_yes_no("\nХотите сыграть ещё раз? ")
        main()
    input("\n\nНажмите Enter, чтобы выйти.")'''
