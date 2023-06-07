from tkinter import *
win = Tk()
win.title("Моё первое графическое приложение)")
win.geometry("400x400")
win.minsize(200, 200)
win.maxsize(800, 800)
win.resizable(True, True)
photo = PhotoImage(file = 'png-clipart-dice-game-dice-game-graphy-playing-card-dice-game-gambling.png')
win.iconphoto(False, photo)
win.config(bg = "white")
'''lbl = Label(win, text = "Привет!",
               bg = "red",
               fg = "white",
               font = ("Times new roman", 15, "bold"),
               width = 10,
               height = 5,
               padx = 20, #отступы от краёв
               pady = 30,
               anchor = "center", #расположение относительно сторон света
               relief = RAISED, #обтекание границ
               bd = 10, #задаёт ширину границ в пикселях
               justify = LEFT #прижимает текст к какой-либо границе
               )
lbl.pack()'''
count = 0
mnoge = 1
coun = 0
def counter():
    global count
    count += 1
    btn2["text"] = f"Счёт: {count}"
def mnog():
    global mnoge
    mnoge *= 2
    btn3["text"] = f"Счёт: {mnoge}"
def kvadrat():
    global coun
    coun += 1
    kdadra = coun*coun
    btn4["text"] = f"Счёт: {kdadra}"
kotik = 0
def kot():
    global kotik
    kotik += 1
    if kotik % 2 == 0:
        t = "Разморозка"
        btn2["state"] = NORMAL
        btn3["state"] = NORMAL
        btn4["state"] = NORMAL
    else:
        t = "Заморозка"
        btn2["state"] = DISABLED
        btn3["state"] = DISABLED
        btn4["state"] = DISABLED
    btn1["text"] = f"{t}"
btn1 = Button(win, text="Заморозь все кнопки", command=kot, state = NORMAL)
btn2 = Button(win, text="Нажми на кнопку", command=counter, state = NORMAL)
btn3 = Button(win, text="Нажми на кнопку", command=mnog, state = NORMAL)
btn4 = Button(win, text="Нажми на кнопку", command=kvadrat, state = NORMAL)
btn1.grid(row = 0, column = 2, columnspan = 3, sticky = "n")
btn2.grid(row = 1, column = 2, sticky = "n")
btn3.grid(row = 2, column = 2, sticky = "n")
btn4.grid(row = 3, column = 2, sticky = "n")
win.mainloop()
#show="символ" - маскирует введенное значение в Entry(в поле ввода)

