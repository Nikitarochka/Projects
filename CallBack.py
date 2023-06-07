from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
import random
HELP_COMMANDS = '''
<b>/start</b> - <em>запускает бота</em>
<b>/help</b> - <em>вызывает список команд</em>
<b>/random</b> - <em>присылает рандомную картинку</em>
<b>/location</b> - <em>отправляет геопозицию</em>
'''
flag = False
flag1 = False
PHOTOS = ["https://vsegda-pomnim.com/uploads/posts/2022-04/1649671638_101-vsegda-pomnim-com-p-kotenok-s-tsvetami-foto-112.jpg",
          "https://pazlyigra.ru/uploads/posts/2022-12/479679246.jpg",
          "https://funart.pro/uploads/posts/2021-07/1625869424_12-funart-pro-p-oboi-s-kotikami-zhivotnie-krasivo-foto-17.jpg",
          "https://funart.pro/uploads/posts/2021-07/1627478074_14-funart-pro-p-kotik-s-rozoi-zhivotnie-krasivo-foto-17.jpg",
          "https://chudo-prirody.com/uploads/posts/2021-08/1628831756_77-p-foto-kotiki-obnimayutsya-82.jpg"]
CAPTION = ["Котик 1.", "Котик 2.", "Котик 3.", "Котик 4.", "Котик 5."]
pht_lig = dict(zip(PHOTOS, CAPTION))
TOKEN_API = '5644718551:AAERzY8iR9rK9PYJqH13tzXcS_dmt4Jv8XY'
bot = Bot(TOKEN_API)
dp = Dispatcher(bot)
random_photo = random.choice(list(pht_lig.keys()))
async def on_startup(_):
    print("Бот успешно запущен!")
ikb = InlineKeyboardMarkup(row_width=2)
ikb1 = InlineKeyboardButton(text="Да", callback_data="like")
ikb2 = InlineKeyboardButton(text="Нет", callback_data="dislike")
ikb3 = InlineKeyboardButton(text="Следующая фотография", callback_data="next")
ikb4 = InlineKeyboardButton(text="Главное меню", callback_data="main")
ikb.add(ikb1, ikb2).insert(ikb3).add(ikb4)
kb = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
kb1 = KeyboardButton(text="/help")
kb2 = KeyboardButton(text="/random")
kb3 = KeyboardButton(text="/location")
kb4 = KeyboardButton('❤')
kb.add(kb1, kb2).add(kb3,kb4)
kbr = ReplyKeyboardRemove()
kb_photo = ReplyKeyboardMarkup(resize_keyboard=True)
bp1 = KeyboardButton(text="Рандом")
bp2 = KeyboardButton(text="Помощь")
kb_photo.add(bp1, bp2)
@dp.message_handler(commands="random")
async def random_photos(message: types.Message):
    global random_photo
    await bot.send_photo(message.from_user.id, photo=random_photo, caption=pht_lig[random_photo] + " Тебе нравится фотография?", reply_markup=ikb)
    await message.delete()
@dp.message_handler(commands="start")
async def starting(message: types.Message):
    await message.answer("Приветствую!", reply_markup=kb)
    await message.delete()
@dp.message_handler(commands="help")
async def helping(message: types.Message):
    await message.answer(text=HELP_COMMANDS, reply_markup=kb, parse_mode="HTMl")
    await message.delete()
@dp.message_handler(commands="random")
async def delete_menu(message: types.Message):
    global random_photo
    message.answer("Рандомная фотка", reply_markup=kbr)
    await random_photos(message)
@dp.message_handler(commands= ['location'])
async def bot_location(message: types.Message):
        await bot.send_location(chat_id=message.from_user.id,latitude=59.952937,longitude=30.233413, reply_markup=kb)
        await message.delete()
@dp.message_handler()
async def send_random_emoji(message: types.Message):
    if message.text == "❤":
        await bot.send_message(chat_id=message.from_user.id,
                               text="<em>Смотри какой смешной клоун ❤</em>",
                               parse_mode="HTML")
        await bot.send_sticker(chat_id=message.from_user.id,
                               sticker="CAACAgIAAxkBAAEIx55kTVXL4zhlhyg7PrDbirGzgo3mGQACgAEAAuW5CBoh6SYYor2_7y8E")
@dp.callback_query_handler()
async def photo_callback(callback: types.CallbackQuery):
    global random_photo
    global flag
    global flag1
    if callback.data == "like":
        if not flag:
            await callback.answer(show_alert=True, text="Тебе понравилась фотография")
            flag = True
        else:
            await callback.answer(show_alert=True, text="Вы уже нажимали кнопку")
    elif callback.data == "dislike":
        if not flag:
            await callback.answer(show_alert=True, text="Тебе не понравилась фотография")
            flag = True
        else:
            await callback.answer(show_alert=True, text="Вы уже нажимали кнопку")

    elif callback.data == "main":
        await callback.message.answer(text="Добро пожаловать в главное меню",
                                      reply_markup=kb)
        await callback.message.delete()
        await callback.answer()
    else:
        flag = False
        flag1 = False
        random_photo = random.choice(list(filter(lambda x: x != random_photo, list(pht_lig.keys()))))
        await bot.send_photo(callback.message.chat.id, photo=random_photo,
                             caption=pht_lig[random_photo] + " Тебе нравится фотография?", reply_markup=ikb)
        await callback.answer()
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True, on_startup=on_startup)

