from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
TOKEN_API = '5644718551:AAERzY8iR9rK9PYJqH13tzXcS_dmt4Jv8XY'
bot = Bot(TOKEN_API)
dp = Dispatcher(bot)
count = 1
async def on_startup(_):
        print("Бот успешно запущен!")
'''
@dp.message_handler(commands=["count"])
async def send_random_letter(message: types.Message):
        global count
        await message.answer(f"COUNT:{count}")
        count += 1
@dp.message_handler()
async def send_random_letter(message: types.Message):
        if "0" in message.text:
                await message.answer("YES")
        elif "1" in message.text:
                await message.answer("NO")
@dp.message_handler()
async def send_random_letter(message: types.Message):
        await message.reply(random.choice(string.ascii_letters))
@dp.message_handler(commands=["give"])
async def send_random_emoji(message: types.Message):
        await message.reply("Смотри какой смешной клоун ❤")
        await bot.send_sticker(message.from_user.id,
                               sticker="CAACAgIAAxkBAAEIx55kTVXL4zhlhyg7PrDbirGzgo3mGQACgAEAAuW5CBoh6SYYor2_7y8E")
@dp.message_handler()
async def send_random_emoji(message: types.Message):
        if "❤" in message.text:
                await message.reply("🖤")
@dp.message_handler(commands="help")
async def send_commands(message: types.Message):
        await message.reply(text=HELP_COMMANDS, parse_mode = "HTML")
@dp.message_handler()
async def send_random_emoji(message: types.Message):
        global count
        if "✅" in message.text:
                await message.reply(f"Count:{count}")
                count += 1
@dp.message_handler(content_types= ['sticker'])
async def send_random_emoji(message: types.Message):
        await message.reply(message.sticker.file_id)
'''

'''kb = ReplyKeyboardMarkup(resize_keyboard=True)
kb2 = ReplyKeyboardRemove()
b1 = KeyboardButton('/help')
b2 = KeyboardButton('/pictures')
b3 = KeyboardButton('/location')
b4 = KeyboardButton('/start')
b5 = KeyboardButton('❤')
kb.add(b1).insert(b2).add(b3).insert(b4).add(b5)
HELP_COMMANDS =''' '''
<b>/start</b> - <em>запускает бота</em>
<b>/help</b> - <em>вызывает список команд</em>
<b>/pictures</b> - <em>присылает картинку</em>
<b>/location</b> - <em>отправляет геопозицию</em>
<b>❤</b> - <em>отправляет смешной стикер</em>'''
'''@dp.message_handler(commands=["start"])
async def send_random_letter(message: types.Message):
        await bot.send_message(chat_id=message.from_user.id,
                               text="Привет, пользователь!",
                               parse_mode="HTML",
                               reply_markup=kb)
@dp.message_handler(commands= ['help'])
async def bot_answer(message: types.Message):
        await bot.send_message(chat_id=message.from_user.id,
                               text=HELP_COMMANDS,
                               parse_mode="HTML",
                               reply_markup=kb2)
        await message.delete()
@dp.message_handler(commands= ['pictures'])
async def bot_pictures(message: types.Message):
        await bot.send_photo(chat_id=message.from_user.id,
                             photo="https://vsegda-pomnim.com/uploads/posts/2022-04/1649282154_4-vsegda-pomnim-com-p-plyazhi-brazilii-foto-4.jpg")
        await message.delete()
@dp.message_handler(commands= ['location'])
async def bot_location(message: types.Message):
        await bot.send_location(chat_id=message.from_user.id,latitude=59.952937,longitude=30.233413)
        await message.delete()
@dp.message_handler()
async def send_random_emoji(message: types.Message):
    if message.text == "❤":
        await bot.send_message(chat_id=message.from_user.id,
                               text="<em>Смотри какой смешной клоун ❤</em>",
                               parse_mode="HTML")
        await bot.send_sticker(chat_id=message.from_user.id,
                               sticker="CAACAgIAAxkBAAEIx55kTVXL4zhlhyg7PrDbirGzgo3mGQACgAEAAuW5CBoh6SYYor2_7y8E")'''
ikbm = InlineKeyboardMarkup(row_width=2)
ikb1 = InlineKeyboardButton(text="Youtube",
                           url = "https://www.youtube.com/watch?v=5_EHfHbzUCo&t=33s")
ikb2 = InlineKeyboardButton(text="ВКонтакте",
                           url = "https://vk.com/nikitarochka")
ikbm.add(ikb1).add(ikb2)
@dp.message_handler(commands="start")
async def send_kb(message: types.Message):
    await bot.send_message(chat_id=message.from_user.id,
                           text = "Привет, мир!",
                           reply_markup=ikbm)
if __name__ == "__main__":
        executor.start_polling(dp, skip_updates=True, on_startup=on_startup)



