from aiogram import Bot, Dispatcher, types, executor
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
import random
TOKEN_API = ''
bot = Bot(TOKEN_API)
dp = Dispatcher(bot)
number = 0
async def on_startup(_):
    print("Бот успешно запущен!")
def get_inline_keyboard() -> InlineKeyboardMarkup:
    ikb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton("Increase", callback_data="btn_increase"), InlineKeyboardButton("Decrease", callback_data="btn_decrease")],
        [InlineKeyboardButton("Random value", callback_data="btn_random")]
    ])
    return ikb
@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message) -> None:
    await message.answer(f"The current number is {number}",
                         reply_markup=get_inline_keyboard())
@dp.callback_query_handler(lambda callback_query: callback_query.data.startswith("btn"))
async def ikb_cb_handler(callback: types.CallbackQuery) -> None:
    global number
    if callback.data == "btn_increase":
        number += 1
        await callback.message.edit_text(f"The current number is {number}", reply_markup=get_inline_keyboard())
    elif callback.data == "btn_decrease":
        number -= 1
        await callback.message.edit_text(f"The current number is {number}", reply_markup=get_inline_keyboard())
    elif callback.data == "btn_random":
        number = random.randint(1, 100)
        await callback.message.edit_text(f"The current number is {number}", reply_markup=get_inline_keyboard())
    else:
        1/0
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True, on_startup=on_startup)
