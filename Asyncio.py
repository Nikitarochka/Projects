import asyncio
import hashlib
from aiogram import executor, Bot, types, Dispatcher
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.types import InlineQueryResultArticle, InputTextMessageContent
from aiogram.utils.exceptions import BotBlocked
from aiogram.utils.callback_data import CallbackData
from CallBack import TOKEN_API
'''
async def fnc1() -> None:
    for i in 6
    await asyncio.sleep(2)
    print("Hello")
async def fnc2() -> None:
    await asyncio.sleep(1)
    print("Buy")
async def main() -> None:
    task1 = asyncio.create_task(fnc1())
    task2 = asyncio.create_task(fnc2())
    await task1
    await task2
'''
'''async def fnc1() -> None:
    n = 0
    while True:
        await asyncio.sleep(1)
        n += 1
        if n % 3 != 0:
            print(f"Прошло {n} секунд")
async def fnc2() -> None:
    while True:
        await asyncio.sleep(3)
        print("Прошло ещё 3 секунды")
async def main() -> None:
    task1 = asyncio.create_task(fnc1())
    task2 = asyncio.create_task(fnc2())
    await task1
    await task2
if __name__ == "__main__":
    asyncio.run(main())'''
cb = CallbackData('ikb', 'action')
bot = Bot(TOKEN_API)
dp = Dispatcher(bot)
def get_ikb() -> InlineKeyboardMarkup:
    ikb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton('Button 1', callback_data=cb.new('push_1'))],
        [InlineKeyboardButton('Button 2', callback_data=cb.new('push_2'))]
    ])
    return ikb
@dp.message_handler(commands=["start"])
async def cmd_start(message: types.Message) -> None:
    await asyncio.sleep(3)
    await message.answer(text='Welcome to my Aiogram Bot!', reply_markup=get_ikb())
@dp.errors_handler(exception=BotBlocked)
async def error_bot_blocked_handler(update: types.Update, exception: BotBlocked) -> bool:
    print("Бот заблокирован!")
    return True
@dp.callback_query_handler(cb.filter(action='push_1'))
async def push_first_cb_handler(callback: types.CallbackQuery) -> None:
    await callback.answer("Hello!")
@dp.callback_query_handler(cb.filter(action='push_2'))
async def push_second_cb_handler(callback: types.CallbackQuery) -> None:
    await callback.answer("World!")
@dp.inline_handler()
async def inline_echo(inline_query: types.InlineQuery) -> None:
    text = inline_query.query or "Echo"
    input_content = InputTextMessageContent(text)
    result_id: str = hashlib.md5(text.encode()).hexdigest()
    item = InlineQueryResultArticle(
        input_message_content=input_content,
        id=result_id,
        title="Echo!!!"
    )
    await bot.answer_inline_query(inline_query_id=inline_query.id,
                                  results=[item],
                                  cache_time=1)
if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)