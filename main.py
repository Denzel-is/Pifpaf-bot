# main.py
import os
import logging
import io
from dataclasses import dataclass
from pathlib import Path
from datetime import time, datetime, timedelta
from typing import Optional, Tuple, List, Dict

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
    BotCommand,
    BotCommandScopeAllPrivateChats,
    BotCommandScopeAllGroupChats,
)
from telegram.constants import ChatType, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)

import psycopg
from psycopg.rows import dict_row

# =========================
# ХАРДКОД НАСТРОЕК (ЗАМЕНИ ПРИ НУЖДЕ)
# =========================
TELEGRAM_BOT_TOKEN = "8289606829:AAGYJ1HeKinYItnhuIbD7-MrpMtYUZJmcy4"
DATABASE_URL = "postgresql://neondb_owner:npg_0Dgy3HuIrEMw@ep-calm-band-ado557e5-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
ADMIN_TG_ID = 940253198  # целое число (без кавычек)

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("bot")

# -----------------------
# Database
# -----------------------
def get_db_connection():
    dsn = DATABASE_URL
    if not dsn:
        raise RuntimeError("DATABASE_URL is empty")
    return psycopg.connect(dsn, row_factory=dict_row)

def init_db() -> None:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                tg_id BIGINT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                registered_at TIMESTAMPTZ NOT NULL
            );
            """
        )
        cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS username TEXT;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schedule_intervals (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                weekday INTEGER NOT NULL,          -- 0 = Пн ... 6 = Вс
                start_time TEXT NOT NULL,          -- "HH:MM"
                end_time TEXT NOT NULL             -- "HH:MM"
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_group (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                chat_id BIGINT NOT NULL
            );
            """
        )
        conn.commit()

def upsert_user_on_event(update: Update) -> Optional[int]:
    if not update.effective_user:
        return None
    tg = update.effective_user
    tg_id = tg.id
    name = tg.full_name or tg.username or "User"
    username = tg.username

    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, name, username FROM users WHERE tg_id = %s", (tg_id,))
        row = cur.fetchone()
        if row:
            need = (row["name"] != name) or (row.get("username") != username)
            if need:
                cur.execute("UPDATE users SET name=%s, username=%s WHERE id=%s", (name, username, row["id"]))
                conn.commit()
            return int(row["id"])
        cur.execute(
            "INSERT INTO users (tg_id, name, username, registered_at) VALUES (%s, %s, %s, NOW()) RETURNING id",
            (tg_id, name, username),
        )
        new_id = int(cur.fetchone()["id"])
        conn.commit()
        return new_id

def get_or_create_user_id(tg_id: int, name: Optional[str] = None, username: Optional[str] = None) -> Optional[int]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM users WHERE tg_id = %s", (tg_id,))
        row = cur.fetchone()
        if row:
            return int(row["id"])
        if name is None:
            return None
        cur.execute(
            "INSERT INTO users (tg_id, name, username, registered_at) VALUES (%s, %s, %s, NOW()) RETURNING id",
            (tg_id, name, username),
        )
        new_id = cur.fetchone()["id"]
        conn.commit()
        return int(new_id)

def get_user_by_tg_id(tg_id: int):
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE tg_id = %s", (tg_id,))
        return cur.fetchone()

def get_users_with_classes_on_weekday(weekday: int) -> List[dict]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT u.id, u.tg_id, u.name, u.username
            FROM users u
            JOIN schedule_intervals si ON si.user_id = u.id
            WHERE si.weekday = %s
            ORDER BY u.name
            """,
            (weekday,),
        )
        return cur.fetchall() or []

def set_day_intervals(user_id: int, weekday: int, intervals: List[Tuple[str, str]]) -> None:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM schedule_intervals WHERE user_id = %s AND weekday = %s", (user_id, weekday))
        for start, end in intervals:
            cur.execute(
                "INSERT INTO schedule_intervals (user_id, weekday, start_time, end_time) VALUES (%s, %s, %s, %s)",
                (user_id, weekday, start, end),
            )
        conn.commit()

def fetch_day_intervals(user_id: int, weekday: int) -> List[Tuple[str, str]]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT start_time, end_time FROM schedule_intervals WHERE user_id = %s AND weekday = %s ORDER BY start_time",
            (user_id, weekday),
        )
        rows = cur.fetchall() or []
        return [(r["start_time"], r["end_time"]) for r in rows]

def delete_interval_by_id(user_id: int, interval_id: int) -> None:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM schedule_intervals WHERE id = %s AND user_id = %s", (interval_id, user_id))
        conn.commit()

def fetch_all_intervals_joined() -> List[dict]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT u.name, u.username, u.tg_id, si.weekday, si.start_time, si.end_time
            FROM users u
            JOIN schedule_intervals si ON si.user_id = u.id
            ORDER BY si.weekday, u.name, si.start_time
            """
        )
        return cur.fetchall() or []

def fetch_day_interval_rows_with_ids(user_id: int, weekday: int) -> List[dict]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, start_time, end_time FROM schedule_intervals WHERE user_id = %s AND weekday = %s ORDER BY start_time",
            (user_id, weekday),
        )
        return cur.fetchall() or []

def get_group_chat_id() -> Optional[int]:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT chat_id FROM bot_group WHERE id = 1")
        row = cur.fetchone()
        return int(row["chat_id"]) if row else None

def set_group_chat_id(chat_id: int) -> None:
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO bot_group (id, chat_id) VALUES (1, %s) "
            "ON CONFLICT(id) DO UPDATE SET chat_id = EXCLUDED.chat_id",
            (chat_id,),
        )
        conn.commit()

# -----------------------
# States
# -----------------------
ASK_NAME, ASK_DAY_HAS, ASK_DAY_TIME = range(3)
EDIT_MENU, EDIT_SELECT_DAY, EDIT_DAY_SCREEN, EDIT_AWAIT_INPUT = range(20, 24)

WEEKDAYS_RU = [
    "Понедельник",
    "Вторник",
    "Среда",
    "Четверг",
    "Пятница",
    "Суббота",
    "Воскресенье",
]

@dataclass
class RegState:
    weekday_index: int = 0
    name: Optional[str] = None

# -----------------------
# Utilities
# -----------------------
def parse_time_range(text: str) -> Optional[Tuple[str, str]]:
    try:
        cleaned = text.replace(" ", "")
        parts = cleaned.split("-")
        if len(parts) != 2:
            return None
        start_s, end_s = parts
        s_h, s_m = [int(x) for x in start_s.split(":")]
        e_h, e_m = [int(x) for x in end_s.split(":")]
        if not (0 <= s_h < 24 and 0 <= s_m < 60 and 0 <= e_h < 24 and 0 <= e_m < 60):
            return None
        if (e_h, e_m) <= (s_h, s_m):
            return None
        return (f"{s_h:02d}:{s_m:02d}", f"{e_h:02d}:{e_m:02d}")
    except Exception:
        return None

def parse_intervals(text: str) -> Optional[List[Tuple[str, str]]]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None
    result: List[Tuple[str, str]] = []
    last_end: Optional[Tuple[int, int]] = None
    for p in parts:
        rng = parse_time_range(p)
        if not rng:
            return None
        s, e = rng
        s_h, s_m = [int(x) for x in s.split(":")]
        e_h, e_m = [int(x) for x in e.split(":")]
        if last_end and (s_h, s_m) < last_end:
            return None
        result.append((s, e))
        last_end = (e_h, e_m)
    return result

def merge_and_validate_intervals(existing: List[Tuple[str, str]], new_intervals: List[Tuple[str, str]]) -> Optional[List[Tuple[str, str]]]:
    def to_min(t: str) -> int:
        h, m = map(int, t.split(":"))
        return h * 60 + m
    all_intervals = existing + new_intervals
    all_intervals.sort(key=lambda x: to_min(x[0]))
    merged: List[Tuple[str, str]] = []
    for s, e in all_intervals:
        if not merged:
            merged.append((s, e)); continue
        _, last_e = merged[-1]
        if to_min(s) <= to_min(last_e):
            return None
        merged.append((s, e))
    return merged

def time_str_to_time(t: str) -> time:
    h, m = [int(x) for x in t.split(":")]
    local_tz = datetime.now().astimezone().tzinfo
    return time(hour=h, minute=m, tzinfo=local_tz)

def sub_minutes(t: time, minutes: int) -> Tuple[time, int]:
    dt = datetime(2000, 1, 2, t.hour, t.minute) - timedelta(minutes=minutes)
    shift = -1 if dt.day < 2 else 0
    local_tz = datetime.now().astimezone().tzinfo
    return time(dt.hour, dt.minute, tzinfo=local_tz), shift

def mention_html(name: str, tg_id: int, username: Optional[str]) -> str:
    if username:
        return f"@{username}"
    safe_name = (name or "User").replace("<", " ").replace(">", " ")
    return f'<a href="tg://user?id={tg_id}">{safe_name}</a>'

# -----------------------
# Registration
# -----------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user_on_event(update)
    user = update.effective_user
    existing = get_user_by_tg_id(user.id)
    if existing:
        await update.effective_message.reply_text(
            f"Вы уже зарегистрированы как {existing['name']}.\n"
            "Используйте /edit для редактирования расписания или /status, чтобы посмотреть его."
        )
        return ConversationHandler.END
    await update.effective_message.reply_text("Привет! Введите ваше имя для регистрации:")
    context.user_data["reg_state"] = RegState()
    return ASK_NAME

async def ask_name(update: Update, context: ContextTypes.DEFAULT_TYPE):
    name = update.effective_message.text.strip()
    if not name:
        await update.effective_message.reply_text("Имя не должно быть пустым. Введите имя:")
        return ASK_NAME
    context.user_data["reg_state"].name = name
    tg_user = update.effective_user
    user_id = get_or_create_user_id(tg_user.id, name, tg_user.username)
    context.user_data["user_id"] = user_id
    await update.effective_message.reply_text("Отлично! Теперь для каждого дня скажите, есть ли пары (да/нет).")
    await update.effective_message.reply_text(f"{WEEKDAYS_RU[0]}: пары есть? (да/нет)")
    return ASK_DAY_HAS

async def ask_day_has(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.effective_message.text.strip().lower()
    st: RegState = context.user_data["reg_state"]
    user_id: int = context.user_data["user_id"]
    if text not in ("да", "нет"):
        await update.effective_message.reply_text("Ответьте 'да' или 'нет'.")
        return ASK_DAY_HAS
    if text == "нет":
        set_day_intervals(user_id, st.weekday_index, [])
        st.weekday_index += 1
        if st.weekday_index >= 7:
            await update.effective_message.reply_text("Регистрация завершена! Спасибо.")
            await rebuild_jobs(context)
            return ConversationHandler.END
        await update.effective_message.reply_text(f"{WEEKDAYS_RU[st.weekday_index]}: пары есть? (да/нет)")
        return ASK_DAY_HAS
    await update.effective_message.reply_text(
        "Введите интервал(ы): HH:MM-HH:MM, можно несколько через запятую.\nНапример: 08:00-08:50, 09:00-09:50"
    )
    return ASK_DAY_TIME

async def ask_day_time(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.effective_message.text.strip()
    st: RegState = context.user_data["reg_state"]
    user_id: int = context.user_data["user_id"]
    ivals = parse_intervals(text)
    if not ivals:
        await update.effective_message.reply_text("Неверный формат. Пример: 08:00-08:50 или 08:00-08:50, 10:00-10:50")
        return ASK_DAY_TIME
    set_day_intervals(user_id, st.weekday_index, ivals)
    st.weekday_index += 1
    if st.weekday_index >= 7:
        await update.effective_message.reply_text("Регистрация завершена! Спасибо.")
        await rebuild_jobs(context)
        return ConversationHandler.END
    await update.effective_message.reply_text(f"{WEEKDAYS_RU[st.weekday_index]}: пары есть? (да/нет)")
    return ASK_DAY_HAS

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text("Операция отменена.")
    return ConversationHandler.END

# -----------------------
# Pretty UI helpers
# -----------------------
def kb_main_edit_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📅 Выбрать день", callback_data="edit_select_day")],
        [InlineKeyboardButton("📤 Экспорт моего расписания (текст)", callback_data="edit_export_me")],
        [InlineKeyboardButton("❌ Выйти", callback_data="edit_cancel")]
    ])

def kb_days() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(f"{i} — {day}", callback_data=f"edit_day_{i}")] for i, day in enumerate(WEEKDAYS_RU)]
    rows.append([InlineKeyboardButton("🔙 Назад", callback_data="edit_back_main")])
    return InlineKeyboardMarkup(rows)

def kb_day_actions(day_num: int, has_intervals: bool) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton("➕ Добавить интервал", callback_data=f"edit_add_time_{day_num}")]]
    if has_intervals:
        rows.append([InlineKeyboardButton("🗑 Удалить интервал", callback_data=f"edit_delete_menu_{day_num}")])
        rows.append([InlineKeyboardButton("🧹 Очистить день", callback_data=f"edit_clear_day_{day_num}")])
    rows.append([InlineKeyboardButton("🔙 К дням", callback_data="edit_select_day")])
    return InlineKeyboardMarkup(rows)

def format_intervals_list(intervals: List[Tuple[str, str]]) -> str:
    if not intervals:
        return "—"
    return "\n".join([f"• {s}-{e}" for s, e in intervals])

# -----------------------
# Edit flow
# -----------------------
async def edit_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user_on_event(update)
    user_row = get_user_by_tg_id(update.effective_user.id)
    if not user_row:
        await update.effective_message.reply_text("Вы не зарегистрированы. Используйте /start.")
        return ConversationHandler.END
    await update.effective_message.reply_html("🔧 <b>Редактирование расписания</b>\n\nВыберите действие:", reply_markup=kb_main_edit_menu())
    context.user_data.pop("await_input", None); context.user_data.pop("edit_selected_day", None)
    return EDIT_MENU

async def edit_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query: return
    await query.answer()
    upsert_user_on_event(update)
    data = query.data or ""
    user_row = get_user_by_tg_id(query.from_user.id)
    if not user_row:
        await query.edit_message_text("Вы не зарегистрированы. Используйте /start.")
        return ConversationHandler.END
    user_id = user_row["id"]

    if data == "edit_cancel":
        await query.edit_message_text("❌ Редактирование отменено.")
        context.user_data.clear()
        return ConversationHandler.END

    if data == "edit_back_main":
        await query.edit_message_text("🔧 <b>Редактирование расписания</b>\n\nВыберите действие:",
                                      reply_markup=kb_main_edit_menu(), parse_mode=ParseMode.HTML)
        context.user_data.pop("await_input", None); context.user_data.pop("edit_selected_day", None)
        return EDIT_MENU

    if data == "edit_export_me":
        lines = [f"📄 <b>Ваше расписание</b>, {user_row['name']}:"]
        for d in range(7):
            ivals = fetch_day_intervals(user_id, d)
            day_line = f"<b>{WEEKDAYS_RU[d]}:</b>"
            lines.append(f"{day_line} " + (", ".join([f"{s}-{e}" for s, e in ivals]) if ivals else "нет пар"))
        await query.edit_message_text("\n".join(lines), parse_mode=ParseMode.HTML, reply_markup=kb_main_edit_menu())
        return EDIT_MENU

    if data == "edit_select_day":
        await query.edit_message_text("📅 <b>Выберите день недели:</b>", reply_markup=kb_days(), parse_mode=ParseMode.HTML)
        return EDIT_SELECT_DAY

    if data.startswith("edit_day_"):
        day = int(data.split("_")[2])
        context.user_data["edit_selected_day"] = day
        ivals = fetch_day_intervals(user_id, day)
        await query.edit_message_text(
            f"📅 <b>{WEEKDAYS_RU[day]}</b>\n\nТекущие интервалы:\n{format_intervals_list(ivals)}",
            parse_mode=ParseMode.HTML, reply_markup=kb_day_actions(day, bool(ivals))
        )
        return EDIT_DAY_SCREEN

    if data.startswith("edit_clear_day_"):
        day = int(data.split("_")[-1])
        set_day_intervals(user_id, day, [])
        await rebuild_jobs(context)
        await query.edit_message_text(
            f"🧹 День <b>{WEEKDAYS_RU[day]}</b> очищен.\n\nТекущие интервалы:\n—",
            parse_mode=ParseMode.HTML, reply_markup=kb_day_actions(day, False)
        )
        return EDIT_DAY_SCREEN

    if data.startswith("edit_delete_menu_"):
        day = int(data.split("_")[-1])
        rows = fetch_day_interval_rows_with_ids(user_id, day)
        if not rows:
            await query.edit_message_text(
                f"❌ В <b>{WEEKDAYS_RU[day]}</b> нет интервалов.",
                parse_mode=ParseMode.HTML, reply_markup=kb_day_actions(day, False)
            ); return EDIT_DAY_SCREEN
        kb = [[InlineKeyboardButton(f"🗑 {r['start_time']}-{r['end_time']}", callback_data=f"edit_del_{day}_{r['id']}")] for r in rows]
        kb.append([InlineKeyboardButton("🔙 Назад", callback_data=f"edit_day_{day}")])
        await query.edit_message_text(f"🗑 <b>Удалить интервал из {WEEKDAYS_RU[day]}</b>:", parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb))
        return EDIT_DAY_SCREEN

    if data.startswith("edit_del_"):
        _, day_s, id_s = data.split("_", 2)
        day = int(day_s); interval_id = int(id_s)
        delete_interval_by_id(user_id, interval_id)
        await rebuild_jobs(context)
        ivals = fetch_day_intervals(user_id, day)
        await query.edit_message_text(
            f"✅ Интервал удалён.\n\n📅 <b>{WEEKDAYS_RU[day]}</b>\nТекущие интервалы:\n{format_intervals_list(ivals)}",
            parse_mode=ParseMode.HTML, reply_markup=kb_day_actions(day, bool(ivals))
        )
        return EDIT_DAY_SCREEN

    if data.startswith("edit_add_time_"):
        day = int(data.split("_")[-1])
        context.user_data["edit_selected_day"] = day
        context.user_data["await_input"] = "add_interval"
        await query.edit_message_text(
            "➕ <b>Добавить интервал</b>\n\nВведите <code>HH:MM-HH:MM</code> или несколько через запятую.\n"
            "Например: <code>08:00-08:50, 09:00-09:50</code>",
            parse_mode=ParseMode.HTML
        )
        return EDIT_AWAIT_INPUT

    return ConversationHandler.END

async def edit_text_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.user_data.get("await_input"): return
    upsert_user_on_event(update)
    user_row = get_user_by_tg_id(update.effective_user.id)
    if not user_row:
        await update.effective_message.reply_text("Вы не зарегистрированы. /start")
        context.user_data.pop("await_input", None)
        return ConversationHandler.END
    user_id = user_row["id"]
    day = context.user_data.get("edit_selected_day")
    if day is None:
        await update.effective_message.reply_text("Не выбран день. /edit")
        context.user_data.pop("await_input", None)
        return ConversationHandler.END

    new_ivals = parse_intervals(update.effective_message.text.strip())
    if not new_ivals:
        await update.effective_message.reply_text("❌ Неверный формат. Пример: 08:00-08:50 или 08:00-08:50, 10:00-10:50")
        return EDIT_AWAIT_INPUT

    existing = fetch_day_intervals(user_id, day)
    merged = merge_and_validate_intervals(existing, new_ivals)
    if not merged:
        await update.effective_message.reply_text("⚠️ Пересечение интервалов. Убедись, что они не накладываются.")
        return EDIT_AWAIT_INPUT

    set_day_intervals(user_id, day, merged)
    await rebuild_jobs(context)
    context.user_data.pop("await_input", None)
    ivals = fetch_day_intervals(user_id, day)
    await update.effective_message.reply_html(
        f"✅ Добавлено в <b>{WEEKDAYS_RU[day]}</b>.\n\nТекущие интервалы:\n{format_intervals_list(ivals)}",
        reply_markup=kb_day_actions(day, True)
    )
    return EDIT_DAY_SCREEN

# -----------------------
# Group binding and misc
# -----------------------
async def bind_group(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user_on_event(update)
    chat = update.effective_chat
    if chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
        await update.effective_message.reply_text("Команда должна выполняться в группе, куда добавлен бот.")
        return
    set_group_chat_id(chat.id)
    await update.effective_message.reply_text("✅ Группа успешно привязана для уведомлений.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user_on_event(update)
    msg = (
        "/start — регистрация\n"
        "/status — моё расписание\n"
        "/edit — редактировать расписание\n"
        "/schedule — общее расписание (таблицы)\n"
        "/bindgroup — привязать эту группу\n"
        "/myid — мой Telegram ID\n"
        "/admin — админ-панель (ЛС)\n"
        "/resetdb — сброс БД (админ)\n"
    )
    await update.effective_message.reply_text(msg)

async def myid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user_on_event(update)
    await update.effective_message.reply_text(f"Ваш Telegram ID: {update.effective_user.id}")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user_on_event(update)
    user_row = get_user_by_tg_id(update.effective_user.id)
    if not user_row:
        await update.effective_message.reply_text("Вы не зарегистрированы. /start")
        return
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT weekday, start_time, end_time FROM schedule_intervals WHERE user_id=%s ORDER BY weekday, start_time", (user_row["id"],))
        rows = cur.fetchall()
    grouped: Dict[int, List[Tuple[str, str]]] = {}
    for r in rows or []:
        grouped.setdefault(int(r["weekday"]), []).append((r["start_time"], r["end_time"]))
    lines = [f"Имя: {user_row['name']}"]
    for i in range(7):
        lines.append(f"{WEEKDAYS_RU[i]}: " + (", ".join([f"{s}-{e}" for s, e in grouped[i]]) if i in grouped else "нет пар"))
    await update.effective_message.reply_text("\n".join(lines))

# -----------------------
# /schedule (таблично)
# -----------------------
def build_day_table_block(day_num: int, entries: List[Tuple[str, str]]) -> str:
    if not entries:
        return f"{WEEKDAYS_RU[day_num]}:\nсвободно\n"
    name_w = max(4, min(24, max(len(n) for n, _ in entries)))
    lines = [f"{WEEKDAYS_RU[day_num]}:", "```", f"{'Имя'.ljust(name_w)} | Интервалы", f"{'-'*name_w}-+-{'-'*30}"]
    for name, times in entries:
        lines.append(f"{name.ljust(name_w)} | {times}")
    lines.append("```")
    return "\n".join(lines) + "\n"

async def schedule_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user_on_event(update)
    rows = fetch_all_intervals_joined()
    by_day: Dict[int, Dict[str, List[str]]] = {i: {} for i in range(7)}
    for r in rows:
        by_day[int(r["weekday"])].setdefault(str(r["name"]), []).append(f"{r['start_time']}-{r['end_time']}")
    parts: List[str] = ["📅 Общее расписание группы\n"]
    for d in range(7):
        if not by_day[d]:
            parts.append(f"{WEEKDAYS_RU[d]}:\nсвободно\n"); continue
        entries = [(name, ", ".join(by_day[d][name])) for name in sorted(by_day[d].keys(), key=str.lower)]
        parts.append(build_day_table_block(d, entries))
    current = ""
    for block in parts:
        if len(current) + len(block) > 3800:
            await update.effective_message.reply_text(current, parse_mode=ParseMode.MARKDOWN); current = block
        else:
            current += block
    if current.strip():
        await update.effective_message.reply_text(current, parse_mode=ParseMode.MARKDOWN)

# -----------------------
# Notifications (T-2 min)
# -----------------------
async def notify_before_end(context: ContextTypes.DEFAULT_TYPE):
    job_data = context.job.data or {}
    user_name = job_data.get("user_name", "Кто-то")
    weekday = job_data.get("weekday")
    end_time = job_data.get("end_time")
    group_id = get_group_chat_id()
    if not group_id: return
    users_same_day = get_users_with_classes_on_weekday(weekday)
    if not users_same_day: return
    mentions = ", ".join([mention_html(u["name"], u["tg_id"], u.get("username")) for u in users_same_day]) or "—"
    text = f"⏳ Пара у <b>{user_name}</b> заканчивается в <b>{end_time}</b> (через ~2 мин).\nКто идёт? {mentions}"
    try:
        await context.bot.send_message(chat_id=group_id, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception:
        pass

async def rebuild_jobs(context: ContextTypes.DEFAULT_TYPE):
    for job in list(context.job_queue.jobs()):
        job.schedule_removal()
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT si.weekday, si.end_time, u.id AS user_id, u.name FROM schedule_intervals si JOIN users u ON u.id = si.user_id")
        rows = cur.fetchall()
    for r in rows or []:
        weekday = int(r["weekday"])
        end_t = time_str_to_time(r["end_time"])
        remind_time, shift = sub_minutes(end_t, 2)
        target_weekday = (weekday + shift) % 7
        context.job_queue.run_daily(
            notify_before_end,
            time=remind_time,
            days=(target_weekday,),
            name=f"remind_{r['user_id']}_{weekday}_{r['end_time']}",
            data={"user_name": r["name"], "user_id": r["user_id"], "weekday": weekday, "end_time": r["end_time"]},
        )

# -----------------------
# Admin
# -----------------------
def require_admin(user_id: int) -> bool:
    return ADMIN_TG_ID and int(user_id) == int(ADMIN_TG_ID)

async def admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    upsert_user_on_event(update)
    if update.effective_chat and update.effective_chat.type not in (ChatType.PRIVATE,):
        await update.effective_message.reply_text("Открыть /admin можно только в личном чате с ботом."); return
    if not require_admin(update.effective_user.id):
        await update.effective_message.reply_text("Только администратор может использовать этот раздел."); return
    args = context.args or []
    if not args:
        await update.effective_message.reply_text(
            "Команды админа:\n"
            "/admin list — список пользователей\n"
            "/admin delete <tg_id> — удалить пользователя\n"
            "/admin export — экспорт пользователей и расписаний в CSV"
        ); return
    sub = args[0].lower()
    if sub == "list":
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT id, tg_id, username, name FROM users ORDER BY id")
            rows = cur.fetchall()
        if not rows: await update.effective_message.reply_text("Пользователей нет."); return
        lines = ["Пользователи:"] + [f"id={r['id']} tg_id={r['tg_id']} username={r['username']} name={r['name']}" for r in rows]
        await update.effective_message.reply_text("\n".join(lines)); return
    if sub == "export":
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT u.tg_id, u.username, u.name, si.weekday, si.start_time, si.end_time
                FROM users u
                LEFT JOIN schedule_intervals si ON si.user_id = u.id
                ORDER BY u.id, si.weekday, si.start_time
                """
            ); rows = cur.fetchall()
        out = io.StringIO(); out.write("tg_id,username,name,weekday,start_time,end_time\n")
        for r in rows or []:
            tg_id = r.get("tg_id"); username = r.get("username") or ""; name = (r.get("name") or "").replace(",", " ")
            weekday = r.get("weekday"); start_t = r.get("start_time") or ""; end_t = r.get("end_time") or ""
            weekday_s = "" if weekday is None else str(int(weekday))
            out.write(f"{tg_id},{username},{name},{weekday_s},{start_t},{end_t}\n")
        data = out.getvalue().encode("utf-8"); out.close()
        doc = InputFile(io.BytesIO(data), filename="schedules_export.csv")
        try:
            await update.effective_message.reply_document(document=doc, caption="Экспорт расписаний")
        except Exception as e:
            logger.exception("Failed to send export: %s", e)
            await update.effective_message.reply_text("Не удалось отправить файл экспорта.")
        return
    if sub == "delete" and len(args) >= 2:
        tg_id_s = args[1]
        try:
            tg_id_i = int(tg_id_s)
        except ValueError:
            await update.effective_message.reply_text("tg_id должен быть числом."); return
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE tg_id = %s RETURNING id", (tg_id_i,)); row = cur.fetchone(); conn.commit()
        if row: await update.effective_message.reply_text("Пользователь удален."); await rebuild_jobs(context)
        else:   await update.effective_message.reply_text("Пользователь не найден.")
        return
    await update.effective_message.reply_text("Неизвестная подкоманда. Введите /admin для помощи.")

async def reset_db(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not require_admin(update.effective_user.id):
        await update.effective_message.reply_text("Недостаточно прав."); return
    with get_db_connection() as conn, conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS schedule_intervals CASCADE;")
        cur.execute("DROP TABLE IF EXISTS bot_group CASCADE;")
        cur.execute("DROP TABLE IF EXISTS users CASCADE;")
        conn.commit()
    init_db()
    await update.effective_message.reply_text("База данных очищена и инициализирована заново.")

# -----------------------
# Application
# -----------------------
def build_application() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is empty")
    init_db()
    async def _post_init(app_: Application):
        private_cmds = [
            BotCommand("start", "Регистрация и расписание"),
            BotCommand("help", "Подсказка по командам"),
            BotCommand("myid", "Показать мой Telegram ID"),
            BotCommand("status", "Моё расписание"),
            BotCommand("edit", "Редактировать моё расписание"),
            BotCommand("schedule", "Общее расписание группы"),
            BotCommand("admin", "Админ-панель (только ЛС)"),
            BotCommand("resetdb", "Сброс базы (админ)"),
        ]
        group_cmds = [
            BotCommand("help", "Подсказка по командам"),
            BotCommand("bindgroup", "Привязать эту группу"),
            BotCommand("status", "Моё расписание"),
            BotCommand("edit", "Редактировать моё расписание"),
            BotCommand("schedule", "Общее расписание группы"),
        ]
        try:
            await app_.bot.set_my_commands(private_cmds, scope=BotCommandScopeAllPrivateChats())
            await app_.bot.set_my_commands(group_cmds, scope=BotCommandScopeAllGroupChats())
        except Exception:
            pass
        async def _rebuild_once(context: ContextTypes.DEFAULT_TYPE):
            await rebuild_jobs(context)
        if app_.job_queue is not None:
            app_.job_queue.run_once(_rebuild_once, when=0)
        logger.info("Бот запущен (polling)")
    app: Application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).post_init(_post_init).build()

    # Регистрация
    conv_reg = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            ASK_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_name)],
            ASK_DAY_HAS: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_day_has)],
            ASK_DAY_TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_day_time)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        name="registration",
        persistent=False,
    )

    # Редактор расписания
    conv_edit = ConversationHandler(
        entry_points=[CommandHandler("edit", edit_cmd)],
        states={
            EDIT_MENU: [CallbackQueryHandler(edit_cb, pattern="^edit_")],
            EDIT_SELECT_DAY: [CallbackQueryHandler(edit_cb, pattern="^edit_")],
            EDIT_DAY_SCREEN: [CallbackQueryHandler(edit_cb, pattern="^edit_")],
            EDIT_AWAIT_INPUT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, edit_text_input),
                CallbackQueryHandler(edit_cb, pattern="^edit_"),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        name="edit",
        persistent=False,
    )

    app.add_handler(conv_reg)
    app.add_handler(conv_edit)

    # Общие команды
    app.add_handler(CommandHandler("bindgroup", bind_group))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("myid", myid_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("schedule", schedule_cmd))
    app.add_handler(CommandHandler("admin", admin_cmd))
    app.add_handler(CommandHandler("resetdb", reset_db))

    # Заглушка под старые vote_* колбэки
    app.add_handler(CallbackQueryHandler(lambda u, c: None, pattern="^vote_"))

    return app

def main():
    app = build_application()
    try:
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    finally:
        logger.warning("Бот остановлен")

if __name__ == "__main__":
    main()
