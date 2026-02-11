# -*- coding: utf-8 -*-
"""
数据库连接管理
"""

from typing import List, Dict, Any, Optional, Union
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor

from ..config import get_settings


@contextmanager
def get_connection():
    """
    获取数据库连接的上下文管理器

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM table")
    """
    settings = get_settings()
    db_config = settings.database

    conn = pymysql.connect(
        host=db_config.host,
        port=db_config.port,
        user=db_config.user,
        password=db_config.password,
        database=db_config.database,
        charset=db_config.charset,
        cursorclass=DictCursor,
        autocommit=False,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute_query(
    sql: str,
    params: Optional[tuple] = None,
    fetch_one: bool = False
) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
    """
    执行查询SQL

    Args:
        sql: SQL语句
        params: 参数元组
        fetch_one: 是否只获取一条记录

    Returns:
        查询结果列表或单条记录
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            if fetch_one:
                return cursor.fetchone()
            return cursor.fetchall()


def execute_many(sql: str, params_list: List[tuple]) -> int:
    """
    批量执行SQL

    Args:
        sql: SQL语句
        params_list: 参数列表

    Returns:
        影响的行数
    """
    if not params_list:
        return 0

    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(sql, params_list)
            return cursor.rowcount


def execute_update(sql: str, params: Optional[tuple] = None) -> int:
    """
    执行更新SQL

    Args:
        sql: SQL语句
        params: 参数元组

    Returns:
        影响的行数
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.rowcount


def execute_insert(sql: str, params: Optional[tuple] = None) -> int:
    """
    执行插入SQL并返回自增ID

    Args:
        sql: SQL语句
        params: 参数元组

    Returns:
        自增主键ID
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.lastrowid
