# =====================================================
# Hot News Module - Platform Adapters
# 平台数据适配器 - 将各平台字段映射到统一结构
# =====================================================

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class UnifiedContent:
    """统一内容模型"""

    platform: str
    content_id: str
    content_type: str  # note/video/image

    # 作者信息
    user_id: str
    nickname: str
    avatar: str = ""
    ip_location: str = ""
    gender: str = ""

    # 内容信息
    title: str = ""
    content: str = ""
    content_url: str = ""

    # 媒体信息
    media_type: str = ""  # video/image/text
    cover_url: str = ""
    video_url: str = ""
    video_download_url: str = ""
    image_list: str = ""
    music_url: str = ""
    tag_list: str = ""

    # 互动数据
    liked_count: int = 0
    comment_count: int = 0
    share_count: int = 0
    collect_count: int = 0
    view_count: int = 0
    coin_count: int = 0  # B站投币
    danmaku_count: int = 0  # B站弹幕

    # 来源追踪
    source_keyword: str = ""

    # 时间信息
    original_created_at: Optional[datetime] = None
    add_ts: int = 0
    last_modify_ts: int = 0


@dataclass
class UnifiedComment:
    """统一评论模型"""

    platform: str
    comment_id: str
    content_id: str
    parent_comment_id: str = ""

    # 评论者信息
    user_id: str = ""
    nickname: str = ""
    avatar: str = ""
    ip_location: str = ""
    gender: str = ""

    # 评论内容
    content: str = ""
    pictures: str = ""

    # 互动数据
    liked_count: int = 0
    reply_count: int = 0

    # 时间信息
    original_created_at: Optional[datetime] = None
    add_ts: int = 0
    last_modify_ts: int = 0


class ContentAdapter:
    """
    内容适配器 - 将各平台数据转换为统一格式

    每个平台定义自己的字段映射表：
    - 键: 统一字段名
    - 值: 平台原始字段名
    """

    # ==================== 字段映射配置 ====================

    # 小红书字段映射
    XHS_MAPPING = {
        "content_id": "note_id",
        "content_type": "type",  # normal/video
        "user_id": "user_id",
        "nickname": "nickname",
        "avatar": "avatar",
        "ip_location": "ip_location",
        "title": "title",
        "content": "desc",
        "content_url": "note_url",
        # 媒体字段
        "media_type": lambda r: "video" if r.get("type") == "video" else "image",
        "cover_url": lambda r: "",
        "video_url": "video_url",
        "video_download_url": lambda r: "",
        "image_list": "image_list",
        "music_url": lambda r: "",
        "tag_list": "tag_list",
        # 互动数据
        "liked_count": "liked_count",
        "comment_count": "comment_count",
        "share_count": "share_count",
        "collect_count": "collected_count",
        "view_count": lambda r: 0,
        "coin_count": lambda r: 0,
        "danmaku_count": lambda r: 0,
        # 其他
        "source_keyword": "source_keyword",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "time",
    }

    # 抖音字段映射
    DOUYIN_MAPPING = {
        "content_id": "aweme_id",
        "content_type": "aweme_type",  # video/image
        "user_id": "user_id",
        "nickname": "nickname",
        "avatar": "avatar",
        "ip_location": "ip_location",
        "title": "title",
        "content": "desc",
        "content_url": "aweme_url",
        # 媒体字段
        "media_type": lambda r: "video" if r.get("aweme_type") != "image" else "image",
        "cover_url": "cover_url",
        "video_url": "aweme_url",
        "video_download_url": "video_download_url",
        "image_list": "note_download_url",  # 图文笔记
        "music_url": "music_download_url",
        "tag_list": lambda r: "",
        # 互动数据
        "liked_count": "liked_count",
        "comment_count": "comment_count",
        "share_count": "share_count",
        "collect_count": "collected_count",
        "view_count": lambda r: 0,
        "coin_count": lambda r: 0,
        "danmaku_count": lambda r: 0,
        # 其他
        "source_keyword": "source_keyword",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "create_time",
    }

    # 微博字段映射
    WEIBO_MAPPING = {
        "content_id": "note_id",
        "content_type": lambda r: "note",
        "user_id": "user_id",
        "nickname": "nickname",
        "avatar": "avatar",
        "ip_location": "ip_location",
        "gender": "gender",
        "title": lambda r: "",
        "content": "content",
        "content_url": "note_url",
        # 媒体字段
        "media_type": lambda r: "text",
        "cover_url": lambda r: "",
        "video_url": lambda r: "",
        "video_download_url": lambda r: "",
        "image_list": lambda r: "",
        "music_url": lambda r: "",
        "tag_list": lambda r: "",
        # 互动数据
        "liked_count": "liked_count",
        "comment_count": "comments_count",
        "share_count": "shared_count",
        "collect_count": lambda r: 0,
        "view_count": lambda r: 0,
        "coin_count": lambda r: 0,
        "danmaku_count": lambda r: 0,
        # 其他
        "source_keyword": "source_keyword",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "create_time",
    }

    # B站字段映射
    BILIBILI_MAPPING = {
        "content_id": "video_id",
        "content_type": lambda r: "video",
        "user_id": "user_id",
        "nickname": "nickname",
        "avatar": "avatar",
        "ip_location": lambda r: "",
        "title": "title",
        "content": "desc",
        "content_url": "video_url",
        # 媒体字段
        "media_type": lambda r: "video",
        "cover_url": "video_cover_url",
        "video_url": "video_url",
        "video_download_url": lambda r: "",
        "image_list": lambda r: "",
        "music_url": lambda r: "",
        "tag_list": lambda r: "",
        # 互动数据
        "liked_count": "liked_count",
        "comment_count": "video_comment",
        "share_count": "video_share_count",
        "collect_count": "video_favorite_count",
        "view_count": "video_play_count",
        "coin_count": "video_coin_count",
        "danmaku_count": "video_danmaku",
        # 其他
        "source_keyword": "source_keyword",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "create_time",
    }

    # 快手字段映射
    KUAISHOU_MAPPING = {
        "content_id": "video_id",
        "content_type": lambda r: "video",
        "user_id": "user_id",
        "nickname": "nickname",
        "avatar": "avatar",
        "ip_location": lambda r: "",
        "title": "title",
        "content": "desc",
        "content_url": "video_url",
        # 媒体字段
        "media_type": lambda r: "video",
        "cover_url": "video_cover_url",
        "video_url": "video_url",
        "video_download_url": "video_play_url",
        "image_list": lambda r: "",
        "music_url": lambda r: "",
        "tag_list": lambda r: "",
        # 互动数据
        "liked_count": "liked_count",
        "comment_count": lambda r: 0,
        "share_count": lambda r: 0,
        "collect_count": lambda r: 0,
        "view_count": "viewd_count",
        "coin_count": lambda r: 0,
        "danmaku_count": lambda r: 0,
        # 其他
        "source_keyword": "source_keyword",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "create_time",
    }

    # 贴吧字段映射
    TIEBA_MAPPING = {
        "content_id": "note_id",
        "content_type": lambda r: "note",
        "user_id": lambda r: "",  # 贴吧无user_id，用user_link代替
        "nickname": "user_nickname",
        "avatar": "user_avatar",
        "ip_location": "ip_location",
        "title": "title",
        "content": "desc",
        "content_url": "note_url",
        # 媒体字段
        "media_type": lambda r: "text",
        "cover_url": lambda r: "",
        "video_url": lambda r: "",
        "video_download_url": lambda r: "",
        "image_list": lambda r: "",
        "music_url": lambda r: "",
        "tag_list": lambda r: "",
        # 互动数据
        "liked_count": lambda r: 0,
        "comment_count": "total_replay_num",
        "share_count": lambda r: 0,
        "collect_count": lambda r: 0,
        "view_count": lambda r: 0,
        "coin_count": lambda r: 0,
        "danmaku_count": lambda r: 0,
        # 其他
        "source_keyword": "source_keyword",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "publish_time",  # 字符串格式，需特殊处理
    }

    # 知乎字段映射
    ZHIHU_MAPPING = {
        "content_id": "content_id",
        "content_type": "content_type",  # answer/article
        "user_id": "user_id",
        "nickname": "user_nickname",
        "avatar": "user_avatar",
        "ip_location": lambda r: "",
        "title": "title",
        "content": "content_text",
        "content_url": "content_url",
        # 媒体字段
        "media_type": lambda r: "text",
        "cover_url": lambda r: "",
        "video_url": lambda r: "",
        "video_download_url": lambda r: "",
        "image_list": lambda r: "",
        "music_url": lambda r: "",
        "tag_list": lambda r: "",
        # 互动数据
        "liked_count": "voteup_count",
        "comment_count": "comment_count",
        "share_count": lambda r: 0,
        "collect_count": lambda r: 0,
        "view_count": lambda r: 0,
        "coin_count": lambda r: 0,
        "danmaku_count": lambda r: 0,
        # 其他
        "source_keyword": "source_keyword",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "created_time",  # 字符串格式，需特殊处理
    }

    # 平台映射表汇总
    PLATFORM_MAPPINGS = {
        "xhs": XHS_MAPPING,
        "dy": DOUYIN_MAPPING,
        "wb": WEIBO_MAPPING,
        "bili": BILIBILI_MAPPING,
        "ks": KUAISHOU_MAPPING,
        "tieba": TIEBA_MAPPING,
        "zhihu": ZHIHU_MAPPING,
    }

    # 平台对应的数据库表名
    PLATFORM_TABLES = {
        "xhs": "xhs_note",
        "dy": "douyin_aweme",
        "wb": "weibo_note",
        "bili": "bilibili_video",
        "ks": "kuaishou_video",
        "tieba": "tieba_note",
        "zhihu": "zhihu_content",
    }

    @classmethod
    def _safe_int(cls, value) -> int:
        """安全转换为整数"""
        if value is None:
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    @classmethod
    def _safe_str(cls, value) -> str:
        """安全转换为字符串"""
        if value is None:
            return ""
        return str(value)

    @classmethod
    def _get_value(cls, row: Dict[str, Any], mapping_value) -> Any:
        """
        从原始数据行获取值

        Args:
            row: 原始数据行（字典）
            mapping_value: 映射值，可以是字段名字符串或 lambda 函数
        """
        if callable(mapping_value):
            return mapping_value(row)
        return row.get(mapping_value)

    @classmethod
    def _ts_to_datetime(cls, ts) -> Optional[datetime]:
        """时间戳或字符串转 datetime"""
        if not ts:
            return None
        try:
            # 如果是字符串格式的日期时间
            if isinstance(ts, str):
                # 尝试多种常见格式
                for fmt in [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M",
                    "%Y-%m-%d",
                    "%Y/%m/%d %H:%M:%S",
                    "%Y/%m/%d",
                ]:
                    try:
                        return datetime.strptime(ts, fmt)
                    except ValueError:
                        continue
                return None

            ts = int(ts)
            # 判断是秒还是毫秒
            if ts > 10000000000:
                ts = ts / 1000
            return datetime.fromtimestamp(ts)
        except (ValueError, TypeError, OSError):
            return None

    @classmethod
    def adapt(cls, platform: str, row: Dict[str, Any]) -> UnifiedContent:
        """
        将平台数据转换为统一格式

        Args:
            platform: 平台代码 (xhs/dy/wb/bili/ks/tieba/zhihu)
            row: 原始数据行（字典格式）

        Returns:
            UnifiedContent 对象
        """
        mapping = cls.PLATFORM_MAPPINGS.get(platform)
        if not mapping:
            raise ValueError(f"不支持的平台: {platform}")

        return UnifiedContent(
            platform=platform,
            content_id=cls._safe_str(cls._get_value(row, mapping["content_id"])),
            content_type=cls._safe_str(cls._get_value(row, mapping["content_type"])),
            user_id=cls._safe_str(cls._get_value(row, mapping["user_id"])),
            nickname=cls._safe_str(cls._get_value(row, mapping["nickname"])),
            avatar=cls._safe_str(
                cls._get_value(row, mapping.get("avatar", lambda r: ""))
            ),
            ip_location=cls._safe_str(
                cls._get_value(row, mapping.get("ip_location", lambda r: ""))
            ),
            gender=cls._safe_str(
                cls._get_value(row, mapping.get("gender", lambda r: ""))
            ),
            title=cls._safe_str(cls._get_value(row, mapping["title"])),
            content=cls._safe_str(cls._get_value(row, mapping["content"])),
            content_url=cls._safe_str(cls._get_value(row, mapping["content_url"])),
            # 媒体字段
            media_type=cls._safe_str(cls._get_value(row, mapping["media_type"])),
            cover_url=cls._safe_str(cls._get_value(row, mapping["cover_url"])),
            video_url=cls._safe_str(cls._get_value(row, mapping["video_url"])),
            video_download_url=cls._safe_str(
                cls._get_value(row, mapping["video_download_url"])
            ),
            image_list=cls._safe_str(cls._get_value(row, mapping["image_list"])),
            music_url=cls._safe_str(cls._get_value(row, mapping["music_url"])),
            tag_list=cls._safe_str(cls._get_value(row, mapping["tag_list"])),
            # 互动数据
            liked_count=cls._safe_int(cls._get_value(row, mapping["liked_count"])),
            comment_count=cls._safe_int(cls._get_value(row, mapping["comment_count"])),
            share_count=cls._safe_int(cls._get_value(row, mapping["share_count"])),
            collect_count=cls._safe_int(cls._get_value(row, mapping["collect_count"])),
            view_count=cls._safe_int(cls._get_value(row, mapping["view_count"])),
            coin_count=cls._safe_int(cls._get_value(row, mapping["coin_count"])),
            danmaku_count=cls._safe_int(cls._get_value(row, mapping["danmaku_count"])),
            # 其他
            source_keyword=cls._safe_str(
                cls._get_value(row, mapping["source_keyword"])
            ),
            add_ts=cls._safe_int(cls._get_value(row, mapping["add_ts"])),
            last_modify_ts=cls._safe_int(
                cls._get_value(row, mapping["last_modify_ts"])
            ),
            original_created_at=cls._ts_to_datetime(
                cls._get_value(row, mapping["original_created_at"])
            )
            or datetime.now(),
        )


class CommentAdapter:
    """评论适配器 - 将各平台评论数据转换为统一格式"""

    # 小红书评论字段映射
    XHS_MAPPING = {
        "comment_id": "comment_id",
        "content_id": "note_id",
        "parent_comment_id": "parent_comment_id",
        "user_id": "user_id",
        "nickname": "nickname",
        "avatar": "avatar",
        "ip_location": "ip_location",
        "content": "content",
        "pictures": "pictures",
        "liked_count": "like_count",
        "reply_count": "sub_comment_count",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "create_time",
    }

    # 抖音评论字段映射
    DOUYIN_MAPPING = {
        "comment_id": "comment_id",
        "content_id": "aweme_id",
        "parent_comment_id": "parent_comment_id",
        "user_id": "user_id",
        "nickname": "nickname",
        "avatar": "avatar",
        "ip_location": "ip_location",
        "content": "content",
        "pictures": "pictures",
        "liked_count": "like_count",
        "reply_count": "sub_comment_count",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "create_time",
    }

    # 微博评论字段映射
    WEIBO_MAPPING = {
        "comment_id": "comment_id",
        "content_id": "note_id",
        "parent_comment_id": "parent_comment_id",
        "user_id": "user_id",
        "nickname": "nickname",
        "avatar": "avatar",
        "ip_location": "ip_location",
        "gender": "gender",
        "content": "content",
        "pictures": lambda r: "",
        "liked_count": "comment_like_count",
        "reply_count": "sub_comment_count",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "create_time",
    }

    # B站评论字段映射
    BILIBILI_MAPPING = {
        "comment_id": "comment_id",
        "content_id": "video_id",
        "parent_comment_id": "parent_comment_id",
        "user_id": "user_id",
        "nickname": "nickname",
        "avatar": "avatar",
        "ip_location": lambda r: "",
        "gender": "sex",
        "content": "content",
        "pictures": lambda r: "",
        "liked_count": "like_count",
        "reply_count": "sub_comment_count",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "create_time",
    }

    # 快手评论字段映射
    KUAISHOU_MAPPING = {
        "comment_id": "comment_id",
        "content_id": "video_id",
        "parent_comment_id": lambda r: "",  # 快手无父评论字段
        "user_id": "user_id",
        "nickname": "nickname",
        "avatar": "avatar",
        "ip_location": lambda r: "",
        "content": "content",
        "pictures": lambda r: "",
        "liked_count": lambda r: 0,  # 快手评论无点赞数
        "reply_count": "sub_comment_count",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "create_time",
    }

    # 贴吧评论字段映射
    TIEBA_MAPPING = {
        "comment_id": "comment_id",
        "content_id": "note_id",
        "parent_comment_id": "parent_comment_id",
        "user_id": lambda r: "",  # 贴吧无user_id
        "nickname": "user_nickname",
        "avatar": "user_avatar",
        "ip_location": "ip_location",
        "content": "content",
        "pictures": lambda r: "",
        "liked_count": lambda r: 0,
        "reply_count": "sub_comment_count",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "publish_time",
    }

    # 知乎评论字段映射
    ZHIHU_MAPPING = {
        "comment_id": "comment_id",
        "content_id": "content_id",
        "parent_comment_id": "parent_comment_id",
        "user_id": "user_id",
        "nickname": "user_nickname",
        "avatar": "user_avatar",
        "ip_location": "ip_location",
        "content": "content",
        "pictures": lambda r: "",
        "liked_count": "like_count",
        "reply_count": "sub_comment_count",
        "add_ts": "add_ts",
        "last_modify_ts": "last_modify_ts",
        "original_created_at": "publish_time",
    }

    PLATFORM_MAPPINGS = {
        "xhs": XHS_MAPPING,
        "dy": DOUYIN_MAPPING,
        "wb": WEIBO_MAPPING,
        "bili": BILIBILI_MAPPING,
        "ks": KUAISHOU_MAPPING,
        "tieba": TIEBA_MAPPING,
        "zhihu": ZHIHU_MAPPING,
    }

    PLATFORM_TABLES = {
        "xhs": "xhs_note_comment",
        "dy": "douyin_aweme_comment",
        "wb": "weibo_note_comment",
        "bili": "bilibili_video_comment",
        "ks": "kuaishou_video_comment",
        "tieba": "tieba_comment",
        "zhihu": "zhihu_comment",
    }

    @classmethod
    def _safe_int(cls, value) -> int:
        if value is None:
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    @classmethod
    def _safe_str(cls, value) -> str:
        if value is None:
            return ""
        return str(value)

    @classmethod
    def _get_value(cls, row: Dict[str, Any], mapping_value) -> Any:
        if callable(mapping_value):
            return mapping_value(row)
        return row.get(mapping_value)

    @classmethod
    def _ts_to_datetime(cls, ts) -> Optional[datetime]:
        if not ts:
            return None
        try:
            ts = int(ts)
            if ts > 10000000000:
                ts = ts / 1000
            return datetime.fromtimestamp(ts)
        except (ValueError, TypeError, OSError):
            return None

    @classmethod
    def adapt(cls, platform: str, row: Dict[str, Any]) -> UnifiedComment:
        """将平台评论数据转换为统一格式"""
        mapping = cls.PLATFORM_MAPPINGS.get(platform)
        if not mapping:
            raise ValueError(f"不支持的平台: {platform}")

        return UnifiedComment(
            platform=platform,
            comment_id=cls._safe_str(cls._get_value(row, mapping["comment_id"])),
            content_id=cls._safe_str(cls._get_value(row, mapping["content_id"])),
            parent_comment_id=cls._safe_str(
                cls._get_value(row, mapping["parent_comment_id"])
            ),
            user_id=cls._safe_str(cls._get_value(row, mapping["user_id"])),
            nickname=cls._safe_str(cls._get_value(row, mapping["nickname"])),
            avatar=cls._safe_str(
                cls._get_value(row, mapping.get("avatar", lambda r: ""))
            ),
            ip_location=cls._safe_str(
                cls._get_value(row, mapping.get("ip_location", lambda r: ""))
            ),
            gender=cls._safe_str(
                cls._get_value(row, mapping.get("gender", lambda r: ""))
            ),
            content=cls._safe_str(cls._get_value(row, mapping["content"])),
            pictures=cls._safe_str(cls._get_value(row, mapping["pictures"])),
            liked_count=cls._safe_int(cls._get_value(row, mapping["liked_count"])),
            reply_count=cls._safe_int(cls._get_value(row, mapping["reply_count"])),
            add_ts=cls._safe_int(cls._get_value(row, mapping["add_ts"])),
            last_modify_ts=cls._safe_int(
                cls._get_value(row, mapping["last_modify_ts"])
            ),
            original_created_at=cls._ts_to_datetime(
                cls._get_value(row, mapping["original_created_at"])
            )
            or datetime.now(),
        )
