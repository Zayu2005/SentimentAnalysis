# -*- coding: utf-8 -*-
# Copyright (c) 2025 relakkes@gmail.com
#
# This file is part of MediaCrawler project.
# Repository: https://github.com/NanmiCoder/MediaCrawler/blob/main/database/models.py
# GitHub: https://github.com/NanmiCoder
# Licensed under NON-COMMERCIAL LEARNING LICENSE 1.1
#
# 声明：本代码仅供学习和研究目的使用。使用者应遵守以下原则：
# 1. 不得用于任何商业用途。
# 2. 使用时应遵守目标平台的使用条款和robots.txt规则。
# 3. 不得进行大规模爬取或对平台造成运营干扰。
# 4. 应合理控制请求频率，避免给目标平台带来不必要的负担。
# 5. 不得用于任何非法或不当的用途。
#
# 详细许可条款请参阅项目根目录下的LICENSE文件。
# 使用本代码即表示您同意遵守上述原则和LICENSE中的所有条款。

from sqlalchemy import create_engine, Column, Integer, Text, String, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


# =====================================================
# B站相关表
# =====================================================

class BilibiliVideo(Base):
    """B站视频内容表"""
    __tablename__ = 'bilibili_video'
    __table_args__ = {'comment': 'B站视频内容表，存储从B站爬取的视频信息'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    video_id = Column(BigInteger, nullable=False, index=True, unique=True, comment='B站视频唯一ID(BV号对应的数字ID)')
    video_url = Column(Text, nullable=False, comment='视频播放页URL地址')
    user_id = Column(BigInteger, index=True, comment='UP主用户ID')
    nickname = Column(Text, comment='UP主昵称')
    avatar = Column(Text, comment='UP主头像URL')
    liked_count = Column(Integer, comment='点赞数量')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    video_type = Column(Text, comment='视频类型分区')
    title = Column(Text, comment='视频标题')
    desc = Column(Text, comment='视频简介描述')
    create_time = Column(BigInteger, index=True, comment='视频发布时间戳')
    disliked_count = Column(Text, comment='踩/不喜欢数量')
    video_play_count = Column(Text, comment='播放次数')
    video_favorite_count = Column(Text, comment='收藏数量')
    video_share_count = Column(Text, comment='分享数量')
    video_coin_count = Column(Text, comment='投币数量')
    video_danmaku = Column(Text, comment='弹幕数量')
    video_comment = Column(Text, comment='评论数量')
    video_cover_url = Column(Text, comment='视频封面图URL')
    source_keyword = Column(Text, default='', comment='搜索来源关键词')


class BilibiliVideoComment(Base):
    """B站视频评论表"""
    __tablename__ = 'bilibili_video_comment'
    __table_args__ = {'comment': 'B站视频评论表，存储视频下的用户评论'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(255), comment='评论用户ID')
    nickname = Column(Text, comment='评论用户昵称')
    sex = Column(Text, comment='用户性别')
    sign = Column(Text, comment='用户个性签名')
    avatar = Column(Text, comment='用户头像URL')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    comment_id = Column(BigInteger, index=True, comment='评论唯一ID')
    video_id = Column(BigInteger, index=True, comment='所属视频ID')
    content = Column(Text, comment='评论内容')
    create_time = Column(BigInteger, comment='评论发布时间戳')
    sub_comment_count = Column(Text, comment='子评论/回复数量')
    parent_comment_id = Column(String(255), comment='父评论ID，一级评论为空')
    like_count = Column(Text, default='0', comment='评论点赞数')


class BilibiliUpInfo(Base):
    """B站UP主信息表"""
    __tablename__ = 'bilibili_up_info'
    __table_args__ = {'comment': 'B站UP主信息表，存储UP主的基本资料'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(BigInteger, index=True, comment='UP主用户ID')
    nickname = Column(Text, comment='UP主昵称')
    sex = Column(Text, comment='性别')
    sign = Column(Text, comment='个性签名')
    avatar = Column(Text, comment='头像URL')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    total_fans = Column(Integer, comment='粉丝总数')
    total_liked = Column(Integer, comment='获赞总数')
    user_rank = Column(Integer, comment='用户等级')
    is_official = Column(Integer, comment='是否官方认证：0-否，1-是')


class BilibiliContactInfo(Base):
    """B站粉丝关注关系表"""
    __tablename__ = 'bilibili_contact_info'
    __table_args__ = {'comment': 'B站粉丝关注关系表，存储UP主与粉丝的关注关系'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    up_id = Column(BigInteger, index=True, comment='UP主用户ID')
    fan_id = Column(BigInteger, index=True, comment='粉丝用户ID')
    up_name = Column(Text, comment='UP主昵称')
    fan_name = Column(Text, comment='粉丝昵称')
    up_sign = Column(Text, comment='UP主个性签名')
    fan_sign = Column(Text, comment='粉丝个性签名')
    up_avatar = Column(Text, comment='UP主头像URL')
    fan_avatar = Column(Text, comment='粉丝头像URL')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')


class BilibiliUpDynamic(Base):
    """B站UP主动态表"""
    __tablename__ = 'bilibili_up_dynamic'
    __table_args__ = {'comment': 'B站UP主动态表，存储UP主发布的动态内容'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    dynamic_id = Column(BigInteger, index=True, comment='动态唯一ID')
    user_id = Column(String(255), comment='发布者用户ID')
    user_name = Column(Text, comment='发布者昵称')
    text = Column(Text, comment='动态文本内容')
    type = Column(Text, comment='动态类型')
    pub_ts = Column(BigInteger, comment='发布时间戳')
    total_comments = Column(Integer, comment='评论总数')
    total_forwards = Column(Integer, comment='转发总数')
    total_liked = Column(Integer, comment='点赞总数')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')


# =====================================================
# 抖音相关表
# =====================================================

class DouyinAweme(Base):
    """抖音视频内容表"""
    __tablename__ = 'douyin_aweme'
    __table_args__ = {'comment': '抖音视频内容表，存储从抖音爬取的视频信息'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(255), comment='作者用户ID')
    sec_uid = Column(String(255), comment='作者加密用户ID')
    short_user_id = Column(String(255), comment='作者短ID')
    user_unique_id = Column(String(255), comment='作者唯一ID')
    nickname = Column(Text, comment='作者昵称')
    avatar = Column(Text, comment='作者头像URL')
    user_signature = Column(Text, comment='作者个性签名')
    ip_location = Column(Text, comment='IP属地')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    aweme_id = Column(BigInteger, index=True, comment='视频唯一ID')
    aweme_type = Column(Text, comment='视频类型')
    title = Column(Text, comment='视频标题')
    desc = Column(Text, comment='视频描述')
    create_time = Column(BigInteger, index=True, comment='视频发布时间戳')
    liked_count = Column(Text, comment='点赞数量')
    comment_count = Column(Text, comment='评论数量')
    share_count = Column(Text, comment='分享数量')
    collected_count = Column(Text, comment='收藏数量')
    aweme_url = Column(Text, comment='视频播放页URL')
    cover_url = Column(Text, comment='视频封面URL')
    video_download_url = Column(Text, comment='视频下载URL')
    music_download_url = Column(Text, comment='背景音乐下载URL')
    note_download_url = Column(Text, comment='图文笔记下载URL')
    source_keyword = Column(Text, default='', comment='搜索来源关键词')


class DouyinAwemeComment(Base):
    """抖音视频评论表"""
    __tablename__ = 'douyin_aweme_comment'
    __table_args__ = {'comment': '抖音视频评论表，存储视频下的用户评论'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(255), comment='评论用户ID')
    sec_uid = Column(String(255), comment='评论用户加密ID')
    short_user_id = Column(String(255), comment='评论用户短ID')
    user_unique_id = Column(String(255), comment='评论用户唯一ID')
    nickname = Column(Text, comment='评论用户昵称')
    avatar = Column(Text, comment='评论用户头像URL')
    user_signature = Column(Text, comment='评论用户个性签名')
    ip_location = Column(Text, comment='评论IP属地')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    comment_id = Column(BigInteger, index=True, comment='评论唯一ID')
    aweme_id = Column(BigInteger, index=True, comment='所属视频ID')
    content = Column(Text, comment='评论内容')
    create_time = Column(BigInteger, comment='评论发布时间戳')
    sub_comment_count = Column(Text, comment='子评论/回复数量')
    parent_comment_id = Column(String(255), comment='父评论ID，一级评论为空')
    like_count = Column(Text, default='0', comment='评论点赞数')
    pictures = Column(Text, default='', comment='评论配图URL列表')


class DyCreator(Base):
    """抖音创作者信息表"""
    __tablename__ = 'dy_creator'
    __table_args__ = {'comment': '抖音创作者信息表，存储抖音达人的基本资料'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(255), comment='创作者用户ID')
    nickname = Column(Text, comment='创作者昵称')
    avatar = Column(Text, comment='创作者头像URL')
    ip_location = Column(Text, comment='IP属地')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    desc = Column(Text, comment='个人简介')
    gender = Column(Text, comment='性别')
    follows = Column(Text, comment='关注数')
    fans = Column(Text, comment='粉丝数')
    interaction = Column(Text, comment='获赞数')
    videos_count = Column(String(255), comment='作品数量')


# =====================================================
# 快手相关表
# =====================================================

class KuaishouVideo(Base):
    """快手视频内容表"""
    __tablename__ = 'kuaishou_video'
    __table_args__ = {'comment': '快手视频内容表，存储从快手爬取的视频信息'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(64), comment='作者用户ID')
    nickname = Column(Text, comment='作者昵称')
    avatar = Column(Text, comment='作者头像URL')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    video_id = Column(String(255), index=True, comment='视频唯一ID')
    video_type = Column(Text, comment='视频类型')
    title = Column(Text, comment='视频标题')
    desc = Column(Text, comment='视频描述')
    create_time = Column(BigInteger, index=True, comment='视频发布时间戳')
    liked_count = Column(Text, comment='点赞数量')
    viewd_count = Column(Text, comment='播放/观看数量')
    video_url = Column(Text, comment='视频页面URL')
    video_cover_url = Column(Text, comment='视频封面URL')
    video_play_url = Column(Text, comment='视频播放URL')
    source_keyword = Column(Text, default='', comment='搜索来源关键词')


class KuaishouVideoComment(Base):
    """快手视频评论表"""
    __tablename__ = 'kuaishou_video_comment'
    __table_args__ = {'comment': '快手视频评论表，存储视频下的用户评论'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(Text, comment='评论用户ID')
    nickname = Column(Text, comment='评论用户昵称')
    avatar = Column(Text, comment='评论用户头像URL')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    comment_id = Column(BigInteger, index=True, comment='评论唯一ID')
    video_id = Column(String(255), index=True, comment='所属视频ID')
    content = Column(Text, comment='评论内容')
    create_time = Column(BigInteger, comment='评论发布时间戳')
    sub_comment_count = Column(Text, comment='子评论/回复数量')


# =====================================================
# 微博相关表
# =====================================================

class WeiboNote(Base):
    """微博内容表"""
    __tablename__ = 'weibo_note'
    __table_args__ = {'comment': '微博内容表，存储从微博爬取的博文信息'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(255), comment='博主用户ID')
    nickname = Column(Text, comment='博主昵称')
    avatar = Column(Text, comment='博主头像URL')
    gender = Column(Text, comment='博主性别')
    profile_url = Column(Text, comment='博主主页URL')
    ip_location = Column(Text, default='', comment='发布IP属地')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    note_id = Column(BigInteger, index=True, comment='微博唯一ID')
    content = Column(Text, comment='微博正文内容')
    create_time = Column(BigInteger, index=True, comment='发布时间戳')
    create_date_time = Column(String(255), index=True, comment='发布时间(格式化字符串)')
    liked_count = Column(Text, comment='点赞数量')
    comments_count = Column(Text, comment='评论数量')
    shared_count = Column(Text, comment='转发数量')
    note_url = Column(Text, comment='微博详情页URL')
    source_keyword = Column(Text, default='', comment='搜索来源关键词')


class WeiboNoteComment(Base):
    """微博评论表"""
    __tablename__ = 'weibo_note_comment'
    __table_args__ = {'comment': '微博评论表，存储微博下的用户评论'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(255), comment='评论用户ID')
    nickname = Column(Text, comment='评论用户昵称')
    avatar = Column(Text, comment='评论用户头像URL')
    gender = Column(Text, comment='评论用户性别')
    profile_url = Column(Text, comment='评论用户主页URL')
    ip_location = Column(Text, default='', comment='评论IP属地')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    comment_id = Column(BigInteger, index=True, comment='评论唯一ID')
    note_id = Column(BigInteger, index=True, comment='所属微博ID')
    content = Column(Text, comment='评论内容')
    create_time = Column(BigInteger, comment='评论发布时间戳')
    create_date_time = Column(String(255), index=True, comment='评论发布时间(格式化字符串)')
    comment_like_count = Column(Text, comment='评论点赞数')
    sub_comment_count = Column(Text, comment='子评论/回复数量')
    parent_comment_id = Column(String(255), comment='父评论ID，一级评论为空')


class WeiboCreator(Base):
    """微博博主信息表"""
    __tablename__ = 'weibo_creator'
    __table_args__ = {'comment': '微博博主信息表，存储微博用户的基本资料'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(255), comment='博主用户ID')
    nickname = Column(Text, comment='博主昵称')
    avatar = Column(Text, comment='博主头像URL')
    ip_location = Column(Text, comment='IP属地')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    desc = Column(Text, comment='个人简介')
    gender = Column(Text, comment='性别')
    follows = Column(Text, comment='关注数')
    fans = Column(Text, comment='粉丝数')
    tag_list = Column(Text, comment='标签列表，JSON格式')


# =====================================================
# 小红书相关表
# =====================================================

class XhsCreator(Base):
    """小红书创作者信息表"""
    __tablename__ = 'xhs_creator'
    __table_args__ = {'comment': '小红书创作者信息表，存储小红书博主的基本资料'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(255), comment='创作者用户ID')
    nickname = Column(Text, comment='创作者昵称')
    avatar = Column(Text, comment='创作者头像URL')
    ip_location = Column(Text, comment='IP属地')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    desc = Column(Text, comment='个人简介')
    gender = Column(Text, comment='性别')
    follows = Column(Text, comment='关注数')
    fans = Column(Text, comment='粉丝数')
    interaction = Column(Text, comment='获赞与收藏数')
    tag_list = Column(Text, comment='标签列表，JSON格式')


class XhsNote(Base):
    """小红书笔记内容表"""
    __tablename__ = 'xhs_note'
    __table_args__ = {'comment': '小红书笔记内容表，存储从小红书爬取的笔记信息'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(255), comment='作者用户ID')
    nickname = Column(Text, comment='作者昵称')
    avatar = Column(Text, comment='作者头像URL')
    ip_location = Column(Text, comment='发布IP属地')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    note_id = Column(String(255), index=True, comment='笔记唯一ID')
    type = Column(Text, comment='笔记类型：normal-图文，video-视频')
    title = Column(Text, comment='笔记标题')
    desc = Column(Text, comment='笔记正文内容')
    video_url = Column(Text, comment='视频播放URL(视频笔记)')
    time = Column(BigInteger, index=True, comment='发布时间戳')
    last_update_time = Column(BigInteger, comment='最后更新时间戳')
    liked_count = Column(Text, comment='点赞数量')
    collected_count = Column(Text, comment='收藏数量')
    comment_count = Column(Text, comment='评论数量')
    share_count = Column(Text, comment='分享数量')
    image_list = Column(Text, comment='图片URL列表，JSON格式')
    tag_list = Column(Text, comment='话题标签列表，JSON格式')
    note_url = Column(Text, comment='笔记详情页URL')
    source_keyword = Column(Text, default='', comment='搜索来源关键词')
    xsec_token = Column(Text, comment='小红书安全令牌')


class XhsNoteComment(Base):
    """小红书笔记评论表"""
    __tablename__ = 'xhs_note_comment'
    __table_args__ = {'comment': '小红书笔记评论表，存储笔记下的用户评论'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(255), comment='评论用户ID')
    nickname = Column(Text, comment='评论用户昵称')
    avatar = Column(Text, comment='评论用户头像URL')
    ip_location = Column(Text, comment='评论IP属地')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    comment_id = Column(String(255), index=True, comment='评论唯一ID')
    create_time = Column(BigInteger, index=True, comment='评论发布时间戳')
    note_id = Column(String(255), comment='所属笔记ID')
    content = Column(Text, comment='评论内容')
    sub_comment_count = Column(Integer, comment='子评论/回复数量')
    pictures = Column(Text, comment='评论配图URL列表')
    parent_comment_id = Column(String(255), comment='父评论ID，一级评论为空')
    like_count = Column(Text, comment='评论点赞数')


# =====================================================
# 贴吧相关表
# =====================================================

class TiebaNote(Base):
    """贴吧帖子内容表"""
    __tablename__ = 'tieba_note'
    __table_args__ = {'comment': '贴吧帖子内容表，存储从百度贴吧爬取的帖子信息'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    note_id = Column(String(644), index=True, comment='帖子唯一ID')
    title = Column(Text, comment='帖子标题')
    desc = Column(Text, comment='帖子内容摘要')
    note_url = Column(Text, comment='帖子详情页URL')
    publish_time = Column(String(255), index=True, comment='发布时间(格式化字符串)')
    user_link = Column(Text, default='', comment='楼主主页链接')
    user_nickname = Column(Text, default='', comment='楼主昵称')
    user_avatar = Column(Text, default='', comment='楼主头像URL')
    tieba_id = Column(String(255), default='', comment='所属贴吧ID')
    tieba_name = Column(Text, comment='所属贴吧名称')
    tieba_link = Column(Text, comment='所属贴吧链接')
    total_replay_num = Column(Integer, default=0, comment='总回复数')
    total_replay_page = Column(Integer, default=0, comment='总回复页数')
    ip_location = Column(Text, default='', comment='发布IP属地')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    source_keyword = Column(Text, default='', comment='搜索来源关键词')


class TiebaComment(Base):
    """贴吧帖子评论表"""
    __tablename__ = 'tieba_comment'
    __table_args__ = {'comment': '贴吧帖子评论表，存储帖子下的楼层回复'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    comment_id = Column(String(255), index=True, comment='评论/回复唯一ID')
    parent_comment_id = Column(String(255), default='', comment='父评论ID，楼中楼使用')
    content = Column(Text, comment='评论内容')
    user_link = Column(Text, default='', comment='评论者主页链接')
    user_nickname = Column(Text, default='', comment='评论者昵称')
    user_avatar = Column(Text, default='', comment='评论者头像URL')
    tieba_id = Column(String(255), default='', comment='所属贴吧ID')
    tieba_name = Column(Text, comment='所属贴吧名称')
    tieba_link = Column(Text, comment='所属贴吧链接')
    publish_time = Column(String(255), index=True, comment='评论发布时间(格式化字符串)')
    ip_location = Column(Text, default='', comment='评论IP属地')
    sub_comment_count = Column(Integer, default=0, comment='楼中楼回复数')
    note_id = Column(String(255), index=True, comment='所属帖子ID')
    note_url = Column(Text, comment='所属帖子URL')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')


class TiebaCreator(Base):
    """贴吧用户信息表"""
    __tablename__ = 'tieba_creator'
    __table_args__ = {'comment': '贴吧用户信息表，存储贴吧用户的基本资料'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(64), comment='用户ID')
    user_name = Column(Text, comment='用户名(登录名)')
    nickname = Column(Text, comment='用户昵称(显示名)')
    avatar = Column(Text, comment='用户头像URL')
    ip_location = Column(Text, comment='IP属地')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
    gender = Column(Text, comment='性别')
    follows = Column(Text, comment='关注数')
    fans = Column(Text, comment='粉丝数')
    registration_duration = Column(Text, comment='贴吧注册时长/吧龄')


# =====================================================
# 知乎相关表
# =====================================================

class ZhihuContent(Base):
    """知乎内容表"""
    __tablename__ = 'zhihu_content'
    __table_args__ = {'comment': '知乎内容表，存储知乎的回答、文章等内容'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    content_id = Column(String(64), index=True, comment='内容唯一ID')
    content_type = Column(Text, comment='内容类型：answer-回答，article-文章，zvideo-视频')
    content_text = Column(Text, comment='内容正文')
    content_url = Column(Text, comment='内容详情页URL')
    question_id = Column(String(255), comment='所属问题ID(回答类型)')
    title = Column(Text, comment='标题(文章)或问题标题(回答)')
    desc = Column(Text, comment='内容摘要描述')
    created_time = Column(String(32), index=True, comment='创建时间(格式化字符串)')
    updated_time = Column(Text, comment='更新时间(格式化字符串)')
    voteup_count = Column(Integer, default=0, comment='赞同数')
    comment_count = Column(Integer, default=0, comment='评论数')
    source_keyword = Column(Text, comment='搜索来源关键词')
    user_id = Column(String(255), comment='作者用户ID')
    user_link = Column(Text, comment='作者主页链接')
    user_nickname = Column(Text, comment='作者昵称')
    user_avatar = Column(Text, comment='作者头像URL')
    user_url_token = Column(Text, comment='作者URL标识')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')


class ZhihuComment(Base):
    """知乎评论表"""
    __tablename__ = 'zhihu_comment'
    __table_args__ = {'comment': '知乎评论表，存储回答和文章下的用户评论'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    comment_id = Column(String(64), index=True, comment='评论唯一ID')
    parent_comment_id = Column(String(64), comment='父评论ID，一级评论为空')
    content = Column(Text, comment='评论内容')
    publish_time = Column(String(32), index=True, comment='发布时间(格式化字符串)')
    ip_location = Column(Text, comment='评论IP属地')
    sub_comment_count = Column(Integer, default=0, comment='子评论数量')
    like_count = Column(Integer, default=0, comment='点赞数')
    dislike_count = Column(Integer, default=0, comment='踩/反对数')
    content_id = Column(String(64), index=True, comment='所属内容ID')
    content_type = Column(Text, comment='所属内容类型')
    user_id = Column(String(64), comment='评论用户ID')
    user_link = Column(Text, comment='评论用户主页链接')
    user_nickname = Column(Text, comment='评论用户昵称')
    user_avatar = Column(Text, comment='评论用户头像URL')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')


class ZhihuCreator(Base):
    """知乎用户信息表"""
    __tablename__ = 'zhihu_creator'
    __table_args__ = {'comment': '知乎用户信息表，存储知乎用户的基本资料'}

    id = Column(Integer, primary_key=True, comment='自增主键ID')
    user_id = Column(String(64), unique=True, index=True, comment='用户唯一ID')
    user_link = Column(Text, comment='用户主页链接')
    user_nickname = Column(Text, comment='用户昵称')
    user_avatar = Column(Text, comment='用户头像URL')
    url_token = Column(Text, comment='用户URL标识')
    gender = Column(Text, comment='性别')
    ip_location = Column(Text, comment='IP属地')
    follows = Column(Integer, default=0, comment='关注数')
    fans = Column(Integer, default=0, comment='粉丝数')
    anwser_count = Column(Integer, default=0, comment='回答数')
    video_count = Column(Integer, default=0, comment='视频数')
    question_count = Column(Integer, default=0, comment='提问数')
    article_count = Column(Integer, default=0, comment='文章数')
    column_count = Column(Integer, default=0, comment='专栏数')
    get_voteup_count = Column(Integer, default=0, comment='获得的赞同数')
    add_ts = Column(BigInteger, comment='数据入库时间戳(毫秒)')
    last_modify_ts = Column(BigInteger, comment='最后更新时间戳(毫秒)')
