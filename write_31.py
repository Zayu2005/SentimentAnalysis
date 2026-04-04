# -*- coding: utf-8 -*-
content = r"""# 3.1 多源异构数据采集与标准化清洗

## 一、数据结构变化全过程（JSON示例）

---

### 第1步：平台原始数据采集

各平台爬虫采集的原始数据结构完全不同，以下是各平台原始数据的JSON示例：

#### 1.1 小红书原始数据 (xhs_note)
```json
{
  "note_id": "6789012345678901234",
  "type": "normal",
  "user_id": "user_12345",
  "nickname": "小明笔记",
  "avatar": "https://头像URL",
  "ip_location": "广东",
  "title": "推荐一款好用的产品",
  "desc": "这个产品真的太好用了，强烈推荐！👍",
  "note_url": "https://www.xiaohongshu.com/discovery/item/6789012345678901234",
  "image_list": "[\"img1.jpg\", \"img2.jpg\"]",
  "video_url": "",
  "liked_count": 1234,
  "comment_count": 56,
  "share_count": 23,
  "collected_count": 89,
  "tag_list": "[\"好物分享\", \"购物笔记\"]",
  "add_ts": 1704067200000,
  "time": "2024-01-01 10:00:00"
}
```

#### 1.2 抖音原始数据 (douyin_aweme)
```json
{
  "aweme_id": "7345678901234567890",
  "aweme_type": 0,
  "user_id": "user_67890",
  "nickname": "抖音达人",
  "avatar": "https://头像URL",
  "ip_location": "北京",
  "title": "太香了",
  "desc": "这个视频太香了，一定要看！#话题",
  "aweme_url": "https://www.douyin.com/video/7345678901234567890",
  "cover_url": "https://cover.jpg",
  "video_download_url": "https://video.mp4",
  "image_list": "",
  "music_download_url": "https://music.mp3",
  "liked_count": 50000,
  "comment_count": 1234,
  "share_count": 567,
  "collected_count": 890,
  "add_ts": 1704067200000,
  "create_time": 1704000000
}
```

#### 1.3 微博原始数据 (weibo_note)
```json
{
  "note_id": "N1234567890",
  "user_id": "user_11111",
  "nickname": "微博用户",
  "avatar": "https://头像URL",
  "ip_location": "上海",
  "gender": "m",
  "content": "今天热搜第一了！这件事你怎么看？",
  "note_url": "https://m.weibo.cn/detail/1234567890",
  "liked_count": 9999,
  "comments_count": 888,
  "shared_count": 77,
  "add_ts": 1704067200000,
  "last_modify_ts": 1704067200000,
  "create_time": 1704000000
}
```

#### 1.4 B站原始数据 (bilibili_video)
```json
{
  "video_id": "BV1xx411c7mD",
  "user_id": "user_22222",
  "nickname": "B站UP主",
  "avatar": "https://头像URL",
  "title": "【实测】这款产品到底好不好用？",
  "desc": "本期视频带来详细测评...",
  "video_url": "https://www.bilibili.com/video/BV1xx411c7mD",
  "video_cover_url": "https://cover.jpg",
  "liked_count": 50000,
  "video_comment": 2345,
  "video_share_count": 123,
  "video_favorite_count": 456,
  "video_play_count": 100000,
  "video_coin_count": 789,
  "video_danmaku": 2345,
  "add_ts": 1704067200000,
  "create_time": 1704000000
}
```

---

### 第2步：适配器转换（字段映射）

通过 ContentAdapter 适配器，将异构数据转换为统一数据模型。

#### 2.1 映射配置示例

```python
XHS_MAPPING = {
    "content_id": "note_id",
    "content_type": "type",
    "content": "desc",
    "media_type": lambda r: "video" if r.get("type") == "video" else "image",
    "liked_count": "liked_count",
    ...
}

WEIBO_MAPPING = {
    "content_id": "note_id",
    "content": "content",
    "comment_count": "comments_count",
    "share_count": "shared_count",
    ...
}
```

#### 2.2 转换后的统一数据模型 (UnifiedContent)

所有平台转换后都变成统一的结构：

```json
{
  "platform": "xhs",
  "content_id": "6789012345678901234",
  "content_type": "normal",
  "user_id": "user_12345",
  "nickname": "小明笔记",
  "avatar": "https://头像URL",
  "ip_location": "广东",
  "gender": "",
  "title": "推荐一款好用的产品",
  "content": "这个产品真的太好用了，强烈推荐！👍",
  "content_url": "https://www.xiaohongshu.com/discovery/item/6789012345678901234",
  "media_type": "image",
  "cover_url": "",
  "video_url": "",
  "video_download_url": "",
  "image_list": "[\"img1.jpg\", \"img2.jpg\"]",
  "music_url": "",
  "tag_list": "[\"好物分享\", \"购物笔记\"]",
  "liked_count": 1234,
  "comment_count": 56,
  "share_count": 23,
  "collect_count": 89,
  "view_count": 0,
  "coin_count": 0,
  "danmaku_count": 0,
  "source_keyword": "",
  "original_created_at": "2024-01-01T10:00:00",
  "add_ts": 1704067200000,
  "last_modify_ts": 0
}
```

---

### 第3步：同步引擎处理

#### 3.1 增量同步查询

```sql
SELECT * FROM xhs_note WHERE add_ts > last_sync_ts ORDER BY add_ts ASC
```

#### 3.2 幂等写入

```sql
INSERT INTO unified_content (
    platform, content_id, content_type, user_id, nickname, 
    title, content, liked_count, comment_count, ...
) VALUES (...)
ON DUPLICATE KEY UPDATE
    content = VALUES(content),
    liked_count = VALUES(liked_count),
    updated_at = CURRENT_TIMESTAMP
```

---

### 第4步：统一数据存储

```json
{
  "id": 1,
  "platform": "xhs",
  "content_id": "6789012345678901234",
  "content_type": "normal",
  "user_id": "user_12345",
  "nickname": "小明笔记",
  "title": "推荐一款好用的产品",
  "content": "这个产品真的太好用了，强烈推荐！👍",
  "liked_count": 1234,
  "comment_count": 56,
  "share_count": 23,
  "collect_count": 89,
  "topic_id": null,
  "sentiment": null,
  "sentiment_score": null,
  "created_at": "2024-01-01 12:00:00",
  "updated_at": "2024-01-01 12:00:00"
}
```

---

### 第5步：数据清洗处理

#### 5.1 TextCleaner 文本清洗

输入（清洗前）：
```json
{
  "content": "这个产品真的太好用了，强烈推荐！👍 https://example.com @用户A #好物分享#",
  "title": "推荐一款好用的产品"
}
```

输出（清洗后）：
```json
{
  "content_cleaned": "这个产品真的太好用了，强烈推荐",
  "title_cleaned": "推荐一款好用的产品"
}
```

清洗过程说明：
| 步骤 | 处理前 | 处理后 |
|------|--------|--------|
| HTML清除 | `<div>内容</div>` | `内容` |
| URL移除 | `推荐 http://xx.com` | `推荐` |
| @提及移除 | `@用户名 很好` | `很好` |
| Emoji移除 | `太棒了👍` | `太棒了` |
| 话题标签 | `#好物分享#推荐` | `好物分享推荐` |
| 繁简转换 | `這個產品` | `这个产品` |
| 网络用语 | `yyds` | `永远的神` |

---

#### 5.2 Segmenter 分词

输入：
```json
{"content_cleaned": "这个产品真的太好用了"}
```

输出：
```json
{
  "segments": ["这个", "产品", "真的", "太", "好用", "了"],
  "segments_with_pos": [("这个","r"), ("产品","n"), ("真的","d"), ("太","d"), ("好用","a"), ("了","u")]
}
```

---

#### 5.3 KeywordExtractor 关键词提取

输入：
```json
{"content_cleaned": "这个产品真的太好用了，强烈推荐"}
```

输出（TF-IDF）：
```json
{
  "keywords": [
    {"word": "产品", "weight": 0.85},
    {"word": "好用", "weight": 0.72},
    {"word": "推荐", "weight": 0.55}
  ]
}
```

---

### 第6步：清洗结果存储

```json
{
  "id": 1,
  "unified_id": 1,
  "content_cleaned": "这个产品真的太好用了，强烈推荐",
  "title_cleaned": "推荐一款好用的产品",
  "segments": "[\"这个\", \"产品\", \"真的\", \"太\", \"好用\", \"了\"]",
  "keywords": "[{\"word\":\"产品\",\"weight\":0.85}]",
  "created_at": "2024-01-01 12:00:00"
}
```

---

## 二、数据结构变化总览

```
【第1步】平台原始数据
  xhs_note ─┐
  douyin_aweme ─┼─→ 异构数据结构 (note_id/aweme_id, desc/content)
  weibo_note ─┤
  bilibili_video ┘

        ↓ 适配转换

【第2步】统一数据模型 UnifiedContent
  platform | content_id | content | liked_count | ...

        ↓ 同步引擎

【第3步】统一数据表 MySQL
  unified_content (10,000+条)

        ↓ SentimentProcessor

【第4步】数据清洗
  TextCleaner → Segmenter → KeywordExtractor

        ↓

【第5步】清洗结果
  processed_content (content_cleaned, keywords, segments)
```

---

## 三、技术架构图

### 3.1 分层架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     多源异构数据采集与标准化清洗                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  【1.数据采集层】SentimentSpider                                         │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐           │
│  │小红书 │ │ 抖音  │ │ 微博  │ │  B站  │ │ 快手  │ │ 贴吧  │  ...      │
│  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘           │
│        │        │        │        │        │        │                       │
│        └────────┴────────┴────────┴────────┴────────┘                       │
│                            │                                              │
│                            ▼                                              │
│  【2.原始数据存储】各平台独立表                                           │
│  xhs_note | douyin_aweme | weibo_note | bilibili_video | ...            │
│                            │                                              │
│                            ▼                                              │
│  【3.适配转换层】Platform Adapters                                        │
│  ContentAdapter / CommentAdapter                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  XHS_MAPPING | WEIBO_MAPPING | DOUYIN_MAPPING | BILIBILI_...   │   │
│  │  (字段名映射 + lambda条件映射)                                      │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                            │                                              │
│                            ▼                                              │
│  【4.统一数据模型】UnifiedContent / UnifiedComment                        │
│  platform | content_id | content | media_type | liked_count | ...       │
│                            │                                              │
│                            ▼                                              │
│  【5.同步引擎】UnifiedDataSync                                           │
│  增量同步(基于时间戳) | 全量重建 | 幂等写入(ON DUPLICATE KEY)            │
│                            │                                              │
│                            ▼                                              │
│  【6.统一数据存储】MySQL                                                  │
│  unified_content (10,000+条) | unified_comment (50,000+条)            │
│                            │                                              │
│                            ▼                                              │
│  【7.数据清洗层】SentimentProcessor                                      │
│  TextCleaner | Segmenter | KeywordExtractor                            │
│  HTML清除 | URL移除 | @提及移除 | Emoji移除 | 繁简转换 | jieba分词     │
│                            │                                              │
│                            ▼                                              │
│  【8.清洗结果存储】processed_content                                      │
│  content_cleaned | title_cleaned | keywords | segments                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Mermaid流程图

```mermaid
graph LR
    subgraph Crawl[数据采集层]
        C1[小红书爬虫]
        C2[抖音爬虫]
        C3[微博爬虫]
        C4[B站爬虫]
    end

    subgraph RawDB[原始数据]
        R1[xhs_note]
        R2[douyin_aweme]
        R3[weibo_note]
        R4[bilibili_video]
    end

    subgraph Adapter[适配转换层]
        A1[ContentAdapter]
        M[字段映射配置]
    end

    subgraph Unified[统一数据模型]
        U[UnifiedContent]
    end

    subgraph Sync[同步引擎]
        S1[增量同步]
        S2[幂等写入]
    end

    subgraph Clean[数据清洗层]
        T1[TextCleaner]
        T2[Segmenter]
        T3[KeywordExtractor]
    end

    C1 --> R1
    C2 --> R2
    C3 --> R3
    C4 --> R4

    R1 --> A1
    R2 --> A1
    R3 --> A1
    R4 --> A1

    A1 --> M
    M --> U
    U --> S1
    S1 --> S2
    S2 --> T1
    T1 --> T2
    T2 --> T3
```

---

## 四、创新点总结

### 4.1 设计模式创新：双层适配器架构

- 传统方式：每个平台写独立转换代码，代码重复
- 本系统方式：配置化映射 + 统一适配器

### 4.2 映射机制创新：支持条件映射

```python
# 支持lambda条件映射
"media_type": lambda r: "video" if r.get("type") == "video" else "image"
"view_count": lambda r: 0
```

### 4.3 同步策略创新

- 增量同步：基于时间戳自动识别新数据
- 全量重建：支持清空后重新同步
- 幂等写入：ON DUPLICATE KEY UPDATE 防止重复

---

## 五、完整JSON变化示例

### 输入：微博原始数据
```json
{
  "note_id": "N1234567890",
  "user_id": "user_11111",
  "content": "今天热搜第一了！这件事你怎么看？https://t.cn/xxx @明星 #热搜",
  "comments_count": 888,
  "shared_count": 77,
  "liked_count": 9999,
  "add_ts": 1704067200000
}
```

### 中间：UnifiedContent
```json
{
  "platform": "wb",
  "content_id": "N1234567890",
  "content": "今天热搜第一了！这件事你怎么看？https://t.cn/xxx @明星 #热搜",
  "comment_count": 888,
  "share_count": 77,
  "liked_count": 9999
}
```

### 中间：清洗后
```json
{
  "content_cleaned": "今天热搜第一了！这件事你怎么看？"
}
```

### 最终：预处理结果
```json
{
  "content_cleaned": "今天热搜第一了！这件事你怎么看？",
  "segments": ["今天", "热搜", "第一", "了", "这", "件", "事", "你", "怎么", "看"],
  "keywords": [
    {"word": "热搜", "weight": 0.92},
    {"word": "第一", "weight": 0.65}
  ]
}
```

---

"""

with open("3.1_多源异构数据采集与标准化清洗.md", "w", encoding="utf-8") as f:
    f.write(content)
print("File created successfully")
